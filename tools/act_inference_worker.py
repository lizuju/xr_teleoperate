#!/usr/bin/env python3
"""ACT inference in its own process. Fixed-size binary frames on stdin/stdout; see act_runtime."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop.utils.act_policy import ChunkPredictor, load_policy
from teleop.utils.act_runtime import decode_request, encode_response, read_exact, request_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, data = load_policy(args.checkpoint, device)
    predict = ChunkPredictor(model, data["normalization"], device)
    config, dataset = data["config"], data["dataset"]
    image_hw = tuple(dataset["image_size_hw"])
    state_dim, cameras = config["state_dim"], config["num_cameras"]
    # The first passes are slow on any device; do them before the runner starts timing.
    for _ in range(3):
        predict(np.zeros((cameras, *image_hw, 3), dtype=np.uint8), np.asarray(data["normalization"]["qpos_mean"], dtype=np.float32))
    output = sys.stdout.buffer
    output.write((json.dumps({"device": device, "chunk_size": config["chunk_size"], "action_dim": config["action_dim"],
                              "state_dim": state_dim, "num_cameras": cameras, "step": data["step"],
                              "dataset": dataset, "normalization": data["normalization"]}) + "\n").encode())
    output.flush()
    size = request_size(state_dim, cameras, image_hw)
    while True:
        payload = read_exact(sys.stdin.buffer, size)
        if payload is None:
            return 0
        qpos, images = decode_request(payload, state_dim, cameras, image_hw)
        started = time.perf_counter()
        chunk = predict(images, qpos)
        if not np.isfinite(chunk).all():
            print("[ACT WORKER] policy produced non-finite actions", file=sys.stderr, flush=True)
            return 3
        output.write(encode_response((time.perf_counter() - started) * 1e3, chunk))
        output.flush()


if __name__ == "__main__":
    sys.exit(main())
