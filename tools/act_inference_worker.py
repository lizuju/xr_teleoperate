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
from teleop.utils.act_runtime import decode_guide, decode_request, encode_response, read_exact, request_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--rtc", action="store_true",
                        help="Real-time chunking: every request ends with a guide (prefix and weights) the flow "
                             "integration continues")
    parser.add_argument("--rtc-beta", type=float, default=5.0, help="Largest real-time chunking guidance weight")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, data = load_policy(args.checkpoint, device)
    config, dataset = data["config"], data["dataset"]
    head = config.get("head", "act")
    if args.rtc and head != "flow":
        print("[ACT WORKER] real-time chunking needs a flow policy", file=sys.stderr, flush=True)
        return 2
    if args.rtc and not args.rtc_beta > 0:
        print("[ACT WORKER] --rtc-beta must be positive", file=sys.stderr, flush=True)
        return 2
    # Guidance differentiates through the model's inputs only.
    model.requires_grad_(False)
    predict = ChunkPredictor(model, data["normalization"], device)
    image_hw = tuple(dataset["image_size_hw"])
    state_dim, cameras = config["state_dim"], config["num_cameras"]
    context_dim = config["context_dim"]
    chunk_size, action_dim = config["chunk_size"], config["action_dim"]
    # The first passes are slow on any device; do them before the runner starts timing.
    warm = (np.zeros((cameras, *image_hw, 3), dtype=np.uint8), np.asarray(data["normalization"]["qpos_mean"], dtype=np.float32),
            np.zeros(context_dim, dtype=np.float32))
    for _ in range(3):
        predict(*warm)
        if args.rtc:
            predict.guided(*warm, np.asarray([data["normalization"]["action_mean"]] * chunk_size, dtype=np.float32),
                           np.ones(chunk_size, dtype=np.float32), args.rtc_beta)
    output = sys.stdout.buffer
    output.write((json.dumps({"device": device, "chunk_size": chunk_size, "action_dim": action_dim,
                              "state_dim": state_dim, "context_dim": context_dim, "num_cameras": cameras, "step": data["step"],
                              "head": head, "rtc": args.rtc, "rtc_beta": args.rtc_beta if args.rtc else None,
                              "dataset": dataset, "normalization": data["normalization"]}) + "\n").encode())
    output.flush()
    size = request_size(state_dim, cameras, image_hw, context_dim, (chunk_size, action_dim) if args.rtc else None)
    while True:
        payload = read_exact(sys.stdin.buffer, size)
        if payload is None:
            return 0
        qpos, images, context = decode_request(payload, state_dim, cameras, image_hw, context_dim)
        started = time.perf_counter()
        if args.rtc:
            prefix, weights = decode_guide(payload, chunk_size, action_dim)
            chunk, complete = predict.guided(images, qpos, context, prefix, weights, args.rtc_beta)
        else:
            chunk, complete = predict(images, qpos, context)
        if not np.isfinite(chunk).all():
            print("[ACT WORKER] policy produced non-finite actions", file=sys.stderr, flush=True)
            return 3
        output.write(encode_response((time.perf_counter() - started) * 1e3, chunk, complete))
        output.flush()


if __name__ == "__main__":
    sys.exit(main())
