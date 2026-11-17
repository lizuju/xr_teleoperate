#!/usr/bin/env python3
"""Create a corrected copy of a legacy 24D ACT dataset without rewriting source recordings."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop.utils.act_contract import validate_contract
from teleop.utils.camera_color import IMAGE_COLOR_PROFILE, recorded_red_blue_swaps
from teleop.utils.icecream_task import source_digest


def correct_dataset(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    original = json.loads((source / "dataset.json").read_text())
    if original.get("schema") != "r1_act_hdf5_v4" or original.get("status") != "complete":
        raise ValueError("Expected a complete v4 ACT export")
    if original.get("image_color_profile") is not None:
        raise ValueError("Dataset already has a color profile; do not apply a second correction")
    if output.exists() or source in output.parents:
        raise ValueError("Choose a new output outside the source dataset")
    dataset = copy.deepcopy(original)
    dataset["image_color_profile"] = IMAGE_COLOR_PROFILE
    validate_contract(dataset)
    changes = {}
    for entry in dataset["sources"]:
        directory = Path(entry["episode"])
        if (source_digest(directory / "episode.json") != entry["manifest_sha256"]
                or source_digest(directory / "frames.jsonl") != entry["frames_sha256"]):
            raise ValueError(f"Source recording changed since export: {directory}")
        info = json.loads((directory / "episode.json").read_text())["info"]
        changes[entry["source_id"]] = recorded_red_blue_swaps(info)
        entry["red_blue_swapped_on_export"] = list(changes[entry["source_id"]])
    if not any(changes.values()):
        raise ValueError("No verified legacy right Dex1 images to correct")
    required = sum((source / e["file"]).stat().st_size for e in dataset["episodes"])
    if shutil.disk_usage(output.parent).free < required + 1024**3:
        raise ValueError("Not enough space for a separate corrected dataset plus 1 GiB reserve")
    output.mkdir()
    shutil.copytree(source / "quality", output / "quality")
    dataset.update(status="building", derived_from={
        "dataset": str(source), "manifest_sha256": source_digest(source / "dataset.json"),
        "operation": "Swap right_wrist red/blue only for verified legacy Dex1 source versions; no JPEG recompression"})
    manifest = output / "dataset.json"
    manifest.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")
    try:
        for number, episode in enumerate(dataset["episodes"]):
            path = source / episode["file"]
            temporary = output / ("." + episode["file"] + ".tmp")
            if "color_3" not in changes[episode["source_id"]]:
                shutil.copy2(path, temporary)
                with h5py.File(temporary, "r+") as h:
                    h.attrs["image_color_profile"] = IMAGE_COLOR_PROFILE
            else:
                with h5py.File(path, "r") as old, h5py.File(temporary, "w") as new:
                    new.attrs.update(dict(old.attrs), image_color_profile=IMAGE_COLOR_PROFILE)
                    for name in old:
                        if name != "observations":
                            old.copy(name, new)
                    observations = new.create_group("observations")
                    observations.attrs.update(dict(old["observations"].attrs))
                    for name in old["observations"]:
                        if name != "images":
                            old.copy("observations/" + name, observations, name=name)
                    images = observations.create_group("images")
                    images.attrs.update(dict(old["observations/images"].attrs))
                    for name in old["observations/images"]:
                        if name != "right_wrist":
                            old.copy("observations/images/" + name, images, name=name)
                    raw = old["observations/images/right_wrist"]
                    fixed = images.create_dataset("right_wrist", shape=raw.shape, dtype=raw.dtype,
                                                  chunks=raw.chunks, compression=raw.compression,
                                                  compression_opts=raw.compression_opts)
                    fixed.attrs.update(dict(raw.attrs))
                    for begin in range(0, len(raw), 32):
                        expected = raw[begin:begin + 32][..., ::-1]
                        fixed[begin:begin + 32] = expected
                        if not np.array_equal(fixed[begin:begin + 32], expected):
                            raise RuntimeError(f"Color write verification failed: {path}")
            temporary.replace(output / episode["file"])
            if number % 20 == 0 or number + 1 == len(dataset["episodes"]):
                print(f"[COLOR] {number + 1}/{len(dataset['episodes'])} segments", flush=True)
        dataset["status"] = "complete"
    except Exception as error:
        dataset.update(status="failed", error=str(error))
        manifest.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")
        raise
    temporary = output / ".dataset.json.tmp"
    temporary.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(manifest)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = correct_dataset(args.source, args.output)
    print(f"[COLOR] {result['status']}: {args.output}")
