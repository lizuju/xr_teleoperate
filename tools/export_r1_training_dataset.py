#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random

import cv2
import h5py
import numpy as np

from check_teleop_episode import (TRAINING_CAMERAS, TRAINING_GROUPS,
                                 episode_file, validate_episode)


IMAGE_HW = (240, 320)


def export_dataset(source, output, min_frames=40):
    source, output = Path(source).resolve(), Path(output).resolve()
    if min_frames < 2:
        raise ValueError("min_frames must be at least 2")
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the original recording directory")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    manifests = ([source / "episode.json"] if (source / "episode.json").is_file()
                 else sorted(source.glob("episode_*/episode.json")))
    if not manifests:
        raise ValueError("No episodes found; pass one task directory or one episode directory")
    output.mkdir(parents=True)
    (output / "quality").mkdir()
    dataset = {
        "schema": "r1_act_hdf5_v1", "status": "building", "source": str(source),
        "state_dim": sum(TRAINING_GROUPS.values()), "action_dim": sum(TRAINING_GROUPS.values()),
        "joint_groups": TRAINING_GROUPS, "camera_names": list(TRAINING_CAMERAS.values()),
        "image_size_hw": list(IMAGE_HW), "image_color_order": "RGB",
        "training_inputs": ["observations/qpos", "observations/images"], "imu_used": False,
        "action_semantics": "Same-row requested qpos before publisher limiting and hand smoothing; no time shift.",
        "observation_semantics": "Same-row latest received states at the control tick; NOT aligned_states at camera time.",
        "timing": "Original samples and measured intervals retained; no interpolation. Split at rejected rows and gaps >1.5 nominal periods.",
        "deployment_contract": "Policy outputs must enter the same target shaping/limiting and hand smoothing path as recorded requests; not direct motor writes.",
        "hand_request_time": "No separate hand-request timestamp in source; observed at sample time. Publication times are not request times.",
        "qvel_semantics": "Offline finite differences of qpos using actual sample times; not measured velocity and not a training input.",
        "image_transform": "OpenCV INTER_AREA resize from each original image to 320x240, then BGR to RGB; original calibration needs the recorded x/y pixel scaling.",
        "min_segment_frames": min_frames, "sources": [], "episodes": [],
        "splits": {"train": [], "val": []}, "split_unit": "original_episode", "split_seed": 42,
    }
    manifest_path = output / "dataset.json"

    def save_manifest():
        temporary = output / ".dataset.json.tmp"
        temporary.write_text(json.dumps(dataset, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(manifest_path)

    save_manifest()
    canonical_signature = None
    try:
        for source_id, path in enumerate(manifests):
            directory = path.parent
            manifest = json.loads(path.read_text())
            report = validate_episode(directory)
            report_path = Path("quality") / f"source_{source_id:04d}.json"
            (output / report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            entry = {"source_id": source_id, "episode": str(directory), "outcome": manifest.get("outcome"),
                     "quality_report": str(report_path), "exported_frames": 0, "short_segment_frames": 0,
                     "text": manifest.get("text"),
                     "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "frames_sha256": hashlib.sha256((directory / "frames.jsonl").read_bytes()).hexdigest()
                     if (directory / "frames.jsonl").is_file() else None}
            dataset["sources"].append(entry)
            if not report["valid"] or manifest.get("status") != "complete":
                entry["excluded_reason"] = "invalid_or_incomplete"
                continue
            if manifest.get("outcome") != "success":
                entry["excluded_reason"] = "not_success"
                continue
            info = manifest["info"]
            if info.get("robot") != "R1_A7" or info.get("end_effector") != "linker_o6":
                entry["excluded_reason"] = "unsupported_robot"
                continue
            frequency = info.get("frequency")
            if not isinstance(frequency, (int, float)) or frequency <= 0:
                raise ValueError(f"{directory}: missing nominal frequency")
            names = info.get("joint_names") or {}
            if any(len(names.get(group, [])) != size for group, size in TRAINING_GROUPS.items()):
                raise ValueError(f"{directory}: expected joint group sizes 7/7/6/6/3")
            joint_names = [f"{group}/{name}" for group in TRAINING_GROUPS for name in names[group]]
            signature = (frequency, tuple(joint_names))
            if canonical_signature is None:
                canonical_signature = signature
                dataset.update(fps=frequency, joint_names=joint_names, task=source.parent.name if (source / "episode.json").is_file() else source.name,
                               units=["normalized_0_to_1" if "ee" in group else "rad"
                                      for group, size in TRAINING_GROUPS.items() for _ in range(size)])
            elif signature != canonical_signature:
                raise ValueError("Mixed frequency or joint order; export these recordings separately")
            entry["timing_basis"] = report["training_no_imu"]["timing_basis"]
            entry["calibration"] = info.get("camera_calibration")
            entry["launch_arguments"] = info.get("launch_arguments")
            entry["code_sha256"] = info.get("code_sha256")
            entry["image_pixel_scale_xy"] = {
                key: [IMAGE_HW[1] / spec["width"], IMAGE_HW[0] / spec["height"]]
                for key, spec in info.get("images", {}).items() if key in TRAINING_CAMERAS}
            rows = [json.loads(line) for line in (directory / "frames.jsonl").open()]
            image_cache = OrderedDict()
            for segment in report["training_no_imu"]["segments"]:
                begin, end = segment["start_idx"], segment["stop_idx"]
                selected = rows[begin:end]
                count = len(selected)
                if count < min_frames:
                    entry["short_segment_frames"] += count
                    continue
                episode_id = len(dataset["episodes"])
                filename = f"episode_{episode_id}.hdf5"
                temporary = output / ("." + filename + ".tmp")
                qpos = np.array([sum((r["states"][g]["qpos"] for g in TRAINING_GROUPS), [])
                                 for r in selected], dtype=np.float32)
                actions = np.array([sum((r["actions"][g]["qpos"] for g in TRAINING_GROUPS), [])
                                    for r in selected], dtype=np.float32)
                if not np.isfinite(qpos).all() or not np.isfinite(actions).all():
                    raise ValueError(f"{directory}: joint values cannot be represented as finite float32")
                stamps = np.array([r["sample"]["monotonic_ns"] for r in selected], dtype=np.int64)
                seconds = (stamps - stamps[0]).astype(np.float64) / 1e9
                with h5py.File(temporary, "w") as h:
                    h.attrs.update(sim=False, compress=False, fps=frequency, imu_used=False,
                                   action_time_shift=0, color_order="RGB", source_episode=str(directory))
                    h.create_dataset("observations/qpos", data=qpos)
                    h.create_dataset("observations/qvel", data=np.gradient(qpos, seconds, axis=0).astype(np.float32))
                    h["observations/qvel"].attrs["meaning"] = dataset["qvel_semantics"]
                    h.create_dataset("action", data=actions)
                    h.create_dataset("timestamp", data=seconds)
                    h.create_dataset("source/frame_index", data=[r["idx"] for r in selected])
                    h.create_dataset("source/monotonic_ns", data=stamps)
                    h.create_dataset("source/arm_request_ns", data=[r["sample"]["commands"]["arm"]["requested"]["monotonic_ns"] for r in selected])
                    for name in ("robot", "left_hand_feedback", "right_hand_feedback", "image", "left_wrist_image", "right_wrist_image"):
                        h.create_dataset("source/received_ns/" + name,
                                         data=[r["sample"]["sources"][name]["received_monotonic_ns"] for r in selected])
                    for key, camera in TRAINING_CAMERAS.items():
                        images = h.create_dataset("observations/images/" + camera,
                                                  shape=(count, *IMAGE_HW, 3), dtype="uint8",
                                                  chunks=(1, *IMAGE_HW, 3), compression="gzip", compression_opts=1)
                        for i, row in enumerate(selected):
                            relative = row["colors"][key]
                            if relative not in image_cache:
                                pixels = cv2.imread(str(episode_file(directory, relative)), cv2.IMREAD_COLOR)
                                if pixels is None:
                                    raise ValueError(f"Unreadable image: {directory / relative}")
                                pixels = cv2.resize(pixels, IMAGE_HW[::-1], interpolation=cv2.INTER_AREA)
                                image_cache[relative] = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                                if len(image_cache) > 16:
                                    image_cache.popitem(last=False)
                            images[i] = image_cache[relative]
                            image_cache.move_to_end(relative)
                temporary.replace(output / filename)
                dataset["episodes"].append({"episode_id": episode_id, "file": filename, "source_id": source_id,
                                            "start_idx": begin, "stop_idx": end, "frames": count,
                                            "duration_s": float(seconds[-1])})
                entry["exported_frames"] += count
            if entry["exported_frames"] == 0:
                entry["excluded_reason"] = "no_long_enough_eligible_segment"
            print(f"[EXPORT] {directory.name}: {entry['exported_frames']} frames", flush=True)
            save_manifest()
        if not dataset["episodes"]:
            raise ValueError("No eligible segments exported; see quality reports and excluded_reason in dataset.json")
        source_ids = sorted({e["source_id"] for e in dataset["episodes"]})
        random.Random(42).shuffle(source_ids)
        val_count = max(1, round(len(source_ids) * .2)) if len(source_ids) > 1 else 0
        val_ids = set(source_ids[:val_count])
        for episode in dataset["episodes"]:
            split = "val" if episode["source_id"] in val_ids else "train"
            dataset["splits"][split].append(episode["episode_id"])
        if not val_ids:
            dataset["validation_note"] = "Only one source episode; no independent validation set. Collect more episodes."
        sums = {k: np.zeros(dataset["state_dim"], dtype=np.float64) for k in ("qpos", "action")}
        squares = {k: value.copy() for k, value in sums.items()}
        count = 0
        for episode_id in dataset["splits"]["train"]:
            with h5py.File(output / f"episode_{episode_id}.hdf5", "r") as h:
                for key, path in (("qpos", "observations/qpos"), ("action", "action")):
                    values = h[path][:].astype(np.float64)
                    sums[key] += values.sum(axis=0)
                    squares[key] += (values * values).sum(axis=0)
                count += len(h["action"])
        dataset["normalization"] = {"split": "train", "frames": count}
        for key in sums:
            mean = sums[key] / count
            std = np.maximum(np.sqrt(np.maximum(squares[key] / count - mean * mean, 0)), .01)
            dataset["normalization"][key + "_mean"] = mean.tolist()
            dataset["normalization"][key + "_std"] = std.tolist()
        dataset["status"] = "complete"
        save_manifest()
        return dataset
    except Exception as exc:
        dataset["status"] = "failed"
        dataset["error"] = str(exc)
        save_manifest()
        raise


def main():
    parser = argparse.ArgumentParser(description="Export successful R1/O6 demonstrations to ACT HDF5; no robot connection, no IMU inputs.")
    parser.add_argument("source", type=Path, help="One task directory or one episode directory")
    parser.add_argument("--output", type=Path, required=True, help="A new directory outside the recordings")
    parser.add_argument("--min-frames", type=int, default=40, help="Minimum consecutive eligible samples per segment (default: 40)")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = export_dataset(args.source, args.output, args.min_frames)
    print(f"[EXPORT] 完成: {len(result['episodes'])} 个连续片段; train={len(result['splits']['train'])}, val={len(result['splits']['val'])}; IMU 已排除")
    print(args.output.resolve() / "dataset.json")


if __name__ == "__main__":
    main()
