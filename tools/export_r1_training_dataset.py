#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from fnmatch import fnmatch
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from check_teleop_episode import (DEFAULT_MIN_SEGMENT_FRAMES, TRAINING_CAMERAS, TRAINING_GROUPS,
                                 TRAINING_TIME_LIMITS, episode_file, training_frame_rejections, training_observation,
                                 training_action, training_commands, training_hand_inputs, training_timing, validate_episode)
from teleop.utils.act_runtime import IMAGE_HW, prepare_training_image


def export_dataset(source, output, min_frames=DEFAULT_MIN_SEGMENT_FRAMES, val_sources=()):
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
        "schema": "r1_act_hdf5_v3", "status": "building", "source": str(source),
        "state_dim": sum(TRAINING_GROUPS.values()), "action_dim": sum(TRAINING_GROUPS.values()),
        "joint_groups": TRAINING_GROUPS, "camera_names": list(TRAINING_CAMERAS.values()),
        "image_size_hw": list(IMAGE_HW), "image_color_order": "RGB",
        "training_inputs": ["observations/qpos", "observations/images"], "imu_used": False,
        "action_semantics": "Bounded causal accepted-request alignment from sample.action_alignment: the latest historical arm target accepted at or before the completed hand request. Arm targets already include arm target shaping; hand targets precede hand smoothing. Only older rows without this block use original same-row actions. Actual acceptance timestamps are retained; no interpolation or fabricated times.",
        "observation_semantics": "Camera-source-time paired images and aligned_states feedback, all acquired and received at or before the completed hand request. Software timestamps, not exposure synchronization.",
        "timing": "Original samples and measured intervals retained; no interpolation. A segment is a run of usable actions, split at unusable actions and gaps >1.5 nominal periods.",
        "anchor_semantics": "observations/* and the per-observation source/* datasets hold only rows that passed every training check. source/anchor_index gives each one's row in action, timestamp and the per-row source/* datasets. A row whose observation failed still contributes its real action to earlier chunks but is never trained from.",
        "deployment_contract": "Reproduce recorded source-time observation pairing and freshness checks. Consume the already accepted targets at the recorded action anchor, preserving publisher limits and hand smoothing safeguards.",
        "hand_request_time": "Recorded paired-target acceptance time and sequence, linked to successful hand publications; input age is checked against hand_target_inputs when present. Not physical execution time.",
        "time_limits_ms": TRAINING_TIME_LIMITS.copy(),
        "qvel_semantics": "Finite differences of aligned qpos over decision ticks, including repeated observations; not physical joint velocity or a training input.",
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
            report = validate_episode(directory, min_segment_frames=min_frames)
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
            for segment in report["training_no_imu"]["action_segments"]:
                begin, end = segment["start_idx"], segment["stop_idx"]
                selected = rows[begin:end]
                count = len(selected)
                anchor_index = [i for i, row in enumerate(selected) if not training_frame_rejections(row, info)]
                anchors = [selected[i] for i in anchor_index]
                if count < min_frames:
                    entry["short_segment_frames"] += len(anchors)
                    continue
                if not anchors:
                    continue
                episode_id = len(dataset["episodes"])
                filename = f"episode_{episode_id}.hdf5"
                temporary = output / ("." + filename + ".tmp")
                observations = [training_observation(r) for r in anchors]
                qpos = np.array([sum((o[g] for g in TRAINING_GROUPS), [])
                                 for o in observations], dtype=np.float32)
                aligned_actions = [training_action(r) for r in selected]
                actions = np.array([sum((action[g] for g in TRAINING_GROUPS), [])
                                    for action in aligned_actions], dtype=np.float32)
                if not np.isfinite(qpos).all() or not np.isfinite(actions).all():
                    raise ValueError(f"{directory}: joint values cannot be represented as finite float32")
                stamps = np.array([r["sample"]["monotonic_ns"] for r in selected], dtype=np.int64)
                seconds = (stamps - stamps[0]).astype(np.float64) / 1e9
                with h5py.File(temporary, "w") as h:
                    h.attrs.update(sim=False, compress=False, fps=frequency, imu_used=False,
                                   action_semantics=dataset["action_semantics"], color_order="RGB", source_episode=str(directory))
                    h.create_dataset("observations/qpos", data=qpos)
                    h.create_dataset("observations/qvel", data=(np.gradient(qpos, seconds[anchor_index], axis=0)
                                                                if len(anchors) > 1 else np.zeros_like(qpos)).astype(np.float32))
                    h.create_dataset("source/anchor_index", data=anchor_index, dtype="int64")
                    h["observations/qvel"].attrs["meaning"] = dataset["qvel_semantics"]
                    h.create_dataset("action", data=actions)
                    h.create_dataset("timestamp", data=seconds)
                    h.create_dataset("source/frame_index", data=[r["idx"] for r in selected])
                    h.create_dataset("source/monotonic_ns", data=stamps)
                    commands = [training_commands(r["sample"]) for r in selected]
                    h.create_dataset("source/arm_request_ns", data=[c["arm"]["requested"]["monotonic_ns"] for c in commands])
                    h.create_dataset("source/hand_request_ns", data=[c["hands"]["requested"]["monotonic_ns"] for c in commands])
                    h.create_dataset("source/hand_request_sequence", data=[c["hands"]["requested"]["sequence"] for c in commands])
                    h.create_dataset("source/arm_request_sequence", data=[c["arm"]["requested"].get("sequence", 0)
                                                                         for c in commands]).attrs["missing_value"] = 0
                    h.create_dataset("source/action_alignment_used", data=["action_alignment" in r["sample"] for r in selected])
                    h.create_dataset("source/action_anchor_ns", data=[r["sample"]["action_alignment"]["anchor_monotonic_ns"]
                                     if "action_alignment" in r["sample"] else r["sample"]["monotonic_ns"] for r in selected])
                    timings = [training_timing(r["sample"]) for r in anchors]
                    inputs = [training_hand_inputs(r["sample"]) for r in anchors]
                    input_stamps = h.create_dataset("source/hand_input_ns", data=[
                        [(value.get(side) or {}).get("received_monotonic_ns", 0) for side in ("left", "right")]
                        for value in inputs])
                    input_stamps.attrs.update(sides=["left", "right"], missing_value=0)
                    input_ages = h.create_dataset("source/hand_input_age_ms", data=[
                        [t[side + "_hand_input_age_ms"] if t[side + "_hand_input_age_ms"] is not None else -1.0
                         for side in ("left", "right")] for t in timings])
                    input_ages.attrs.update(sides=["left", "right"], missing_value=-1.0)
                    h.create_dataset("source/observation_ns", data=[t["observation_ns"] for t in timings])
                    for kind in ("camera_ns", "feedback_ns", "request_ns"):
                        for name in timings[0][kind]:
                            h.create_dataset("source/" + kind + "/" + name, data=[t[kind][name] for t in timings])
                    for metric in ("observation_skew_ms", "observation_age_ms", "request_skew_ms",
                                   "arm_observation_delay_ms", "hands_observation_delay_ms"):
                        h.create_dataset("source/" + metric, data=[t[metric] for t in timings])
                    for name in timings[0]["camera_ns"]:
                        for field in ("source_monotonic_ns", "clock_offset_ns", "clock_uncertainty_ns",
                                      "clock_measured_monotonic_ns"):
                            h.create_dataset("source/camera_timing/" + name + "/" + field,
                                             data=[r["sample"]["sources"][name]["timing"][field] for r in anchors])
                        for field in ("clock_id", "timestamp_kind"):
                            h.create_dataset("source/camera_timing/" + name + "/" + field,
                                             dtype=h5py.string_dtype("utf-8"),
                                             data=[r["sample"]["sources"][name]["timing"].get(field, "unknown")
                                                   for r in anchors])
                    h.create_dataset("source/camera_timing/image/stereo_skew_ns",
                                     data=[r["sample"]["sources"]["image"]["timing"]["stereo_skew_ns"]
                                           for r in anchors])
                    for name in ("robot", "left_hand_feedback", "right_hand_feedback", "image", "left_wrist_image", "right_wrist_image"):
                        h.create_dataset("source/received_ns/" + name,
                                         data=[r["sample"]["sources"][name]["received_monotonic_ns"] for r in anchors])
                    for key, camera in TRAINING_CAMERAS.items():
                        images = h.create_dataset("observations/images/" + camera,
                                                  shape=(len(anchors), *IMAGE_HW, 3), dtype="uint8",
                                                  chunks=(1, *IMAGE_HW, 3), compression="gzip", compression_opts=1)
                        for i, row in enumerate(anchors):
                            relative = row["colors"][key]
                            if relative not in image_cache:
                                pixels = cv2.imread(str(episode_file(directory, relative)), cv2.IMREAD_COLOR)
                                if pixels is None:
                                    raise ValueError(f"Unreadable image: {directory / relative}")
                                image_cache[relative] = prepare_training_image(pixels)
                                if len(image_cache) > 16:
                                    image_cache.popitem(last=False)
                            images[i] = image_cache[relative]
                            image_cache.move_to_end(relative)
                temporary.replace(output / filename)
                dataset["episodes"].append({"episode_id": episode_id, "file": filename, "source_id": source_id,
                                            "start_idx": begin, "stop_idx": end, "frames": count,
                                            "anchors": len(anchors),
                                            "duration_s": float(seconds[-1])})
                entry["exported_frames"] += len(anchors)
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
        if val_sources:
            # Named holdouts, normally a whole capture session, replace the random pick.
            val_ids = {i for i in source_ids if any(fnmatch(Path(dataset["sources"][i]["episode"]).name, pattern)
                                                    for pattern in val_sources)}
            if not val_ids or len(val_ids) == len(source_ids):
                raise ValueError(f"--val-source must leave both splits non-empty; matched {len(val_ids)} of {len(source_ids)}")
            dataset["split_unit"] = "named_source"
        for episode in dataset["episodes"]:
            split = "val" if episode["source_id"] in val_ids else "train"
            dataset["splits"][split].append(episode["episode_id"])
        if not val_ids:
            dataset["validation_note"] = "Only one source episode; no independent validation set. Collect more episodes."
        values = {key: [] for key in ("qpos", "action")}
        for episode_id in dataset["splits"]["train"]:
            with h5py.File(output / f"episode_{episode_id}.hdf5", "r") as h:
                values["qpos"].append(h["observations/qpos"][:].astype(np.float64))
                values["action"].append(h["action"][:].astype(np.float64))
        dataset["normalization"] = {"split": "train", "frames": sum(map(len, values["qpos"]))}
        for key, parts in values.items():
            stacked = np.concatenate(parts)
            dataset["normalization"][key + "_mean"] = stacked.mean(axis=0).tolist()
            dataset["normalization"][key + "_std"] = np.maximum(stacked.std(axis=0), .01).tolist()
            dataset["normalization"][key + "_min"] = stacked.min(axis=0).tolist()
            dataset["normalization"][key + "_max"] = stacked.max(axis=0).tolist()
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
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_SEGMENT_FRAMES,
                        help=f"Minimum consecutive usable actions per segment (default: {DEFAULT_MIN_SEGMENT_FRAMES})")
    parser.add_argument("--val-source", action="append", default=[], metavar="GLOB",
                        help="Hold out source episodes whose directory name matches, e.g. '*_20261107_*'. "
                             "Default: a random 20%% of sources")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = export_dataset(args.source, args.output, args.min_frames, args.val_source)
    print(f"[EXPORT] 完成: {len(result['episodes'])} 个连续片段; train={len(result['splits']['train'])}, val={len(result['splits']['val'])}; IMU 已排除")
    print(args.output.resolve() / "dataset.json")


if __name__ == "__main__":
    main()
