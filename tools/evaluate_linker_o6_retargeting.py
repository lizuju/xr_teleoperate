#!/usr/bin/env python3
"""Offline O6 optimizer checks; no DDS, serial, Vuer or simulator is started."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TIP_INDICES = np.array([4, 9, 14, 19, 24])
ROBOT_FROM_WRAPPER = np.array([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])


def forward_hand(hand, normalized):
    sequence = hand.retargeting
    robot = sequence.optimizer.robot
    qpos = np.zeros(robot.dof)
    indices = [sequence.joint_names.index(name) for name in hand.hardware_joint_order]
    qpos[indices] = hand.hardware_lower + np.asarray(normalized) * (hand.hardware_upper - hand.hardware_lower)
    qpos = sequence.optimizer.adaptor.forward_qpos(qpos)
    robot.compute_forward_kinematics(qpos)
    tips = np.array([robot.get_link_pose(robot.get_link_index(name))[:3, 3] for name in hand.tip_link_names])
    return tips, qpos


def synthetic_hand_points(hand, normalized):
    tips, _ = forward_hand(hand, normalized)
    robot = hand.retargeting.optimizer.robot
    prefix = hand.hardware_joint_order[0].split("_", 1)[0] + "_"
    points = np.zeros((25, 3))
    points[0] = robot.get_link_pose(robot.get_link_index(hand.wrist_link_name))[:3, 3]
    for index, link in enumerate(("thumb_metacarpals_base2", "thumb_metacarpals", "thumb_distal"), start=1):
        points[index] = robot.get_link_pose(robot.get_link_index(prefix + link))[:3, 3]
    points[4] = tips[0]
    for finger, start, tip in zip(("index", "middle", "ring", "pinky"), (5, 10, 15, 20), tips[1:]):
        proximal = robot.get_link_pose(robot.get_link_index(prefix + finger + "_proximal"))[:3, 3]
        distal = robot.get_link_pose(robot.get_link_index(prefix + finger + "_distal"))[:3, 3]
        points[start:start + 5] = [
            (points[0] + proximal) * .5, proximal, distal, (distal + tip) * .5, tip
        ]
    return points @ ROBOT_FROM_WRAPPER


def load_recording(path):
    frames = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("armed") is False:
            continue
        if "left_hand_points" not in row or "right_hand_points" not in row:
            raise ValueError(
                f"Line {line_number} has no recorded left_hand_points/right_hand_points. "
                "Old normalized targets cannot be used to evaluate a new retargeting algorithm."
            )
        frame = row.get("hand_points_frame")
        if frame != "televuer_unitree_hand_wrist_local_meters":
            raise ValueError(f"Line {line_number} has unsupported hand_points_frame: {frame!r}")
        timestamp = row.get("monotonic_timestamp")
        if timestamp is not None and timestamp in seen:
            continue
        if timestamp is not None:
            seen.add(timestamp)
        frames.append((np.asarray(row["left_hand_points"], dtype=float), np.asarray(row["right_hand_points"], dtype=float)))
    if not frames:
        raise ValueError("Recording contains no usable paired hand frames")
    return frames


def summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)), "max": float(np.max(values)),
    }


def evaluate(urdf_root, recording=None, frames=120):
    from teleop.robot_control.linker_o6_retargeting import DualLinkerO6Retargeter

    source = load_recording(recording) if recording else None
    results = {}
    for method in ("vector", "position", "dexpilot"):
        dual = DualLinkerO6Retargeter(urdf_root, method=method)
        if source is None:
            samples = []
            for phase in np.linspace(0., 2. * np.pi, frames, endpoint=False):
                target = .25 + .2 * np.sin(phase + np.arange(6) * .45)
                samples.append((synthetic_hand_points(dual.left, target), synthetic_hand_points(dual.right, target)))
        else:
            samples = source
        for _ in range(5):
            dual.retarget(*samples[0])
        dual.reset()
        latencies, tip_error, normalized_outputs = [], [], []
        bounds_ok, finite_ok = True, True
        for left_points, right_points in samples:
            start = time.perf_counter()
            left, right = dual.retarget(left_points, right_points)
            latencies.append((time.perf_counter() - start) * 1000.)
            normalized_outputs.append(np.concatenate((left, right)))
            for hand, points, output in ((dual.left, left_points, left), (dual.right, right_points, right)):
                actual_tips, qpos = forward_hand(hand, output)
                target_tips = hand.to_robot_points(points)[TIP_INDICES]
                tip_error.extend(np.linalg.norm(actual_tips - target_tips, axis=1) * 1000.)
                limits = hand.retargeting.optimizer.robot.joint_limits
                finite_ok &= bool(np.isfinite(output).all() and np.isfinite(qpos).all())
                bounds_ok &= bool(np.all((output >= 0.) & (output <= 1.)) and np.all(qpos >= limits[:, 0] - 1e-6) and np.all(qpos <= limits[:, 1] + 1e-6))
        outputs = np.asarray(normalized_outputs)
        results[method] = {
            "frames": len(samples), "mapping": dual.mapping_name,
            "dual_retarget_ms": summary(latencies),
            "tip_target_distance_mm": summary(tip_error),
            "finite": finite_ok, "all_active_and_mimic_bounds": bounds_ok,
            "output_min_12": outputs.min(0).tolist(), "output_max_12": outputs.max(0).tolist(),
            "solver_only_hz": 1000. / float(np.mean(latencies)),
        }
    return {
        "schema": "linker_o6_dex_retargeting_evaluation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(recording).resolve()) if recording else "synthetic_nonzero_fk",
        "recording_sha256": hashlib.sha256(Path(recording).read_bytes()).hexdigest() if recording else None,
        "input_frame": "televuer_unitree_hand_wrist_local_meters",
        "results": results,
        "passed": all(row["finite"] and row["all_active_and_mimic_bounds"] for row in results.values()),
        "scope": "Offline optimizer only; excludes XR/network latency, simulator tracking and physical hand validation. "
                 "Tip distance is diagnostic, not a common objective score: DexPilot intentionally projects pinch vectors.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf-root", default="/home/hnh/unitree_r1_dev/linkerhand-urdf/O6")
    parser.add_argument("--recording", help="JSONL containing paired 25-point wrist-local inputs, not old 12-axis targets")
    parser.add_argument("--frames", type=int, default=120, help="Synthetic FK frame count, ignored when --recording is provided")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    report = evaluate(args.urdf_root, args.recording, args.frames)
    result = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
