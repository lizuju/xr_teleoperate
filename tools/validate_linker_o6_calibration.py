#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


AXES = ("thumb_pitch", "thumb_yaw", "index", "middle", "ring", "pinky")
JOINTS = tuple(f"{side}_{axis}" for side in ("left", "right") for axis in AXES)


def parse_args():
    parser = argparse.ArgumentParser(description="Regress recorded Linker O6 calibration outputs.")
    parser.add_argument("--recording", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--expected-frames", type=int, default=377)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--report-dir", default="/home/hnh/unitree_r1_dev/reports")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_unique_frames(path):
    frames = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("armed") is not True:
            continue
        timestamp = row.get("monotonic_timestamp")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp):
            raise ValueError("Every armed row must have a finite monotonic timestamp")
        if float(timestamp) in frames:
            previous = frames[float(timestamp)]
            for field in ("raw_target_12", "target_12"):
                if not np.array_equal(np.asarray(previous.get(field)), np.asarray(row.get(field))):
                    raise ValueError(f"Duplicate timestamp has inconsistent {field}: {timestamp}")
        frames[float(timestamp)] = row
    return [frames[timestamp] for timestamp in sorted(frames)]


def markdown_report(report):
    lines = [
        "# Linker O6 calibration regression",
        "",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Generated: {report['generated_at']}",
        f"- Mapping: `{report['mapping']}`",
        f"- Unique frames: `{report['unique_frames']}`",
        f"- Tolerance: `{report['tolerance']:.1e}`",
        f"- Overall max error: `{report['overall']['max_abs_error']:.3e}`",
        f"- Calibration SHA-256: `{report['calibration_sha256']}`",
        f"- Recording SHA-256: `{report['recording_sha256']}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Per-joint absolute error",
            "",
            "| Joint | Mean | RMSE | p95 | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for joint in JOINTS:
        metric = report["joints"][joint]
        lines.append(
            f"| {joint} | {metric['mean_abs_error']:.3e} | {metric['rmse']:.3e} | "
            f"{metric['p95_abs_error']:.3e} | {metric['max_abs_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "This is a software calibration regression from recorded geometric targets to calibrated targets. "
            "It does not validate Vision Pro tracking, Vuer/WebSocket transport, or physical O6 direction/accuracy.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.expected_frames <= 0:
        raise ValueError("--expected-frames must be positive")
    if not (math.isfinite(args.tolerance) and args.tolerance >= 0.0):
        raise ValueError("--tolerance must be a finite non-negative value")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from teleop.robot_control.linker_o6_retargeting import LinkerO6Calibration

    calibration = LinkerO6Calibration(args.calibration)
    rows = load_unique_frames(args.recording)
    expected = []
    calculated = []
    row_metadata_valid = True
    for row in rows:
        raw = np.asarray(row.get("raw_target_12"), dtype=np.float64)
        target = np.asarray(row.get("target_12"), dtype=np.float64)
        if raw.shape != (12,) or target.shape != (12,) or not np.isfinite(raw).all() or not np.isfinite(target).all():
            raise ValueError("Each calibrated frame must have finite raw_target_12 and target_12 values")
        if not np.array_equal(raw, np.asarray(row.get("raw_left_target") + row.get("raw_right_target"))):
            raise ValueError("raw_target_12 does not match left/right raw targets")
        if not np.array_equal(target, np.asarray(row.get("left_target") + row.get("right_target"))):
            raise ValueError("target_12 does not match left/right calibrated targets")
        if row.get("mapping") != calibration.name:
            row_metadata_valid = False
        if row.get("visionpro_input_calibrated") is not True:
            row_metadata_valid = False
        if row.get("thumb_yaw_hardware_direction_validated") is not False:
            row_metadata_valid = False
        left, right = calibration.apply(raw[:6], raw[6:])
        calculated.append(np.concatenate((left, right)))
        expected.append(target)

    expected = np.asarray(expected, dtype=np.float64)
    calculated = np.asarray(calculated, dtype=np.float64)
    error = calculated - expected
    absolute_error = np.abs(error)
    joints = {}
    for index, joint in enumerate(JOINTS):
        joint_error = error[:, index]
        joint_abs = absolute_error[:, index]
        joints[joint] = {
            "mean_abs_error": float(np.mean(joint_abs)),
            "rmse": float(np.sqrt(np.mean(np.square(joint_error)))),
            "p95_abs_error": float(np.percentile(joint_abs, 95)),
            "max_abs_error": float(np.max(joint_abs)),
        }

    overall = {
        "mean_abs_error": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "p95_abs_error": float(np.percentile(absolute_error, 95)),
        "max_abs_error": float(np.max(absolute_error)),
    }
    checks = {
        "expected unique frame count": len(rows) == args.expected_frames,
        "mapping and calibration metadata match": row_metadata_valid,
        "recorded targets remain normalized": bool(np.all((expected >= 0.0) & (expected <= 1.0))),
        "calculated targets remain normalized": bool(np.all((calculated >= 0.0) & (calculated <= 1.0))),
        "all outputs reproduce within tolerance": overall["max_abs_error"] <= args.tolerance,
    }
    report = {
        "schema": "linker_o6_calibration_regression_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mapping": calibration.name,
        "recording": str(Path(args.recording)),
        "calibration": str(Path(args.calibration)),
        "recording_sha256": file_sha256(args.recording),
        "calibration_sha256": file_sha256(args.calibration),
        "unique_frames": len(rows),
        "tolerance": args.tolerance,
        "overall": overall,
        "joints": joints,
        "checks": checks,
        "passed": all(checks.values()),
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"o6_calibration_regression_{stamp}.json"
    markdown_path = report_dir / f"o6_calibration_regression_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"[O6 CALIBRATION] result={'PASS' if report['passed'] else 'FAIL'}")
    print(f"[O6 CALIBRATION] frames={len(rows)} max_abs_error={overall['max_abs_error']:.3e}")
    print(f"[O6 CALIBRATION] report={markdown_path}")
    print(f"[O6 CALIBRATION] json={json_path}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[O6 CALIBRATION] BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
