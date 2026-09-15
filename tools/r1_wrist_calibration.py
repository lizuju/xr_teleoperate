#!/usr/bin/env python3
"""Estimate fixed Vision Pro wrist to R1 end-effector transforms from diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np


def _rotation(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 pose")
    return matrix[:3, :3], matrix[:3, 3]


def _project_to_so3(matrix):
    u, _, vh = np.linalg.svd(matrix)
    result = u @ vh
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result


def estimate(records):
    activations = [record for record in records if record.get("event") == "activation"]
    if not activations:
        raise ValueError("No activation record found")

    output = {
        "schema": "r1_a7_wrist_calibration_v1",
        "sample_count": len(activations),
        "method": "SVD projection of R_vision.T @ R_robot",
        "sides": {},
    }
    for side in ("left", "right"):
        corrections = []
        offsets = []
        for record in activations:
            vision_rotation, vision_position = _rotation(record[f"vision_{side}_reference"], f"vision_{side}_reference")
            robot_rotation, robot_position = _rotation(record[f"robot_{side}_reference"], f"robot_{side}_reference")
            corrections.append(vision_rotation.T @ robot_rotation)
            offsets.append(robot_position - vision_position)
        correction = _project_to_so3(np.mean(corrections, axis=0))
        offset = np.mean(offsets, axis=0)
        orientation_errors = []
        for record in activations:
            vision_rotation, _ = _rotation(record[f"vision_{side}_reference"], f"vision_{side}_reference")
            robot_rotation, _ = _rotation(record[f"robot_{side}_reference"], f"robot_{side}_reference")
            residual = (vision_rotation @ correction).T @ robot_rotation
            angle = np.arccos(np.clip((np.trace(residual) - 1.0) / 2.0, -1.0, 1.0))
            orientation_errors.append(np.degrees(angle))
        position_rms = float(np.sqrt(np.mean(np.sum((np.asarray(offsets) - offset) ** 2, axis=1))))
        output["sides"][side] = {
            "rotation_correction": correction.tolist(),
            "translation_offset_m": offset.tolist(),
            "rotation_correction_det": float(np.linalg.det(correction)),
            "orientation_rms_deg": float(np.sqrt(np.mean(np.square(orientation_errors)))),
            "position_rms_m": position_rms,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+", help="JSONL file(s) produced by --arm-diagnostic-dir")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Calibration JSON output path")
    args = parser.parse_args()

    records = []
    for input_path in args.input:
        with input_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON on {input_path} line {line_number}") from exc
    result = estimate(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
