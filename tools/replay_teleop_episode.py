#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np

from check_teleop_episode import episode_directory, episode_file, validate_episode


def replay_episode(path, output, fps=30.0, max_gap_seconds=10.0):
    report = validate_episode(path)
    result = {"validation": report}
    if not report["valid"]:
        return result
    if report["status"] == "recording":
        result["error"] = "Close the recording before replaying its changing files."
        return result
    if not math.isfinite(fps) or not 0 < fps <= 120:
        result["error"] = "Output FPS must be finite and in (0, 120]."
        return result
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        result["error"] = "Maximum gap must be a positive finite number of seconds."
        return result
    if report["sampling"]["max_gap_ms"] / 1000 > max_gap_seconds:
        result["error"] = (f"Sample gap exceeds {max_gap_seconds:g}s; review the timestamps, or explicitly "
                           "raise --max-gap-seconds to preserve that interval in the video.")
        return result
    directory = episode_directory(path)
    manifest = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    output = Path(output).resolve()
    if output.suffix.lower() != ".mp4":
        result["error"] = "Output path must have an .mp4 extension."
        return result
    if output.exists():
        result["error"] = f"Output already exists: {output}"
        return result
    width = manifest["info"]["image"]["width"]
    height = manifest["info"]["image"]["height"]
    output_size = (width * 2, height + 100)
    if any(value % 2 for value in output_size):
        result["error"] = "MP4 output needs even image width and height; no implicit cropping is performed."
        return result
    start_ns = report["sampling"]["first_monotonic_ns"]
    last_ns = report["sampling"]["last_monotonic_ns"]
    frame_count = math.ceil((last_ns - start_ns) / 1e9 * fps) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    try:
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open the mp4v encoder")
        with episode_file(directory, manifest["frames"]).open(encoding="utf-8") as stream:
            rows = (json.loads(line) for line in stream)
            current = next(rows)
            following = next(rows, None)
            loaded_idx = None
            for output_index in range(frame_count):
                tick_ns = start_ns + round(output_index * 1e9 / fps)
                while following is not None and following["sample"]["monotonic_ns"] <= tick_ns:
                    current = following
                    following = next(rows, None)
                if current["idx"] != loaded_idx:
                    images = []
                    for key in ("color_0", "color_1"):
                        if key not in current["colors"]:
                            raise ValueError(f"Frame {current['idx']} is missing stereo image {key}")
                        pixels = cv2.imread(str(episode_file(directory, current["colors"][key])), cv2.IMREAD_COLOR)
                        if pixels is None or pixels.shape[:2] != (height, width):
                            raise ValueError(f"Image {key} changed after validation")
                        images.append(pixels)
                    stereo = np.hstack(images)
                    errors = []
                    for group in ("left_arm", "right_arm", "left_ee", "right_ee", "body"):
                        actual = current["states"].get(group, {}).get("qpos", [])
                        target = current["actions"].get(group, {}).get("qpos", [])
                        if actual and len(actual) == len(target):
                            difference = np.asarray(target, dtype=float) - np.asarray(actual, dtype=float)
                            errors.append(f"{group}={np.sqrt(np.mean(difference ** 2)):.4f}")
                        else:
                            errors.append(f"{group}=n/a")
                    loaded_idx = current["idx"]
                canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
                canvas[:height] = stereo
                sample = current["sample"]
                elapsed = (tick_ns - start_ns) / 1e9
                sample_age = (tick_ns - sample["monotonic_ns"]) / 1e9
                lines = [f"episode={manifest['episode_id']} outcome={manifest['outcome']} status={manifest['status']}",
                         f"t={elapsed:.3f}s sample={current['idx']} mode={sample['mode']} held={sample_age:.3f}s",
                         "qpos RMS error: " + "  ".join(errors[:2]),
                         "  ".join(errors[2:]) + "  (arms/body: rad; hands: normalized)"]
                font_scale = min(0.52, output_size[0] / 1300)
                for line_index, line in enumerate(lines):
                    cv2.putText(canvas, line, (8, height + 19 + line_index * 23), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, (240, 240, 240), 1, cv2.LINE_AA)
                writer.write(canvas)
        writer.release()
        verification = cv2.VideoCapture(str(temporary_path))
        try:
            ok, first_frame = verification.read()
            encoded_count = round(verification.get(cv2.CAP_PROP_FRAME_COUNT))
            if not ok or first_frame.shape[:2] != output_size[::-1] or encoded_count != frame_count:
                raise RuntimeError("Encoded MP4 failed frame-count/dimension verification")
        finally:
            verification.release()
        temporary_path.replace(output)
        result["replay"] = {"path": str(output), "frames": frame_count, "fps": fps,
                            "size": list(output_size), "source_duration_seconds": (last_ns - start_ns) / 1e9,
                            "video_duration_seconds": frame_count / fps,
                            "timeline": "monotonic sample timestamps; hold the latest available image between samples",
                            "codec": "mp4v"}
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        result["error"] = str(exc)
    finally:
        writer.release()
        temporary_path.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Render a recorded stereo episode to MP4 offline; never sends robot commands.")
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-gap-seconds", type=float, default=10.0,
                        help="Reject larger timestamp gaps unless explicitly increased (default: 10)")
    args = parser.parse_args()
    result = replay_episode(args.episode, args.output, args.fps, args.max_gap_seconds)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if "replay" in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
