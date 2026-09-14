#!/usr/bin/env python3
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import cv2


SOURCE_NAMES = ("image", "xr", "left_hand_tracking", "right_hand_tracking", "robot",
                "left_hand_feedback", "right_hand_feedback")


def episode_directory(path):
    path = Path(path).resolve()
    return path.parent if path.name == "episode.json" else path


def episode_file(directory, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("expected a relative file path")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("file path escapes the episode directory")
    return path


def validate_episode(path):
    directory = episode_directory(path)
    report = {"episode": str(directory), "valid": False, "trainable": False, "errors": [],
              "error_count": 0, "warnings": [], "excluded_reasons": [], "frames_checked": 0,
              "images_checked": 0, "modes": {}, "tracking_invalid_frames": 0,
              "usable_following_frames": 0, "requires_frame_filtering": True,
              "command_inactive_frames": 0,
              "sources": {}, "sampling": {}}

    def error(message):
        report["error_count"] += 1
        if len(report["errors"]) < 50:
            report["errors"].append(message)

    def finite(value, location):
        if isinstance(value, float) and not math.isfinite(value):
            error(f"{location}: non-finite number")
        elif isinstance(value, dict):
            for key, child in value.items():
                finite(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                finite(child, f"{location}[{index}]")

    def timestamp(value, location, optional=False):
        if optional and value is None:
            return False
        if type(value) is not int or not 0 <= value <= 2 ** 63 - 1:
            error(f"{location}: expected a non-negative int64 timestamp")
            return False
        return True

    def command_vector(value, length, location):
        if (not isinstance(value, list) or len(value) != length
                or any(type(item) not in (int, float) for item in value)):
            error(f"{location}: expected {length} numeric values")

    try:
        manifest = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        error(f"episode.json: {exc}")
        return report
    if not isinstance(manifest, dict):
        error("episode.json: expected an object")
        return report
    finite(manifest, "manifest")
    report.update({key: manifest.get(key) if type(manifest.get(key)) in (str, int) else None
                   for key in ("schema", "status", "outcome", "episode_id")})
    if manifest.get("schema") != "xr_teleop_episode_v2":
        error("manifest.schema: expected xr_teleop_episode_v2")
    if manifest.get("status") not in ("recording", "complete", "incomplete"):
        error("manifest.status: expected recording, complete or incomplete")
    if manifest.get("status") != "complete":
        report["excluded_reasons"].append(f"episode status is {manifest.get('status')!r}, not complete")
    if manifest.get("outcome") not in ("unspecified", "success", "failure", "discarded"):
        error("manifest.outcome: unknown outcome")
    if manifest.get("outcome") == "discarded":
        report["excluded_reasons"].append("episode outcome is discarded")
    if manifest.get("outcome") == "unspecified":
        report["warnings"].append("Episode outcome has not been labeled; it is not a success label.")
    if type(manifest.get("episode_id")) is not int or manifest["episode_id"] < 0:
        error("manifest.episode_id: expected a non-negative integer")
    count = manifest.get("frame_count")
    if type(count) is not int or count < 0:
        error("manifest.frame_count: expected a non-negative integer")
    if not isinstance(manifest.get("text"), dict):
        error("manifest.text: expected an object")
    info = manifest.get("info")
    if not isinstance(info, dict):
        error("manifest.info: expected an object")
        info = {}
    require_r1_commands = info.get("robot") == "R1_A7" and info.get("end_effector") == "linker_o6"
    image = info.get("image", {})
    image = image if isinstance(image, dict) else {}
    size = (image.get("width"), image.get("height"))
    if any(type(value) is not int or value <= 0 for value in size):
        error("manifest.info.image: expected positive integer width and height")
        size = None
    frequency = info.get("frequency")
    if frequency is not None and (not isinstance(frequency, (int, float)) or isinstance(frequency, bool)
                                  or not math.isfinite(frequency) or frequency <= 0):
        error("manifest.info.frequency: expected a positive finite sampling frequency")
        frequency = None
    joint_names = info.get("joint_names", {})
    if not isinstance(joint_names, dict):
        error("manifest.info.joint_names: expected an object")
        joint_names = {}
    if manifest.get("frames") != "frames.jsonl":
        error("manifest.frames: expected frames.jsonl")
    try:
        frames_path = episode_file(directory, manifest.get("frames"))
        stream = frames_path.open("rb")
    except (OSError, ValueError) as exc:
        error(f"manifest.frames: {exc}")
        return report

    previous = {}
    modes = Counter()
    gap_count = gap_sum = gap_max = gap_abnormal = zero_gaps = 0
    first_ns = last_ns = None
    sources = {}
    with stream:
        for line_number, line in enumerate(stream, 1):
            label = f"line {line_number}"
            report["frames_checked"] += 1
            try:
                frame = json.loads(line.decode("utf-8"))
            except ValueError as exc:
                error(f"{label}: invalid JSON: {exc}")
                continue
            if not isinstance(frame, dict):
                error(f"{label}: expected a frame object")
                continue
            if type(frame.get("idx")) is not int or frame["idx"] != line_number - 1:
                error(f"{label}.idx: expected {line_number - 1}, got {frame.get('idx')!r}")
            colors = frame.get("colors")
            if not isinstance(colors, dict) or not colors:
                error(f"{label}.colors: expected image paths")
            else:
                for key, relative in colors.items():
                    try:
                        image_path = episode_file(directory, relative)
                        if not image_path.is_file():
                            raise ValueError("image does not exist")
                        pixels = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                        if pixels is None:
                            raise ValueError("image cannot be decoded")
                        if size and (pixels.shape[1], pixels.shape[0]) != size:
                            raise ValueError(f"image size {(pixels.shape[1], pixels.shape[0])} differs from {size}")
                        report["images_checked"] += 1
                    except (OSError, ValueError, cv2.error) as exc:
                        error(f"{label}.colors.{key}: {exc}")
            for field in ("states", "actions"):
                values = frame.get(field)
                if not isinstance(values, dict) or not values:
                    error(f"{label}.{field}: expected non-empty joint states")
                    continue
                finite(values, f"{label}.{field}")
                for group, names in joint_names.items():
                    if isinstance(names, list) and names and group not in values:
                        error(f"{label}.{field}.{group}: missing declared joint group")
                for group, state in values.items():
                    if not isinstance(state, dict):
                        error(f"{label}.{field}.{group}: expected joint vectors")
                        continue
                    for name in ("qpos", "qvel", "torque"):
                        if name not in state:
                            continue
                        vector = state[name]
                        if not isinstance(vector, list) or any(type(value) not in (int, float) for value in vector):
                            error(f"{label}.{field}.{group}.{name}: expected a numeric list")
                    expected = joint_names.get(group)
                    qpos = state.get("qpos")
                    if not isinstance(qpos, list) or not qpos:
                        error(f"{label}.{field}.{group}.qpos: expected a non-empty numeric list")
                    if isinstance(expected, list) and expected and (not isinstance(qpos, list) or len(qpos) != len(expected)):
                        error(f"{label}.{field}.{group}.qpos: joint count differs from info.joint_names")
            sample = frame.get("sample")
            if not isinstance(sample, dict):
                error(f"{label}.sample: expected an object")
                continue
            finite(sample, f"{label}.sample")
            for name in ("xr", "commands"):
                if not isinstance(sample.get(name), dict):
                    error(f"{label}.sample.{name}: expected an object")
            commands = sample.get("commands", {})
            if isinstance(commands, dict):
                for controller, stages in commands.items():
                    if not isinstance(stages, dict):
                        error(f"{label}.sample.commands.{controller}: expected command snapshots")
                        continue
                    if controller == "arm":
                        for stage in ("requested", "published"):
                            command = stages.get(stage)
                            if command is None:
                                continue
                            if not isinstance(command, dict):
                                error(f"{label}.sample.commands.arm.{stage}: expected an object or null")
                                continue
                            for name, length in (("arm_q", 14), ("arm_tau", 14), ("head_q", 2)):
                                command_vector(command.get(name), length, f"{label}.sample.commands.arm.{stage}.{name}")
                    elif controller == "hands":
                        requested = stages.get("requested")
                        if requested is not None:
                            if not isinstance(requested, dict):
                                error(f"{label}.sample.commands.hands.requested: expected an object or null")
                            else:
                                for side in ("left", "right"):
                                    command_vector(requested.get(f"{side}_q"), 6,
                                                   f"{label}.sample.commands.hands.requested.{side}_q")
                        published = stages.get("published", {})
                        if not isinstance(published, dict):
                            error(f"{label}.sample.commands.hands.published: expected per-side snapshots")
                        else:
                            for side, command in published.items():
                                if command is not None:
                                    if not isinstance(command, dict):
                                        error(f"{label}.sample.commands.hands.published.{side}: expected an object or null")
                                    else:
                                        command_vector(command.get("q"), 6,
                                                       f"{label}.sample.commands.hands.published.{side}.q")
            for name in ("timestamp_ns", "monotonic_ns"):
                value = sample.get(name)
                if timestamp(value, f"{label}.sample.{name}"):
                    if name in previous and value < previous[name]:
                        error(f"{label}.sample.{name}: timestamp moved backwards")
                    if name == "monotonic_ns":
                        if first_ns is None:
                            first_ns = value
                        if name in previous:
                            gap = value - previous[name]
                            if gap >= 0:
                                gap_count += 1
                                gap_sum += gap
                                gap_max = max(gap_max, gap)
                                zero_gaps += gap == 0
                                gap_abnormal += bool(frequency and gap > 1.5e9 / frequency)
                        last_ns = value
                    previous[name] = value
            mode = sample.get("mode")
            if mode not in ("following", "paused", "tracking_hold"):
                error(f"{label}.sample.mode: unknown mode")
            else:
                modes[mode] += 1
            source_data = sample.get("sources")
            if not isinstance(source_data, dict):
                error(f"{label}.sample.sources: expected an object")
                continue
            for name in SOURCE_NAMES:
                source = source_data.get(name)
                if not isinstance(source, dict):
                    error(f"{label}.sample.sources.{name}: missing source metadata")
                    continue
                stats = sources.setdefault(name, {"samples": 0, "not_fresh": 0, "repeated": 0,
                                                  "sequence_repeats": 0, "sequence_regressions": 0,
                                                  "max_age_ms": None, "age_count": 0, "age_sum_ms": 0.0})
                stats["samples"] += 1
                if type(source.get("fresh")) is not bool:
                    error(f"{label}.sample.sources.{name}.fresh: expected a boolean")
                stats["not_fresh"] += source.get("fresh") is not True
                if "repeated" in source and type(source["repeated"]) is not bool:
                    error(f"{label}.sample.sources.{name}.repeated: expected a boolean")
                stats["repeated"] += source.get("repeated") is True
                stamp = source.get("received_monotonic_ns")
                timestamp(stamp, f"{label}.sample.sources.{name}.received_monotonic_ns", optional=True)
                age = source.get("age_ms")
                if age is not None:
                    if type(age) not in (int, float) or not math.isfinite(age) or age < 0:
                        error(f"{label}.sample.sources.{name}.age_ms: expected a non-negative finite age or null")
                    else:
                        stats["max_age_ms"] = max(stats["max_age_ms"] or 0, age)
                        stats["age_count"] += 1
                        stats["age_sum_ms"] += age
                if source.get("fresh") and (stamp is None or age is None):
                    error(f"{label}.sample.sources.{name}: fresh source needs receive time and age")
                sample_time = sample.get("monotonic_ns")
                if (type(stamp) is int and stamp > 0 and type(sample_time) is int
                        and type(age) in (int, float) and math.isfinite(age)):
                    expected_age = (sample_time - stamp) / 1e6
                    if abs(age - expected_age) > 1.0:
                        error(f"{label}.sample.sources.{name}: age_ms disagrees with receive/sample timestamps")
                    if source.get("fresh") and stamp > sample_time:
                        error(f"{label}.sample.sources.{name}: fresh receive timestamp is in the future")
                sequence = source.get("sequence")
                if sequence is not None:
                    if type(sequence) is not int or sequence < 0:
                        error(f"{label}.sample.sources.{name}.sequence: expected a non-negative integer")
                    else:
                        old_sequence = previous.get(f"sequence:{name}")
                        if old_sequence is not None:
                            stats["sequence_repeats"] += sequence == old_sequence
                            stats["sequence_regressions"] += sequence < old_sequence
                        previous[f"sequence:{name}"] = sequence
            tracking = ("xr", "left_hand_tracking", "right_hand_tracking")
            report["tracking_invalid_frames"] += any(
                not isinstance(source_data.get(name), dict) or source_data[name].get("fresh") is not True
                for name in tracking)
            commands_active = True
            if require_r1_commands:
                arm_commands = commands.get("arm", {}) if isinstance(commands, dict) else {}
                hand_commands = commands.get("hands", {}) if isinstance(commands, dict) else {}
                arm_published = arm_commands.get("published") if isinstance(arm_commands, dict) else None
                hands_published = hand_commands.get("published", {}) if isinstance(hand_commands, dict) else {}
                left_published = hands_published.get("left") if isinstance(hands_published, dict) else None
                right_published = hands_published.get("right") if isinstance(hands_published, dict) else None
                sample_time = sample.get("monotonic_ns")
                commands_active = type(sample_time) is int and all(
                    isinstance(command, dict) and type(command.get("monotonic_ns")) is int
                    and 0 <= sample_time - command["monotonic_ns"] <= 250_000_000
                    for command in (arm_published, left_published, right_published)
                ) and all(
                    isinstance(command, dict) and type(command.get("mode")) is int and command["mode"] == 1
                    for command in (left_published, right_published)
                )
                report["command_inactive_frames"] += not commands_active
            report["usable_following_frames"] += commands_active and mode == "following" and all(
                isinstance(source_data.get(name), dict) and source_data[name].get("fresh") is True
                for name in SOURCE_NAMES)

    if report["frames_checked"] == 0:
        error("frames.jsonl: episode has no frames")
    if manifest.get("status") != "recording" and count != report["frames_checked"]:
        error(f"manifest.frame_count: declared {count}, found {report['frames_checked']} lines")
    if manifest.get("status") == "recording" and count != report["frames_checked"]:
        report["warnings"].append("Recording manifest count is provisional; validate again after closing the episode.")
    for stats in sources.values():
        age_count = stats.pop("age_count")
        age_sum = stats.pop("age_sum_ms")
        stats["mean_age_ms"] = age_sum / age_count if age_count else None
    report["sources"] = sources
    report["modes"] = dict(modes)
    report["sampling"] = {"frequency_hz": frequency, "first_monotonic_ns": first_ns,
                          "last_monotonic_ns": last_ns,
                          "duration_seconds": (last_ns - first_ns) / 1e9 if first_ns is not None else None,
                          "gap_count": gap_count, "max_gap_ms": gap_max / 1e6,
                          "mean_gap_ms": gap_sum / gap_count / 1e6 if gap_count else None,
                          "gaps_over_1_5_periods": gap_abnormal, "zero_gaps": zero_gaps}
    report["requires_frame_filtering"] = report["usable_following_frames"] != report["frames_checked"]
    if report["command_inactive_frames"]:
        report["warnings"].append(
            f"Exclude {report['command_inactive_frames']} R1/O6 frames with unpublished, released or expired commands. "
            "Require an arm publication, both hand modes=1, and publication ages between 0 and 250 ms."
        )
        if report["command_inactive_frames"] == report["frames_checked"]:
            report["excluded_reasons"].append("all R1/O6 frames have unpublished, released or expired commands")
    if not report["usable_following_frames"]:
        report["excluded_reasons"].append("episode has no following frames with fresh XR, both hands and feedback sources")
    elif report["requires_frame_filtering"]:
        report["warnings"].append(
            f"Only {report['usable_following_frames']}/{report['frames_checked']} frames have following mode and "
            "all sources fresh and, for R1/O6, active fresh published commands. "
            "Filter by sample.mode, sample.sources and sample.commands before training; do not train on the full episode."
        )
    report["valid"] = report["error_count"] == 0
    report["trainable"] = report["valid"] and not report["excluded_reasons"]
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate a v2 teleoperation episode offline; never connects to a robot.")
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this path")
    args = parser.parse_args()
    report = validate_episode(args.episode)
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 1 if not report["valid"] else 0 if report["trainable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
