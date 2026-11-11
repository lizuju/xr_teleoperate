#!/usr/bin/env python3
"""Label segment-by-segment ice-cream recordings with their stage, read-only.

The task is recorded as four consecutive episodes, each starting where the last
one ended (grasp -> turn right and deliver -> dispense -> turn left and place).
The arrow-key waist turns always stop at the start heading or -90 deg, so where
the waist is at the start and at the end of an episode tells the stage; the
gripper and the left hand cross-check it:

  stage  waist start -> end   cross-check
  1      front -> front       gripper closes (picks the cup up)
  2      front -> right       gripper stays closed
  3      right -> right       left hand grips and releases the lever
  4      right -> front       gripper opens at the end (puts the cup down)

Nothing in the recordings is modified. --write saves stage_labels.json in the
task directory; --link-dir makes one folder per stage holding symlinks to its
successful episodes, which export_r1_training_dataset.py accepts as a task
directory, so each stage can become its own dataset.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

STAGES = {
    1: {"name": "grasp_cup", "zh": "抓杯：右手夹爪抓起杯子，抬到胸前",
        "instruction": "pick up the cup with the right gripper and lift it in front of the chest"},
    2: {"name": "turn_right_deliver", "zh": "右转送杯：右转 90°，把杯子送到出雪糕口下面",
        "instruction": "turn right and hold the cup under the ice cream outlet"},
    3: {"name": "dispense", "zh": "接雪糕：左手拉杆，杯子装满后推回，左手收回",
        "instruction": "pull the lever with the left hand until the cup is full, then push it back"},
    4: {"name": "turn_left_place", "zh": "回转放杯：左转 90°，放下杯子，张开夹爪，右手收回",
        "instruction": "turn left, put the cup on the table and release it"},
}
BY_HEADINGS = {("front", "front"): 1, ("front", "right"): 2, ("right", "right"): 3, ("right", "front"): 4}
HEADING_TOLERANCE = math.radians(30.0)
END_WINDOW_S = 0.5
GRIPPER_CHANGE_RAD = 0.3
LEFT_ACTIVITY = 0.25


def heading(waist, front):
    offset = math.atan2(math.sin(waist - front), math.cos(waist - front))
    for name, centre in (("front", 0.0), ("right", -math.pi / 2), ("left", math.pi / 2)):
        if abs(offset - centre) <= HEADING_TOLERANCE:
            return name
    return "between"


def frame_time(frame, index, frequency):
    sample = frame.get("sample") or {}
    stamp = sample.get("monotonic_ns") or sample.get("timestamp_ns")
    return stamp / 1e9 if isinstance(stamp, (int, float)) else index / frequency


def qpos(frame, group):
    values = ((frame.get("states") or {}).get(group) or {}).get("qpos")
    if not isinstance(values, list) or not values:
        return None
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        return None
    return [float(value) for value in values]


def window_median(times, values, start, stop):
    chosen = [value for time, value in zip(times, values) if start <= time <= stop and value is not None]
    return statistics.median(chosen) if chosen else None


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))]


def read_episode(directory):
    manifest = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    info = manifest.get("info") or {}
    frequency = float(info.get("frequency") or 40.0)
    frames = []
    with (directory / manifest.get("frames", "frames.jsonl")).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                frames.append(json.loads(line))
    return manifest, info, frequency, frames


def label_episode(directory, front=0.0):
    directory = Path(directory)
    manifest, info, frequency, frames = read_episode(directory)
    result = {"episode": directory.name, "episode_id": manifest.get("episode_id"),
              "outcome": manifest.get("outcome"), "status": manifest.get("status"),
              "started_at": manifest.get("started_at"), "frames": len(frames),
              "stage": None, "warnings": []}
    right_device = ((info.get("end_effectors") or {}).get("right")
                    or ("dex1_1" if info.get("end_effector") == "linker_o6_left+dex1_1_right" else None))
    if not frames:
        result["reason"] = "no frames"
        return result
    times = [frame_time(frame, index, frequency) for index, frame in enumerate(frames)]
    first, last = times[0], times[-1]
    result["duration_s"] = round(last - first, 3)
    waist = [(qpos(frame, "body") or [None])[0] for frame in frames]
    start_waist = window_median(times, waist, first, first + END_WINDOW_S)
    end_waist = window_median(times, waist, last - END_WINDOW_S, last)
    if start_waist is None or end_waist is None:
        result["reason"] = "no waist angle recorded"
        return result
    result["waist_deg"] = [round(math.degrees(start_waist - front), 1), round(math.degrees(end_waist - front), 1)]
    headings = (heading(start_waist, front), heading(end_waist, front))
    result["headings"] = list(headings)

    gripper_change = None
    if right_device == "dex1_1":
        gripper = [(qpos(frame, "right_ee") or [None])[0] for frame in frames]
        start_q = window_median(times, gripper, first, first + END_WINDOW_S)
        end_q = window_median(times, gripper, last - END_WINDOW_S, last)
        if start_q is not None and end_q is not None:
            gripper_change = end_q - start_q
            result["gripper_rad"] = [round(start_q, 3), round(end_q, 3)]
    left_activity = None
    closures = [statistics.fmean(values) for values in (qpos(frame, "left_ee") for frame in frames) if values]
    if closures:
        left_activity = percentile(closures, 0.95) - percentile(closures, 0.05)
        result["left_hand_range"] = round(left_activity, 3)

    stage = BY_HEADINGS.get(headings)
    if stage == 1 and gripper_change is None:
        # Only the gripper tells a grasp from any other episode that stays at the front.
        stage = None
        result["reason"] = "waist stayed at the front and there is no Dex1-1 gripper angle to show a grasp"
    elif stage == 1 and gripper_change > -GRIPPER_CHANGE_RAD:
        stage = None
        result["reason"] = "waist stayed at the front but the gripper did not close"
    elif stage is None:
        result["reason"] = f"waist went {headings[0]} -> {headings[1]}, which is no stage"
    result["stage"] = stage
    if stage is None:
        return result
    result.update(name=STAGES[stage]["name"], instruction=STAGES[stage]["instruction"])
    warnings = result["warnings"]
    if gripper_change is None:
        warnings.append("no Dex1-1 gripper angle to cross-check")
    elif stage in (2, 3) and gripper_change > GRIPPER_CHANGE_RAD:
        warnings.append("gripper opened while it should hold the cup")
    elif stage == 4 and gripper_change < GRIPPER_CHANGE_RAD:
        warnings.append("gripper did not open at the end")
    if stage == 3 and (left_activity is None or left_activity < LEFT_ACTIVITY):
        warnings.append("left hand barely moved; was the lever pulled?")
    if stage in (1, 2, 4) and left_activity is not None and left_activity >= LEFT_ACTIVITY:
        warnings.append("left hand moved a lot for this stage")
    return result


def episode_directories(source):
    source = Path(source)
    if (source / "episode.json").is_file():
        return [source]
    return sorted(path.parent for path in source.glob("episode_*/episode.json"))


def link_stages(task_dir, labels, link_dir):
    link_dir = Path(link_dir)
    task_dir = Path(task_dir).resolve()
    if link_dir.resolve() == task_dir:
        raise ValueError("--link-dir must not be the task directory")
    made = {}
    for label in labels:
        if label["stage"] is None or label["outcome"] != "success" or label["status"] != "complete":
            continue
        folder = link_dir / f"stage{label['stage']}_{STAGES[label['stage']]['name']}"
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / label["episode"]
        target = task_dir / label["episode"]
        if link.is_symlink() or link.exists():
            if link.resolve() != target:
                raise ValueError(f"{link} already exists and points elsewhere")
        else:
            link.symlink_to(target, target_is_directory=True)
        made[folder.name] = made.get(folder.name, 0) + 1
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="A task directory of episodes, or one episode directory")
    parser.add_argument("--front-deg", type=float, default=0.0,
                        help="Waist angle of the start heading in deg (default 0: the activation recentre)")
    parser.add_argument("--write", action="store_true", help="Save stage_labels.json in the task directory")
    parser.add_argument("--link-dir", type=Path,
                        help="Make one folder per stage of symlinks to its successful, complete episodes")
    args = parser.parse_args(argv)
    directories = episode_directories(args.source)
    if not directories:
        parser.error("no episodes found")
    front = math.radians(args.front_deg)
    labels = []
    for directory in directories:
        try:
            labels.append(label_episode(directory, front))
        except (OSError, ValueError) as exc:
            labels.append({"episode": directory.name, "stage": None, "outcome": None, "status": None,
                           "reason": f"unreadable: {exc}", "warnings": []})
    for label in labels:
        stage = f"stage {label['stage']} {STAGES[label['stage']]['name']}" if label["stage"] else "unlabelled"
        waist = label.get("waist_deg")
        waist = f"waist {waist[0]:+.0f}->{waist[1]:+.0f} deg" if waist else ""
        notes = "; ".join(label["warnings"] + ([label["reason"]] if label.get("reason") else []))
        print(f"{label['episode']:42s} {str(label['outcome']):8s} {stage:30s} {waist:24s} {notes}")
    counts = {}
    for label in labels:
        if label["stage"] and label["outcome"] == "success":
            counts[label["stage"]] = counts.get(label["stage"], 0) + 1
    print("successful episodes per stage:",
          ", ".join(f"{stage} {STAGES[stage]['name']}: {counts.get(stage, 0)}" for stage in STAGES))
    task_dir = args.source if not (args.source / "episode.json").is_file() else args.source.parent
    if args.write:
        output = task_dir / "stage_labels.json"
        output.write_text(json.dumps({"schema": "r1_icecream_stage_labels_v1", "stages": STAGES,
                                      "front_deg": args.front_deg, "episodes": labels},
                                     ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    if args.link_dir:
        made = link_stages(task_dir, labels, args.link_dir)
        print("linked:", made or "nothing (no successful labelled episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
