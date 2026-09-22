#!/usr/bin/env python3
"""Install day-dir hand_eye into assets/r1/camera_calibration.json (no motion).

Refuses to write if wrist translation RMS > 0.03 m. Backs up the current assets
file with a timestamped sidecar after verifying the expected pre-install sha.
Marks wrist streams/hand_eye with image_mirror=horizontal so consumers use the
shared helper on raw JPEGs (no per-call-site flip).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

EXPECTED_PRE_SHA = "3cf7df22a492bada4ab84aa2a35786ef88de81157aa94e79c93abd3017d0911d"
MAX_WRIST_TRANS_RMS_M = 0.03
WRISTS = ("left_wrist", "right_wrist")
HEAD = ("head_left", "head_right")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(t):
    return math.sqrt(sum(float(x) * float(x) for x in t))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--day-json",
        type=Path,
        default=Path("/home/hnh/r1-cam-calib/20260922/camera_calibration.json"),
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("/home/hnh/unitree_r1_dev/xr_teleoperate/assets/r1/camera_calibration.json"),
    )
    parser.add_argument("--expected-sha", default=EXPECTED_PRE_SHA)
    parser.add_argument("--max-wrist-trans-rms-m", type=float, default=MAX_WRIST_TRANS_RMS_M)
    parser.add_argument("--force-sha", action="store_true",
                        help="Allow install when assets sha differs from expected")
    args = parser.parse_args(argv)

    assets = args.assets.expanduser().resolve()
    day_json = args.day_json.expanduser().resolve()
    if not assets.is_file():
        print(f"missing assets: {assets}", file=sys.stderr)
        return 1
    if not day_json.is_file():
        print(f"missing day json: {day_json}", file=sys.stderr)
        return 1

    current_sha = sha256_file(assets)
    print(f"assets sha256: {current_sha}")
    if current_sha != args.expected_sha and not args.force_sha:
        print(
            f"refusing: assets sha != expected {args.expected_sha} "
            f"(pass --force-sha if intentional)",
            file=sys.stderr,
        )
        return 2

    day = json.loads(day_json.read_text(encoding="utf-8"))
    base = json.loads(assets.read_text(encoding="utf-8"))
    hand_eye = day.get("hand_eye") or {}
    if not hand_eye:
        print("day json has empty hand_eye", file=sys.stderr)
        return 2

    for name in WRISTS:
        entry = hand_eye.get(name)
        if not entry:
            print(f"missing hand_eye.{name}", file=sys.stderr)
            return 2
        rms = float(entry.get("translation_rms_m", 1e9))
        if rms > args.max_wrist_trans_rms_m:
            print(
                f"refusing: {name} translation_rms_m={rms:.4f} > {args.max_wrist_trans_rms_m}",
                file=sys.stderr,
            )
            return 2
        entry = dict(entry)
        entry["image_mirror"] = "horizontal"
        hand_eye[name] = entry
        print(
            f"{name}: |t|={norm(entry['translation_m']):.5f} m  "
            f"t={entry['translation_m']}  "
            f"trans_rms={rms:.4f}  rot_rms={entry.get('rotation_rms_deg')}"
        )

    for name in HEAD:
        if name not in hand_eye:
            print(f"missing hand_eye.{name}", file=sys.stderr)
            return 2
        print(
            f"{name}: |t|={norm(hand_eye[name]['translation_m']):.5f} m  "
            f"t={hand_eye[name]['translation_m']}  "
            f"trans_rms={hand_eye[name].get('translation_rms_m')}  "
            f"rot_rms={hand_eye[name].get('rotation_rms_deg')}"
        )

    # Preserve assets intrinsics/stereo exactly; only add wrist mirror flags + hand_eye.
    out = json.loads(json.dumps(base))
    for name in WRISTS:
        cam = out["cameras"].get(name)
        if not isinstance(cam, dict):
            print(f"assets missing cameras.{name}", file=sys.stderr)
            return 2
        # Keep K/D/size; only annotate mirror.
        cam["image_mirror"] = "horizontal"
    out["hand_eye"] = hand_eye
    out["created"] = day.get("created") or out.get("created")
    note_bit = (
        "hand_eye eye-in-hand (wrist→wrist_yaw_link solved after H-unmirror; "
        "image_mirror=horizontal — use prepare_image_for_hand_eye / "
        "project_link_points_to_pixels on raw wrist JPEGs; "
        "head→head_yaw_link, no mirror)"
    )
    note = out.get("note") or ""
    if note_bit not in note:
        out["note"] = (note.rstrip(".") + "; " + note_bit).lstrip("; ")

    # Head intrinsics must be byte-identical to pre-install cameras block fields
    # except we only touched wrists.
    for name in ("head_left", "head_right"):
        if out["cameras"][name] != base["cameras"][name]:
            print(f"bug: mutated {name} intrinsics", file=sys.stderr)
            return 3
    if out["stereo"] != base["stereo"]:
        print("bug: mutated stereo", file=sys.stderr)
        return 3

    stamp = time.strftime("%Y%m%d-%H%M%S")
    sidecar = assets.with_name(assets.name + f".pre-hand-eye-install-{stamp}")
    shutil.copy2(assets, sidecar)
    print(f"backup: {sidecar}")

    temporary = assets.with_name("." + assets.name + ".tmp")
    temporary.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, assets)
    new_sha = sha256_file(assets)
    print(f"installed: {assets}")
    print(f"new sha256: {new_sha}")

    # Also refresh day-dir JSON with the same mirror annotations for consistency.
    day_out = json.loads(json.dumps(day))
    day_out["hand_eye"] = hand_eye
    for name in WRISTS:
        if name in day_out.get("cameras", {}):
            day_out["cameras"][name]["image_mirror"] = "horizontal"
    if note_bit not in (day_out.get("note") or ""):
        day_note = day_out.get("note") or ""
        day_out["note"] = (day_note.rstrip(".") + "; " + note_bit).lstrip("; ")
    day_backup = day_json.with_name(day_json.name + ".pre-mirror-annotate")
    if not day_backup.exists():
        shutil.copy2(day_json, day_backup)
    temporary = day_json.with_name("." + day_json.name + ".tmp")
    temporary.write_text(json.dumps(day_out, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, day_json)
    print(f"day-dir updated: {day_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
