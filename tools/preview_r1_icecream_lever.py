#!/usr/bin/env python3
"""DRY-RUN preview of a right-lever ice-cream sequence (order + distances only).

Loads ``icecream_waypoints.json`` and the right-hand ``o6_grip_cap.json`` entry,
then prints the planned motion in Chinese/plain text.

Does NOT publish lowcmd, enter debug mode, close the hand, start teleop,
import Unitree ChannelPublisher, or move the robot. ``q`` is not an e-stop
(this tool has no interactive loop).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from pathlib import Path

SCHEMA_V2 = "icecream_waypoints_v2"
SCHEMA_V1 = "icecream_waypoints_v1"
DEFAULT_WAYPOINTS = Path.home() / ".config" / "xr_teleoperate" / "icecream_waypoints.json"
DEFAULT_GRIP_CAP = Path.home() / ".config" / "xr_teleoperate" / "o6_grip_cap.json"
DEFAULT_SLOT = "right_lever_3"
LEVER_SLOTS = frozenset({"right_lever_1", "right_lever_2", "right_lever_3"})
AXIS_COUNT = 6


class PreviewError(RuntimeError):
    """Raised when the preview cannot be built (missing fields, bad JSON)."""


def default_waypoints_path():
    return Path(os.environ.get("XR_ICECREAM_WAYPOINTS", str(DEFAULT_WAYPOINTS))).expanduser()


def default_grip_cap_path():
    return Path(os.environ.get("XR_O6_GRIP_CAP", str(DEFAULT_GRIP_CAP))).expanduser()


def _finite_vec3(values, label):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise PreviewError(f"{label} must be a length-3 list")
    out = []
    for i, raw in enumerate(values):
        try:
            x = float(raw)
        except (TypeError, ValueError) as error:
            raise PreviewError(f"{label}[{i}] must be a number") from error
        if not math.isfinite(x):
            raise PreviewError(f"{label}[{i}] must be finite")
        out.append(x)
    return out


def _finite_close_q(values, label):
    if not isinstance(values, (list, tuple)) or len(values) != AXIS_COUNT:
        raise PreviewError(f"{label} must be a length-{AXIS_COUNT} list")
    out = []
    for i, raw in enumerate(values):
        try:
            x = float(raw)
        except (TypeError, ValueError) as error:
            raise PreviewError(f"{label}[{i}] must be a number") from error
        if not math.isfinite(x):
            raise PreviewError(f"{label}[{i}] must be finite")
        out.append(x)
    return out


def travel_m(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)


def delta_xyz_m(a, b):
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]


def load_waypoints(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise PreviewError(f"waypoints file missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PreviewError(f"waypoints JSON invalid: {error}") from error
    if not isinstance(document, dict):
        raise PreviewError("waypoints root must be an object")
    schema = document.get("schema")
    if schema not in (SCHEMA_V1, SCHEMA_V2):
        raise PreviewError(f"unsupported waypoints schema: {schema!r}")
    if not isinstance(document.get("slots"), dict):
        raise PreviewError("waypoints.slots must be an object")
    return document


def load_right_max_close_q(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise PreviewError(f"grip cap file missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PreviewError(f"grip cap JSON invalid: {error}") from error
    if not isinstance(document, dict):
        raise PreviewError("grip cap root must be an object")
    sides = document.get("sides")
    if not isinstance(sides, dict):
        raise PreviewError("grip cap.sides must be an object")
    right = sides.get("right")
    if right is None:
        raise PreviewError("grip cap missing sides.right (preview needs right max_close_q)")
    if not isinstance(right, dict):
        raise PreviewError("grip cap.sides.right must be an object")
    if "max_close_q" not in right or right["max_close_q"] is None:
        raise PreviewError("grip cap.sides.right.max_close_q missing")
    return _finite_close_q(right["max_close_q"], "sides.right.max_close_q")


def _require_nested_position(entry, field, slot_name):
    nested = entry.get(field)
    if nested is None:
        raise PreviewError(f"{slot_name}.{field} missing — refuse preview")
    if not isinstance(nested, dict):
        raise PreviewError(f"{slot_name}.{field} must be an object")
    if "position_m" not in nested or nested["position_m"] is None:
        raise PreviewError(f"{slot_name}.{field}.position_m missing — refuse preview")
    return _finite_vec3(nested["position_m"], f"{slot_name}.{field}.position_m")


def build_preview(waypoints, right_max_close_q, slot_name=DEFAULT_SLOT):
    if slot_name not in LEVER_SLOTS:
        raise PreviewError(f"slot {slot_name!r} is not a right lever slot")
    slots = waypoints.get("slots") or {}
    entry = slots.get(slot_name)
    if entry is None:
        raise PreviewError(f"slot {slot_name} missing in waypoints")
    if not isinstance(entry, dict):
        raise PreviewError(f"slot {slot_name} must be an object")

    grasp = _finite_vec3(entry.get("position_m"), f"{slot_name}.position_m")
    pulled = _require_nested_position(entry, "pulled", slot_name)
    if entry.get("wait_s") is None:
        raise PreviewError(f"{slot_name}.wait_s missing — refuse preview")
    try:
        wait_s = float(entry["wait_s"])
    except (TypeError, ValueError) as error:
        raise PreviewError(f"{slot_name}.wait_s must be a number") from error
    if not math.isfinite(wait_s) or wait_s <= 0.0:
        raise PreviewError(f"{slot_name}.wait_s must be > 0")
    withdraw = _require_nested_position(entry, "withdraw", slot_name)

    if waypoints.get("turn_waist_yaw_rad") is None:
        raise PreviewError("turn_waist_yaw_rad missing — refuse preview")
    try:
        turn_rad = float(waypoints["turn_waist_yaw_rad"])
    except (TypeError, ValueError) as error:
        raise PreviewError("turn_waist_yaw_rad must be a number") from error
    if not math.isfinite(turn_rad):
        raise PreviewError("turn_waist_yaw_rad must be finite")

    pull_travel = travel_m(grasp, pulled)
    pull_delta = delta_xyz_m(grasp, pulled)
    withdraw_from_grasp = travel_m(grasp, withdraw)
    withdraw_from_pulled = travel_m(pulled, withdraw)
    turn_deg = math.degrees(turn_rad)

    left_present = "left_cup" in slots and slots.get("left_cup") is not None

    return {
        "slot": slot_name,
        "right_max_close_q": list(right_max_close_q),
        "grasp_m": grasp,
        "pulled_m": pulled,
        "pull_travel_m": pull_travel,
        "pull_delta_m": pull_delta,
        "wait_s": wait_s,
        "withdraw_m": withdraw,
        "withdraw_from_grasp_m": withdraw_from_grasp,
        "withdraw_from_pulled_m": withdraw_from_pulled,
        "turn_waist_yaw_rad": turn_rad,
        "turn_waist_yaw_deg": turn_deg,
        "left_cup_present": left_present,
        "sends_robot_commands": False,
    }


def format_preview(preview):
    q = preview["right_max_close_q"]
    q_txt = "[" + ", ".join(f"{x:.4g}" for x in q) + "]"
    grasp = preview["grasp_m"]
    pulled = preview["pulled_m"]
    withdraw = preview["withdraw_m"]
    d_cm = preview["pull_travel_m"] * 100.0
    dx, dy, dz = preview["pull_delta_m"]
    w_grasp_cm = preview["withdraw_from_grasp_m"] * 100.0
    w_pulled_cm = preview["withdraw_from_pulled_m"] * 100.0
    wait_s = preview["wait_s"]
    turn_deg = preview["turn_waist_yaw_deg"]
    turn_rad = preview["turn_waist_yaw_rad"]

    lines = [
        f"DRY-RUN 预览：槽位 {preview['slot']}（只打印顺序与距离，不发命令）",
        f"sends_robot_commands: {str(preview['sends_robot_commands']).lower()}",
        "",
        "右手 / 腰序列：",
        (
            f"1. 右手按已保存 grip cap 闭合到 max_close_q={q_txt}，"
            f"同时腕部到 grasp xyz=[{grasp[0]:.4f}, {grasp[1]:.4f}, {grasp[2]:.4f}] m"
        ),
        (
            f"2. 腕部移到 pulled xyz=[{pulled[0]:.4f}, {pulled[1]:.4f}, {pulled[2]:.4f}] m；"
            f"行程 {d_cm:.2f} cm；"
            f"Δxyz=[{dx:+.4f}, {dy:+.4f}, {dz:+.4f}] m"
        ),
        f"3. 等待 {wait_s:.1f} 秒",
        (
            f"4. 回到 grasp（推回拨杆）；同样行程 {d_cm:.2f} cm；"
            f"Δxyz=[{-dx:+.4f}, {-dy:+.4f}, {-dz:+.4f}] m"
        ),
        (
            f"5. 松开右手（本预览不使用左手 cap）并到 withdraw "
            f"xyz=[{withdraw[0]:.4f}, {withdraw[1]:.4f}, {withdraw[2]:.4f}] m；"
            f"距 grasp {w_grasp_cm:.2f} cm；距 pulled {w_pulled_cm:.2f} cm"
        ),
        (
            f"6. 腰 yaw 转到已保存角度 turn_waist_yaw_rad={turn_rad:.4f} rad "
            f"（约 {turn_deg:.2f}°）"
        ),
        "",
        "左手：",
        (
            "本预览不命令左手 / 不套用左手 grip cap"
            + ("（文件里虽有 left_cup，但不参与本次运动）" if preview["left_cup_present"] else "（waypoints 中无 left_cup）")
            + "；左手保持 uncapped-by-this-preview。"
        ),
        "",
        "说明：在后续回放工具存在且你明确启动之前，机械臂不会动。"
        "本工具不发 lowcmd。q 不是急停。",
    ]
    return "\n".join(lines)


def module_publishes_lowcmd(source_text=None):
    """Static check used by unit tests: this file must not publish lowcmd."""
    text = source_text if source_text is not None else Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden = ("lowcmd", "LowCmd", "ChannelPublisher", "Enter_Debug_Mode", "publish")
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            hits.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in ("lowcmd", "LowCmd", "publish"):
            hits.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Allow documentary mentions of lowcmd in strings/docstrings.
            continue
    return hits


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="DRY-RUN preview of right-lever ice-cream sequence (no robot motion)."
    )
    parser.add_argument(
        "--slot",
        default=DEFAULT_SLOT,
        choices=sorted(LEVER_SLOTS),
        help=f"lever slot (default {DEFAULT_SLOT})",
    )
    parser.add_argument(
        "--waypoints",
        type=Path,
        default=None,
        help="JSON path (default ~/.config/xr_teleoperate/icecream_waypoints.json)",
    )
    parser.add_argument(
        "--grip-cap",
        type=Path,
        default=None,
        help="JSON path (default ~/.config/xr_teleoperate/o6_grip_cap.json)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    waypoints_path = args.waypoints or default_waypoints_path()
    grip_path = args.grip_cap or default_grip_cap_path()
    waypoints = load_waypoints(waypoints_path)
    right_q = load_right_max_close_q(grip_path)
    preview = build_preview(waypoints, right_q, slot_name=args.slot)
    print(format_preview(preview))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreviewError as error:
        print(f"preview refused: {error}", file=sys.stderr)
        raise SystemExit(2)
