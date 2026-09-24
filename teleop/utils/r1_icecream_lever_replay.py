"""Plan right-lever ice-cream playback from taught waypoints (no DDS).

Replays only the recorded right-arm joints. Left arm, left hand, and head stay
at the live measured values captured at start. Waist holds the grasp-pose yaw
through approach → pull → wait → return → withdraw, then ramps to
``turn_waist_yaw_rad``.

This module never imports Unitree publishers or robot controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_V2 = "icecream_waypoints_v2"
SCHEMA_V1 = "icecream_waypoints_v1"
DEFAULT_WAYPOINTS = Path.home() / ".config" / "xr_teleoperate" / "icecream_waypoints.json"
DEFAULT_GRIP_CAP = Path.home() / ".config" / "xr_teleoperate" / "o6_grip_cap.json"
DEFAULT_SLOT = "right_lever_3"
LEVER_SLOTS = frozenset({"right_lever_1", "right_lever_2", "right_lever_3"})

PINOCCHIO_DOF = 17
ARM_DOF = 7
DUAL_ARM_DOF = 14
HEAD_DOF = 2
HAND_DOF = 6

DEFAULT_HZ = 40.0
MIN_APPROACH_S = 1.0
DEFAULT_MAX_APPROACH_S = 15.0
# Conservative. Not the revoked 3.0 rad/s position cap.
APPROACH_RIGHT_ARM_VEL = 0.6  # rad/s
APPROACH_WAIST_VEL = math.radians(10.0)  # 10 deg/s
SEGMENT_RIGHT_ARM_VEL = 0.6
SEGMENT_WAIST_VEL = math.radians(10.0)
HAND_CLOSE_VEL = 0.4
MIN_SEGMENT_S = 0.25
# Informational only: withdraw may store a different waist; playback keeps grasp waist
# through withdraw and only ramps on step 6. Do not refuse on this spread.
WAIST_SPREAD_NOTE_DEG = 5.0


class IcecreamLeverError(ValueError):
    """The lever sequence cannot be replayed safely."""


@dataclass(frozen=True)
class LeverPose:
    """One commanded sample: dual-arm 14, head 2, waist, hands."""

    arm_q: np.ndarray  # live_left(7) + right(7)
    head_q: np.ndarray
    waist_q: float
    left_q: np.ndarray
    right_q: np.ndarray
    left_fresh: bool
    right_fresh: bool

    def copy(self):
        return LeverPose(
            arm_q=np.array(self.arm_q, dtype=np.float64, copy=True),
            head_q=np.array(self.head_q, dtype=np.float64, copy=True),
            waist_q=float(self.waist_q),
            left_q=np.array(self.left_q, dtype=np.float64, copy=True),
            right_q=np.array(self.right_q, dtype=np.float64, copy=True),
            left_fresh=bool(self.left_fresh),
            right_fresh=bool(self.right_fresh),
        )


@dataclass
class LeverSegment:
    name: str
    duration_s: float
    poses: list = field(default_factory=list)


@dataclass
class LeverPlan:
    slot: str
    right_max_close_q: list
    grasp_m: list
    pulled_m: list
    pull_travel_m: float
    pull_delta_m: list
    wait_s: float
    withdraw_m: list
    withdraw_from_grasp_m: float
    turn_waist_yaw_rad: float
    turn_waist_yaw_deg: float
    grasp_waist_rad: float
    grasp_waist_deg: float
    pulled_waist_rad: float
    withdraw_waist_rad: float
    grasp_right_arm: list
    pulled_right_arm: list
    withdraw_right_arm: list
    left_cup_present: bool
    segments: list = field(default_factory=list)
    sends_robot_commands: bool = False
    warnings: list = field(default_factory=list)


def default_waypoints_path():
    import os

    return Path(os.environ.get("XR_ICECREAM_WAYPOINTS", str(DEFAULT_WAYPOINTS))).expanduser()


def default_grip_cap_path():
    import os

    return Path(os.environ.get("XR_O6_GRIP_CAP", str(DEFAULT_GRIP_CAP))).expanduser()


def _finite_vec(values, length, label):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise IcecreamLeverError(f"{label} must be a length-{length} list")
    out = []
    for i, raw in enumerate(values):
        try:
            x = float(raw)
        except (TypeError, ValueError) as error:
            raise IcecreamLeverError(f"{label}[{i}] must be a number") from error
        if not math.isfinite(x):
            raise IcecreamLeverError(f"{label}[{i}] must be finite")
        out.append(x)
    return out


def travel_m(a, b):
    return math.sqrt(sum((bi - ai) ** 2 for ai, bi in zip(a, b)))


def delta_xyz_m(a, b):
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]


def load_waypoints(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise IcecreamLeverError(f"waypoints file missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise IcecreamLeverError(f"waypoints JSON invalid: {error}") from error
    if not isinstance(document, dict):
        raise IcecreamLeverError("waypoints root must be an object")
    schema = document.get("schema")
    if schema not in (SCHEMA_V1, SCHEMA_V2):
        raise IcecreamLeverError(f"unsupported waypoints schema: {schema!r}")
    if not isinstance(document.get("slots"), dict):
        raise IcecreamLeverError("waypoints.slots must be an object")
    return document


def load_right_max_close_q(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise IcecreamLeverError(f"grip cap file missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise IcecreamLeverError(f"grip cap JSON invalid: {error}") from error
    if not isinstance(document, dict):
        raise IcecreamLeverError("grip cap root must be an object")
    sides = document.get("sides")
    if not isinstance(sides, dict):
        raise IcecreamLeverError("grip cap.sides must be an object")
    right = sides.get("right")
    if not isinstance(right, dict) or right.get("max_close_q") is None:
        raise IcecreamLeverError("grip cap.sides.right.max_close_q missing")
    return _finite_vec(right["max_close_q"], HAND_DOF, "sides.right.max_close_q")


def _pinocchio_parts(pinocchio_q, label):
    q = _finite_vec(pinocchio_q, PINOCCHIO_DOF, label)
    return {
        "waist_yaw": q[0],
        "head": q[1:3],
        "left_arm": q[3:10],
        "right_arm": q[10:17],
        "pinocchio_q": q,
    }


def _require_nested_pose(entry, field, slot_name):
    nested = entry.get(field)
    if nested is None:
        raise IcecreamLeverError(f"{slot_name}.{field} missing")
    if not isinstance(nested, dict):
        raise IcecreamLeverError(f"{slot_name}.{field} must be an object")
    if nested.get("position_m") is None:
        raise IcecreamLeverError(f"{slot_name}.{field}.position_m missing")
    if nested.get("pinocchio_q") is None:
        raise IcecreamLeverError(f"{slot_name}.{field}.pinocchio_q missing")
    position = _finite_vec(nested["position_m"], 3, f"{slot_name}.{field}.position_m")
    parts = _pinocchio_parts(nested["pinocchio_q"], f"{slot_name}.{field}.pinocchio_q")
    return position, parts


def extract_lever_slot(waypoints, right_max_close_q, slot_name=DEFAULT_SLOT):
    """Validate waypoints and return a motion-ready LeverPlan shell (no poses)."""
    if slot_name not in LEVER_SLOTS:
        raise IcecreamLeverError(f"slot {slot_name!r} is not a right lever slot")
    slots = waypoints.get("slots") or {}
    entry = slots.get(slot_name)
    if entry is None:
        raise IcecreamLeverError(f"slot {slot_name} missing in waypoints")
    if not isinstance(entry, dict):
        raise IcecreamLeverError(f"slot {slot_name} must be an object")
    if entry.get("pinocchio_q") is None:
        raise IcecreamLeverError(f"{slot_name}.pinocchio_q missing")
    if entry.get("position_m") is None:
        raise IcecreamLeverError(f"{slot_name}.position_m missing")
    if entry.get("wait_s") is None:
        raise IcecreamLeverError(f"{slot_name}.wait_s missing")
    try:
        wait_s = float(entry["wait_s"])
    except (TypeError, ValueError) as error:
        raise IcecreamLeverError(f"{slot_name}.wait_s must be a number") from error
    if not math.isfinite(wait_s) or wait_s <= 0.0:
        raise IcecreamLeverError(f"{slot_name}.wait_s must be > 0")
    if waypoints.get("turn_waist_yaw_rad") is None:
        raise IcecreamLeverError("turn_waist_yaw_rad missing")
    try:
        turn_rad = float(waypoints["turn_waist_yaw_rad"])
    except (TypeError, ValueError) as error:
        raise IcecreamLeverError("turn_waist_yaw_rad must be a number") from error
    if not math.isfinite(turn_rad):
        raise IcecreamLeverError("turn_waist_yaw_rad must be finite")

    grasp = _finite_vec(entry["position_m"], 3, f"{slot_name}.position_m")
    grasp_parts = _pinocchio_parts(entry["pinocchio_q"], f"{slot_name}.pinocchio_q")
    pulled, pulled_parts = _require_nested_pose(entry, "pulled", slot_name)
    withdraw, withdraw_parts = _require_nested_pose(entry, "withdraw", slot_name)

    grasp_w = grasp_parts["waist_yaw"]
    pulled_w = pulled_parts["waist_yaw"]
    withdraw_w = withdraw_parts["waist_yaw"]
    spreads_deg = {
        "grasp_vs_pulled_deg": abs(math.degrees(grasp_w - pulled_w)),
        "grasp_vs_withdraw_deg": abs(math.degrees(grasp_w - withdraw_w)),
        "pulled_vs_withdraw_deg": abs(math.degrees(pulled_w - withdraw_w)),
    }
    max_spread = max(spreads_deg.values())
    warnings = []
    if max_spread > WAIST_SPREAD_NOTE_DEG + 1e-9:
        # Pass through: approach / pull / wait / return / withdraw keep grasp waist;
        # withdraw only moves the right arm. Step 6 ramps to turn_waist_yaw_rad.
        warnings.append(
            "recorded waist yaws for grasp/pulled/withdraw differ "
            f"(max spread {max_spread:.2f}° > {WAIST_SPREAD_NOTE_DEG:.1f}° note); "
            f"using grasp waist {math.degrees(grasp_w):.2f}° through withdraw "
            f"(stored withdraw waist {math.degrees(withdraw_w):.2f}° ignored for yaw). "
            f"details={spreads_deg}"
        )

    pull_travel = travel_m(grasp, pulled)
    pull_delta = delta_xyz_m(grasp, pulled)
    left_present = "left_cup" in slots and slots.get("left_cup") is not None

    return LeverPlan(
        slot=slot_name,
        right_max_close_q=list(right_max_close_q),
        grasp_m=grasp,
        pulled_m=pulled,
        pull_travel_m=pull_travel,
        pull_delta_m=pull_delta,
        wait_s=wait_s,
        withdraw_m=withdraw,
        withdraw_from_grasp_m=travel_m(grasp, withdraw),
        turn_waist_yaw_rad=turn_rad,
        turn_waist_yaw_deg=math.degrees(turn_rad),
        grasp_waist_rad=grasp_w,
        grasp_waist_deg=math.degrees(grasp_w),
        pulled_waist_rad=pulled_w,
        withdraw_waist_rad=withdraw_w,
        grasp_right_arm=list(grasp_parts["right_arm"]),
        pulled_right_arm=list(pulled_parts["right_arm"]),
        withdraw_right_arm=list(withdraw_parts["right_arm"]),
        left_cup_present=left_present,
        segments=[],
        sends_robot_commands=False,
        warnings=warnings,
    )


def build_arm14(live_left7, right7):
    left = np.asarray(live_left7, dtype=np.float64).reshape(ARM_DOF)
    right = np.asarray(right7, dtype=np.float64).reshape(ARM_DOF)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise IcecreamLeverError("arm joints must be finite")
    return np.concatenate([left, right])


def live_to_grasp_deltas(live_right7, live_waist, grasp_right7, grasp_waist):
    live_r = np.asarray(live_right7, dtype=np.float64).reshape(ARM_DOF)
    grasp_r = np.asarray(grasp_right7, dtype=np.float64).reshape(ARM_DOF)
    return {
        "right_arm_max_abs_dq": float(np.max(np.abs(grasp_r - live_r))),
        "waist_delta_rad": float(grasp_waist - live_waist),
        "waist_delta_deg": float(math.degrees(grasp_waist - live_waist)),
    }


def approach_duration_right_waist(
    live_right7,
    live_waist,
    target_right7,
    target_waist,
    *,
    arm_vel=APPROACH_RIGHT_ARM_VEL,
    waist_vel=APPROACH_WAIST_VEL,
    min_seconds=MIN_APPROACH_S,
    max_seconds=DEFAULT_MAX_APPROACH_S,
):
    if not math.isfinite(arm_vel) or arm_vel <= 0.0:
        raise IcecreamLeverError("approach arm velocity must be positive")
    if not math.isfinite(waist_vel) or waist_vel <= 0.0:
        raise IcecreamLeverError("approach waist velocity must be positive")
    if not math.isfinite(min_seconds) or min_seconds < 0.0:
        raise IcecreamLeverError("min approach duration must be non-negative")
    if not math.isfinite(max_seconds) or max_seconds <= 0.0:
        raise IcecreamLeverError("max approach duration must be positive")
    deltas = live_to_grasp_deltas(live_right7, live_waist, target_right7, target_waist)
    needed = max(
        deltas["right_arm_max_abs_dq"] / arm_vel,
        abs(deltas["waist_delta_rad"]) / waist_vel,
        min_seconds,
    )
    if needed > max_seconds + 1e-9:
        raise IcecreamLeverError(
            "live pose is too far from grasp for a safe approach "
            f"(need {needed:.2f}s at limits; max {max_seconds:.2f}s). "
            f"right_arm_max|Δq|={deltas['right_arm_max_abs_dq']:.3f} rad, "
            f"waist Δ={deltas['waist_delta_deg']:.2f}°. "
            "Move closer to the grasp heading / posture, or raise --max-approach-seconds."
        )
    return needed, deltas


def _segment_duration(right_a, right_b, waist_a, waist_b, hand_a=None, hand_b=None,
                      arm_vel=SEGMENT_RIGHT_ARM_VEL, waist_vel=SEGMENT_WAIST_VEL,
                      hand_vel=HAND_CLOSE_VEL, min_seconds=MIN_SEGMENT_S):
    ra = np.asarray(right_a, dtype=np.float64).reshape(ARM_DOF)
    rb = np.asarray(right_b, dtype=np.float64).reshape(ARM_DOF)
    needed = max(
        float(np.max(np.abs(rb - ra))) / arm_vel,
        abs(float(waist_b) - float(waist_a)) / waist_vel,
        min_seconds,
    )
    if hand_a is not None and hand_b is not None:
        ha = np.asarray(hand_a, dtype=np.float64).reshape(HAND_DOF)
        hb = np.asarray(hand_b, dtype=np.float64).reshape(HAND_DOF)
        needed = max(needed, float(np.max(np.abs(hb - ha))) / hand_vel)
    return needed


def lerp_lever_pose(start: LeverPose, end: LeverPose, alpha: float) -> LeverPose:
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise IcecreamLeverError("interpolation alpha must be finite")
    alpha = min(1.0, max(0.0, alpha))
    # Fresh flags follow the end pose once motion toward it has begun (alpha>0),
    # so left stays held and right close/open matches the segment goal.
    if alpha <= 0.0:
        left_fresh, right_fresh = start.left_fresh, start.right_fresh
    else:
        left_fresh, right_fresh = end.left_fresh, end.right_fresh
    return LeverPose(
        arm_q=start.arm_q + alpha * (end.arm_q - start.arm_q),
        head_q=start.head_q + alpha * (end.head_q - start.head_q),
        waist_q=start.waist_q + alpha * (end.waist_q - start.waist_q),
        left_q=start.left_q + alpha * (end.left_q - start.left_q),
        right_q=start.right_q + alpha * (end.right_q - start.right_q),
        left_fresh=left_fresh,
        right_fresh=right_fresh,
    )


def build_ramp_poses(start: LeverPose, end: LeverPose, duration_s: float, hz=DEFAULT_HZ):
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise IcecreamLeverError("segment duration must be positive")
    if not math.isfinite(hz) or hz <= 0.0:
        raise IcecreamLeverError("hz must be positive")
    steps = max(1, int(round(duration_s * hz)))
    # k=1..steps: first command is a small step away from start (no snap).
    return [lerp_lever_pose(start, end, k / steps) for k in range(1, steps + 1)]


def build_hold_poses(pose: LeverPose, duration_s: float, hz=DEFAULT_HZ):
    if not math.isfinite(duration_s) or duration_s < 0.0:
        raise IcecreamLeverError("hold duration must be non-negative")
    if duration_s == 0.0:
        return []
    steps = max(1, int(round(duration_s * hz)))
    return [pose.copy() for _ in range(steps)]


def _make_pose(live_left, live_head, live_left_hand, right7, waist, right_hand, right_fresh):
    return LeverPose(
        arm_q=build_arm14(live_left, right7),
        head_q=np.asarray(live_head, dtype=np.float64).reshape(HEAD_DOF),
        waist_q=float(waist),
        left_q=np.asarray(live_left_hand, dtype=np.float64).reshape(HAND_DOF),
        right_q=np.asarray(right_hand, dtype=np.float64).reshape(HAND_DOF),
        left_fresh=True,
        right_fresh=bool(right_fresh),
    )


def build_motion_segments(
    plan: LeverPlan,
    *,
    live_left7,
    live_right7,
    live_head2,
    live_waist,
    live_left_hand,
    live_right_hand,
    max_approach_s=DEFAULT_MAX_APPROACH_S,
    hz=DEFAULT_HZ,
):
    """Fill ``plan.segments`` from a live snapshot. Mutates and returns plan."""
    live_left7 = _finite_vec(list(np.asarray(live_left7, dtype=float).reshape(ARM_DOF)), ARM_DOF, "live_left")
    live_right7 = _finite_vec(list(np.asarray(live_right7, dtype=float).reshape(ARM_DOF)), ARM_DOF, "live_right")
    live_head2 = _finite_vec(list(np.asarray(live_head2, dtype=float).reshape(HEAD_DOF)), HEAD_DOF, "live_head")
    live_left_hand = _finite_vec(
        list(np.asarray(live_left_hand, dtype=float).reshape(HAND_DOF)), HAND_DOF, "live_left_hand"
    )
    live_right_hand = _finite_vec(
        list(np.asarray(live_right_hand, dtype=float).reshape(HAND_DOF)), HAND_DOF, "live_right_hand"
    )
    if not math.isfinite(float(live_waist)):
        raise IcecreamLeverError("live_waist must be finite")

    grasp_w = plan.grasp_waist_rad
    close_q = plan.right_max_close_q
    open_q = [0.0] * HAND_DOF

    approach_s, deltas = approach_duration_right_waist(
        live_right7,
        live_waist,
        plan.grasp_right_arm,
        grasp_w,
        max_seconds=max_approach_s,
    )
    plan.warnings.append(
        f"live_to_grasp right_arm_max|Δq|={deltas['right_arm_max_abs_dq']:.4f} rad, "
        f"waist Δ={deltas['waist_delta_deg']:.2f}°, approach_s={approach_s:.2f}"
    )

    start = _make_pose(
        live_left7, live_head2, live_left_hand, live_right7, live_waist, live_right_hand, True
    )
    at_grasp_closed = _make_pose(
        live_left7, live_head2, live_left_hand, plan.grasp_right_arm, grasp_w, close_q, True
    )
    at_pulled = _make_pose(
        live_left7, live_head2, live_left_hand, plan.pulled_right_arm, grasp_w, close_q, True
    )
    at_grasp_return = at_grasp_closed.copy()
    at_withdraw_open = _make_pose(
        live_left7, live_head2, live_left_hand, plan.withdraw_right_arm, grasp_w, open_q, False
    )
    at_turn = _make_pose(
        live_left7,
        live_head2,
        live_left_hand,
        plan.withdraw_right_arm,
        plan.turn_waist_yaw_rad,
        open_q,
        False,
    )

    segments = []

    # 1. Approach live → grasp while closing right hand to cap (no first-frame snap).
    approach_poses = build_ramp_poses(start, at_grasp_closed, approach_s, hz=hz)
    segments.append(LeverSegment(name="approach_grasp_close", duration_s=approach_s, poses=approach_poses))

    # 2. Pull.
    pull_s = _segment_duration(
        plan.grasp_right_arm, plan.pulled_right_arm, grasp_w, grasp_w, close_q, close_q
    )
    segments.append(
        LeverSegment(
            name="pull",
            duration_s=pull_s,
            poses=build_ramp_poses(at_grasp_closed, at_pulled, pull_s, hz=hz),
        )
    )

    # 3. Wait.
    wait_poses = build_hold_poses(at_pulled, plan.wait_s, hz=hz)
    segments.append(LeverSegment(name="wait", duration_s=plan.wait_s, poses=wait_poses))

    # 4. Return to grasp.
    return_s = _segment_duration(
        plan.pulled_right_arm, plan.grasp_right_arm, grasp_w, grasp_w, close_q, close_q
    )
    segments.append(
        LeverSegment(
            name="return_grasp",
            duration_s=return_s,
            poses=build_ramp_poses(at_pulled, at_grasp_return, return_s, hz=hz),
        )
    )

    # 5. Withdraw + release right hand. Waist still at grasp yaw.
    withdraw_s = _segment_duration(
        plan.grasp_right_arm,
        plan.withdraw_right_arm,
        grasp_w,
        grasp_w,
        close_q,
        open_q,
    )
    segments.append(
        LeverSegment(
            name="withdraw_release",
            duration_s=withdraw_s,
            poses=build_ramp_poses(at_grasp_return, at_withdraw_open, withdraw_s, hz=hz),
        )
    )

    # 6. Turn waist only.
    turn_s = _segment_duration(
        plan.withdraw_right_arm,
        plan.withdraw_right_arm,
        grasp_w,
        plan.turn_waist_yaw_rad,
        open_q,
        open_q,
        min_seconds=MIN_SEGMENT_S,
    )
    # Also gate the final waist ramp with the same max-approach ceiling.
    if turn_s > max_approach_s + 1e-9:
        raise IcecreamLeverError(
            "waist turn from grasp heading to turn_waist_yaw_rad would take "
            f"{turn_s:.2f}s at {math.degrees(SEGMENT_WAIST_VEL):.1f} deg/s; "
            f"max allowed {max_approach_s:.2f}s "
            f"(Δ={plan.grasp_waist_deg - plan.turn_waist_yaw_deg:.2f}°). "
            "This is a large yaw change — refuse rather than snap."
        )
    segments.append(
        LeverSegment(
            name="turn_waist",
            duration_s=turn_s,
            poses=build_ramp_poses(at_withdraw_open, at_turn, turn_s, hz=hz),
        )
    )

    # Sanity: every commanded 14-vector keeps live left arm.
    live_left_arr = np.asarray(live_left7, dtype=np.float64)
    for segment in segments:
        for pose in segment.poses:
            if not np.allclose(pose.arm_q[:ARM_DOF], live_left_arr, atol=1e-9):
                raise IcecreamLeverError("internal error: left arm joints changed in plan")
            if not np.allclose(pose.head_q, np.asarray(live_head2, dtype=np.float64), atol=1e-9):
                raise IcecreamLeverError("internal error: head joints changed in plan")
            if not bool(pose.left_fresh):
                raise IcecreamLeverError("internal error: left hand must stay tracking_fresh")

    # After approach arrives, waist stays at grasp through withdraw; only
    # turn_waist may leave it. Approach itself ramps live waist → grasp waist.
    if segments and segments[0].name == "approach_grasp_close" and segments[0].poses:
        if abs(segments[0].poses[-1].waist_q - grasp_w) > 1e-6:
            raise IcecreamLeverError("internal error: approach did not end at grasp waist")
    for segment in segments:
        if segment.name in ("approach_grasp_close", "turn_waist"):
            continue
        for pose in segment.poses:
            if abs(pose.waist_q - grasp_w) > 1e-9:
                raise IcecreamLeverError(
                    f"internal error: waist left grasp value during {segment.name}"
                )

    plan.segments = segments
    return plan, deltas, approach_s


def format_check_only(plan: LeverPlan):
    q = plan.right_max_close_q
    q_txt = "[" + ", ".join(f"{x:.4g}" for x in q) + "]"
    grasp = plan.grasp_m
    pulled = plan.pulled_m
    withdraw = plan.withdraw_m
    d_cm = plan.pull_travel_m * 100.0
    dx, dy, dz = plan.pull_delta_m
    w_grasp_cm = plan.withdraw_from_grasp_m * 100.0
    wait_s = plan.wait_s
    turn_deg = plan.turn_waist_yaw_deg
    turn_rad = plan.turn_waist_yaw_rad
    grasp_w_deg = plan.grasp_waist_deg
    waist_delta = plan.grasp_waist_deg - plan.turn_waist_yaw_deg

    lines = [
        f"CHECK-ONLY 回放计划：槽位 {plan.slot}（不连接 DDS，不发 lowcmd）",
        f"sends_robot_commands: {str(plan.sends_robot_commands).lower()}",
        "",
        "腰航向（重要）：",
        (
            f"  示教 grasp 腰 yaw = {plan.grasp_waist_rad:.4f} rad "
            f"（约 {grasp_w_deg:.2f}°）— 接近/抓取/拉/等/回/抽手全程保持此腰"
        ),
        (
            f"  最终 turn_waist_yaw_rad = {turn_rad:.4f} rad "
            f"（约 {turn_deg:.2f}°）— 仅第 6 步从 grasp 腰斜坡转到此值"
        ),
        (
            f"  若开始时已在 grasp 航向，第 6 步约转 {abs(waist_delta):.1f}° "
            f"（grasp→turn；禁止瞬切）"
        ),
        (
            f"  记录参考：pulled 腰 {math.degrees(plan.pulled_waist_rad):.2f}°，"
            f"withdraw 腰 {math.degrees(plan.withdraw_waist_rad):.2f}°"
            f"（与 grasp 差仅作提示；抽手不跳到 withdraw 记录腰）"
        ),
        "",
        "右手 / 腰序列：",
        (
            f"1. 右手闭合到 max_close_q={q_txt}，同时右腕到 grasp "
            f"xyz=[{grasp[0]:.4f}, {grasp[1]:.4f}, {grasp[2]:.4f}] m"
            f"（从 LIVE 右臂+腰斜坡接近，无首帧瞬切）"
        ),
        (
            f"2. 右腕移到 pulled xyz=[{pulled[0]:.4f}, {pulled[1]:.4f}, {pulled[2]:.4f}] m；"
            f"行程 {d_cm:.2f} cm；"
            f"Δxyz=[{dx:+.4f}, {dy:+.4f}, {dz:+.4f}] m"
        ),
        f"3. 等待 {wait_s:.1f} 秒",
        (
            f"4. 回到 grasp（推回拨杆）；同样行程 {d_cm:.2f} cm；"
            f"Δxyz=[{-dx:+.4f}, {-dy:+.4f}, {-dz:+.4f}] m"
        ),
        (
            f"5. 松开右手并到 withdraw "
            f"xyz=[{withdraw[0]:.4f}, {withdraw[1]:.4f}, {withdraw[2]:.4f}] m；"
            f"距 grasp {w_grasp_cm:.2f} cm；腰仍为 grasp 腰（不采用 withdraw 记录腰）"
        ),
        (
            f"6. 腰 yaw 斜坡转到 turn_waist_yaw_rad={turn_rad:.4f} rad "
            f"（约 {turn_deg:.2f}°）"
        ),
        "",
        "不动：",
        "  左臂关节 = 全程 LIVE 锁定；左手 = LIVE q + tracking_fresh True（不套用左手 cap）；头 = LIVE。",
        "",
        "说明：本工具内 q / Ctrl+C = 软件急停（停轨迹、冻结 LIVE 关节、松右手、停 publisher、"
        "尽量退出 debug）；不是机器人本体断电按钮。未按 r 前不会发运动。"
        "本 --check-only 路径不连接 DDS、不发 lowcmd。",
    ]
    if plan.warnings:
        lines.append("")
        lines.append("警告：")
        for warning in plan.warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines)
