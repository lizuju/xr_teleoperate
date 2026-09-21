"""Load published R1/O6 commands from an xr_teleop_episode_v2 recording.

This module never talks to DDS, IK, cameras, or the robot. It turns
``frames.jsonl`` into time-stamped joint waypoints so a separate runner can
stream them. Replay uses ``sample.commands.*.published`` (the last SDK Write),
not XR/IK ``actions`` and not measured ``states``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "xr_teleop_episode_v2"
ARM_DOF = 14
HEAD_DOF = 2
HAND_DOF = 6
DEFAULT_APPROACH_HZ = 40.0
MIN_APPROACH_S = 1.0
DEFAULT_MAX_APPROACH_S = 8.0
# Conservative approach speeds. These are not the revoked 3.0 rad/s teleop cap.
APPROACH_ARM_VEL = 1.0
APPROACH_HEAD_VEL = 0.4
APPROACH_WAIST_VEL = math.radians(20.0)
APPROACH_HAND_VEL = 0.4
MAX_PLAYBACK_SPEED = 1.0


class ReplayError(ValueError):
    """The episode cannot be replayed as recorded."""


@dataclass(frozen=True)
class ReplayPose:
    arm_q: np.ndarray
    arm_tau: np.ndarray
    head_q: np.ndarray
    waist_q: float
    left_q: np.ndarray
    right_q: np.ndarray
    left_mode: int
    right_mode: int

    def copy(self):
        return ReplayPose(
            arm_q=np.array(self.arm_q, dtype=np.float64, copy=True),
            arm_tau=np.array(self.arm_tau, dtype=np.float64, copy=True),
            head_q=np.array(self.head_q, dtype=np.float64, copy=True),
            waist_q=float(self.waist_q),
            left_q=np.array(self.left_q, dtype=np.float64, copy=True),
            right_q=np.array(self.right_q, dtype=np.float64, copy=True),
            left_mode=int(self.left_mode),
            right_mode=int(self.right_mode),
        )


@dataclass(frozen=True)
class ReplayWaypoint(ReplayPose):
    monotonic_ns: int
    frame_idx: int


@dataclass
class ReplayPlan:
    episode: str
    schema: str
    status: str
    outcome: str
    waypoints: list = field(default_factory=list)
    skipped_leading: int = 0
    held_unpublished: int = 0
    duration_s: float = 0.0
    max_gap_ms: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def first(self):
        return self.waypoints[0]

    @property
    def last(self):
        return self.waypoints[-1]


def episode_directory(path):
    path = Path(path).resolve()
    return path.parent if path.name == "episode.json" else path


def _finite_vector(value, length, location):
    if not isinstance(value, list) or len(value) != length:
        raise ReplayError(f"{location}: expected {length} numeric values")
    if any(type(item) not in (int, float) or isinstance(item, bool) for item in value):
        raise ReplayError(f"{location}: expected {length} numeric values")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ReplayError(f"{location}: non-finite number")
    return array


def _finite_scalar(value, location):
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ReplayError(f"{location}: expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ReplayError(f"{location}: non-finite number")
    return number


def _mode(value, location):
    if value not in (0, 1):
        raise ReplayError(f"{location}: expected hand mode 0 or 1")
    return int(value)


def _published_pose(commands, location):
    if not isinstance(commands, dict):
        return None
    arm = commands.get("arm")
    hands = commands.get("hands")
    if not isinstance(arm, dict) or not isinstance(hands, dict):
        return None
    published_arm = arm.get("published")
    published_hands = hands.get("published")
    if not isinstance(published_arm, dict) or not isinstance(published_hands, dict):
        return None
    left = published_hands.get("left")
    right = published_hands.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    left_q = _finite_vector(left.get("q"), HAND_DOF, f"{location}.hands.left.q")
    right_q = _finite_vector(right.get("q"), HAND_DOF, f"{location}.hands.right.q")
    if np.any(left_q < -1e-6) or np.any(left_q > 1.0 + 1e-6):
        raise ReplayError(f"{location}.hands.left.q: expected values in [0, 1]")
    if np.any(right_q < -1e-6) or np.any(right_q > 1.0 + 1e-6):
        raise ReplayError(f"{location}.hands.right.q: expected values in [0, 1]")
    return ReplayPose(
        arm_q=_finite_vector(published_arm.get("arm_q"), ARM_DOF, f"{location}.arm_q"),
        arm_tau=_finite_vector(published_arm.get("arm_tau"), ARM_DOF, f"{location}.arm_tau"),
        head_q=_finite_vector(published_arm.get("head_q"), HEAD_DOF, f"{location}.head_q"),
        waist_q=_finite_scalar(published_arm.get("waist_q"), f"{location}.waist_q"),
        left_q=np.clip(left_q, 0.0, 1.0),
        right_q=np.clip(right_q, 0.0, 1.0),
        left_mode=_mode(left.get("mode"), f"{location}.hands.left.mode"),
        right_mode=_mode(right.get("mode"), f"{location}.hands.right.mode"),
    )


def load_replay_plan(path):
    directory = episode_directory(path)
    manifest_path = directory / "episode.json"
    frames_path = directory / "frames.jsonl"
    if not manifest_path.is_file():
        raise ReplayError(f"missing episode.json in {directory}")
    if not frames_path.is_file():
        raise ReplayError(f"missing frames.jsonl in {directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReplayError(f"episode.json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ReplayError("episode.json: expected an object")
    schema = manifest.get("schema")
    status = manifest.get("status")
    outcome = manifest.get("outcome")
    if schema != SCHEMA:
        raise ReplayError(f"manifest.schema: expected {SCHEMA}, got {schema!r}")
    if status != "complete":
        raise ReplayError(f"episode status is {status!r}, not complete")
    if outcome == "discarded":
        raise ReplayError("episode outcome is discarded")
    info = manifest.get("info") if isinstance(manifest.get("info"), dict) else {}
    if info.get("robot") != "R1_A7" or info.get("end_effector") != "linker_o6":
        raise ReplayError("robot replay requires info.robot=R1_A7 and end_effector=linker_o6")

    waypoints = []
    skipped_leading = 0
    held_unpublished = 0
    previous = None
    previous_ns = None
    max_gap_ms = 0.0
    try:
        stream = frames_path.open(encoding="utf-8")
    except OSError as exc:
        raise ReplayError(f"frames.jsonl: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
            except ValueError as exc:
                raise ReplayError(f"frames.jsonl line {line_number}: invalid JSON") from exc
            if not isinstance(frame, dict):
                raise ReplayError(f"frames.jsonl line {line_number}: expected an object")
            sample = frame.get("sample")
            if not isinstance(sample, dict):
                raise ReplayError(f"frames.jsonl line {line_number}: missing sample")
            monotonic_ns = sample.get("monotonic_ns")
            if type(monotonic_ns) is not int or monotonic_ns < 0:
                raise ReplayError(f"frames.jsonl line {line_number}: monotonic_ns must be a non-negative int")
            if previous_ns is not None and monotonic_ns < previous_ns:
                raise ReplayError(f"frames.jsonl line {line_number}: timestamp moved backwards")
            if previous_ns is not None:
                max_gap_ms = max(max_gap_ms, (monotonic_ns - previous_ns) / 1e6)
            location = f"frame {frame.get('idx', line_number - 1)}"
            pose = _published_pose(sample.get("commands"), location)
            if pose is None:
                if previous is None:
                    skipped_leading += 1
                    previous_ns = monotonic_ns
                    continue
                pose = previous
                held_unpublished += 1
            waypoint = ReplayWaypoint(
                arm_q=pose.arm_q,
                arm_tau=pose.arm_tau,
                head_q=pose.head_q,
                waist_q=pose.waist_q,
                left_q=pose.left_q,
                right_q=pose.right_q,
                left_mode=pose.left_mode,
                right_mode=pose.right_mode,
                monotonic_ns=monotonic_ns,
                frame_idx=frame.get("idx") if type(frame.get("idx")) is int else line_number - 1,
            )
            waypoints.append(waypoint)
            previous = pose
            previous_ns = monotonic_ns

    if not waypoints:
        raise ReplayError("episode has no published arm and hand commands to replay")
    duration_s = (waypoints[-1].monotonic_ns - waypoints[0].monotonic_ns) / 1e9
    warnings = []
    if outcome not in ("success", "failure"):
        warnings.append(f"episode outcome is {outcome!r}; replaying unlabeled data")
    if skipped_leading:
        warnings.append(f"skipped {skipped_leading} leading frames without published commands")
    if held_unpublished:
        warnings.append(f"held previous command through {held_unpublished} unpublished frames")
    return ReplayPlan(
        episode=str(directory),
        schema=schema,
        status=status,
        outcome=outcome,
        waypoints=waypoints,
        skipped_leading=skipped_leading,
        held_unpublished=held_unpublished,
        duration_s=duration_s,
        max_gap_ms=max_gap_ms,
        warnings=warnings,
    )


def lerp_pose(start, end, alpha):
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ReplayError("interpolation alpha must be finite")
    alpha = min(1.0, max(0.0, alpha))
    return ReplayPose(
        arm_q=start.arm_q + alpha * (end.arm_q - start.arm_q),
        arm_tau=start.arm_tau + alpha * (end.arm_tau - start.arm_tau),
        head_q=start.head_q + alpha * (end.head_q - start.head_q),
        waist_q=start.waist_q + alpha * (end.waist_q - start.waist_q),
        left_q=start.left_q + alpha * (end.left_q - start.left_q),
        right_q=start.right_q + alpha * (end.right_q - start.right_q),
        left_mode=end.left_mode if alpha >= 1.0 else start.left_mode if alpha <= 0.0 else end.left_mode,
        right_mode=end.right_mode if alpha >= 1.0 else start.right_mode if alpha <= 0.0 else end.right_mode,
    )


def approach_limits(arm=None, head=None, waist=None, hand=None):
    return {
        "arm": APPROACH_ARM_VEL if arm is None else float(arm),
        "head": APPROACH_HEAD_VEL if head is None else float(head),
        "waist": APPROACH_WAIST_VEL if waist is None else float(waist),
        "hand": APPROACH_HAND_VEL if hand is None else float(hand),
    }


def pose_deltas(current, target):
    return {
        "arm": float(np.max(np.abs(target.arm_q - current.arm_q))),
        "head": float(np.max(np.abs(target.head_q - current.head_q))),
        "waist": float(abs(target.waist_q - current.waist_q)),
        "hand": float(max(
            np.max(np.abs(target.left_q - current.left_q)),
            np.max(np.abs(target.right_q - current.right_q)),
        )),
    }


def approach_duration(current, target, limits=None, min_seconds=MIN_APPROACH_S, max_seconds=DEFAULT_MAX_APPROACH_S):
    limits = approach_limits() if limits is None else limits
    for name, value in limits.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ReplayError(f"approach {name} velocity must be positive")
    if not math.isfinite(min_seconds) or min_seconds < 0.0:
        raise ReplayError("min approach duration must be non-negative")
    if not math.isfinite(max_seconds) or max_seconds <= 0.0:
        raise ReplayError("max approach duration must be positive")
    if min_seconds > max_seconds:
        raise ReplayError("min approach duration exceeds max")
    deltas = pose_deltas(current, target)
    needed = max(
        deltas["arm"] / limits["arm"],
        deltas["head"] / limits["head"],
        deltas["waist"] / limits["waist"],
        deltas["hand"] / limits["hand"],
        min_seconds,
    )
    if needed > max_seconds + 1e-9:
        raise ReplayError(
            "start pose is too far from the first recorded command "
            f"(need {needed:.2f}s at approach limits; max {max_seconds:.2f}s). "
            "Move the robot closer to the recorded start, or raise --max-approach-seconds."
        )
    return needed, deltas


def build_approach_poses(current, target, duration, hz=DEFAULT_APPROACH_HZ):
    if not math.isfinite(duration) or duration <= 0.0:
        raise ReplayError("approach duration must be positive")
    if not math.isfinite(hz) or hz <= 0.0:
        raise ReplayError("approach hz must be positive")
    steps = max(1, int(round(duration * hz)))
    # k=1..steps so the first commanded pose is a small step away from current,
    # never a snap onto the recorded first frame.
    return [lerp_pose(current, target, k / steps) for k in range(1, steps + 1)]


def sample_at_time(waypoints, query_ns):
    if not waypoints:
        raise ReplayError("no waypoints")
    if type(query_ns) is not int:
        raise ReplayError("query timestamp must be an int")
    if query_ns < waypoints[0].monotonic_ns:
        return waypoints[0]
    selected = waypoints[0]
    for waypoint in waypoints:
        if waypoint.monotonic_ns > query_ns:
            break
        selected = waypoint
    return selected


def playback_clock_ns(start_ns, elapsed_s, speed=1.0):
    if not math.isfinite(speed) or speed <= 0.0:
        raise ReplayError("playback speed must be positive")
    if speed > MAX_PLAYBACK_SPEED:
        raise ReplayError(f"playback speed must be <= {MAX_PLAYBACK_SPEED} (no speeding up)")
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ReplayError("elapsed time must be a non-negative finite number")
    return start_ns + int(elapsed_s * speed * 1e9)


def plan_summary(plan):
    first = plan.first
    last = plan.last
    return {
        "episode": plan.episode,
        "schema": plan.schema,
        "status": plan.status,
        "outcome": plan.outcome,
        "waypoints": len(plan.waypoints),
        "duration_s": plan.duration_s,
        "max_gap_ms": plan.max_gap_ms,
        "skipped_leading": plan.skipped_leading,
        "held_unpublished": plan.held_unpublished,
        "first_frame": first.frame_idx,
        "last_frame": last.frame_idx,
        "first_waist_q": first.waist_q,
        "last_waist_q": last.waist_q,
        "warnings": list(plan.warnings),
        "sends_robot_commands": False,
    }
