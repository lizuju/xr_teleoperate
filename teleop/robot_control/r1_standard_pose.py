"""Operator key that moves the R1 to one standard start pose between recordings.

Teleop is relative: an episode starts wherever the robot is when following begins, so starts
drift from episode to episode. o (O too) pauses following and eases the arms and head to one
fixed pose, the mean start of the demos a working model was trained on, and turns the waist
back to its activation heading (the 0 deg stop). Then s starts the next episode (r resumes
without recording); following re-anchors there, at the operator's current hands, as after any
pause. The move runs in the control loop only while paused. o again, p, or an input-loss pause
stops it where it is.

With --policy (takeover), o also moves the hands to the file's recorded start hands, so o then g
starts the policy from the same pose as run_r1_act_policy.py --start-pose standard. Without it the
hands are left alone: they follow the operator's fingers again after the resume.
"""
import json
import math
import threading
from pathlib import Path

import numpy as np

STANDARD_POSE_KEYS = ("o", "O")
DEFAULT_POSE_PATH = Path(__file__).with_name("r1_standard_start_pose.json")
ARM_SPEED_RAD_S = 0.3
HEAD_SPEED_RAD_S = 0.3
WAIST_SPEED_RAD_S = math.radians(20.0)
# O6 joints in their 0..1 units per second, the Dex1-1 gripper in rad/s.
HAND_SPEED = 1.0
MIN_SECONDS = 2.0
MAX_SECONDS = 20.0
# After the timed move the last target is sent a little longer, so the arm target shaper
# reaches it exactly before the pause holds it.
SETTLE_SECONDS = 0.5
# The move ends once the controller's own arm target is this close to the pose, or, if it
# never gets there, this long after the settle time, with a warning saying how far off it is.
CONVERGED_RAD = 0.01
MAX_EXTRA_SECONDS = 2.0
# The controller clips head targets to these (pitch, yaw) limits.
HEAD_LOWER = np.array([-0.62832, -2.0071])
HEAD_UPPER = np.array([0.62832, 2.0071])


def _vector(value, length, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must hold {length} finite values")
    return array


def load_standard_pose(path):
    """The standard pose file as {"arm_q": (14,), "head_q": (2,), "source": str}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pose = {"arm_q": _vector(data.get("arm_q"), 14, "arm_q"), "head_q": _vector(data.get("head_q"), 2, "head_q"),
            "source": str(data.get("description", path))}
    if np.any(pose["head_q"] < HEAD_LOWER) or np.any(pose["head_q"] > HEAD_UPPER):
        raise ValueError("head_q is outside the controller's head limits")
    return pose


def load_standard_hands(path):
    """The file's mean recorded start hands, (left O6 (6,), right Dex1-1 (1,)), for a policy start."""
    from teleop.utils.r1_episode_replay import O6_DEX1

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("end_effectors") != O6_DEX1:
        raise ValueError(f"a policy start needs end_effectors {O6_DEX1} in the standard pose file")
    left = _vector(data.get("left_hand_q"), 6, "left_hand_q")
    right = _vector(data.get("right_hand_q"), 1, "right_hand_q")
    if np.any(left < 0.0) or np.any(left > 1.0):
        raise ValueError("left_hand_q must be within 0..1")
    return left, right


def standard_start_pose(path):
    """The standard pose as a policy start (run_r1_act_policy.py --start-pose): (ReplayPose, end_effectors).

    Arms and head come from the file, the waist sits on its 0 stop, and the hands go to the
    file's mean recorded start. As with a recorded start, arm_tau is zero and the hand modes are 1.
    """
    from teleop.utils.r1_episode_replay import O6_DEX1, ReplayPose

    pose = load_standard_pose(path)
    left, right = load_standard_hands(path)
    return ReplayPose(arm_q=pose["arm_q"], arm_tau=np.zeros(14), head_q=pose["head_q"], waist_q=0.0,
                      left_q=left, right_q=right, left_mode=1, right_mode=1), dict(O6_DEX1)


def arm_gravity(arm_ik):
    """Gravity torque for an arm_q, from the IK's reduced model (as the IK's own feedforward)."""
    import pinocchio as pin

    model, data = arm_ik.reduced_robot.model, arm_ik.reduced_robot.data
    zeros = np.zeros(model.nv)
    return lambda q: np.asarray(pin.rnea(model, data, np.asarray(q, dtype=float), zeros, zeros))


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class StandardPoseMove:
    """A timed move from one pose to another; velocity is zero at both ends.

    Smoothstep peaks at 1.5 times the average speed, so the duration is 1.5 times the
    slowest group's distance over its speed, and at least MIN_SECONDS. waist None leaves
    the waist alone, and hands None (both start_hands and hands) leaves the hands alone.
    """

    def __init__(self, start_arm, start_head, start_waist, arm, head, waist, now,
                 arm_speed=ARM_SPEED_RAD_S, head_speed=HEAD_SPEED_RAD_S, waist_speed=WAIST_SPEED_RAD_S,
                 max_seconds=MAX_SECONDS, start_hands=None, hands=None, hand_speed=HAND_SPEED):
        self.start_arm, self.arm = _vector(start_arm, 14, "start arm_q"), _vector(arm, 14, "arm_q")
        self.start_head, self.head = _vector(start_head, 2, "start head_q"), _vector(head, 2, "head_q")
        if (start_waist is None) != (waist is None):
            raise ValueError("give both waist angles or neither")
        self.start_waist = None if start_waist is None else float(start_waist)
        self.waist = None if waist is None else float(waist)
        if (start_hands is None) != (hands is None):
            raise ValueError("give both hand poses or neither")
        self.start_hands = self.hands = None
        if hands is not None:
            self.start_hands = tuple(np.asarray(side, dtype=float).reshape(-1) for side in start_hands)
            self.hands = tuple(np.asarray(side, dtype=float).reshape(-1) for side in hands)
            if (len(self.start_hands) != 2 or len(self.hands) != 2
                    or any(a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
                           for a, b in zip(self.start_hands, self.hands))):
                raise ValueError("hand poses must be (left, right) finite targets of matching sizes")
        for speed in (arm_speed, head_speed, waist_speed, hand_speed):
            if not (math.isfinite(speed) and speed > 0.0):
                raise ValueError("speeds must be positive")
        self.deltas = {
            "arm": float(np.abs(self.arm - self.start_arm).max()),
            "head": float(np.abs(self.head - self.start_head).max()),
            "waist": 0.0 if waist is None else abs(self.waist - self.start_waist),
            "hands": 0.0 if hands is None else max(float(np.abs(b - a).max())
                                                   for a, b in zip(self.start_hands, self.hands)),
        }
        slowest = max(self.deltas["arm"] / arm_speed, self.deltas["head"] / head_speed,
                      self.deltas["waist"] / waist_speed, self.deltas["hands"] / hand_speed)
        self.duration = max(MIN_SECONDS, 1.5 * slowest)
        if self.duration > max_seconds:
            raise ValueError(f"the move would take {self.duration:.1f} s, more than {max_seconds:.0f} s")
        self.started = float(now)

    def at(self, now):
        """(arm_q, head_q, waist or None, finished) to command at now."""
        elapsed = float(now) - self.started
        alpha = smoothstep(elapsed / self.duration)
        arm = self.start_arm + alpha * (self.arm - self.start_arm)
        head = self.start_head + alpha * (self.head - self.start_head)
        waist = None if self.waist is None else self.start_waist + alpha * (self.waist - self.start_waist)
        return arm, head, waist, elapsed >= self.duration + SETTLE_SECONDS

    def hands_at(self, now):
        """(left, right) hand targets at now, on the same path, or None when the hands are left alone."""
        if self.hands is None:
            return None
        alpha = smoothstep((float(now) - self.started) / self.duration)
        return tuple(a + alpha * (b - a) for a, b in zip(self.start_hands, self.hands))


class R1StandardPose:
    """o key state (keyboard thread) and the move itself (control loop, while paused)."""

    def __init__(self, pose, gravity, logger, arm_speed=ARM_SPEED_RAD_S, head_speed=HEAD_SPEED_RAD_S,
                 hands=None, hand_ctrl=None):
        self.pose, self.gravity, self.logger = pose, gravity, logger
        self.arm_speed, self.head_speed = float(arm_speed), float(head_speed)
        # hands: (left, right) standard hand targets, moved through hand_ctrl (teleop --policy only).
        if (hands is None) != (hand_ctrl is None):
            raise ValueError("give the standard hands and the hand controller together")
        self.hands = None if hands is None else tuple(np.asarray(side, dtype=float).reshape(-1) for side in hands)
        self.hand_ctrl = hand_ctrl
        self._lock = threading.Lock()
        self._request = None   # "go" or "stop", taken by the control loop
        self._busy = False     # a move is requested or running
        self._hands_moving = False
        self._move = None
        self._generation = None

    def press(self):
        """Keyboard thread: start a move, or stop the one requested or running. Returns which."""
        with self._lock:
            self._request = "stop" if self._busy else "go"
            self._busy = not self._busy
            return self._request

    def moving(self):
        with self._lock:
            return self._busy

    def commanding_hands(self):
        """Hand loop thread: True while a move drives the hands; the hand loop then sends nothing."""
        with self._lock:
            return self._hands_moving

    def _finish(self, message, level="info"):
        self._move = None
        with self._lock:
            self._busy = False
            self._hands_moving = False
        getattr(self.logger, level)(message)

    def step(self, arm_ctrl, generation, waist_reference, now, head_locked=False, hand_locked=(False, False)):
        """Control loop, while paused: start, advance or stop the move.

        Returns True when it commanded the robot this tick (the caller then neither holds
        nor resumes). waist_reference None leaves the waist alone; a locked head stays put,
        and so does a locked hand ([ or ]) keeps its grip when the hands move.
        """
        with self._lock:
            request, self._request = self._request, None
        if self._move is not None and (request == "stop" or generation != self._generation):
            why = "o pressed again" if request == "stop" else "a new pause"
            self._finish(f"[R1 START POSE] stopped by {why}; holding here. o moves to the standard pose again.",
                         "warning")
            return False
        if request == "go":
            held = arm_ctrl.get_recording_snapshot()["requested"]
            start_waist = None if waist_reference is None else arm_ctrl.get_current_waist_yaw()
            start_head = np.asarray(held["head_q"], dtype=float)
            start_hands = hands = None
            if self.hands is not None:
                start_hands = tuple(np.asarray(side, dtype=float).reshape(-1) for side in self.hand_ctrl.get_action())
                hands = tuple(start if locked else target
                              for start, target, locked in zip(start_hands, self.hands, hand_locked))
            try:
                self._move = StandardPoseMove(
                    held["arm_q"], start_head, start_waist, self.pose["arm_q"],
                    start_head if head_locked else self.pose["head_q"], waist_reference, now,
                    arm_speed=self.arm_speed, head_speed=self.head_speed,
                    start_hands=start_hands, hands=hands,
                )
            except ValueError as exc:
                self._finish(f"[R1 START POSE] not moving: {exc}", "warning")
                return False
            self._generation = generation
            with self._lock:
                self._hands_moving = hands is not None
            deltas = self._move.deltas
            hand_note = ""
            if hands is not None:
                hand_note = ", hands %.2f%s" % (deltas["hands"], " (locked hand kept)" if any(hand_locked) else "")
            self.logger.info(
                "[R1 START POSE] moving to the standard start pose over %.1f s (arm %.2f rad, head %.2f rad%s, "
                "waist %.1f deg%s). s then starts the next episode, r resumes without recording; o again stops.",
                self._move.duration, deltas["arm"], deltas["head"], ", head locked" if head_locked else "",
                math.degrees(deltas["waist"]), hand_note,
            )
        if self._move is None:
            return False
        arm, head, waist, finished = self._move.at(now)
        arm_ctrl.ctrl_dual_arm_and_head(arm, self.gravity(arm), head, waist_yaw_target=waist)
        hand_targets = self._move.hands_at(now)
        if hand_targets is not None:
            self.hand_ctrl.update(*hand_targets, tracking_fresh=(True, True))
        if finished:
            # Judge arrival by the target the controller actually holds, which passes the arm
            # target shaper, not by what was sent.
            requested = np.asarray(arm_ctrl.get_recording_snapshot()["requested"]["arm_q"], dtype=float)
            residual = float(np.abs(requested - self._move.arm).max())
            if residual <= CONVERGED_RAD:
                self._finish("[R1 START POSE] at the standard start pose and holding. s starts the next episode; "
                             "r resumes without recording. Following re-anchors at your hands.")
            elif float(now) - self._move.started >= self._move.duration + SETTLE_SECONDS + MAX_EXTRA_SECONDS:
                self._finish(f"[R1 START POSE] the arm target stayed {residual:.3f} rad from the standard pose; "
                             "holding where it is. Press o to try again.", "warning")
        return True
