"""Unitree Dex1-1 gripper on one R1_A7 wrist, next to a Linker O6 on the other.

PC2 runs unitreerobotics/dex1_1_service, which bridges the gripper's serial bus to DDS
(rt/dex1/<side>/cmd and rt/dex1/<side>/state, one unitree_go motor each). Its q is the
output shaft angle in radians: 0 is the calibrated, tightly closed position and about
5.4 rad is fully open (9 cm between the jaws). The service brakes the motor when no
command has arrived for 1 s.

The gripper follows the operator's thumb-index distance. This module keeps the
LinkerO6Controller interface (paired left/right arguments and results), so the O6
control loop, pause and shutdown paths drive both wrists without knowing which device
is on which side.
"""
import logging
import math
import threading
import time
from collections import deque

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


STATE_TIMEOUT = 0.25
COMMAND_TIMEOUT = 0.1
CLOSED_Q = 0.0
OPEN_Q = 5.4
# Gains from the vendor xr_teleoperate Dex1_1_Gripper_Controller.
DEFAULT_KP = 5.0
DEFAULT_KD = 0.05
# The command may lead the measured position by at most this much (vendor value). On an
# object this lead is the squeeze: 5 * 0.18 = 0.9 N*m at the shaft, about 3 mm of jaw travel.
DEFAULT_GRIP_DELTA = 0.18
# Feedback outside this band means a missing calibration or the wrong motor on the port.
STATE_RANGE = (-0.5, 6.5)
# Thumb tip to index tip in Vision Pro hand joints, in metres. The joints sit inside the
# finger pads, so touching fingertips read about 1.5-2.5 cm rather than zero.
DEFAULT_PINCH_CLOSED_M = 0.025
DEFAULT_PINCH_OPEN_M = 0.10
DEFAULT_PINCH_SMOOTHING_S = 0.02
THUMB_TIP = 4
INDEX_TIP = 9
logger = logging.getLogger(__name__)


class Dex1PinchRetargeter:
    """Maps the thumb-index distance linearly onto the gripper stroke."""

    method = "pinch"

    def __init__(self, closed_m=DEFAULT_PINCH_CLOSED_M, open_m=DEFAULT_PINCH_OPEN_M,
                 smoothing_s=DEFAULT_PINCH_SMOOTHING_S, closed_q=CLOSED_Q, open_q=OPEN_Q):
        values = (closed_m, open_m, smoothing_s, closed_q, open_q)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Dex1 pinch mapping values must be finite")
        if not 0.0 <= closed_m < open_m:
            raise ValueError("Dex1 pinch distances need 0 <= closed < open")
        if smoothing_s < 0.0:
            raise ValueError("Dex1 pinch smoothing must be non-negative")
        if not closed_q < open_q:
            raise ValueError("Dex1 gripper needs closed_q < open_q")
        self.closed_m, self.open_m = float(closed_m), float(open_m)
        self.closed_q, self.open_q = float(closed_q), float(open_q)
        self.smoothing_s = float(smoothing_s)
        self._value = None
        self._time = None

    @staticmethod
    def pinch_distance(points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[0] <= INDEX_TIP or points.shape[1] != 3:
            raise ValueError("hand points must be an N x 3 array with N > 9")
        distance = float(np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP]))
        if not math.isfinite(distance):
            raise ValueError("hand points must be finite")
        return distance

    def target_for_distance(self, distance):
        fraction = (distance - self.closed_m) / (self.open_m - self.closed_m)
        fraction = min(max(fraction, 0.0), 1.0)
        return self.closed_q + fraction * (self.open_q - self.closed_q)

    def retarget(self, points, now=None):
        target = self.target_for_distance(self.pinch_distance(points))
        current = time.monotonic() if now is None else now
        if self._value is None or self.smoothing_s == 0.0:
            self._value = target
        else:
            dt = min(max(current - self._time, 0.0), 0.1)
            alpha = -math.expm1(-dt / self.smoothing_s)
            self._value += alpha * (target - self._value)
        self._time = current
        return np.array([self._value])

    def reset(self):
        self._value = None
        self._time = None


class Dex1GripperController:
    """One gripper with the LinkerO6Controller single-side semantics.

    get_action() is the operator's request (the target the gripper is driven towards),
    not the force-limited command. The control loop feeds get_action() back while paused;
    feeding back the force-limited command instead would ratchet the squeeze deeper on
    every cycle against a soft object.
    """

    def __init__(self, side="right", kp=DEFAULT_KP, kd=DEFAULT_KD, grip_delta=DEFAULT_GRIP_DELTA,
                 closed_q=CLOSED_Q, open_q=OPEN_Q):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        for name, value in (("kp", kp), ("kd", kd), ("grip_delta", grip_delta)):
            if not (math.isfinite(float(value)) and float(value) >= 0.0):
                raise ValueError(f"Dex1 {name} must be finite and non-negative")
        if not closed_q < open_q:
            raise ValueError("Dex1 gripper needs closed_q < open_q")
        self.side = side
        self.kp, self.kd, self.grip_delta = float(kp), float(kd), float(grip_delta)
        self.closed_q, self.open_q = float(closed_q), float(open_q)
        self.cmd_topic = f"rt/dex1/{side}/cmd"
        self.state_topic = f"rt/dex1/{side}/state"
        self.publisher = ChannelPublisher(self.cmd_topic, MotorCmds_)
        self.publisher.Init()

        self.state_lock = threading.Lock()
        self.state = None
        self.state_aux = None
        self.state_time = None
        self.state_count = 0
        self.state_error = None
        self.recording_history = deque(maxlen=256)
        self.desired = None
        self.command_q = None
        self.action_time = None
        self.requested_target = None
        self.requested_monotonic_ns = None
        self.requested_sequence = 0
        self.published_command = None
        self.published_sequence = 0
        self.ready = False
        self.active = False
        self.closed = False
        self.subscriber = ChannelSubscriber(self.state_topic, MotorStates_)
        self.subscriber.Init(self._on_state, 1)

    @staticmethod
    def _state_value(message):
        states = getattr(message, "states", None)
        if not states:
            raise RuntimeError("Dex1 state must contain one motor")
        q = float(states[0].q)
        if not math.isfinite(q):
            raise RuntimeError("Dex1 state must be finite")
        if not STATE_RANGE[0] <= q <= STATE_RANGE[1]:
            raise RuntimeError(
                f"Dex1 state q={q:.3f} rad is outside {STATE_RANGE}; calibrate the gripper "
                "(dex1_1_gripper_server -c with the jaws closed)"
            )

        def number(attribute):
            try:
                value = getattr(states[0], attribute, None)
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        return q, {"qvel": number("dq"), "torque": number("tau_est")}

    def _on_state(self, message):
        try:
            q, aux = self._state_value(message)
        except (RuntimeError, TypeError, ValueError, AttributeError) as error:
            with self.state_lock:
                self.state_error = error
            return
        with self.state_lock:
            now = time.monotonic()
            gap = None if self.state_time is None else now - self.state_time
            self.state = q
            self.state_aux = aux
            self.state_time = now
            self.state_count += 1
            self.state_error = None
            count = self.state_count
            self.recording_history.append({
                "q": [q], "monotonic_ns": int(now * 1e9), "sequence": count, **aux,
            })
        if gap is not None and gap >= STATE_TIMEOUT:
            logger.warning("[DEX1 FEEDBACK] %s: callback gap_ms=%.1f count=%d", self.side, gap * 1000, count)

    def _wait_for_state(self, timeout, require_new=False):
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout must be positive")
        with self.state_lock:
            floor = self.state_count
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.state_lock:
                if self.state_error is not None:
                    raise self.state_error
                ready = self.state is not None and (not require_new or self.state_count > floor)
            if ready:
                return
            time.sleep(0.01)
        raise TimeoutError(
            f"Timed out waiting for {self.state_topic}; is dex1_1_gripper_server running on PC2?"
        )

    def _snapshot(self):
        with self.state_lock:
            if self.state_error is not None:
                raise self.state_error
            if self.state is None:
                raise RuntimeError("Dex1 state is not ready")
            return self.state, self.state_time, self.state_count

    def _publish(self, q, request_sequence=None):
        message = MotorCmds_()
        message.cmds = [unitree_go_msg_dds__MotorCmd_()]
        command = message.cmds[0]
        command.q = float(q)
        command.dq = 0.0
        command.tau = 0.0
        command.kp = self.kp
        command.kd = self.kd
        if self.publisher.Write(message, timeout=COMMAND_TIMEOUT) is not True:
            raise RuntimeError(f"Failed to publish {self.side} Dex1 command")
        with self.state_lock:
            self.published_sequence += 1
            self.published_command = {
                "q": [float(q)], "kp": self.kp, "kd": self.kd,
                "monotonic_ns": int(time.monotonic() * 1e9),
                "sequence": self.published_sequence,
                "request_sequence": request_sequence,
            }

    def _limited(self, state):
        return float(np.clip(self.desired, state - self.grip_delta, state + self.grip_delta))

    def wait_until_ready(self, timeout=3.0):
        if self.closed:
            raise RuntimeError("Dex1 controller is closed")
        self._wait_for_state(timeout)
        self.ready = True

    def activate(self):
        if self.closed:
            raise RuntimeError("Dex1 controller is closed")
        if self.active:
            return
        if not self.ready:
            raise RuntimeError("wait_until_ready() must succeed before activate()")
        self._wait_for_state(3.0, require_new=True)
        state, _, _ = self._snapshot()
        # Start where the jaws are, so activation itself moves nothing.
        self.desired = float(np.clip(state, self.closed_q, self.open_q))
        with self.state_lock:
            self.requested_target = self.desired
            self.requested_monotonic_ns = int(time.monotonic() * 1e9)
            self.requested_sequence += 1
        self.command_q = state
        self._publish(state)
        self.action_time = time.monotonic()
        self.active = True

    def update(self, target, tracking_fresh=True):
        if not self.active:
            raise RuntimeError("Dex1 controller is not active")
        values = np.asarray(target, dtype=float).reshape(-1)
        if values.shape != (1,) or not math.isfinite(values[0]):
            raise ValueError(f"{self.side} Dex1 target must be one finite value")
        state, state_time, count = self._snapshot()
        now = time.monotonic()
        age = now - state_time
        if age >= STATE_TIMEOUT:
            message = (
                f"Dex1 {self.side} state feedback is stale: age_ms={age * 1000:.1f} count={count} "
                f"update_gap_ms={(now - self.action_time) * 1000:.1f} limit_ms={STATE_TIMEOUT * 1000:.0f}"
            )
            # Publishing nothing lets dex1_1_service keep the last command for 1 s and then brake.
            logger.error("[DEX1 FEEDBACK] %s", message)
            raise TimeoutError(message)
        if tracking_fresh:
            self.desired = float(np.clip(values[0], self.closed_q, self.open_q))
        # A stale hand keeps the last request, so a held object stays gripped.
        with self.state_lock:
            self.requested_target = self.desired
            self.requested_monotonic_ns = int(now * 1e9)
            self.requested_sequence += 1
            request_sequence = self.requested_sequence
        command = self._limited(state)
        self._publish(command, request_sequence)
        self.command_q = command
        self.action_time = now

    def get_state(self):
        state, _, _ = self._snapshot()
        return np.array([state])

    def get_action(self):
        if self.desired is None:
            raise RuntimeError("Dex1 action is not ready")
        return np.array([self.desired])

    def get_recording_snapshot(self):
        with self.state_lock:
            if self.state is None:
                raise RuntimeError("Dex1 state is not ready")
            aux = self.state_aux or {}
            return {
                "state": {
                    "q": [self.state], "monotonic_ns": int(self.state_time * 1e9),
                    "sequence": self.state_count,
                    "qvel": aux.get("qvel"), "torque": aux.get("torque"),
                },
                "requested": None if self.requested_target is None else {
                    "q": [self.requested_target], "monotonic_ns": self.requested_monotonic_ns,
                    "sequence": self.requested_sequence,
                },
                "published": None if self.published_command is None else dict(self.published_command),
            }

    def get_recording_state_at(self, target_ns, end_ns):
        with self.state_lock:
            return min((entry for entry in self.recording_history if entry["monotonic_ns"] <= end_ns),
                       key=lambda entry: abs(entry["monotonic_ns"] - target_ns), default=None)

    def stop(self):
        if self.closed:
            return
        self.closed = True
        error = None
        try:
            if self.active:
                self.active = False
                try:
                    state, _, _ = self._snapshot()
                    # Hold where the jaws are, without squeeze; dex1_1_service brakes 1 s later.
                    self._publish(state)
                except Exception as write_error:
                    error = write_error
        finally:
            for endpoint in (self.publisher, self.subscriber):
                try:
                    endpoint.Close()
                except Exception as close_error:
                    if error is None:
                        error = close_error
        if error is not None:
            raise error


class O6Dex1Retargeter:
    """Linker O6 retargeting on one wrist, thumb-index distance for the gripper on the other."""

    def __init__(self, o6_retargeter, gripper_retargeter, gripper_side="right"):
        if gripper_side not in ("left", "right"):
            raise ValueError("gripper_side must be left or right")
        self.gripper_side = gripper_side
        self.left = gripper_retargeter if gripper_side == "left" else o6_retargeter
        self.right = gripper_retargeter if gripper_side == "right" else o6_retargeter
        self.method = o6_retargeter.method
        self.mapping_name = f"linker_o6_dex_{self.method}_v1+dex1_pinch_v1_{gripper_side}"

    def retarget(self, left_hand_points, right_hand_points):
        return self.left.retarget(left_hand_points), self.right.retarget(right_hand_points)

    def reset(self):
        self.left.reset()
        self.right.reset()


class O6Dex1Hands:
    """A one-sided LinkerO6Controller and a Dex1GripperController behind the paired interface."""

    def __init__(self, o6, gripper):
        if gripper.side in o6.sides or len(o6.sides) != 1:
            raise ValueError("the O6 and the gripper must be on different wrists")
        self.o6 = o6
        self.gripper = gripper
        self.gripper_side = gripper.side
        self.o6_side = o6.sides[0]
        self._gripper_index = 0 if gripper.side == "left" else 1
        self._o6_index = 1 - self._gripper_index

    def _pair(self, o6_value, gripper_value):
        pair = [None, None]
        pair[self._o6_index] = o6_value
        pair[self._gripper_index] = gripper_value
        return tuple(pair)

    def wait_until_ready(self, timeout=3.0):
        self.o6.wait_until_ready(timeout=timeout)
        self.gripper.wait_until_ready(timeout=timeout)

    def activate(self):
        self.o6.activate()
        self.gripper.activate()

    def update(self, left_target, right_target, tracking_fresh=(True, True)):
        targets = (left_target, right_target)
        o6_targets = [None, None]
        o6_targets[self._o6_index] = targets[self._o6_index]
        o6_fresh = [False, False]
        o6_fresh[self._o6_index] = tracking_fresh[self._o6_index]
        error = None
        try:
            self.o6.update(*o6_targets, tracking_fresh=tuple(o6_fresh))
        except Exception as o6_error:
            error = o6_error
        try:
            self.gripper.update(targets[self._gripper_index], tracking_fresh[self._gripper_index])
        except Exception as gripper_error:
            if error is None:
                error = gripper_error
        if error is not None:
            raise error

    def hold(self):
        self.update(*self.get_action(), tracking_fresh=(False, False))

    def get_state(self):
        return self._pair(self.o6.get_state()[self._o6_index], self.gripper.get_state())

    def get_action(self):
        return self._pair(self.o6.get_action()[self._o6_index], self.gripper.get_action())

    def get_recording_snapshot(self):
        o6 = self.o6.get_recording_snapshot()
        gripper = self.gripper.get_recording_snapshot()
        o6_side, gripper_side = self.o6_side, self.gripper_side
        requested = o6["requested"]
        return {
            "devices": {o6_side: "linker_o6", gripper_side: "dex1_1"},
            "state": {o6_side: o6["state"][o6_side], gripper_side: gripper["state"]},
            "requested": None if requested is None else {
                f"{o6_side}_q": requested[f"{o6_side}_q"],
                f"{gripper_side}_q": None if gripper["requested"] is None else gripper["requested"]["q"],
                "monotonic_ns": requested["monotonic_ns"],
                "sequence": requested["sequence"],
            },
            "published": {o6_side: o6["published"][o6_side], gripper_side: gripper["published"]},
        }

    def get_recording_states_at(self, target_ns, end_ns):
        states = self.o6.get_recording_states_at(target_ns, end_ns)
        states[self.gripper_side] = self.gripper.get_recording_state_at(target_ns, end_ns)
        return states

    def stop(self):
        error = None
        for controller in (self.o6, self.gripper):
            try:
                controller.stop()
            except Exception as stop_error:
                if error is None:
                    error = stop_error
        if error is not None:
            raise error
