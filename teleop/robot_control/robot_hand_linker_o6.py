import logging
import math
import threading
import time
from collections import deque

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

from teleop.utils.o6_grip_cap import GripCapError, clamp_close_q, limits_from_document, load_grip_cap


STATE_TIMEOUT = 0.25
COMMAND_TIMEOUT = 0.1
GATE_REQUIRE_RELEASE = 0
GATE_READY = 1
GATE_ARMED = 2
DEFAULT_COMMAND_TORQUE = 1.0
# Rest noise in the retargeted target is about 0.001, below one device byte (0.004)
# and the motor's deadband, so this can stay short without adding chatter.
COMMAND_SMOOTHING_S = 0.005
SIDES = ("left", "right")
logger = logging.getLogger(__name__)


def _copy(values):
    return None if values is None else values.copy()


class LinkerO6Controller:
    def __init__(self, grip_cap_path=None, apply_grip_cap=True, sides=SIDES):
        # A side left out has no O6 on it (for example a gripper on that wrist):
        # it gets no DDS endpoints, its state and action read as None, and it
        # never holds up readiness or the feedback timeout.
        sides = tuple(sides)
        if not sides or len(set(sides)) != len(sides) or any(side not in SIDES for side in sides):
            raise ValueError("sides must be a non-empty subset of ('left', 'right')")
        self.sides = sides
        self.left_publisher = self._publisher("left")
        self.right_publisher = self._publisher("right")

        self.state_lock = threading.Lock()
        self.recording_history = {side: deque(maxlen=256) for side in ("left", "right")}
        self.left_state = None
        self.right_state = None
        self.left_state_count = 0
        self.right_state_count = 0
        self.left_state_time = None
        self.right_state_time = None
        self.left_gate_mode = None
        self.right_gate_mode = None
        # Speeds, joint torque and temperature the O6 reports next to position.
        self.left_aux = None
        self.right_aux = None
        self.release_times = [None, None]
        self.left_action = None
        self.right_action = None
        self.action_time = None
        self.requested_targets = None
        self.requested_monotonic_ns = None
        self.requested_sequence = 0
        self.published_commands = {"left": None, "right": None}
        self.published_sequences = {"left": 0, "right": 0}
        self.ready = False
        self.active = False
        self.closed = False
        # Operator-confirmed per-axis max normalised close q; None means uncapped.
        self._grip_max_close = {"left": None, "right": None}
        self._command_torque = DEFAULT_COMMAND_TORQUE
        self._grip_cap_path = None

        self.left_state_error = None
        self.right_state_error = None
        self.left_subscriber = self._subscriber("left", self._on_left_state)
        self.right_subscriber = self._subscriber("right", self._on_right_state)
        if apply_grip_cap:
            self.load_grip_cap(grip_cap_path)

    def _publisher(self, side):
        if side not in self.sides:
            return None
        publisher = ChannelPublisher(f"rt/linker/{side}/cmd", MotorCmds_)
        publisher.Init()
        return publisher

    def _subscriber(self, side, handler):
        if side not in self.sides:
            return None
        subscriber = ChannelSubscriber(f"rt/linker/{side}/state", MotorStates_)
        subscriber.Init(handler, 1)
        return subscriber

    def _raise_state_error(self):
        # Caller holds state_lock.
        for side in self.sides:
            error = getattr(self, f"{side}_state_error")
            if error is not None:
                raise error

    def load_grip_cap(self, path=None):
        """Load an operator-confirmed max close. Missing file leaves hands uncapped."""
        try:
            document = load_grip_cap(path)
        except GripCapError as error:
            logger.warning("[LINKER O6 GRIP] ignoring invalid grip cap: %s", error)
            self.clear_grip_cap()
            return None
        limits = limits_from_document(document)
        self._grip_max_close = {"left": limits["left"], "right": limits["right"]}
        torque = limits["command_torque"]
        self._command_torque = DEFAULT_COMMAND_TORQUE if torque is None else float(torque)
        self._grip_cap_path = str(path) if path is not None else None
        if document is None:
            logger.info("[LINKER O6 GRIP] no saved max close; teleop uncapped")
        else:
            def _fmt(limit):
                if limit is None:
                    return None
                return [round(float(v), 3) for v in limit]

            logger.info(
                "[LINKER O6 GRIP] max_close_q left=%s right=%s command_torque=%.3f path=%s",
                _fmt(self._grip_max_close["left"]),
                _fmt(self._grip_max_close["right"]),
                self._command_torque,
                path if path is not None else "default",
            )
        return document

    def clear_grip_cap(self):
        self._grip_max_close = {"left": None, "right": None}
        self._command_torque = DEFAULT_COMMAND_TORQUE
        self._grip_cap_path = None

    def set_grip_max_close(self, left=None, right=None, command_torque=None):
        """Override in-memory per-axis limits (used by the cup-grip calibration tool)."""
        for side, value in (("left", left), ("right", right)):
            if value is None:
                self._grip_max_close[side] = None
            else:
                vector = np.asarray(value, dtype=float)
                if vector.shape == ():
                    vector = np.full(6, float(vector), dtype=float)
                if vector.shape != (6,):
                    raise ValueError(f"{side} max close must contain six axes")
                if not np.isfinite(vector).all() or np.any(vector < 0.0) or np.any(vector > 1.0):
                    raise ValueError(f"{side} max close must be within [0, 1]")
                self._grip_max_close[side] = vector.copy()
        if command_torque is not None:
            number = float(command_torque)
            if not np.isfinite(number) or number < 0.0 or number > 1.0:
                raise ValueError("command_torque must be within [0, 1]")
            self._command_torque = number

    def _capped_target(self, values, side):
        return clamp_close_q(values, self._grip_max_close.get(side))

    @staticmethod
    def _values(values, name):
        result = np.asarray(values, dtype=float)
        if result.shape != (6,):
            raise ValueError(f"{name} must contain six axes")
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
        if np.any(result < 0.0) or np.any(result > 1.0):
            raise ValueError(f"{name} must be within [0, 1]")
        return result.copy()

    @classmethod
    def _state_values(cls, message, name):
        if len(message.states) != 6:
            raise RuntimeError(f"{name} state must contain six axes")
        modes = {state.mode for state in message.states}
        if len(modes) != 1 or not modes.issubset({GATE_REQUIRE_RELEASE, GATE_READY, GATE_ARMED}):
            raise RuntimeError(f"{name} command gate state is invalid")
        values = cls._values([state.q for state in message.states], f"{name} state")

        def channel(attribute):
            # The O6 exposes speeds, joint torque and temperature in the same
            # register block as position. The DDS layer forwards them normalised
            # to 0..1 because the device register is a single byte, so these are
            # relative measures, not calibrated rad/s, N*m or degrees C.
            result = []
            for state in message.states:
                try:
                    value = getattr(state, attribute, None)
                except Exception:
                    value = None
                result.append(None if value is None else float(value))
            return result

        def error_channel():
            # PC2 reserves reserve[0] for the per-joint fault code; the field
            # defaults to zero, which reads as "no fault reported".
            result = []
            for state in message.states:
                try:
                    reserve = getattr(state, "reserve", None)
                    result.append(None if not reserve else int(reserve[0]))
                except Exception:
                    result.append(None)
            return result

        errors = error_channel()
        aux = {"qvel": channel("dq"), "torque": channel("tau_est"),
               "temperature": channel("temperature"),
               "errors": errors if any(value is not None for value in errors) else None}
        return values, aux, modes.pop()

    def _on_left_state(self, message):
        try:
            state, aux, gate_mode = self._state_values(message, "left")
        except (ValueError, RuntimeError) as error:
            with self.state_lock:
                self.left_state_error = error
            return
        with self.state_lock:
            now = time.monotonic()
            gap = None if self.left_state_time is None else now - self.left_state_time
            self.left_state = state
            self.left_aux = aux
            self.left_gate_mode = gate_mode
            self.left_state_count += 1
            self.left_state_time = now
            self.left_state_error = None
            count = self.left_state_count
            self.recording_history["left"].append({
                "q": state.tolist(), "monotonic_ns": int(now * 1e9),
                "sequence": count, "mode": gate_mode, **aux,
            })
        if gap is not None and gap >= STATE_TIMEOUT:
            logger.warning("[LINKER O6 FEEDBACK] left: callback gap_ms=%.1f count=%d gate=%d", gap * 1000, count, gate_mode)

    def _on_right_state(self, message):
        try:
            state, aux, gate_mode = self._state_values(message, "right")
        except (ValueError, RuntimeError) as error:
            with self.state_lock:
                self.right_state_error = error
            return
        with self.state_lock:
            now = time.monotonic()
            gap = None if self.right_state_time is None else now - self.right_state_time
            self.right_state = state
            self.right_aux = aux
            self.right_gate_mode = gate_mode
            self.right_state_count += 1
            self.right_state_time = now
            self.right_state_error = None
            count = self.right_state_count
            self.recording_history["right"].append({
                "q": state.tolist(), "monotonic_ns": int(now * 1e9),
                "sequence": count, "mode": gate_mode, **aux,
            })
        if gap is not None and gap >= STATE_TIMEOUT:
            logger.warning("[LINKER O6 FEEDBACK] right: callback gap_ms=%.1f count=%d gate=%d", gap * 1000, count, gate_mode)

    def _wait_for_state_pair(self, timeout, require_new=False):
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout must be positive")
        with self.state_lock:
            floors = {side: getattr(self, f"{side}_state_count") for side in self.sides}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.state_lock:
                self._raise_state_error()
                ready = all(getattr(self, f"{side}_state") is not None for side in self.sides)
                if require_new:
                    ready = ready and all(
                        getattr(self, f"{side}_state_count") > floor for side, floor in floors.items()
                    )
            if ready:
                return
            time.sleep(0.01)
        if self.sides == SIDES:
            raise TimeoutError("Timed out waiting for both Linker O6 state topics")
        raise TimeoutError(f"Timed out waiting for the {self.sides[0]} Linker O6 state topic")

    @staticmethod
    def _command(values, mode, speed, torque):
        message = MotorCmds_()
        message.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
        for command, value in zip(message.cmds, values):
            command.mode = mode
            command.q = float(value)
            command.dq = speed
            command.tau = torque
        return message

    def _write_pair(self, left_values, right_values, modes, speed, torque, request_sequence=None):
        error = None
        for side, publisher, message in (
            (
                "left",
                self.left_publisher,
                None if self.left_publisher is None else
                self._command(left_values, modes[0], speed if modes[0] else 0.0, torque if modes[0] else 0.0),
            ),
            (
                "right",
                self.right_publisher,
                None if self.right_publisher is None else
                self._command(right_values, modes[1], speed if modes[1] else 0.0, torque if modes[1] else 0.0),
            ),
        ):
            if publisher is None:
                continue
            try:
                if publisher.Write(message, timeout=COMMAND_TIMEOUT) is not True:
                    raise RuntimeError(f"Failed to publish {side} Linker O6 command")
                with self.state_lock:
                    self.published_sequences[side] += 1
                    self.published_commands[side] = {
                        "q": [float(command.q) for command in message.cmds],
                        "mode": int(message.cmds[0].mode),
                        "torque": float(message.cmds[0].tau),
                        "speed": float(message.cmds[0].dq),
                        "monotonic_ns": int(time.monotonic() * 1e9),
                        "sequence": self.published_sequences[side],
                        "request_sequence": request_sequence,
                    }
            except Exception as write_error:
                if error is None:
                    error = write_error
        if error is not None:
            raise error

    def _state_snapshot(self):
        with self.state_lock:
            self._raise_state_error()
            if any(getattr(self, f"{side}_state") is None for side in self.sides):
                raise RuntimeError("Linker O6 state is not ready")
            return (
                _copy(self.left_state),
                _copy(self.right_state),
                self.left_state_time,
                self.right_state_time,
                self.left_gate_mode,
                self.right_gate_mode,
            )

    def wait_until_ready(self, timeout=3.0):
        if self.closed:
            raise RuntimeError("Linker O6 controller is closed")
        self._wait_for_state_pair(timeout)
        self.ready = True

    def activate(self):
        if self.closed:
            raise RuntimeError("Linker O6 controller is closed")
        if self.active:
            return
        if not self.ready:
            raise RuntimeError("wait_until_ready() must succeed before activate()")
        self._wait_for_state_pair(3.0, require_new=True)
        left_state, right_state, _, _, _, _ = self._state_snapshot()
        self._write_pair(
            left_state,
            right_state,
            modes=(0, 0),
            speed=0.0,
            torque=0.0,
        )
        self.release_times = [time.monotonic()] * 2
        self._wait_for_state_pair(3.0, require_new=True)
        left_state, right_state, _, _, _, _ = self._state_snapshot()
        self.left_action = left_state
        self.right_action = right_state
        with self.state_lock:
            self.requested_targets = (_copy(left_state), _copy(right_state))
            self.requested_monotonic_ns = int(time.monotonic() * 1e9)
            self.requested_sequence += 1
        self.action_time = time.monotonic()
        self.active = True

    def update(self, left_target, right_target, tracking_fresh=(True, True)):
        if not self.active:
            raise RuntimeError("Linker O6 controller is not active")
        # Cap the retargeted close target before the command smoother so teleop
        # cannot command a tighter grasp than the operator saved.
        left = (self._capped_target(self._values(left_target, "left target"), "left")
                if "left" in self.sides else None)
        right = (self._capped_target(self._values(right_target, "right target"), "right")
                 if "right" in self.sides else None)
        with self.state_lock:
            self.requested_targets = (_copy(left), _copy(right))
            self.requested_monotonic_ns = int(time.monotonic() * 1e9)
            self.requested_sequence += 1
            request_sequence = self.requested_sequence
        left_state, right_state, left_state_time, right_state_time, left_mode, right_mode = self._state_snapshot()
        now = time.monotonic()

        def stale(state_time, side):
            return side in self.sides and now - state_time >= STATE_TIMEOUT

        def age_text(age, side):
            return f"{age * 1000:.1f}" if side in self.sides else "off"

        if stale(left_state_time, "left") or stale(right_state_time, "right"):
            # A callback may have replaced the snapshot while this thread was descheduled.
            with self.state_lock:
                self._raise_state_error()
                now = time.monotonic()
                left_state, right_state = _copy(self.left_state), _copy(self.right_state)
                left_state_time, right_state_time = self.left_state_time, self.right_state_time
                left_mode, right_mode = self.left_gate_mode, self.right_gate_mode
                left_age = now - left_state_time if "left" in self.sides else 0.0
                right_age = now - right_state_time if "right" in self.sides else 0.0
                counts = (self.left_state_count, self.right_state_count)
            if left_age >= STATE_TIMEOUT or right_age >= STATE_TIMEOUT:
                message = (
                    "Linker O6 state feedback is stale: "
                    f"left_age_ms={age_text(left_age, 'left')} right_age_ms={age_text(right_age, 'right')} "
                    f"left_count={counts[0]} right_count={counts[1]} "
                    f"gate_modes=({left_mode},{right_mode}) "
                    f"update_gap_ms={(now - self.action_time) * 1000:.1f} limit_ms={STATE_TIMEOUT * 1000:.0f}"
                )
                logger.error("[LINKER O6 FEEDBACK] %s; releasing both hands", message)
                # mode=0 is a release, not a stale-position command; PC2 uses its local state.
                try:
                    self._write_pair(left_state, right_state, (0, 0), 0.0, 0.0)
                except Exception as release_error:
                    logger.error("[LINKER O6 FEEDBACK] release failed: %s", release_error)
                    raise TimeoutError(message) from release_error
                raise TimeoutError(message)
        targets, modes, release_times = [], [], []
        dt = min(max(now - self.action_time, 0.0), 1.0 / 30.0)
        alpha = -math.expm1(-dt / COMMAND_SMOOTHING_S)
        for side, target, previous, state, state_time, gate_mode, fresh, released_at in zip(
            SIDES, (left, right), (self.left_action, self.right_action),
            (left_state, right_state), (left_state_time, right_state_time),
            (left_mode, right_mode), tracking_fresh, self.release_times,
        ):
            if side not in self.sides:
                targets.append(None)
                modes.append(0)
                release_times.append(None)
                continue
            enabled = fresh and gate_mode != GATE_REQUIRE_RELEASE and (
                released_at is None or (gate_mode == GATE_READY and state_time > released_at)
            )
            # Restart smoothing from measured position after a gate release.
            start = state if released_at is not None else previous
            targets.append(start + alpha * (target - start) if enabled else state)
            modes.append(1 if enabled else 0)
            release_times.append(None if enabled else (released_at if released_at is not None else now))
        torque = self._command_torque
        self._write_pair(*targets, modes=modes, speed=1.0, torque=torque,
                         request_sequence=request_sequence)
        for side, before, after in zip(("left", "right"), self.release_times, release_times):
            if before is None and after is not None:
                logger.info("[LINKER O6] %s: holding; automatic recovery pending", side)
            elif before is not None and after is None:
                logger.info("[LINKER O6] %s: tracking fresh and gate ready; automatically resumed", side)
        self.release_times = release_times
        self.left_action, self.right_action = targets
        self.action_time = now

    def hold(self):
        self.update(*self.get_action(), tracking_fresh=(False, False))

    def get_state(self):
        left, right, _, _, _, _ = self._state_snapshot()
        return left, right

    def get_action(self):
        if any(getattr(self, f"{side}_action") is None for side in self.sides):
            raise RuntimeError("Linker O6 action is not ready")
        return _copy(self.left_action), _copy(self.right_action)

    def get_recording_states_at(self, target_ns, end_ns):
        with self.state_lock:
            return {side: min((entry for entry in entries if entry["monotonic_ns"] <= end_ns),
                              key=lambda entry: abs(entry["monotonic_ns"] - target_ns), default=None)
                    for side, entries in self.recording_history.items()}

    def get_recording_snapshot(self):
        with self.state_lock:
            if any(getattr(self, f"{side}_state") is None for side in self.sides):
                raise RuntimeError("Linker O6 state is not ready")
            states = {}
            for side, values, aux, timestamp, sequence, mode in (
                ("left", self.left_state, self.left_aux, self.left_state_time,
                 self.left_state_count, self.left_gate_mode),
                ("right", self.right_state, self.right_aux, self.right_state_time,
                 self.right_state_count, self.right_gate_mode),
            ):
                if side not in self.sides:
                    continue
                aux = aux or {}
                states[side] = {
                    "q": values.tolist(), "monotonic_ns": int(timestamp * 1e9),
                    "sequence": sequence, "mode": mode,
                    "qvel": aux.get("qvel"), "torque": aux.get("torque"),
                    "temperature": aux.get("temperature"),
                    "errors": aux.get("errors"),
                }
            return {
                "state": states,
                "requested": None if self.requested_targets is None else {
                    "left_q": None if self.requested_targets[0] is None else self.requested_targets[0].tolist(),
                    "right_q": None if self.requested_targets[1] is None else self.requested_targets[1].tolist(),
                    "monotonic_ns": self.requested_monotonic_ns,
                    "sequence": self.requested_sequence,
                },
                "published": {
                    side: None if command is None else {
                        key: value[:] if isinstance(value, list) else value
                        for key, value in command.items()
                    } for side, command in self.published_commands.items()
                },
            }

    def stop(self):
        if self.closed:
            return
        self.closed = True
        write_error = None
        try:
            if self.active:
                self.active = False
                try:
                    left_state, right_state, _, _, _, _ = self._state_snapshot()
                except (ValueError, RuntimeError):
                    with self.state_lock:
                        left_state = _copy(self.left_state)
                        right_state = _copy(self.right_state)
                try:
                    self._write_pair(left_state, right_state, (0, 0), 0.0, 0.0)
                except Exception as error:
                    write_error = error
        finally:
            for endpoint in (
                self.left_publisher,
                self.right_publisher,
                self.left_subscriber,
                self.right_subscriber,
            ):
                if endpoint is None:
                    continue
                try:
                    endpoint.Close()
                except Exception as error:
                    if write_error is None:
                        write_error = error
        if write_error is not None:
            raise write_error
