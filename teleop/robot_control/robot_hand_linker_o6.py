import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


STATE_TIMEOUT = 0.25
COMMAND_TIMEOUT = 0.1


class LinkerO6Controller:
    def __init__(self):
        self.left_publisher = ChannelPublisher("rt/linker/left/cmd", MotorCmds_)
        self.left_publisher.Init()
        self.right_publisher = ChannelPublisher("rt/linker/right/cmd", MotorCmds_)
        self.right_publisher.Init()

        self.state_lock = threading.Lock()
        self.left_state = None
        self.right_state = None
        self.left_state_count = 0
        self.right_state_count = 0
        self.left_state_time = None
        self.right_state_time = None
        self.left_action = None
        self.right_action = None
        self.ready = False
        self.active = False
        self.closed = False

        self.left_state_error = None
        self.right_state_error = None
        self.left_subscriber = ChannelSubscriber("rt/linker/left/state", MotorStates_)
        self.left_subscriber.Init(self._on_left_state, 1)
        self.right_subscriber = ChannelSubscriber("rt/linker/right/state", MotorStates_)
        self.right_subscriber.Init(self._on_right_state, 1)

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
        return cls._values([state.q for state in message.states], f"{name} state")

    def _on_left_state(self, message):
        try:
            state = self._state_values(message, "left")
        except (ValueError, RuntimeError) as error:
            with self.state_lock:
                self.left_state_error = error
            return
        with self.state_lock:
            self.left_state = state
            self.left_state_count += 1
            self.left_state_time = time.monotonic()
            self.left_state_error = None

    def _on_right_state(self, message):
        try:
            state = self._state_values(message, "right")
        except (ValueError, RuntimeError) as error:
            with self.state_lock:
                self.right_state_error = error
            return
        with self.state_lock:
            self.right_state = state
            self.right_state_count += 1
            self.right_state_time = time.monotonic()
            self.right_state_error = None

    def _wait_for_state_pair(self, timeout, require_new=False):
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout must be positive")
        with self.state_lock:
            left_floor = self.left_state_count
            right_floor = self.right_state_count
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.state_lock:
                if self.left_state_error is not None:
                    raise self.left_state_error
                if self.right_state_error is not None:
                    raise self.right_state_error
                ready = self.left_state is not None and self.right_state is not None
                if require_new:
                    ready = (
                        ready
                        and self.left_state_count > left_floor
                        and self.right_state_count > right_floor
                    )
            if ready:
                return
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for both Linker O6 state topics")

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

    def _write_pair(self, left_values, right_values, mode, speed, torque):
        error = None
        for side, publisher, message in (
            (
                "left",
                self.left_publisher,
                self._command(left_values, mode, speed, torque),
            ),
            (
                "right",
                self.right_publisher,
                self._command(right_values, mode, speed, torque),
            ),
        ):
            try:
                if publisher.Write(message, timeout=COMMAND_TIMEOUT) is not True:
                    raise RuntimeError(f"Failed to publish {side} Linker O6 command")
            except Exception as write_error:
                if error is None:
                    error = write_error
        if error is not None:
            raise error

    def _state_snapshot(self):
        with self.state_lock:
            if self.left_state_error is not None:
                raise self.left_state_error
            if self.right_state_error is not None:
                raise self.right_state_error
            if self.left_state is None or self.right_state is None:
                raise RuntimeError("Linker O6 state is not ready")
            return (
                self.left_state.copy(),
                self.right_state.copy(),
                self.left_state_time,
                self.right_state_time,
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
        left_state, right_state, _, _ = self._state_snapshot()
        self._write_pair(
            left_state,
            right_state,
            mode=0,
            speed=0.0,
            torque=0.0,
        )
        self._wait_for_state_pair(3.0, require_new=True)
        left_state, right_state, _, _ = self._state_snapshot()
        self.left_action = left_state
        self.right_action = right_state
        self.active = True

    def update(self, left_target, right_target):
        if not self.active:
            raise RuntimeError("Linker O6 controller is not active")
        left = self._values(left_target, "left target")
        right = self._values(right_target, "right target")
        _, _, left_state_time, right_state_time = self._state_snapshot()
        now = time.monotonic()
        if (
            now - left_state_time >= STATE_TIMEOUT
            or now - right_state_time >= STATE_TIMEOUT
        ):
            raise TimeoutError("Linker O6 state feedback is stale")
        self._write_pair(left, right, mode=1, speed=1.0, torque=1.0)
        self.left_action = left
        self.right_action = right

    def get_state(self):
        left, right, _, _ = self._state_snapshot()
        return left, right

    def get_action(self):
        if self.left_action is None or self.right_action is None:
            raise RuntimeError("Linker O6 action is not ready")
        return self.left_action.copy(), self.right_action.copy()

    def stop(self):
        if self.closed:
            return
        self.closed = True
        write_error = None
        try:
            if self.active:
                self.active = False
                try:
                    left_state, right_state, _, _ = self._state_snapshot()
                except (ValueError, RuntimeError):
                    with self.state_lock:
                        left_state = self.left_state.copy()
                        right_state = self.right_state.copy()
                try:
                    self._write_pair(left_state, right_state, 0, 0.0, 0.0)
                except Exception as error:
                    write_error = error
        finally:
            for endpoint in (
                self.left_publisher,
                self.right_publisher,
                self.left_subscriber,
                self.right_subscriber,
            ):
                try:
                    endpoint.Close()
                except Exception as error:
                    if write_error is None:
                        write_error = error
        if write_error is not None:
            raise write_error
