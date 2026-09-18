import ast
from contextlib import redirect_stdout
from enum import IntEnum
import io
import math
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ARM_PATH = Path(__file__).resolve().parents[1] / "teleop/robot_control/robot_arm.py"
MAIN_PATH = Path(__file__).resolve().parents[1] / "teleop/teleop_hand_and_arm.py"


def motor_state(value=0.1):
    return SimpleNamespace(mode_machine=7, motor_state=[
        SimpleNamespace(q=value + index / 100, dq=-index / 1000, tau_est=index / 1000) for index in range(35)
    ])


class CallbackSubscriber:
    def __init__(self, *args):
        self.messages = queue.Queue()
        self.handler = None
        self.closed = False
        self.worker = None

    def Init(self, handler, queue_len):
        assert queue_len == 1
        self.handler = handler
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        self.messages.put(motor_state())

    def _run(self):
        while True:
            message = self.messages.get()
            if message is None:
                return
            self.handler(message)

    def Read(self, *args, **kwargs):
        raise AssertionError("R1_A7 must not call blocking/polling SDK Read")

    def Close(self):
        self.closed = True
        self.messages.put(None)
        self.worker.join(timeout=0.2)
        if self.worker.is_alive():
            raise RuntimeError("Fake callback worker did not stop")


def wait_for(predicate, name):
    deadline = time.monotonic() + 1.0
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(name)
        time.sleep(0.001)


def load_controller():
    tree = ast.parse(ARM_PATH.read_text(encoding="utf-8"))
    selected = {"MotorState", "R1_A7_LowState", "DataBuffer", "R1_A7_JointArmIndex",
                "R1_A7_JointHeadIndex", "R1_A7_JointIndex", "R1_A7_ArmController"}
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in selected]
    namespace = {
        "np": np, "threading": threading, "time": time, "IntEnum": IntEnum,
        "logger_mp": Mock(), "ChannelSubscriber": CallbackSubscriber,
        "ChannelPublisher": Mock(side_effect=AssertionError("No real command initialization allowed")),
        "hg_LowState": object, "R1_A7_Num_Motors": 35, "kTopicLowState": "rt/lowstate",
        "wait_for_dds": wait_for,
        "ArmTargetShaper": _load_arm_target_shaper(),
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(ARM_PATH), "exec"), namespace)
    return namespace["R1_A7_ArmController"], namespace


def _load_arm_target_shaper():
    path = ARM_PATH.parent / "arm_target_shaper.py"
    namespace = {"np": np, "math": math, "time": time}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["ArmTargetShaper"]


class R1A7FeedbackShutdownTest(unittest.TestCase):
    def setUp(self):
        self.controller_class, self.namespace = load_controller()

    def bare_controller(self, publisher=None, subscriber=None, publish_thread=None):
        controller = self.controller_class.__new__(self.controller_class)
        controller.lifecycle_lock = threading.Lock()
        controller.publish_running = True
        controller.subscribe_running = True
        controller.active = True
        controller.publish_thread = publish_thread
        controller.lowcmd_publisher = publisher
        controller.lowstate_subscriber = subscriber
        return controller

    def test_deferred_controller_uses_callback_without_a_polling_thread_or_publisher(self):
        controller = self.controller_class(deferred_activation=True)
        try:
            self.assertFalse(hasattr(controller, "subscribe_thread"))
            self.assertTrue(controller.lowstate_sub_ready)
            self.assertIsNotNone(controller.lowstate_subscriber.handler)
            self.namespace["ChannelPublisher"].assert_not_called()
            np.testing.assert_allclose(controller.get_current_motor_q(), [0.1 + i / 100 for i in range(35)])
        finally:
            controller.stop()

    def test_each_callback_copies_the_state_and_advances_sequence_and_timestamp(self):
        controller = self.controller_class(deferred_activation=True)
        try:
            previous = controller.lowstate_buffer.GetData()
            controller.lowstate_subscriber.messages.put(motor_state(0.4))
            wait_for(lambda: controller.lowstate_buffer.GetData().sequence > previous.sequence, "callback")
            current = controller.lowstate_buffer.GetData()
            self.assertEqual(current.mode_machine, 7)
            self.assertEqual(current.sequence, previous.sequence + 1)
            self.assertGreaterEqual(current.monotonic_timestamp, previous.monotonic_timestamp)
            self.assertEqual(current.motor_state[13].q, 0.53)
            self.assertEqual(current.motor_state[13].dq, -0.013)
            self.assertEqual(previous.motor_state[13].q, 0.23)
        finally:
            controller.stop()

    def test_feedback_loss_does_not_block_shutdown_or_print_reader_timeouts(self):
        output = io.StringIO()
        with redirect_stdout(output):
            controller = self.controller_class(deferred_activation=True)
            subscriber = controller.lowstate_subscriber
            last_state = controller.lowstate_buffer.GetData()
            time.sleep(0.02)
            self.assertIs(controller.lowstate_buffer.GetData(), last_state)
            started = time.monotonic()
            controller.stop()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.5)
        self.assertTrue(subscriber.closed)
        self.assertFalse(subscriber.worker.is_alive())
        self.assertIsNone(controller.lowstate_subscriber)
        self.assertNotIn("[Reader]", output.getvalue())

    def test_measured_torque_comes_from_the_tau_est_dds_field(self):
        """unitree_hg MotorState_ has no `.tau`; reading it killed the reader thread."""
        controller = self.controller_class(deferred_activation=True)
        try:
            controller.lowstate_subscriber.messages.put(motor_state(0.4))
            wait_for(lambda: controller.lowstate_buffer.GetData().motor_state[15].tau is not None, "tau")
            tau = controller.get_current_dual_arm_tau()
            self.assertEqual(len(tau), 14)
            self.assertAlmostEqual(tau[0], 15 / 1000)      # first left-arm joint
            self.assertAlmostEqual(tau[-1], 28 / 1000)     # last right-arm joint
        finally:
            controller.stop()

    def test_feedback_without_a_torque_field_is_not_fatal(self):
        controller = self.controller_class(deferred_activation=True)
        try:
            legacy = SimpleNamespace(mode_machine=7, motor_state=[
                SimpleNamespace(q=0.1 + index / 100, dq=0.0) for index in range(35)
            ])
            controller._subscribe_motor_state(legacy)      # must not raise
            np.testing.assert_allclose(controller.get_current_dual_arm_tau(), np.zeros(14))
        finally:
            controller.stop()

    def test_malformed_feedback_is_swallowed_so_the_reader_thread_survives(self):
        controller = self.controller_class(deferred_activation=True)
        try:
            controller._subscribe_motor_state(SimpleNamespace(mode_machine=7, motor_state=[]))
            controller._subscribe_motor_state(motor_state(0.9))
            self.assertEqual(controller.lowstate_buffer.GetData().motor_state[0].q, 0.9)
            self.namespace["logger_mp"].error.assert_called()
        finally:
            controller.stop()

    def test_published_target_is_rate_limited_to_the_arm_capability(self):
        """Commanding 5-11 rad/s at a 1.6-2.7 rad/s servo is what looked like stutter."""
        controller = self.controller_class.__new__(self.controller_class)
        controller.arm_velocity_limit = 3.0
        controller.control_dt = 1.0 / 250.0
        controller.get_current_dual_arm_q = lambda: np.zeros(14)

        clipped = controller.clip_arm_q_target(np.full(14, 0.5), controller.arm_velocity_limit)
        # at most velocity_limit * control_dt per publish cycle for the fastest joint
        self.assertAlmostEqual(float(np.max(np.abs(clipped))), 3.0 / 250.0, places=9)
        # the whole arm keeps its coordination: every joint scaled by the same factor
        np.testing.assert_allclose(clipped, np.full(14, 3.0 / 250.0))

        small = np.full(14, 0.005)
        np.testing.assert_allclose(
            controller.clip_arm_q_target(small, controller.arm_velocity_limit), small,
            err_msg="motion inside the limit must pass through untouched",
        )

    def test_default_position_limit_is_a_safety_net_not_a_tracking_cap(self):
        # Mode B (dq feed-forward) keeps the command; the position cap is only a net.
        self.assertEqual(self.controller_class.default_arm_velocity_limit, 30.0)
        self.assertTrue(self.controller_class.default_dq_feedforward)
        self.assertEqual(self.controller_class.default_target_velocity_limit, 6.0)
        self.assertEqual(self.controller_class.default_target_accel_limit, 40.0)

    def test_wrappers_enable_velocity_feedforward_by_default(self):
        root = MAIN_PATH.parents[1]
        for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
            script = (root / "teleop" / name).read_text(encoding="utf-8")
            self.assertIn('--arm-dq-feedforward "${ARM_DQ_FEEDFORWARD:-on}"', script)
            self.assertIn('--arm-target-velocity-limit "${ARM_TARGET_VELOCITY_LIMIT:-6.0}"', script)
            self.assertIn('--arm-target-accel-limit "${ARM_TARGET_ACCEL_LIMIT:-40.0}"', script)

    def test_main_program_exposes_and_forwards_the_limit_and_feedforward(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        for flag in ("'--arm-velocity-limit'", "'--arm-dq-feedforward'", "'--arm-dq-limit'",
                     "'--arm-dq-filter'", "'--arm-target-velocity-limit'", "'--arm-target-accel-limit'"):
            self.assertIn(flag, source)
        self.assertEqual(source.count("arm_velocity_limit=args.arm_velocity_limit"), 2)
        self.assertEqual(source.count("dq_feedforward=args.arm_dq_feedforward == 'on'"), 2)
        self.assertEqual(source.count("target_velocity_limit=args.arm_target_velocity_limit"), 2)
        self.assertEqual(source.count("target_accel_limit=args.arm_target_accel_limit"), 2)

    def test_late_callback_after_stop_is_ignored(self):
        controller = self.controller_class(deferred_activation=True)
        controller.stop()
        previous = controller.lowstate_buffer.GetData()
        controller._subscribe_motor_state(motor_state(0.6))
        self.assertIs(controller.lowstate_buffer.GetData(), previous)

    def test_publisher_thread_timeout_does_not_skip_either_endpoint_close(self):
        publisher, subscriber = Mock(), Mock()
        thread = Mock(is_alive=Mock(return_value=True))
        controller = self.bare_controller(publisher, subscriber, thread)
        with self.assertRaisesRegex(RuntimeError, "publisher thread did not stop"):
            controller.stop()
        thread.join.assert_called_once_with(timeout=1.0)
        publisher.Close.assert_called_once_with()
        subscriber.Close.assert_called_once_with()
        self.assertFalse(controller.publish_running)
        self.assertFalse(controller.subscribe_running)
        self.assertFalse(controller.active)

    def test_publisher_join_error_does_not_skip_endpoint_cleanup(self):
        publisher, subscriber = Mock(), Mock()
        thread = Mock(join=Mock(side_effect=RuntimeError("join failed")))
        controller = self.bare_controller(publisher, subscriber, thread)
        with self.assertRaisesRegex(RuntimeError, "join failed"):
            controller.stop()
        publisher.Close.assert_called_once_with()
        subscriber.Close.assert_called_once_with()

    def test_publisher_close_error_does_not_skip_subscriber_close(self):
        publisher = Mock(Close=Mock(side_effect=RuntimeError("writer close failed")))
        subscriber = Mock()
        controller = self.bare_controller(publisher, subscriber)
        with self.assertRaisesRegex(RuntimeError, "writer close failed"):
            controller.stop()
        subscriber.Close.assert_called_once_with()
        self.assertIs(controller.lowcmd_publisher, publisher)
        self.assertIsNone(controller.lowstate_subscriber)

    def test_all_cleanup_errors_are_reported_and_failed_close_can_be_retried(self):
        publisher = Mock(Close=Mock(side_effect=RuntimeError("writer close failed")))
        subscriber = Mock(Close=Mock(side_effect=RuntimeError("reader close failed")))
        controller = self.bare_controller(publisher, subscriber)
        with self.assertRaises(RuntimeError) as caught:
            controller.stop()
        self.assertIn("writer close failed", str(caught.exception))
        self.assertIn("reader close failed", str(caught.exception))
        publisher.Close.side_effect = None
        subscriber.Close.side_effect = None
        controller.stop()
        self.assertIsNone(controller.lowcmd_publisher)
        self.assertIsNone(controller.lowstate_subscriber)

    def test_successful_stop_is_idempotent_and_joins_before_closing_writer(self):
        calls = []
        thread = Mock(join=Mock(side_effect=lambda **_: calls.append("join")), is_alive=Mock(return_value=False))
        publisher = Mock(Close=Mock(side_effect=lambda: calls.append("writer")))
        subscriber = Mock(Close=Mock(side_effect=lambda: calls.append("reader")))
        controller = self.bare_controller(publisher, subscriber, thread)
        controller.stop()
        controller.stop()
        self.assertEqual(calls[:3], ["join", "writer", "reader"])
        publisher.Close.assert_called_once_with()
        subscriber.Close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
