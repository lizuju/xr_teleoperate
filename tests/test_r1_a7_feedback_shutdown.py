import ast
from contextlib import redirect_stdout
from enum import IntEnum
import io
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ARM_PATH = Path(__file__).resolve().parents[1] / "teleop/robot_control/robot_arm.py"


def motor_state(value=0.1):
    return SimpleNamespace(mode_machine=7, motor_state=[
        SimpleNamespace(q=value + index / 100, dq=-index / 1000) for index in range(35)
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
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(ARM_PATH), "exec"), namespace)
    return namespace["R1_A7_ArmController"], namespace


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
