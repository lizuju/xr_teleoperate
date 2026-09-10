import importlib
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np


class FakeMotorCmd:
    def __init__(self):
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0


class FakeMotorCmds:
    def __init__(self):
        self.cmds = []


class FakePublisher:
    instances = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.writes = []
        self.timeouts = []
        self.write_result = True
        self.closed = False
        type(self).instances.append(self)

    def Init(self):
        pass

    def Write(self, message, timeout=None):
        self.writes.append(message)
        self.timeouts.append(timeout)
        return self.write_result

    def Close(self):
        self.closed = True


class FakeSubscriber:
    instances = []
    queued_messages = {}

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.handler = None
        self.closed = False
        type(self).instances.append(self)

    def Init(self, handler=None, queueLen=0):
        self.handler = handler
        if handler is not None:
            messages = type(self).queued_messages.setdefault(self.topic, [])
            while messages:
                handler(messages.pop(0))

    def Read(self):
        messages = type(self).queued_messages.setdefault(self.topic, [])
        return messages.pop(0) if messages else None

    def Close(self):
        self.closed = True

    @classmethod
    def deliver(cls, topic, message):
        for subscriber in cls.instances:
            if subscriber.topic == topic and subscriber.handler is not None and not subscriber.closed:
                subscriber.handler(message)
                return
        cls.queued_messages.setdefault(topic, []).append(message)


def state_message(values, mode=1):
    return types.SimpleNamespace(
        states=[types.SimpleNamespace(q=value, mode=mode) for value in values]
    )


class LinkerO6ControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        channel_module = types.ModuleType("unitree_sdk2py.core.channel")
        channel_module.ChannelPublisher = FakePublisher
        channel_module.ChannelSubscriber = FakeSubscriber
        default_module = types.ModuleType("unitree_sdk2py.idl.default")
        default_module.unitree_go_msg_dds__MotorCmd_ = FakeMotorCmd
        dds_module = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
        dds_module.MotorCmds_ = FakeMotorCmds
        dds_module.MotorStates_ = object
        modules = {
            "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
            "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
            "unitree_sdk2py.core.channel": channel_module,
            "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
            "unitree_sdk2py.idl.default": default_module,
            "unitree_sdk2py.idl.unitree_go": types.ModuleType("unitree_sdk2py.idl.unitree_go"),
            "unitree_sdk2py.idl.unitree_go.msg": types.ModuleType("unitree_sdk2py.idl.unitree_go.msg"),
            "unitree_sdk2py.idl.unitree_go.msg.dds_": dds_module,
        }
        cls.module_patch = mock.patch.dict(sys.modules, modules)
        cls.module_patch.start()
        sys.modules.pop("teleop.robot_control.robot_hand_linker_o6", None)
        cls.module = importlib.import_module("teleop.robot_control.robot_hand_linker_o6")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("teleop.robot_control.robot_hand_linker_o6", None)
        cls.module_patch.stop()

    def setUp(self):
        FakePublisher.instances.clear()
        FakeSubscriber.instances.clear()
        FakeSubscriber.queued_messages = {}

    @staticmethod
    def queue_pair(left=None, right=None, modes=(1, 1)):
        if left is not None:
            FakeSubscriber.deliver("rt/linker/left/state", state_message(left, modes[0]))
        if right is not None:
            FakeSubscriber.deliver("rt/linker/right/state", state_message(right, modes[1]))

    def activate_after_release(
        self,
        controller,
        pre_release_left,
        pre_release_right,
        post_release_left,
        post_release_right,
    ):
        errors = []

        def activate():
            try:
                controller.activate()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=activate, daemon=True)
        thread.start()
        time.sleep(0.005)
        self.assertTrue(thread.is_alive())
        self.assertTrue(all(len(publisher.writes) == 0 for publisher in FakePublisher.instances))
        self.queue_pair(pre_release_left, pre_release_right)
        deadline = time.monotonic() + 0.2
        while (
            any(len(publisher.writes) == 0 for publisher in FakePublisher.instances)
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        self.assertTrue(all(len(publisher.writes) == 1 for publisher in FakePublisher.instances))
        self.assertTrue(thread.is_alive())
        self.queue_pair(post_release_left, post_release_right)
        thread.join(timeout=0.2)
        self.assertFalse(thread.is_alive())
        if errors:
            raise errors[0]

    def test_constructor_and_readiness_do_not_publish(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        self.assertEqual(
            [publisher.topic for publisher in FakePublisher.instances],
            ["rt/linker/left/cmd", "rt/linker/right/cmd"],
        )
        self.assertEqual(
            [subscriber.topic for subscriber in FakeSubscriber.instances],
            ["rt/linker/left/state", "rt/linker/right/state"],
        )
        controller.wait_until_ready(timeout=0.1)
        self.assertEqual([publisher.writes for publisher in FakePublisher.instances], [[], []])
        controller.stop()
        self.assertEqual([publisher.writes for publisher in FakePublisher.instances], [[], []])

    def test_activate_release_update_and_idempotent_stop(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(
            controller,
            [0.11] * 6,
            [0.22] * 6,
            [0.12] * 6,
            [0.23] * 6,
        )
        for publisher, expected in zip(FakePublisher.instances, ([0.11] * 6, [0.22] * 6)):
            self.assertEqual([command.mode for command in publisher.writes[0].cmds], [0] * 6)
            np.testing.assert_allclose([command.q for command in publisher.writes[0].cmds], expected)
            np.testing.assert_allclose([command.dq for command in publisher.writes[0].cmds], [0.0] * 6)
            np.testing.assert_allclose([command.tau for command in publisher.writes[0].cmds], [0.0] * 6)

        left_target = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        right_target = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        controller.update(left_target, right_target)
        for publisher, expected, initial in zip(FakePublisher.instances, (left_target, right_target), (0.12, 0.23)):
            command = publisher.writes[1]
            self.assertEqual([item.mode for item in command.cmds], [1] * 6)
            actual = np.array([item.q for item in command.cmds])
            self.assertTrue(np.all(actual >= np.minimum(initial, expected)))
            self.assertTrue(np.all(actual <= np.maximum(initial, expected)))
            self.assertLess(np.linalg.norm(actual - expected), np.linalg.norm(initial - np.array(expected)))
            np.testing.assert_allclose([item.dq for item in command.cmds], [1.0] * 6)
            np.testing.assert_allclose([item.tau for item in command.cmds], [1.0] * 6)
            self.assertEqual(
                publisher.timeouts,
                [self.module.COMMAND_TIMEOUT, self.module.COMMAND_TIMEOUT],
            )

        self.queue_pair([0.15] * 6, [0.25] * 6)
        controller.stop()
        write_counts = [len(publisher.writes) for publisher in FakePublisher.instances]
        controller.stop()
        self.assertEqual([len(publisher.writes) for publisher in FakePublisher.instances], write_counts)
        self.assertEqual(write_counts, [3, 3])
        for publisher, expected in zip(FakePublisher.instances, ([0.15] * 6, [0.25] * 6)):
            release = publisher.writes[-1]
            self.assertEqual([command.mode for command in release.cmds], [0] * 6)
            np.testing.assert_allclose([command.q for command in release.cmds], expected)
            np.testing.assert_allclose([command.dq for command in release.cmds], [0.0] * 6)
            np.testing.assert_allclose([command.tau for command in release.cmds], [0.0] * 6)
        self.assertTrue(all(publisher.closed for publisher in FakePublisher.instances))

    def test_failed_write_rejects_action_after_attempting_both_hands(self):
        for failed_side in (0, 1):
            with self.subTest(failed_side=failed_side):
                FakePublisher.instances.clear()
                FakeSubscriber.instances.clear()
                FakeSubscriber.queued_messages = {}
                self.queue_pair([0.1] * 6, [0.2] * 6)
                controller = self.module.LinkerO6Controller()
                controller.wait_until_ready(timeout=0.1)
                self.activate_after_release(
                    controller,
                    [0.11] * 6,
                    [0.22] * 6,
                    [0.12] * 6,
                    [0.23] * 6,
                )
                previous_action = controller.get_action()
                before = [len(publisher.writes) for publisher in FakePublisher.instances]
                FakePublisher.instances[failed_side].write_result = False
                with self.assertRaises(RuntimeError):
                    controller.update([0.3] * 6, [0.4] * 6)
                self.assertEqual(
                    [len(publisher.writes) for publisher in FakePublisher.instances],
                    [count + 1 for count in before],
                )
                np.testing.assert_allclose(controller.get_action()[0], previous_action[0])
                np.testing.assert_allclose(controller.get_action()[1], previous_action[1])
                FakePublisher.instances[failed_side].write_result = True
                controller.stop()

    def test_activate_requires_successful_readiness(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        with self.assertRaises(RuntimeError):
            controller.activate()
        self.assertEqual([publisher.writes for publisher in FakePublisher.instances], [[], []])
        controller.stop()

    def test_missing_side_times_out_without_writes(self):
        self.queue_pair(left=[0.1] * 6)
        controller = self.module.LinkerO6Controller()
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            controller.wait_until_ready(timeout=0.02)
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertEqual([publisher.writes for publisher in FakePublisher.instances], [[], []])
        controller.stop()

    def test_invalid_targets_are_rejected_before_writes(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(
            controller,
            [0.1] * 6,
            [0.2] * 6,
            [0.1] * 6,
            [0.2] * 6,
        )
        write_counts = [len(publisher.writes) for publisher in FakePublisher.instances]
        for side in ("left", "right"):
            for bad in ([0.0] * 5, [0.0] * 7, [-0.1] * 6, [float("nan")] * 6, [float("inf")] * 6, [1.1] * 6):
                with self.subTest(side=side, target=bad):
                    left = bad if side == "left" else [0.5] * 6
                    right = bad if side == "right" else [0.5] * 6
                    with self.assertRaises(ValueError):
                        controller.update(left, right)
                    self.assertEqual(
                        [len(publisher.writes) for publisher in FakePublisher.instances],
                        write_counts,
                    )
        controller.stop()

    def test_invalid_state_is_rejected_without_writes(self):
        self.queue_pair([float("nan")] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        with self.assertRaises(ValueError):
            controller.wait_until_ready(timeout=0.1)
        self.assertEqual([publisher.writes for publisher in FakePublisher.instances], [[], []])
        controller.stop()

    def test_stop_releases_both_hands_when_latest_state_is_invalid(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(
            controller,
            [0.11] * 6,
            [0.22] * 6,
            [0.12] * 6,
            [0.23] * 6,
        )
        self.queue_pair([float("nan")] * 6, [float("nan")] * 6)
        controller.stop()
        for publisher in FakePublisher.instances:
            release = publisher.writes[-1]
            self.assertEqual([command.mode for command in release.cmds], [0] * 6)
            self.assertEqual(len(release.cmds), 6)
            np.testing.assert_allclose([command.dq for command in release.cmds], [0.0] * 6)
            np.testing.assert_allclose([command.tau for command in release.cmds], [0.0] * 6)
            self.assertTrue(publisher.closed)

    def test_state_and_action_accessors_return_copies(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(
            controller,
            [0.11] * 6,
            [0.22] * 6,
            [0.12] * 6,
            [0.23] * 6,
        )
        state = controller.get_state()
        action = controller.get_action()
        state[0][0] = 9.0
        action[0][0] = 9.0
        self.assertNotEqual(controller.get_state()[0][0], 9.0)
        self.assertNotEqual(controller.get_action()[0][0], 9.0)
        controller.stop()

    def test_each_hand_recovers_from_gate_timeout_without_reactivating_controller(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(controller, [0.1] * 6, [0.2] * 6, [0.1] * 6, [0.2] * 6)
        controller.update([0.3] * 6, [0.4] * 6)
        for side in (0, 1, 0):
            with self.subTest(side=side):
                modes = [2, 2]
                modes[side] = 0
                self.queue_pair([0.12] * 6, [0.23] * 6, modes=modes)
                controller.update([0.8] * 6, [0.9] * 6)
                release = FakePublisher.instances[side].writes[-1]
                self.assertEqual([cmd.mode for cmd in release.cmds], [0] * 6)
                np.testing.assert_allclose([cmd.q for cmd in release.cmds], [0.12 if side == 0 else 0.23] * 6)
                np.testing.assert_allclose([cmd.dq for cmd in release.cmds], 0)
                np.testing.assert_allclose([cmd.tau for cmd in release.cmds], 0)
                self.assertEqual(FakePublisher.instances[1 - side].writes[-1].cmds[0].mode, 1)
                self.assertTrue(controller.active)
                # An in-flight armed status cannot acknowledge our release.
                self.queue_pair([0.12] * 6, [0.23] * 6, modes=(2, 2))
                controller.update([0.8] * 6, [0.9] * 6)
                self.assertEqual(FakePublisher.instances[side].writes[-1].cmds[0].mode, 0)
                modes[side] = 1
                self.queue_pair([0.12] * 6, [0.23] * 6, modes=modes)
                controller.update([0.5] * 6, [0.6] * 6)
                resumed = FakePublisher.instances[side].writes[-1]
                self.assertEqual([cmd.mode for cmd in resumed.cmds], [1] * 6)
                actual = np.array([cmd.q for cmd in resumed.cmds])
                self.assertTrue(np.all(actual > (0.12 if side == 0 else 0.23)))
                self.assertTrue(np.all(actual < (0.5 if side == 0 else 0.6)))
        controller.stop()

    def test_tracking_hold_requires_new_ready_feedback_and_fresh_tracking_to_resume(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(controller, [0.1] * 6, [0.2] * 6, [0.1] * 6, [0.2] * 6)
        controller.update([0.7] * 6, [0.8] * 6)
        controller.hold()
        controller.update([0.9] * 6, [0.9] * 6)
        self.assertEqual([pub.writes[-1].cmds[0].mode for pub in FakePublisher.instances], [0, 0])
        self.queue_pair([0.12] * 6, [0.23] * 6)
        controller.update([0.9] * 6, [0.9] * 6, tracking_fresh=(False, True))
        self.assertEqual([pub.writes[-1].cmds[0].mode for pub in FakePublisher.instances], [0, 1])
        np.testing.assert_allclose(controller.get_action()[0], [0.12] * 6)
        self.queue_pair([0.12] * 6, [0.23] * 6, modes=(1, 2))
        controller.update([0.4] * 6, [0.5] * 6)
        self.assertEqual([pub.writes[-1].cmds[0].mode for pub in FakePublisher.instances], [1, 1])
        self.assertTrue(np.all(controller.get_action()[0] > 0.12))
        self.assertTrue(np.all(controller.get_action()[0] < 0.4))
        controller.stop()

    def test_smoothing_reduces_noise_and_reaches_full_range_without_bias(self):
        self.queue_pair([0.5] * 6, [0.5] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(controller, [0.5] * 6, [0.5] * 6, [0.5] * 6, [0.5] * 6)
        started = controller.action_time
        filtered = []
        with mock.patch.object(self.module.time, "monotonic") as clock:
            for step in range(1, 121):
                clock.return_value = started + step / 30
                self.queue_pair([0.5] * 6, [0.5] * 6)
                target = 0.5 + (-1) ** step * 0.02
                controller.update([target] * 6, [target] * 6)
                if step > 30:
                    filtered.append(controller.get_action()[0][0])
            self.assertLess(np.std(filtered), 0.01)
            for step in range(121, 151):
                clock.return_value = started + step / 30
                self.queue_pair([0.5] * 6, [0.5] * 6, modes=(2, 2))
                controller.update([1.0] * 6, [0.0] * 6)
            np.testing.assert_allclose(controller.get_action(), [[1.0] * 6, [0.0] * 6], atol=1e-9)
            controller.stop()

    def test_recovery_smoothing_starts_from_measured_state_not_old_action(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(controller, [0.1] * 6, [0.2] * 6, [0.1] * 6, [0.2] * 6)
        started = controller.action_time
        with mock.patch.object(self.module.time, "monotonic") as clock:
            for step in range(1, 16):
                clock.return_value = started + step / 30
                self.queue_pair([0.1] * 6, [0.2] * 6)
                controller.update([0.9] * 6, [0.9] * 6)
            clock.return_value = started + 1.0
            self.queue_pair([0.2] * 6, [0.2] * 6, modes=(0, 2))
            controller.update([0.6] * 6, [0.9] * 6)
            clock.return_value += 1 / 30
            self.queue_pair([0.3] * 6, [0.2] * 6, modes=(1, 2))
            controller.update([0.6] * 6, [0.9] * 6)
            resumed = controller.get_action()[0]
            self.assertTrue(np.all(resumed > 0.3))
            self.assertTrue(np.all(resumed < 0.6))
            self.assertTrue(np.all(controller.get_action()[1] > 0.89))
            self.assertTrue(controller.active)
            controller.stop()

    def test_invalid_gate_feedback_is_rejected_before_enable(self):
        for modes in ((3, 1), (1, 255)):
            with self.subTest(modes=modes):
                self.queue_pair([0.1] * 6, [0.2] * 6, modes=modes)
                controller = self.module.LinkerO6Controller()
                with self.assertRaises(RuntimeError):
                    controller.wait_until_ready(timeout=0.1)
                self.assertTrue(all(not pub.writes for pub in FakePublisher.instances))
                controller.stop()

    def test_one_sided_feedback_stall_rejects_enable_command(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(
            controller,
            [0.11] * 6,
            [0.22] * 6,
            [0.12] * 6,
            [0.23] * 6,
        )
        time.sleep(self.module.STATE_TIMEOUT + 0.01)
        self.queue_pair(left=[0.13] * 6)
        write_counts = [len(publisher.writes) for publisher in FakePublisher.instances]
        with self.assertRaises(TimeoutError):
            controller.update([0.3] * 6, [0.4] * 6)
        self.assertEqual(
            [len(publisher.writes) for publisher in FakePublisher.instances],
            [count + 1 for count in write_counts],
        )
        for publisher in FakePublisher.instances:
            self.assertEqual([cmd.mode for cmd in publisher.writes[-1].cmds], [0] * 6)
        controller.stop()

    def test_feedback_updated_after_snapshot_does_not_cause_false_timeout(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(controller, [0.1] * 6, [0.2] * 6, [0.1] * 6, [0.2] * 6)
        controller.update([0.3] * 6, [0.4] * 6)
        controller.left_state_time = controller.right_state_time = 99.0
        controller.action_time = 99.99
        snapshot = controller._state_snapshot

        def preempted_snapshot():
            old = snapshot()
            self.queue_pair([0.15] * 6, [0.25] * 6, modes=(2, 2))
            return old

        with mock.patch.object(self.module.time, "monotonic", return_value=100.0):
            with mock.patch.object(controller, "_state_snapshot", side_effect=preempted_snapshot):
                controller.update([0.4] * 6, [0.5] * 6)
        self.assertEqual([pub.writes[-1].cmds[0].mode for pub in FakePublisher.instances], [1, 1])
        controller.stop()

    def test_stale_hold_releases_both_hands_and_reports_feedback_details(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(controller, [0.1] * 6, [0.2] * 6, [0.1] * 6, [0.2] * 6)
        controller.left_state_time = 100.0
        controller.right_state_time = 100.249
        controller.action_time = 100.24
        with mock.patch.object(self.module.time, "monotonic", return_value=100.25):
            with self.assertRaises(TimeoutError) as raised:
                controller.hold()
        self.assertIn("left_age_ms=250.0 right_age_ms=1.0", str(raised.exception))
        self.assertIn("left_count=3 right_count=3", str(raised.exception))
        self.assertIn("update_gap_ms=10.0 limit_ms=250", str(raised.exception))
        for publisher in FakePublisher.instances:
            release = publisher.writes[-1]
            self.assertEqual([cmd.mode for cmd in release.cmds], [0] * 6)
            np.testing.assert_array_equal([cmd.dq for cmd in release.cmds], [0] * 6)
            np.testing.assert_array_equal([cmd.tau for cmd in release.cmds], [0] * 6)
        controller.stop()

    def test_failed_stale_release_attempts_both_sides_and_preserves_timeout(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(controller, [0.1] * 6, [0.2] * 6, [0.1] * 6, [0.2] * 6)
        controller.left_state_time = controller.right_state_time = 100.0
        controller.action_time = 100.0
        counts = [len(pub.writes) for pub in FakePublisher.instances]
        FakePublisher.instances[0].write_result = False
        with mock.patch.object(self.module.time, "monotonic", return_value=101.0):
            with self.assertRaises(TimeoutError) as raised:
                controller.hold()
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual([len(pub.writes) for pub in FakePublisher.instances], [count + 1 for count in counts])
        self.assertEqual([pub.writes[-1].cmds[0].mode for pub in FakePublisher.instances], [0, 0])
        FakePublisher.instances[0].write_result = True
        controller.stop()

    def test_new_callback_on_only_one_side_cannot_hide_other_stale_side(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.wait_until_ready(timeout=0.1)
        self.activate_after_release(controller, [0.1] * 6, [0.2] * 6, [0.1] * 6, [0.2] * 6)
        controller.left_state_time = controller.right_state_time = 100.0
        controller.action_time = 100.0
        snapshot = controller._state_snapshot

        def one_new_side():
            old = snapshot()
            self.queue_pair(left=[0.15] * 6)
            return old

        with mock.patch.object(self.module.time, "monotonic", return_value=101.0):
            with mock.patch.object(controller, "_state_snapshot", side_effect=one_new_side):
                with self.assertRaisesRegex(TimeoutError, "left_age_ms=0.0 right_age_ms=1000.0"):
                    controller.update([0.4] * 6, [0.5] * 6)
        self.assertEqual([pub.writes[-1].cmds[0].mode for pub in FakePublisher.instances], [0, 0])
        controller.stop()

    def test_callback_gap_log_identifies_side_count_and_gate(self):
        self.queue_pair([0.1] * 6, [0.2] * 6)
        controller = self.module.LinkerO6Controller()
        controller.left_state_time = controller.right_state_time = 100.0
        with mock.patch.object(self.module.time, "monotonic", return_value=100.5):
            with self.assertLogs(self.module.logger, level="WARNING") as logs:
                self.queue_pair(left=[0.2] * 6)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("left: callback gap_ms=500.0 count=2 gate=1", logs.output[0])
        controller.stop()


if __name__ == "__main__":
    unittest.main()
