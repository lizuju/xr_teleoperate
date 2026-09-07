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


def state_message(values):
    return types.SimpleNamespace(
        states=[types.SimpleNamespace(q=value) for value in values]
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
    def queue_pair(left=None, right=None):
        if left is not None:
            FakeSubscriber.deliver("rt/linker/left/state", state_message(left))
        if right is not None:
            FakeSubscriber.deliver("rt/linker/right/state", state_message(right))

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
        for publisher, expected in zip(FakePublisher.instances, (left_target, right_target)):
            command = publisher.writes[1]
            self.assertEqual([item.mode for item in command.cmds], [1] * 6)
            np.testing.assert_allclose([item.q for item in command.cmds], expected)
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
            write_counts,
        )
        controller.stop()


if __name__ == "__main__":
    unittest.main()
