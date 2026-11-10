import importlib
import math
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_ABSENT_GRIP_CAP = Path(tempfile.gettempdir()) / "xr-o6-grip-cap-absent-for-dex1-tests.json"


class FakeMotorCmd:
    def __init__(self):
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0


class FakeMotorCmds:
    def __init__(self):
        self.cmds = []


class FakePublisher:
    instances = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.writes = []
        self.closed = False
        type(self).instances.append(self)

    def Init(self):
        pass

    def Write(self, message, timeout=None):
        self.writes.append(message)
        return True

    def Close(self):
        self.closed = True


class FakeSubscriber:
    instances = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.handler = None
        self.closed = False
        type(self).instances.append(self)

    def Init(self, handler=None, queueLen=0):
        self.handler = handler

    def Close(self):
        self.closed = True

    @classmethod
    def deliver(cls, topic, message):
        for subscriber in cls.instances:
            if subscriber.topic == topic and subscriber.handler is not None and not subscriber.closed:
                subscriber.handler(message)
                return True
        return False


def gripper_state(q):
    return types.SimpleNamespace(states=[types.SimpleNamespace(q=q, dq=0.0, tau_est=0.1)])


def o6_state(values, mode=1):
    return types.SimpleNamespace(states=[types.SimpleNamespace(q=value, mode=mode) for value in values])


def hand_points(distance):
    points = np.zeros((25, 3))
    points[9] = [distance, 0.0, 0.0]
    return points


class Dex1GripperTest(unittest.TestCase):
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
        for name in ("teleop.robot_control.robot_hand_linker_o6", "teleop.robot_control.robot_hand_dex1_r1"):
            sys.modules.pop(name, None)
        cls.o6 = importlib.import_module("teleop.robot_control.robot_hand_linker_o6")
        cls.dex1 = importlib.import_module("teleop.robot_control.robot_hand_dex1_r1")

    @classmethod
    def tearDownClass(cls):
        for name in ("teleop.robot_control.robot_hand_linker_o6", "teleop.robot_control.robot_hand_dex1_r1"):
            sys.modules.pop(name, None)
        cls.module_patch.stop()

    def setUp(self):
        FakePublisher.instances.clear()
        FakeSubscriber.instances.clear()
        self.env = mock.patch.dict(os.environ, {"XR_O6_GRIP_CAP": str(_ABSENT_GRIP_CAP)})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    @staticmethod
    def publisher(topic):
        return next(publisher for publisher in FakePublisher.instances if publisher.topic == topic)

    @staticmethod
    def activate_while_feeding(controller, feeds):
        errors = []

        def run():
            try:
                controller.activate()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 2.0
        while thread.is_alive() and time.monotonic() < deadline:
            for topic, message in feeds:
                FakeSubscriber.deliver(topic, message)
            time.sleep(0.01)
        thread.join(1.0)
        if errors:
            raise errors[0]

    def ready_gripper(self, q=1.0, **kwargs):
        gripper = self.dex1.Dex1GripperController("right", **kwargs)
        FakeSubscriber.deliver("rt/dex1/right/state", gripper_state(q))
        gripper.wait_until_ready(timeout=0.5)
        self.activate_while_feeding(gripper, [("rt/dex1/right/state", gripper_state(q))])
        self.assertTrue(gripper.active)
        return gripper

    def last_q(self):
        return self.publisher("rt/dex1/right/cmd").writes[-1].cmds[0].q

    # Thumb-index mapping

    def test_pinch_maps_linearly_onto_the_stroke_and_clips(self):
        retargeter = self.dex1.Dex1PinchRetargeter(0.02, 0.10, smoothing_s=0.0)
        for distance, expected in ((0.01, 0.0), (0.02, 0.0), (0.06, 2.7), (0.10, 5.4), (0.30, 5.4)):
            self.assertAlmostEqual(retargeter.retarget(hand_points(distance))[0], expected, places=6)

    def test_pinch_smoothing_moves_part_of_the_way(self):
        retargeter = self.dex1.Dex1PinchRetargeter(0.02, 0.10, smoothing_s=0.02)
        self.assertAlmostEqual(retargeter.retarget(hand_points(0.02), now=10.0)[0], 0.0)
        value = retargeter.retarget(hand_points(0.10), now=10.02)[0]
        self.assertAlmostEqual(value, 5.4 * (1.0 - math.exp(-1.0)), places=6)
        retargeter.reset()
        self.assertAlmostEqual(retargeter.retarget(hand_points(0.10), now=11.0)[0], 5.4)

    def test_pinch_rejects_short_or_nonfinite_points(self):
        retargeter = self.dex1.Dex1PinchRetargeter()
        with self.assertRaises(ValueError):
            retargeter.retarget(np.zeros((5, 3)))
        points = hand_points(0.05)
        points[4, 0] = np.nan
        with self.assertRaises(ValueError):
            retargeter.retarget(points)

    def test_bad_mapping_values_are_rejected(self):
        with self.assertRaises(ValueError):
            self.dex1.Dex1PinchRetargeter(0.10, 0.05)
        with self.assertRaises(ValueError):
            self.dex1.Dex1GripperController("right", kp=-1.0)

    # Gripper controller

    def test_activation_holds_the_measured_position(self):
        gripper = self.ready_gripper(q=1.0)
        command = self.publisher("rt/dex1/right/cmd").writes[-1].cmds[0]
        self.assertAlmostEqual(command.q, 1.0)
        self.assertAlmostEqual(command.kp, 5.0)
        self.assertAlmostEqual(command.kd, 0.05)
        np.testing.assert_allclose(gripper.get_action(), [1.0])

    def test_command_leads_the_measured_position_by_at_most_grip_delta(self):
        gripper = self.ready_gripper(q=1.0)
        gripper.update([5.4])
        self.assertAlmostEqual(self.last_q(), 1.18)
        gripper.update([0.0])
        self.assertAlmostEqual(self.last_q(), 0.82)
        gripper.update([1.1])
        self.assertAlmostEqual(self.last_q(), 1.1)

    def test_lost_tracking_keeps_the_last_request(self):
        gripper = self.ready_gripper(q=1.0)
        gripper.update([0.0], tracking_fresh=True)
        gripper.update([5.4], tracking_fresh=False)
        self.assertAlmostEqual(self.last_q(), 0.82)
        np.testing.assert_allclose(gripper.get_action(), [0.0])

    def test_feeding_back_the_action_does_not_ratchet_the_squeeze(self):
        gripper = self.ready_gripper(q=1.0)
        gripper.update([0.0])
        for _ in range(5):
            gripper.update(gripper.get_action(), tracking_fresh=True)
        self.assertAlmostEqual(self.last_q(), 0.82)

    def test_targets_are_clamped_to_the_stroke(self):
        gripper = self.ready_gripper(q=5.3)
        gripper.update([9.0])
        np.testing.assert_allclose(gripper.get_action(), [5.4])
        self.assertAlmostEqual(self.last_q(), 5.4)

    def test_stale_feedback_raises_and_publishes_nothing(self):
        gripper = self.ready_gripper(q=1.0)
        writes = len(self.publisher("rt/dex1/right/cmd").writes)
        with gripper.state_lock:
            gripper.state_time -= self.dex1.STATE_TIMEOUT + 0.05
        with self.assertRaises(TimeoutError):
            gripper.update([2.0])
        self.assertEqual(len(self.publisher("rt/dex1/right/cmd").writes), writes)

    def test_out_of_range_feedback_asks_for_calibration(self):
        gripper = self.dex1.Dex1GripperController("right")
        FakeSubscriber.deliver("rt/dex1/right/state", gripper_state(8.0))
        with self.assertRaisesRegex(RuntimeError, "calibrate"):
            gripper.wait_until_ready(timeout=0.2)

    def test_missing_gripper_names_the_service(self):
        gripper = self.dex1.Dex1GripperController("right")
        with self.assertRaisesRegex(TimeoutError, "dex1_1_gripper_server"):
            gripper.wait_until_ready(timeout=0.05)

    def test_stop_holds_position_and_closes_endpoints(self):
        gripper = self.ready_gripper(q=2.0)
        gripper.update([0.0])
        gripper.stop()
        self.assertAlmostEqual(self.last_q(), 2.0)
        self.assertTrue(self.publisher("rt/dex1/right/cmd").closed)
        self.assertTrue(all(sub.closed for sub in FakeSubscriber.instances if sub.topic.startswith("rt/dex1/")))

    # One-sided Linker O6

    def test_sides_must_be_a_subset(self):
        for sides in ((), ("left", "left"), ("middle",)):
            with self.assertRaises(ValueError):
                self.o6.LinkerO6Controller(sides=sides, apply_grip_cap=False)

    def ready_left_o6(self, values=(0.5,) * 6):
        controller = self.o6.LinkerO6Controller(sides=("left",), apply_grip_cap=False)
        FakeSubscriber.deliver("rt/linker/left/state", o6_state(values))
        controller.wait_until_ready(timeout=0.5)
        self.activate_while_feeding(controller, [("rt/linker/left/state", o6_state(values))])
        return controller

    def test_left_only_o6_has_no_right_endpoints(self):
        self.o6.LinkerO6Controller(sides=("left",), apply_grip_cap=False)
        self.assertEqual([p.topic for p in FakePublisher.instances], ["rt/linker/left/cmd"])
        self.assertEqual([s.topic for s in FakeSubscriber.instances], ["rt/linker/left/state"])

    def test_left_only_o6_runs_without_right_feedback(self):
        controller = self.ready_left_o6()
        left = self.publisher("rt/linker/left/cmd")
        writes = len(left.writes)
        controller.update(np.full(6, 0.6), None, tracking_fresh=(True, False))
        self.assertEqual(len(left.writes), writes + 1)
        action_left, action_right = controller.get_action()
        self.assertEqual(action_left.shape, (6,))
        self.assertIsNone(action_right)
        state_left, state_right = controller.get_state()
        self.assertIsNone(state_right)
        snapshot = controller.get_recording_snapshot()
        self.assertEqual(set(snapshot["state"]), {"left"})
        self.assertIsNone(snapshot["requested"]["right_q"])
        controller.stop()
        self.assertTrue(left.closed)

    def test_left_only_o6_stale_check_ignores_the_missing_right(self):
        controller = self.ready_left_o6()
        with controller.state_lock:
            controller.left_state_time -= self.o6.STATE_TIMEOUT + 0.05
        with self.assertRaisesRegex(TimeoutError, "right_age_ms=off"):
            controller.update(np.full(6, 0.6), None, tracking_fresh=(True, False))

    # Pairing

    def test_o6_and_gripper_share_the_paired_interface(self):
        o6 = self.o6.LinkerO6Controller(sides=("left",), apply_grip_cap=False)
        gripper = self.dex1.Dex1GripperController("right")
        hands = self.dex1.O6Dex1Hands(o6, gripper)
        FakeSubscriber.deliver("rt/linker/left/state", o6_state((0.5,) * 6))
        FakeSubscriber.deliver("rt/dex1/right/state", gripper_state(1.0))
        hands.wait_until_ready(timeout=0.5)
        self.activate_while_feeding(hands, [("rt/linker/left/state", o6_state((0.5,) * 6)),
                                            ("rt/dex1/right/state", gripper_state(1.0))])
        hands.update(np.full(6, 0.6), np.array([1.1]), tracking_fresh=(True, True))
        left, right = hands.get_state()
        self.assertEqual((left.shape, right.shape), ((6,), (1,)))
        self.assertEqual(np.concatenate(hands.get_action()).shape, (7,))
        self.assertAlmostEqual(self.last_q(), 1.1)
        snapshot = hands.get_recording_snapshot()
        self.assertEqual(snapshot["devices"], {"left": "linker_o6", "right": "dex1_1"})
        self.assertEqual(snapshot["requested"]["right_q"], [1.1])
        hands.stop()
        self.assertTrue(all(p.closed for p in FakePublisher.instances))

    def test_pairing_needs_different_wrists(self):
        o6 = self.o6.LinkerO6Controller(sides=("right",), apply_grip_cap=False)
        gripper = self.dex1.Dex1GripperController("right")
        with self.assertRaises(ValueError):
            self.dex1.O6Dex1Hands(o6, gripper)

    def test_retargeter_puts_the_gripper_on_its_wrist(self):
        o6_side = types.SimpleNamespace(method="vector", reset=lambda: None)
        gripper_side = self.dex1.Dex1PinchRetargeter()
        retargeter = self.dex1.O6Dex1Retargeter(o6_side, gripper_side, gripper_side="right")
        self.assertIs(retargeter.left, o6_side)
        self.assertIs(retargeter.right, gripper_side)
        self.assertIn("dex1_pinch", retargeter.mapping_name)


class Dex1PreflightTest(unittest.TestCase):
    def test_gripper_feedback_validation(self):
        from tools.check_r1_teleop import validate_gripper_state
        validate_gripper_state(gripper_state(2.0))
        for bad in (gripper_state(float("nan")), gripper_state(9.0), types.SimpleNamespace(states=[])):
            with self.assertRaises(ValueError):
                validate_gripper_state(bad)

    def test_right_wrist_comes_from_the_environment(self):
        from tools.check_r1_teleop import right_ee_from_environment
        with mock.patch.dict(os.environ, {"R1_RIGHT_EE": "dex1"}):
            self.assertEqual(right_ee_from_environment(), "dex1")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("R1_RIGHT_EE", None)
            self.assertEqual(right_ee_from_environment(), "linker_o6")
        with mock.patch.dict(os.environ, {"R1_RIGHT_EE": "gripper"}):
            with self.assertRaises(ValueError):
                right_ee_from_environment()


if __name__ == "__main__":
    unittest.main()
