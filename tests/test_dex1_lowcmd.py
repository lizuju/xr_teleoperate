"""The internally wired Dex1-1: an rt/lowcmd slot driven through the R1_A7 arm controller."""
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
import test_r1_a7_official_activation as arm_fixtures
from check_r1_teleop import dex1_wiring_from_environment, validate_gripper_slot
from teleop.robot_control.robot_hand_dex1_r1 import Dex1LowCmdGripper, INTERNAL_RIGHT_SLOT, STATE_TIMEOUT


class FakeArm:
    """Stands in for R1_A7_ArmController: every read is a fresh 1 kHz-style lowstate sample."""

    def __init__(self, q=1.0, motorstate=0, vol=38.0, present=True):
        self.q, self.motorstate, self.vol, self.present = q, motorstate, vol, present
        self.configured, self.targets, self.history = {}, [], []
        self.sequence, self.age_s = 0, 0.0

    def configure_gripper_slot(self, slot, kp, kd):
        self.configured[slot] = (kp, kd)

    def set_gripper_target(self, slot, q):
        self.targets.append((slot, q))

    def get_gripper_feedback(self, slot):
        if not self.present:
            return None
        self.sequence += 1
        entry = {"q": self.q, "dq": 0.0, "tau": 0.2, "motorstate": self.motorstate, "vol": self.vol,
                 "monotonic_ns": int((time.monotonic() - self.age_s) * 1e9), "sequence": self.sequence}
        self.history.append(entry)
        return dict(entry)

    def get_gripper_feedback_at(self, slot, target_ns, end_ns):
        candidates = [entry for entry in self.history if entry["monotonic_ns"] <= end_ns]
        return dict(min(candidates, key=lambda entry: abs(entry["monotonic_ns"] - target_ns))) if candidates else None


class LowCmdGripperTest(unittest.TestCase):
    def ready(self, **kwargs):
        arm = FakeArm(**kwargs)
        gripper = Dex1LowCmdGripper(arm)
        gripper.wait_until_ready(timeout=0.5)
        gripper.activate()
        return arm, gripper

    def test_slot_is_configured_with_gripper_gains_on_construction(self):
        arm = FakeArm()
        Dex1LowCmdGripper(arm)
        self.assertEqual(arm.configured, {INTERNAL_RIGHT_SLOT: (5.0, 0.05)})
        self.assertEqual(arm.targets, [])

    def test_activation_holds_and_updates_limit_the_squeeze(self):
        arm, gripper = self.ready(q=1.0)
        self.assertEqual(arm.targets[-1], (33, 1.0))
        gripper.update([5.4])
        self.assertAlmostEqual(arm.targets[-1][1], 1.18)
        gripper.update([0.0], request_sequence=7)
        self.assertAlmostEqual(arm.targets[-1][1], 0.82)
        published = gripper.get_recording_snapshot()["published"]
        self.assertEqual((published["mode"], published["request_sequence"]), (1, 7))

    def test_fault_or_unpowered_slot_names_the_app_setting(self):
        for kwargs in ({"motorstate": 0x80000000}, {"vol": 0.0}):
            with self.subTest(**kwargs):
                gripper = Dex1LowCmdGripper(FakeArm(**kwargs))
                with self.assertRaisesRegex(RuntimeError, "Unitree app"):
                    gripper.wait_until_ready(timeout=0.2)

    def test_missing_feedback_times_out_with_the_app_hint(self):
        gripper = Dex1LowCmdGripper(FakeArm(present=False))
        with self.assertRaisesRegex(TimeoutError, "Dex1-1"):
            gripper.wait_until_ready(timeout=0.05)

    def test_uncalibrated_angle_is_refused(self):
        gripper = Dex1LowCmdGripper(FakeArm(q=9.0))
        with self.assertRaisesRegex(RuntimeError, "calibrate"):
            gripper.wait_until_ready(timeout=0.2)

    def test_stale_feedback_stops_without_a_new_target(self):
        arm, gripper = self.ready()
        arm.age_s = STATE_TIMEOUT + 0.05
        count = len(arm.targets)
        with self.assertRaises(TimeoutError):
            gripper.update([2.0])
        self.assertEqual(len(arm.targets), count)

    def test_stop_holds_then_releases_the_slot(self):
        arm, gripper = self.ready(q=2.0)
        gripper.update([0.0])
        gripper.stop()
        self.assertEqual(arm.targets[-2], (33, 2.0))
        self.assertEqual(arm.targets[-1], (33, None))

    def test_recording_reads_come_from_lowstate(self):
        arm, gripper = self.ready(q=1.5)
        state = gripper.get_recording_snapshot()["state"]
        self.assertEqual((state["q"], state["qvel"], state["torque"]), ([1.5], [0.0], [0.2]))
        now_ns = time.monotonic_ns()
        aligned = gripper.get_recording_state_at(now_ns, now_ns)
        self.assertEqual(aligned["q"], [1.5])
        self.assertEqual(aligned["torque"], [0.2])


class ArmControllerGripperSlotTest(unittest.TestCase):
    def setUp(self):
        arm_fixtures.FakePublisher.instances.clear()
        self.namespace = arm_fixtures.load_r1_controller_namespace()
        self.controller = self.namespace["R1_A7_ArmController"](deferred_activation=True)
        self.addCleanup(self.controller.stop)

    def test_arm_head_waist_and_missing_slots_are_refused(self):
        for slot in (15, 28, 29, 30, 13, 35, -1):
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                self.controller.configure_gripper_slot(slot, 5.0, 0.05)
        with self.assertRaises(KeyError):
            self.controller.set_gripper_target(33, 1.0)

    def activate(self):
        controller = self.controller
        controller.configure_gripper_slot(33, 5.0, 0.05)
        original = controller.ctrl_head_and_waist_go_home
        controller.ctrl_head_and_waist_go_home = lambda **kwargs: original(duration=0.0, **kwargs)
        controller.activate()

    def wait_for_write(self, predicate):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            writes = arm_fixtures.FakePublisher.instances[0].writes
            if writes and predicate(writes[-1]):
                return writes[-1]
            time.sleep(0.002)
        self.fail("no matching lowcmd write")

    def test_activation_drives_the_gripper_slot_with_gripper_gains_not_the_hold(self):
        self.activate()
        command = self.controller.msg.motor_cmd[33]
        self.assertEqual((command.mode, command.kp, command.kd), (1, 5.0, 0.05))
        self.assertAlmostEqual(command.q, 0.33)
        self.assertEqual(self.controller.msg.motor_cmd[34].kp, self.controller.kp_high)

    def test_target_is_published_and_none_follows_the_measured_angle(self):
        self.activate()
        self.controller.set_gripper_target(33, 1.5)
        self.wait_for_write(lambda q: q[33] == 1.5)
        self.controller.set_gripper_target(33, None)
        self.wait_for_write(lambda q: abs(q[33] - 0.33) < 1e-9)

    def test_feedback_and_history_of_the_slot(self):
        self.controller.configure_gripper_slot(33, 5.0, 0.05)
        deadline = time.monotonic() + 1.0
        while self.controller.get_gripper_feedback(33) is None and time.monotonic() < deadline:
            time.sleep(0.002)
        time.sleep(0.01)
        feedback = self.controller.get_gripper_feedback(33)
        self.assertAlmostEqual(feedback["q"], 0.33)
        self.assertEqual((feedback["motorstate"], feedback["vol"]), (0, 38.0))
        now_ns = time.monotonic_ns()
        aligned = self.controller.get_gripper_feedback_at(33, now_ns, now_ns)
        self.assertAlmostEqual(aligned["q"], 0.33)


class PreflightGripperSlotTest(unittest.TestCase):
    @staticmethod
    def lowstate(**motor):
        values = {"q": 0.0, "dq": 0.0, "tau_est": 0.0, "vol": 38.0, "motorstate": 0}
        values.update(motor)
        return SimpleNamespace(motor_state=[SimpleNamespace(**values) for _ in range(35)])

    def test_present_calibrated_slot_passes(self):
        validate_gripper_slot(self.lowstate(q=2.0), 33)

    def test_fault_unpowered_or_uncalibrated_slot_fails(self):
        for motor in ({"motorstate": 0x80000000}, {"vol": 0.0}, {"q": 8.0}, {"q": float("nan")}):
            with self.subTest(**motor), self.assertRaises(ValueError):
                validate_gripper_slot(self.lowstate(**motor), 33)

    def test_wiring_defaults_to_internal_slot_33(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("R1_DEX1_TRANSPORT", None)
            os.environ.pop("R1_DEX1_SLOT", None)
            self.assertEqual(dex1_wiring_from_environment(), ("lowcmd", 33))
        with mock.patch.dict(os.environ, {"R1_DEX1_TRANSPORT": "service"}):
            self.assertEqual(dex1_wiring_from_environment()[0], "service")
        for env in ({"R1_DEX1_TRANSPORT": "usb"}, {"R1_DEX1_SLOT": "15"}):
            with mock.patch.dict(os.environ, env), self.assertRaises(ValueError):
                dex1_wiring_from_environment()


if __name__ == "__main__":
    unittest.main()
