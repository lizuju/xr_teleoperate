import ast
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


MAIN_PATH = Path(__file__).resolve().parents[1] / "teleop/teleop_hand_and_arm.py"


def assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def invokes(node, name):
    return any(
        isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        and child.func.attr == name for child in ast.walk(node)
    )


class R1StartupTrackingWaitTest(unittest.TestCase):
    def run_startup(self, events, *, stale_during_ik=False, early_r_during_ik=False, feedback_error=False):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "__name__ == '__main__'")
        setup = next(node for node in main.body if isinstance(node, ast.Try))
        start = next(i for i, node in enumerate(setup.body) if assigns(node, "r1_arm_request_floor"))
        end = next(i for i in range(start, len(setup.body))
                   if isinstance(setup.body[i], ast.While) and invokes(setup.body[i], "activate"))
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"is_fresh_motion_data", "head_yaw_rotation"}]
        clock = [10.0]
        sample = [None]
        ticks = []
        events = iter(events)
        arm = Mock()
        motor_q = np.zeros(35)
        motor_q[13] = 0.17
        arm.get_current_motor_q.return_value = motor_q
        arm.get_current_dual_arm_q.return_value = np.zeros(14)
        arm.get_current_waist_yaw.return_value = 0.17
        if feedback_error:
            arm.get_current_waist_yaw.side_effect = RuntimeError("robot feedback stale")
        ik = Mock()
        ik.forward_wrist_poses.return_value = (np.eye(4), np.eye(4))
        hand = Mock()
        hand_factory = Mock(return_value=hand)
        controller_module = ModuleType("teleop.robot_control.robot_hand_linker_o6")
        controller_module.LinkerO6Controller = hand_factory
        switcher = Mock()
        switcher.Enter_Debug_Mode.return_value = (0, {})
        switcher_factory = Mock(return_value=switcher)
        wrapper = Mock()
        wrapper.tvuer.get_tracking_diagnostics.return_value = {}
        ns = {
            "np": np, "math": math, "json": json, "STOP": False, "START": False,
            "READY": False, "ARM_REQUEST_GENERATION": 0,
            "r1_a7_anchored": True, "r1_a7_deferred_real": True,
            "r1_vision_left_reference": None, "r1_vision_right_reference": None,
            "args": SimpleNamespace(ee="linker_o6", tracking_timeout=0.25, waist_follow=False,
                                    arm_translation_scale=1.0),
            "arm_ctrl": arm, "hand_ctrl": None, "arm_ik": None,
            "MotionSwitcher": switcher_factory, "tv_wrapper": wrapper,
            "camera_config": {"head_camera": {"enable_zmq": False}},
            "xr_need_local_img": False, "logger_mp": Mock(),
            "arm_diagnostic_file": None,
        }

        def make_sample(timestamp, marker):
            left, right, head = np.eye(4), np.eye(4), np.eye(4)
            left[0, 3], right[0, 3], head[0, 3] = marker, -marker, marker / 2
            return SimpleNamespace(motion_data_ready=True, motion_data_timestamp=timestamp,
                                   left_hand_timestamp=timestamp, right_hand_timestamp=timestamp,
                                   left_wrist_pose=left, right_wrist_pose=right, head_pose=head)

        sample[0] = make_sample(clock[0], 0.0)

        def tick(_):
            try:
                event = next(events)
            except StopIteration:
                ns["STOP"] = True
                return
            ticks.append({name: ns.get(name) for name in (
                "r1_activation_prepared", "r1_startup_tracking_ready", "START",
            )})
            clock[0] += event.get("dt", 0.1)
            source = event.get("sample", "fresh")
            if source == "fresh":
                sample[0] = make_sample(clock[0], event.get("marker", len(ticks) / 100))
            elif source == "missing":
                sample[0] = make_sample(0.0, 0.0)
            elif source == "old":
                sample[0] = make_sample(clock[0] - 1.0, 0.0)
            elif source != "duplicate":
                raise AssertionError(source)
            if event.get("key") == "r":
                ns["ARM_REQUEST_GENERATION"] += 1
                ns["START"] = True
            elif event.get("key") == "q":
                ns["STOP"], ns["START"] = True, False

        def activate():
            clock[0] += 3.0
            sample[0] = make_sample(0.0, 0.0)

        def load_ik(**kwargs):
            clock[0] += 2.0
            sample[0] = make_sample(0.0 if stale_during_ik else clock[0], 0.0)
            if early_r_during_ik:
                ns["ARM_REQUEST_GENERATION"] += 1
                ns["START"] = True
            return ik

        arm.activate.side_effect = activate
        ik_factory = Mock(side_effect=load_ik)
        ns["R1_A7_ArmIK"] = ik_factory
        ns["time"] = SimpleNamespace(monotonic=lambda: clock[0], sleep=tick)
        wrapper.get_tele_data.side_effect = lambda: sample[0]
        code = ast.fix_missing_locations(ast.Module(body=helpers + setup.body[start:end + 1], type_ignores=[]))
        with patch.dict(sys.modules, {controller_module.__name__: controller_module}):
            exec(compile(code, str(MAIN_PATH), "exec"), ns)
        self.assertLessEqual(arm.activate.call_count, 1)
        self.assertLessEqual(ik_factory.call_count, 1)
        self.assertLessEqual(hand_factory.call_count, 1)
        self.assertLessEqual(switcher.Enter_Debug_Mode.call_count, 1)
        return SimpleNamespace(ns=ns, arm=arm, hand=hand, hand_factory=hand_factory,
                               ik_factory=ik_factory, switcher=switcher, ticks=ticks, sample=sample[0])

    def test_no_first_r_emits_no_activation(self):
        result = self.run_startup([{}, {}, {"key": "q"}])
        result.arm.activate.assert_not_called()
        result.hand_factory.assert_not_called()
        result.ik_factory.assert_not_called()

    def test_q_cancels_pending_first_request_before_hands_recover(self):
        result = self.run_startup([{"key": "r", "sample": "missing"}, {"key": "q"}])
        result.arm.activate.assert_not_called()
        result.hand_factory.assert_not_called()

    def test_q_at_stability_threshold_does_not_enable_following(self):
        result = self.run_startup([{"key": "r"}] + [{}] * 4 + [{"key": "q"}])
        result.arm.activate.assert_called_once_with()
        result.hand.activate.assert_not_called()
        self.assertFalse(result.ns["START"])
        self.assertIsNone(result.ns["r1_vision_left_reference"])

    def test_missing_hands_after_recenter_and_ik_wait_without_exiting(self):
        result = self.run_startup(
            [{"key": "r"}] + [{"sample": "missing"}] * 12 + [{"key": "q", "sample": "missing"}],
            stale_during_ik=True,
        )
        result.arm.activate.assert_called_once_with()
        result.ik_factory.assert_called_once_with(waist_yaw=0.17)
        result.hand.wait_until_ready.assert_called_once_with(timeout=3.0)
        result.hand.activate.assert_not_called()
        self.assertTrue(result.ns["r1_activation_prepared"])
        self.assertFalse(result.ns["START"])
        self.assertIsNone(result.ns["r1_vision_left_reference"])

    def test_stable_hands_start_automatically_after_one_r(self):
        result = self.run_startup([{"key": "r"}] + [{}] * 8 + [{"key": "q"}])
        self.assertTrue(result.ns["r1_startup_tracking_ready"])
        self.assertTrue(result.ns["START"])
        self.assertEqual(result.ns["ARM_REQUEST_GENERATION"], 1)
        result.hand.activate.assert_called_once_with()

    def test_first_r_with_missing_hands_waits_then_prepares_without_another_key(self):
        result = self.run_startup([{"key": "r", "sample": "missing"}] + [{"sample": "missing"}] * 4 + [{}] * 8)
        self.assertTrue(result.ns["START"])
        self.assertEqual(result.ns["ARM_REQUEST_GENERATION"], 1)
        result.arm.activate.assert_called_once_with()
        result.hand.activate.assert_called_once_with()

    def test_early_r_during_ik_and_recovery_is_consumed(self):
        result = self.run_startup(
            [{"key": "r"}, {"key": "r", "sample": "missing"}]
            + [{"key": "r", "dt": 0.03}] * 4 + [{"key": "q", "sample": "missing"}],
            stale_during_ik=True, early_r_during_ik=True,
        )
        self.assertFalse(any(tick["r1_startup_tracking_ready"] for tick in result.ticks))
        result.hand.activate.assert_not_called()
        result.arm.activate.assert_called_once_with()

    def test_auto_start_captures_latest_pose_without_reinitializing(self):
        result = self.run_startup([{"key": "r"}] + [{}] * 4 + [{"marker": 0.42}])
        self.assertTrue(result.ns["START"])
        result.arm.activate.assert_called_once_with()
        result.hand.activate.assert_called_once_with()
        result.switcher.Enter_Debug_Mode.assert_called_once_with()
        np.testing.assert_array_equal(result.ns["r1_vision_left_reference"], result.sample.left_wrist_pose)
        np.testing.assert_array_equal(result.ns["r1_vision_right_reference"], result.sample.right_wrist_pose)
        self.assertEqual(result.ns["r1_vision_left_reference"][0, 3], 0.42)

    def test_cached_duplicate_frames_do_not_satisfy_five_sample_requirement(self):
        result = self.run_startup(
            [{"key": "r"}]
            + [{}, {"sample": "duplicate"}, {}, {"sample": "duplicate"}, {}, {"sample": "duplicate"}]
            + [{"key": "q", "sample": "duplicate"}],
        )
        self.assertFalse(any(tick["r1_startup_tracking_ready"] for tick in result.ticks))
        result.hand.activate.assert_not_called()

    def test_five_fast_frames_do_not_satisfy_stability_duration(self):
        result = self.run_startup([{"key": "r"}] + [{"dt": 0.03}] * 6 + [{"key": "q", "dt": 0.01}])
        self.assertFalse(any(tick["r1_startup_tracking_ready"] for tick in result.ticks))
        result.hand.activate.assert_not_called()

    def test_tracking_loss_during_recovery_requires_new_stability_then_auto_starts(self):
        result = self.run_startup(
            [{"key": "r"}] + [{}] * 3
            + [{"sample": "missing", "key": "r"}] + [{}] * 6,
        )
        self.assertTrue(result.ns["START"])
        self.assertGreaterEqual(len(result.ticks), 10)
        result.hand.activate.assert_called_once_with()
        result.arm.activate.assert_called_once_with()

    def test_old_tracking_is_not_accepted_as_recovered(self):
        result = self.run_startup([{"key": "r"}] + [{"sample": "old", "key": "r"}] * 8)
        result.hand.activate.assert_not_called()
        self.assertFalse(result.ns["r1_startup_tracking_ready"])

    def test_sample_from_ik_completion_is_not_counted_as_new(self):
        result = self.run_startup([{"key": "r"}] + [{"sample": "duplicate", "dt": 0.04}] * 6)
        result.hand.activate.assert_not_called()
        self.assertEqual(result.ns["r1_startup_fresh_samples"], 0)

    def test_stale_robot_feedback_remains_fatal_while_waiting_for_hands(self):
        with self.assertRaisesRegex(RuntimeError, "robot feedback stale"):
            self.run_startup([{"key": "r"}, {"sample": "missing"}], feedback_error=True)


if __name__ == "__main__":
    unittest.main()
