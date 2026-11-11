"""Operator hand locks ([ left, ] right): the locked arm and its hand hold; nothing else does."""
import ast
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from teleop.robot_control.linker_o6_control_loop import LinkerO6ControlLoop
from teleop.robot_control.r1_hand_lock import HAND_LOCK_KEYS, R1HandLocks
from teleop.robot_control.r1_head_waist import R1HeadWaistFollower
import test_r1_head_waist_integration as integration

MAIN_PATH = ROOT / "teleop" / "teleop_hand_and_arm.py"


class KeyTest(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "on_press"]
        self.logger = Mock()
        self.ns = {
            "logger_mp": self.logger, "HAND_LOCK_KEYS": HAND_LOCK_KEYS, "HAND_LOCKS": R1HandLocks(),
            "WAIST_TURN_LOCK": threading.Lock(), "WAIST_TURN_REQUEST": None, "R1_PAUSE": None,
            "READY": True, "START": True, "STOP": False, "RECORD_TOGGLE": False,
            "RECORD_OUTCOME": "unspecified", "RECORD_RUNNING": False, "RECORD_ENABLED": False,
            "DRY_RUN_MODE": False, "ARM_REQUEST_GENERATION": 0, "R1_A7_DEFERRED_REAL_MODE": False,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(MAIN_PATH), "exec"), self.ns)

    def test_brackets_toggle_each_side_and_full_width_brackets_work_too(self):
        locks = self.ns["HAND_LOCKS"]
        self.ns["on_press"]("]")
        self.assertEqual(locks.snapshot(), (False, True))
        self.ns["on_press"]("\u3010")
        self.assertEqual(locks.snapshot(), (True, True))
        self.ns["on_press"]("\u3011")
        self.ns["on_press"]("[")
        self.assertEqual(locks.snapshot(), (False, False))
        self.assertFalse(self.ns["STOP"])
        self.assertIsNone(self.ns["WAIST_TURN_REQUEST"])
        self.logger.warning.assert_not_called()

    def test_existing_keys_are_untouched(self):
        self.ns["on_press"]("right")
        self.assertEqual(self.ns["WAIST_TURN_REQUEST"], "right")
        self.assertEqual(self.ns["HAND_LOCKS"].snapshot(), (False, False))


class LoopTest(unittest.TestCase):
    """The production control-loop slice with a locked side."""

    def setUp(self):
        self.case = integration.R1HeadWaistIntegrationTest("test_tracking_resumption_resets_follower_from_actual_waist")
        self.case.setUp()
        self.execute = integration.execute
        # The shared harness stubs the anchoring to "target = wrist"; the unlock re-anchors
        # the references, so these cases run the production anchoring instead.
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "anchored_wrist_target")
        helpers = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(MAIN_PATH), "exec"), helpers)
        self.anchored_wrist_target = helpers["anchored_wrist_target"]

    def context(self, left_time=0.99, right_time=0.99):
        ns, sample = self.case.independent_context(left_time, right_time)
        ns["anchored_wrist_target"] = self.anchored_wrist_target
        return ns, sample

    def run_loop(self, ns, frames=1):
        for _ in range(frames):
            self.execute(self.case.control_nodes, ns, loop=True)

    def lock_logs(self, ns):
        return [call.args for call in ns["logger_mp"].info.call_args_list if "[R1 HAND LOCK]" in str(call.args[0])]

    def test_locked_right_arm_holds_while_the_left_follows_and_nothing_counts_as_a_hold(self):
        ns, sample = self.context()
        ns["HAND_LOCKS"].toggle("right")
        sample.left_wrist_pose = integration.pose(0.8, [0.45, 0.2, 0.8])
        sample.right_wrist_pose = integration.pose(-1.0, [0.6, -0.4, 0.9])
        self.run_loop(ns, 2)
        np.testing.assert_allclose(ns["right_wrist_target"], self.case.right, atol=1e-12)
        self.assertEqual(ns["left_wrist_target"][0, 3], sample.left_wrist_pose[0, 3])
        self.assertFalse(ns["tracking_hold_active"])
        self.assertEqual(ns["tracking_hold_events"], 0)
        self.assertEqual(ns["capture_mode"], "following")
        ns["arm_ctrl"].hold_waist.assert_not_called()
        self.assertEqual(ns["waist_follower"].update.call_count, 2)
        self.assertIn(("[R1 HAND LOCK] %s: %s", "right", "locked; this arm and its hand hold their pose"),
                      self.lock_logs(ns))

    def test_a_locked_hand_out_of_view_is_not_a_tracking_hold(self):
        ns, _ = self.context(0.99, 0.0)
        ns["HAND_LOCKS"].toggle("right")
        self.run_loop(ns)
        self.assertFalse(ns["tracking_hold_active"])
        self.assertEqual(ns["capture_mode"], "following")
        ns["arm_ctrl"].hold_waist.assert_not_called()
        ns["arm_ik"].solve_ik.assert_called_once()
        np.testing.assert_allclose(ns["right_wrist_target"], self.case.right, atol=1e-12)
        # The same hand lost without a lock is still a tracking hold.
        unlocked, _ = self.context(0.99, 0.0)
        self.run_loop(unlocked)
        self.assertTrue(unlocked["tracking_hold_active"])
        unlocked["arm_ctrl"].hold_waist.assert_called()

    def test_unlock_carries_on_from_the_held_pose_without_jumping(self):
        ns, sample = self.context()
        ns["HAND_LOCKS"].toggle("right")
        drifted = integration.pose(-1.0, [0.6, -0.4, 0.9])
        sample.right_wrist_pose = drifted
        self.run_loop(ns, 2)
        ns["HAND_LOCKS"].toggle("right")
        self.run_loop(ns, 2)
        np.testing.assert_allclose(ns["right_wrist_target"], self.case.right, atol=1e-9)
        moved = drifted.copy()
        moved[0, 3] += 0.02
        sample.right_wrist_pose = moved
        self.run_loop(ns)
        np.testing.assert_allclose(ns["right_wrist_target"][:3, 3], self.case.right[:3, 3] + [0.02, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(ns["right_wrist_target"][:3, :3], self.case.right[:3, :3], atol=1e-9)
        self.assertEqual(ns["hand_lock_reanchor"], [False, False])
        self.assertIn(("[R1 HAND LOCK] %s arm re-anchored at its held pose.", "right"), self.lock_logs(ns))

    def test_arrow_turn_runs_while_one_arm_is_locked_but_not_while_a_hand_is_lost(self):
        ns, _ = self.context()
        ns["args"].waist_follow_compensation = "torso"
        ns["waist_follower"] = R1HeadWaistFollower(0.43, 0.98)
        ns["take_waist_turn_request"] = Mock(return_value="right")
        ns["HAND_LOCKS"].toggle("right")
        self.run_loop(ns)
        self.assertTrue(ns["waist_follower"].turning)
        lost, _ = self.context(0.99, 0.0)
        lost["args"].waist_follow_compensation = "torso"
        lost["waist_follower"] = R1HeadWaistFollower(0.43, 0.98)
        lost["take_waist_turn_request"] = Mock(return_value="right")
        self.run_loop(lost)
        self.assertFalse(lost["waist_follower"].turning)


class Retargeter:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)
        self.resets = 0

    def retarget(self, points):
        return self.value.copy()

    def reset(self):
        self.resets += 1


class Controller:
    def __init__(self):
        self.action = [np.zeros(6), np.array([5.0])]
        self.calls = []

    def get_action(self):
        return [value.copy() for value in self.action]

    def get_state(self):
        return self.get_action()

    def get_recording_snapshot(self):
        return {"requested": {"monotonic_ns": 1}}

    def update(self, left, right, tracking_fresh=(True, True)):
        self.action = [np.asarray(left, dtype=float).copy(), np.asarray(right, dtype=float).copy()]
        self.calls.append((self.action[0].copy(), self.action[1].copy(), tuple(tracking_fresh)))


class HandLoopTest(unittest.TestCase):
    """Left O6 and right Dex1-1 in the hand loop, with a pickup tolerance on the gripper."""

    def setUp(self):
        self.locks = [False, False]
        self.visible = [True, True]
        self.log = []
        self.controller = Controller()
        self.left, self.right = Retargeter(np.full(6, 0.2)), Retargeter([5.0])
        self.sample = None
        self.loop = LinkerO6ControlLoop(
            lambda: self.sample, SimpleNamespace(left=self.left, right=self.right), self.controller,
            50.0, 0.25, lambda: False, lambda states, actions: None,
            is_locked=lambda index: self.locks[index], pickup_tolerance=(None, 0.1), log=self.log.append,
        )

    def step(self):
        now = time.monotonic()
        self.sample = SimpleNamespace(
            motion_data_ready=True,
            left_hand_timestamp=now if self.visible[0] else 0.0,
            right_hand_timestamp=now if self.visible[1] else 0.0,
            left_hand_pos=np.zeros((25, 3)), right_hand_pos=np.zeros((25, 3)),
        )
        self.loop._step()
        return self.controller.calls[-1]

    def test_locked_gripper_holds_and_waits_for_the_pinch_after_unlock(self):
        self.assertEqual(self.step()[1][0], 5.0)
        self.locks[1] = True
        self.right.value = np.array([5.4])
        left, right, fresh = self.step()
        self.assertEqual((right[0], fresh), (5.0, (True, True)))
        self.visible[1] = False
        left, right, fresh = self.step()
        self.assertEqual((right[0], fresh), (5.0, (True, True)))
        self.assertTrue(self.loop.get_recording_sample()["tracking_fresh"]["right"])
        self.visible[1] = True
        self.locks[1] = False
        self.assertEqual(self.step()[1][0], 5.0)
        self.assertTrue(any("keeps its grip" in line for line in self.log))
        self.right.value = np.array([5.3])
        self.assertEqual(self.step()[1][0], 5.0)
        self.right.value = np.array([5.05])
        self.assertEqual(self.step()[1][0], 5.05)
        self.assertTrue(any("follows your pinch again" in line for line in self.log))
        self.right.value = np.array([5.3])
        self.assertEqual(self.step()[1][0], 5.3)

    def test_a_pinch_that_crosses_the_held_grip_also_picks_it_up(self):
        self.step()
        self.locks[1] = True
        self.step()
        self.locks[1] = False
        self.right.value = np.array([5.4])
        self.assertEqual(self.step()[1][0], 5.0)
        self.right.value = np.array([4.7])
        self.assertEqual(self.step()[1][0], 4.7)

    def test_the_o6_side_holds_while_locked_and_follows_at_once_after_unlock(self):
        np.testing.assert_allclose(self.step()[0], 0.2)
        self.locks[0] = True
        self.left.value = np.full(6, 0.8)
        np.testing.assert_allclose(self.step()[0], 0.2)
        self.assertEqual(self.left.resets, 1)
        self.locks[0] = False
        np.testing.assert_allclose(self.step()[0], 0.8)

    def test_an_unlocked_hand_out_of_view_is_still_reported_stale(self):
        self.step()
        self.visible[0] = False
        self.assertEqual(self.step()[2], (False, True))


if __name__ == "__main__":
    unittest.main()
