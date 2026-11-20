"""Operator head lock: h holds the robot head; unlocking eases it back to the operator's head."""
import ast
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
from teleop.robot_control.r1_hand_lock import HAND_LOCK_KEYS, R1HandLocks
from teleop.robot_control.r1_head_lock import HEAD_LOCK_KEYS, R1HeadLock
import test_r1_head_waist_integration as integration

MAIN_PATH = ROOT / "teleop" / "teleop_hand_and_arm.py"


class HeadLockUnitTest(unittest.TestCase):
    def test_follows_while_unlocked(self):
        lock = R1HeadLock()
        np.testing.assert_allclose(lock.apply([0.1, -0.2], 1.0), [0.1, -0.2])
        np.testing.assert_allclose(lock.apply([0.3, 0.0], 1.1), [0.3, 0.0])
        self.assertEqual(lock.pop_events(), [])

    def test_lock_holds_the_last_sent_target_and_logs_it_once(self):
        lock = R1HeadLock()
        lock.apply([0.1, -0.2], 1.0)
        self.assertTrue(lock.toggle())
        for t, follow in ((1.1, [0.6, 0.4]), (1.2, [-0.5, 0.9])):
            np.testing.assert_allclose(lock.apply(follow, t), [0.1, -0.2])
        events = lock.pop_events()
        self.assertEqual(len(events), 1)
        self.assertIn("[R1 HEAD LOCK] locked; the head holds pitch 5.7 deg, yaw -11.5 deg", events[0])

    def test_lock_before_any_target_holds_the_first_follow_target(self):
        lock = R1HeadLock()
        lock.toggle()
        np.testing.assert_allclose(lock.apply([0.2, 0.1], 1.0), [0.2, 0.1])
        np.testing.assert_allclose(lock.apply([0.7, 0.7], 1.1), [0.2, 0.1])

    def test_unlock_moves_at_a_limited_speed_then_follows(self):
        lock = R1HeadLock(unlock_speed=1.0)
        lock.apply([0.0, 0.0], 1.0)
        lock.toggle()
        lock.apply([0.5, -0.3], 1.1)
        self.assertFalse(lock.toggle())
        lock.pop_events()
        np.testing.assert_allclose(lock.apply([0.5, -0.3], 1.2), [0.1, -0.1])
        np.testing.assert_allclose(lock.apply([0.5, -0.3], 1.3), [0.2, -0.2])
        np.testing.assert_allclose(lock.apply([0.5, -0.3], 1.4), [0.3, -0.3])
        np.testing.assert_allclose(lock.apply([0.5, -0.3], 1.5), [0.4, -0.3])
        np.testing.assert_allclose(lock.apply([0.5, -0.3], 1.6), [0.5, -0.3])
        self.assertEqual(lock.pop_events(), ["[R1 HEAD LOCK] unlocked; the head follows you again."])
        np.testing.assert_allclose(lock.apply([0.9, 0.2], 1.7), [0.9, 0.2])

    def test_a_target_held_during_a_pause_is_what_the_lock_keeps(self):
        lock = R1HeadLock()
        lock.apply([0.1, 0.1], 1.0)
        lock.observe([0.25, -0.05], 1.5)
        lock.toggle()
        np.testing.assert_allclose(lock.apply([0.9, 0.9], 1.6), [0.25, -0.05])

    def test_unlock_speed_must_be_positive(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                R1HeadLock(unlock_speed=bad)


class HeadLockKeyTest(unittest.TestCase):
    def namespace(self, with_head_lock=True):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "on_press"]
        ns = {
            "logger_mp": Mock(), "HAND_LOCK_KEYS": HAND_LOCK_KEYS, "HAND_LOCKS": R1HandLocks(),
            "WAIST_TURN_LOCK": threading.Lock(), "WAIST_TURN_REQUEST": None, "R1_PAUSE": None,
            "READY": True, "START": True, "STOP": False, "RECORD_TOGGLE": False,
            "RECORD_OUTCOME": "unspecified", "RECORD_RUNNING": False, "RECORD_ENABLED": False,
            "DRY_RUN_MODE": False, "ARM_REQUEST_GENERATION": 0, "R1_A7_DEFERRED_REAL_MODE": False,
        }
        if with_head_lock:
            ns.update(HEAD_LOCK_KEYS=HEAD_LOCK_KEYS, HEAD_LOCK=R1HeadLock())
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(MAIN_PATH), "exec"), ns)
        return ns

    def test_h_toggles_the_head_lock_and_nothing_else(self):
        ns = self.namespace()
        ns["on_press"]("h")
        self.assertTrue(ns["HEAD_LOCK"].locked())
        ns["on_press"]("H")
        self.assertFalse(ns["HEAD_LOCK"].locked())
        self.assertEqual(ns["HAND_LOCKS"].snapshot(), (False, False))
        self.assertFalse(ns["STOP"])
        self.assertIsNone(ns["WAIST_TURN_REQUEST"])
        messages = [call.args[0] for call in ns["logger_mp"].info.call_args_list]
        self.assertEqual(messages, ["[R1 HEAD LOCK] head: lock requested.", "[R1 HEAD LOCK] head: unlock requested."])
        ns["logger_mp"].warning.assert_not_called()

    def test_unknown_keys_still_warn_without_a_head_lock_in_scope(self):
        ns = self.namespace(with_head_lock=False)
        ns["on_press"]("z")
        ns["logger_mp"].warning.assert_called_once()

    def test_main_creates_the_lock_and_records_it(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn("HEAD_LOCK = R1HeadLock()", source)
        self.assertIn("head_lock.apply(head_q_target, time.monotonic())", source)
        self.assertIn("head_lock.observe(head_q_target, time.monotonic())", source)
        self.assertIn("head_locked=", source)
        self.assertIn("Press [h]", source)


class HeadLockLoopTest(unittest.TestCase):
    """The production control-loop slice with the head lock in scope."""

    def setUp(self):
        self.case = integration.R1HeadWaistIntegrationTest("test_tracking_resumption_resets_follower_from_actual_waist")
        self.case.setUp()

    def run_tick(self, ns, now, t):
        now[0] = t
        ns["loop_monotonic"] = t
        integration.execute(self.case.control_nodes, ns, loop=True)
        return np.array(ns["arm_ctrl"].ctrl_dual_arm_and_head.call_args.args[2], dtype=float)

    def context(self, head_lock):
        ns, sample = self.case.independent_context()
        now = [1.0]
        ns["time"].monotonic = lambda: now[0]
        if head_lock is not None:
            ns["HEAD_LOCK"] = head_lock
        return ns, sample, now

    def test_locked_head_holds_while_the_operator_looks_down_then_eases_back(self):
        # The harness follower is a mock: its update() result is the head target that
        # following would send. Changing it stands for the operator looking down.
        lock = R1HeadLock(unlock_speed=1.0)
        ns, _, now = self.context(lock)
        start = self.run_tick(ns, now, 1.0)
        np.testing.assert_allclose(start, [0.2, 0.27], atol=1e-12)
        lock.toggle()
        follow = np.array([0.6, -0.1])
        ns["waist_follower"].update.return_value = (follow.copy(), 1.2)
        for t in (1.025, 1.05, 1.075):
            np.testing.assert_allclose(self.run_tick(ns, now, t), start, atol=1e-12)
        self.assertTrue(any("locked; the head holds" in str(call.args[0])
                            for call in ns["logger_mp"].info.call_args_list))
        lock.toggle()
        eased = self.run_tick(ns, now, 1.1)
        np.testing.assert_allclose(eased, start + [0.025, -0.025], atol=1e-9)
        t = 1.1
        for _ in range(20):
            t += 0.025
            last = self.run_tick(ns, now, t)
        np.testing.assert_allclose(last, follow, atol=1e-9)
        self.assertTrue(any("follows you again" in str(call.args[0])
                            for call in ns["logger_mp"].info.call_args_list))

    def test_no_head_lock_in_scope_leaves_the_loop_as_before(self):
        ns, _, now = self.context(None)
        np.testing.assert_allclose(self.run_tick(ns, now, 1.0), [0.2, 0.27], atol=1e-12)


class HeadLockRecordingTest(unittest.TestCase):
    """A recorded head_lock flag is informational: the episode stays valid and exportable."""

    def test_recorded_head_lock_keeps_frames_valid_and_unrejected(self):
        from check_teleop_episode import training_frame_rejections, validate_episode
        import test_training_export
        helper = test_training_export.TrainingExportTests("test_tracking_sources_share_the_150_ms_hand_input_limit")
        helper.setUp()
        try:
            path, manifest, rows = helper.episode(count=3)
            helper.add_action_alignment(rows)
            before = [training_frame_rejections(row, manifest["info"]) for row in rows]
            for row in rows:
                row["sample"]["head_lock"] = True
            helper.write(path, manifest, rows)
            report = validate_episode(path)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual([training_frame_rejections(row, manifest["info"]) for row in rows], before)
        finally:
            helper.doCleanups()

if __name__ == "__main__":
    unittest.main()
