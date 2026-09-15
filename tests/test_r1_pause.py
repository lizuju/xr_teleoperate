import ast
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from teleop.robot_control.r1_pause import R1PauseState
from teleop.robot_control.r1_hand_tracking import R1WristHold
from teleop.robot_control.r1_head_waist import R1HeadWaistFollower
import test_r1_head_waist_integration as integration_fixture
from test_r1_head_waist_integration import MAIN_PATH, execute, pose


def tracked_sample(now, left=None, right=None):
    return SimpleNamespace(
        motion_data_ready=True,
        left_hand_timestamp=now if left is None else left,
        right_hand_timestamp=now if right is None else right,
    )


class R1PauseStateTest(unittest.TestCase):
    def setUp(self):
        self.state = R1PauseState()

    def stable_frames(self, start=1.0):
        result = None
        for index in range(1, 6):
            now = start + 0.1 * index
            result = self.state.poll_resume(tracked_sample(now), 0.25, now)
        return result

    def test_fresh_hands_cannot_resume_without_an_explicit_request(self):
        self.state.pause()
        self.assertIsNone(self.stable_frames())
        self.assertTrue(self.state.paused)

    def test_resume_needs_new_stable_samples_from_both_hands(self):
        self.state.pause()
        self.state.request_resume(1.0)
        self.assertIsNone(self.state.poll_resume(tracked_sample(0.99), 0.25, 1.0))
        token = self.stable_frames()
        self.assertEqual(token, self.state.generation)
        self.assertTrue(self.state.paused)
        self.assertTrue(self.state.complete_resume(token))
        self.assertFalse(self.state.paused)

    def test_duplicate_or_one_sided_samples_do_not_satisfy_stability(self):
        self.state.pause()
        self.state.request_resume(1.0)
        for index in range(10):
            now = 1.01 + 0.02 * index
            self.assertIsNone(self.state.poll_resume(tracked_sample(now, right=1.01), 0.25, now))
        self.assertEqual(self.state.samples, 1)

    def test_loss_during_recovery_restarts_the_stability_window(self):
        self.state.pause()
        self.state.request_resume(1.0)
        for now in (1.1, 1.2, 1.3):
            self.state.poll_resume(tracked_sample(now), 0.25, now)
        self.assertIsNone(self.state.poll_resume(tracked_sample(1.4, right=0.0), 0.25, 1.4))
        self.assertIsNone(self.state.stable_since)
        self.assertEqual(self.state.samples, 0)
        self.assertIsNotNone(self.stable_frames(1.4))

    def test_a_new_pause_invalidates_an_already_ready_resume(self):
        self.state.pause()
        self.state.request_resume(1.0)
        token = self.stable_frames()
        self.state.pause()
        self.assertFalse(self.state.complete_resume(token))
        self.assertTrue(self.state.paused)


class R1PauseControlIntegrationTest(unittest.TestCase):
    def setUp(self):
        fixture = integration_fixture.R1HeadWaistIntegrationTest()
        fixture.setUp()
        self.nodes = fixture.control_nodes
        self.ns, self.sample = fixture.independent_context()
        self.ns["args"].waist_follow = False
        self.state = R1PauseState()
        self.ns["R1_PAUSE"] = self.state
        self.ns["R1WristHold"] = R1WristHold
        self.ns["R1HeadWaistFollower"] = R1HeadWaistFollower
        self.now = 1.0
        self.ns["time"].monotonic = lambda: self.now
        self.ns["time"].time = lambda: self.now
        self.published = {"arm_q": [0.12] * 14, "arm_tau": [0.03] * 14, "head_q": [0.15, -0.25]}
        self.requested = {"arm_q": [0.9] * 14, "arm_tau": [0.4] * 14, "head_q": [0.4, -0.9]}
        self.ns["arm_ctrl"].get_recording_snapshot.side_effect = lambda: {
            "published": deepcopy(self.published), "requested": deepcopy(self.requested),
        }

        def set_targets(q, tau, head, **kwargs):
            self.requested.update(arm_q=q.tolist(), arm_tau=tau.tolist(), head_q=head.tolist())

        self.ns["arm_ctrl"].ctrl_dual_arm_and_head.side_effect = set_targets
        self.robot_poses = (pose(0.4, [0.4, 0.2, 0.8]), pose(-0.3, [0.4, -0.2, 0.8]))
        self.ns["arm_ik"].forward_wrist_poses.return_value = self.robot_poses
        self.ns["arm_ik"].solve_ik.side_effect = lambda *args, **kwargs: (
            np.array(self.published["arm_q"]), np.array(self.published["arm_tau"]),
        )
        self.ns["arm_diagnostic_file"] = None
        names = {"hold_r1_published_targets", "head_yaw_rotation", "relative_head_pitch_yaw",
                 "wrist_in_reference_head_yaw_frame", "anchored_wrist_target"}
        helpers = [node for node in ast.parse(MAIN_PATH.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
        execute(helpers, self.ns)

    def tick(self, now, tracked=True):
        self.now = now
        self.sample.left_hand_timestamp = now if tracked else 0.0
        self.sample.right_hand_timestamp = now if tracked else 0.0
        self.sample.motion_data_timestamp = now
        execute(self.nodes, self.ns, loop=True)

    def test_pause_freezes_last_published_step_instead_of_far_requested_target(self):
        self.state.pause()
        self.tick(1.0)
        self.assertEqual(self.requested, self.published)
        self.ns["arm_ik"].solve_ik.assert_not_called()
        self.ns["arm_ctrl"].ctrl_dual_arm_and_head.assert_called_once()
        self.assertTrue(self.ns["completed"])
        self.assertEqual(self.ns["capture_mode"], "paused")
        np.testing.assert_allclose(self.ns["sol_q"], self.published["arm_q"])
        np.testing.assert_allclose(self.ns["head_q_target"], self.published["head_q"])

    def test_hand_motion_and_tracking_loss_while_paused_keep_hold_and_reach_recording(self):
        self.state.pause()
        self.tick(1.0)
        self.sample.left_wrist_pose = pose(2.1, [2.0, 3.0, 1.0])
        self.sample.head_pose = pose(-1.4, [0.5, 0.8, 1.8])
        for index in range(1, 10):
            self.tick(1.0 + index * 0.1, tracked=index % 2 == 0)
            self.assertTrue(self.ns["completed"])
            self.assertEqual(self.ns["capture_mode"], "paused")
            self.assertEqual(self.requested, self.published)
        self.ns["arm_ctrl"].ctrl_dual_arm_and_head.assert_called_once()
        self.ns["arm_ik"].solve_ik.assert_not_called()
        self.assertGreaterEqual(self.ns["arm_ctrl"].hold_targets.call_count, 10)

    def test_stable_resume_realigns_without_recenter_or_jump_to_old_reference(self):
        self.state.pause()
        self.tick(1.0)
        self.sample.left_wrist_pose = pose(1.3, [1.1, 0.8, 0.9])
        self.sample.right_wrist_pose = pose(-1.0, [-0.6, 0.7, 0.8])
        self.sample.head_pose = pose(1.2, [0.3, -0.2, 1.7])
        self.state.request_resume(1.0)
        for index in range(1, 6):
            self.tick(1.0 + index * 0.1)
        self.assertFalse(self.state.paused)
        self.ns["arm_ik"].solve_ik.assert_not_called()
        np.testing.assert_allclose(
            self.ns["arm_ik"].reset_smoothing.call_args.kwargs["reference_q"], self.published["arm_q"],
        )
        self.tick(1.6)
        for target, expected in zip(self.ns["arm_ik"].solve_ik.call_args.args[:2], self.robot_poses):
            np.testing.assert_allclose(target, expected, atol=1e-12)
        np.testing.assert_allclose(self.ns["head_q_target"], self.published["head_q"], atol=1e-12)
        self.assertEqual(self.ns["capture_mode"], "following")
        self.ns["arm_ctrl"].activate.assert_not_called()
        self.ns["arm_ctrl"].ctrl_head_and_waist_go_home.assert_not_called()

    def test_pause_during_ik_discards_unpublished_solution(self):
        def delayed_ik(*args, **kwargs):
            self.state.pause()
            return np.ones(14), np.ones(14)

        self.ns["arm_ik"].solve_ik.side_effect = delayed_ik
        self.tick(1.0)
        self.assertEqual(self.requested, self.published)
        self.assertEqual(self.ns["capture_mode"], "paused")
        self.assertTrue(self.ns["completed"])

    def test_waist_follow_resume_preserves_fixed_ik_frame_and_held_head(self):
        self.ns["args"].waist_follow = True
        self.state.pause()
        self.tick(1.0)
        self.sample.head_pose = pose(1.3, [0.3, -0.2, 1.7])
        self.sample.left_wrist_pose = pose(1.5, [1.0, 0.7, 0.9])
        self.sample.right_wrist_pose = pose(-1.1, [-0.6, 0.7, 0.8])
        self.state.request_resume(1.0)
        for index in range(1, 6):
            self.tick(1.0 + index * 0.1)
        self.assertFalse(self.state.paused)
        self.assertEqual(self.ns["r1_waist_yaw_reference"], 0.17)
        self.tick(1.6)
        for target, expected in zip(self.ns["arm_ik"].solve_ik.call_args.args[:2], self.robot_poses):
            np.testing.assert_allclose(target, expected, atol=1e-12)
        np.testing.assert_allclose(self.ns["head_q_target"], self.published["head_q"], atol=1e-12)
        self.assertAlmostEqual(self.ns["waist_yaw_target"], 0.43)

    def test_pause_before_first_ik_still_emits_complete_alignment_diagnostics(self):
        self.ns["arm_diagnostic_file"] = object()
        self.state.pause()
        self.tick(1.0)
        payload = self.ns["write_json_line"].call_args.args[1]
        for name, expected in zip(("left_ik_target", "right_ik_target"), self.robot_poses):
            np.testing.assert_allclose(payload[name], expected)
        self.assertEqual(payload["q_ik_command"], self.published["arm_q"])
        self.assertTrue(self.ns["completed"])

    def test_feedback_fault_is_still_fatal_during_pause(self):
        self.state.pause()
        self.ns["arm_ctrl"].hold_targets.side_effect = RuntimeError("robot feedback stale")
        with self.assertRaisesRegex(RuntimeError, "feedback stale"):
            self.tick(1.0)
        self.ns["arm_ik"].solve_ik.assert_not_called()

    def test_both_hands_lost_reaches_recording_with_held_targets(self):
        self.tick(1.0, tracked=False)
        self.assertEqual(self.ns["capture_mode"], "tracking_hold")
        self.assertTrue(self.ns["completed"])
        self.ns["arm_ik"].solve_ik.assert_not_called()
        np.testing.assert_allclose(self.ns["sol_q"], self.requested["arm_q"])


class R1PauseKeysTest(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(MAIN_PATH.read_text())
        on_press = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "on_press")
        self.state = R1PauseState()
        self.ns = {
            "R1_PAUSE": self.state, "STOP": False, "START": True, "READY": False,
            "RECORD_RUNNING": True, "RECORD_TOGGLE": False, "RECORD_OUTCOME": "unspecified",
            "DRY_RUN_MODE": False, "ARM_REQUEST_GENERATION": 1,
            "R1_A7_DEFERRED_REAL_MODE": True, "logger_mp": Mock(),
        }
        execute([on_press], self.ns)

    def test_pause_and_resume_request_do_not_depend_on_recording_readiness(self):
        self.ns["on_press"]("p")
        self.assertTrue(self.state.paused)
        self.ns["on_press"]("r")
        self.assertIsNotNone(self.state.resume_after)
        self.assertTrue(self.state.paused)
        self.assertEqual(self.ns["ARM_REQUEST_GENERATION"], 1)
        self.assertFalse(self.ns["RECORD_TOGGLE"])

    def test_q_remains_exit_while_paused(self):
        self.state.pause()
        self.ns["on_press"]("q")
        self.assertTrue(self.ns["STOP"])
        self.assertFalse(self.ns["START"])

    def test_outcome_keys_end_only_an_active_episode(self):
        for key, outcome in (("y", "success"), ("n", "failure"), ("x", "discarded")):
            self.ns["RECORD_RUNNING"] = True
            self.ns["on_press"](key)
            self.assertEqual(self.ns["RECORD_OUTCOME"], outcome)
            self.assertTrue(self.ns["RECORD_TOGGLE"])
            self.ns.update(RECORD_RUNNING=False, RECORD_TOGGLE=False, RECORD_OUTCOME="unspecified")
            self.ns["on_press"](key)
            self.assertEqual(self.ns["RECORD_OUTCOME"], "unspecified")
            self.assertFalse(self.ns["RECORD_TOGGLE"])


if __name__ == "__main__":
    unittest.main()
