import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from teleop.robot_control.linker_o6_retargeting import (
    DualLinkerO6Retargeter,
    LinkerO6HandRetargeter,
    is_tracking_fresh,
)
from tools.evaluate_linker_o6_retargeting import forward_hand, load_recording, synthetic_hand_points


URDF_ROOT = Path(os.environ.get(
    "LINKER_O6_URDF_ROOT", "/home/hnh/unitree_r1_dev/linkerhand-urdf/O6"
))
if not URDF_ROOT.exists():
    URDF_ROOT = REPO_ROOT.parent / "unitree-r1-a7-linker-o6/third_party/linkerhand/O6"
HAS_SOLVER = all(importlib.util.find_spec(name) is not None for name in (
    "dex_retargeting", "pinocchio", "nlopt", "torch"
))
METHODS = ("vector", "position", "dexpilot")


@unittest.skipUnless(HAS_SOLVER and URDF_ROOT.exists(), "Real dex-retargeting dependencies and O6 URDF are required")
class LinkerO6RetargetingTest(unittest.TestCase):
    def make_hand(self, side="left", method="vector"):
        return LinkerO6HandRetargeter(
            URDF_ROOT / side / f"linkerhand_o6_{side}.urdf", side, method=method
        )

    def test_three_real_optimizers_use_six_active_axes_and_mimic(self):
        for method in METHODS:
            for side in ("left", "right"):
                with self.subTest(method=method, side=side):
                    hand = self.make_hand(side, method)
                    optimizer = hand.retargeting.optimizer
                    self.assertEqual(optimizer.retargeting_type.lower(), method)
                    self.assertEqual(optimizer.opt_dof, 6)
                    self.assertEqual(len(hand.retargeting.joint_names), 11)
                    self.assertEqual(len(optimizer.adaptor.idx_pin2mimic), 5)
                    self.assertEqual(len(optimizer.idx_pin2fixed), 0)
                    self.assertEqual(len(hand.tip_link_names), 5)
                    self.assertEqual(hand.retargeting.filter.alpha, 1.0)

    def test_hardware_order_is_explicit_and_not_pinocchio_order(self):
        hand = self.make_hand()
        self.assertEqual(hand.hardware_joint_order, (
            "lh_thumb_cmc_pitch", "lh_thumb_cmc_yaw", "lh_index_mcp_pitch",
            "lh_middle_mcp_pitch", "lh_ring_mcp_pitch", "lh_pinky_mcp_pitch",
        ))
        np.testing.assert_allclose(hand.hardware_upper, [.58, 1.3, 1.6, 1.6, 1.6, 1.6])
        self.assertNotEqual(list(hand.hardware_joint_order), hand.retargeting.joint_names[:6])

    def test_wrapper_to_o6_axes_and_wrist_translation(self):
        hand = self.make_hand()
        points = synthetic_hand_points(hand, np.full(6, .3))
        translated = points + [2.5, -1.2, .7]
        np.testing.assert_allclose(hand.to_robot_points(points), hand.to_robot_points(translated), atol=1e-12)
        expected_rotation = np.array([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
        expected = (points - points[0]) @ expected_rotation.T
        np.testing.assert_allclose(hand.to_robot_points(points), expected, atol=1e-12)

    def test_retargets_nontrivial_fk_targets_for_all_methods_and_hands(self):
        normalized = np.array([.30, .40, .22, .34, .45, .55])
        for method in METHODS:
            for side in ("left", "right"):
                with self.subTest(method=method, side=side):
                    hand = self.make_hand(side, method)
                    points = synthetic_hand_points(hand, normalized)
                    desired_tips = hand.to_robot_points(points)[[4, 9, 14, 19, 24]]
                    for _ in range(50):
                        output = hand.retarget(points)
                    actual_tips, qpos = forward_hand(hand, output)
                    self.assertGreater(float(np.linalg.norm(output)), .2)
                    self.assertTrue(np.isfinite(output).all())
                    self.assertTrue(np.all((output >= 0.) & (output <= 1.)))
                    self.assertLess(float(np.max(np.linalg.norm(actual_tips - desired_tips, axis=1))), .005)
                    limits = hand.retargeting.optimizer.robot.joint_limits
                    self.assertTrue(np.all(qpos >= limits[:, 0] - 1e-6))
                    self.assertTrue(np.all(qpos <= limits[:, 1] + 1e-6))

    def test_left_thumb_bound_respects_distal_mimic_limit(self):
        hand = self.make_hand()
        optimizer = hand.retargeting.optimizer
        index = optimizer.target_joint_names.index("lh_thumb_cmc_pitch")
        self.assertAlmostEqual(hand.retargeting.joint_limits[index, 1], 1.08 / 2.29, places=6)
        self.assertAlmostEqual(hand.hardware_upper[0], .58)

    def test_reset_removes_filter_history_and_dexpilot_projection(self):
        for method in METHODS:
            with self.subTest(method=method):
                hand = self.make_hand(method=method)
                fresh = self.make_hand(method=method)
                open_points = synthetic_hand_points(hand, np.zeros(6))
                bent_points = synthetic_hand_points(hand, np.full(6, .5))
                for _ in range(8):
                    hand.retarget(bent_points)
                if method == "dexpilot":
                    hand.retargeting.optimizer.projected[:] = True
                hand.reset()
                if method == "dexpilot":
                    self.assertFalse(hand.retargeting.optimizer.projected.any())
                np.testing.assert_allclose(hand.retarget(open_points), fresh.retarget(open_points), atol=1e-7)

    def test_rejects_invalid_tracking_at_input_boundary(self):
        hand = self.make_hand()
        valid = synthetic_hand_points(hand, np.full(6, .2))
        nan = valid.copy()
        nan[4, 0] = np.nan
        infinity = valid.copy()
        infinity[9, 1] = np.inf
        for invalid in (np.zeros((25, 3)), np.zeros((21, 3)), valid[:, :2], nan, infinity):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    hand.retarget(invalid)

    def test_dexpilot_projection_uses_webxr_thumb_index_tips(self):
        hand = self.make_hand(method="dexpilot")
        points = synthetic_hand_points(hand, np.full(6, .2))
        points[9] = points[4] + [0., .01, 0.]
        hand.retarget(points)
        self.assertTrue(hand.retargeting.optimizer.projected[0])
        points[9] = points[4] + [0., .08, 0.]
        hand.retarget(points)
        self.assertFalse(hand.retargeting.optimizer.projected[0])

    def test_real_optimizer_failure_is_not_returned_as_a_fresh_target(self):
        for method in METHODS:
            with self.subTest(method=method):
                hand = self.make_hand(method=method)
                points = synthetic_hand_points(hand, np.full(6, .3))
                for _ in range(4):
                    hand.retarget(points)
                optimizer = hand.retargeting.optimizer
                make_objective = optimizer.get_objective_function

                def make_stopped_objective(*args):
                    objective = make_objective(*args)

                    def stopped_objective(x, grad):
                        optimizer.opt.force_stop()
                        return objective(x, grad)

                    return stopped_objective

                optimizer.get_objective_function = make_stopped_objective
                with self.assertRaisesRegex(RuntimeError, "optimization failed"):
                    hand.retarget(points)
                self.assertLess(optimizer.opt.last_optimize_result(), 0)
                np.testing.assert_array_equal(hand.retargeting.last_qpos, np.zeros(6))
                self.assertFalse(hand.retargeting.filter.is_init)
                self.assertIsNone(hand.retargeting.filter.y)
                if method == "dexpilot":
                    self.assertFalse(optimizer.projected.any())
                optimizer.get_objective_function = make_objective
                fresh = self.make_hand(method=method)
                np.testing.assert_allclose(hand.retarget(points), fresh.retarget(points), atol=1e-7)

    def test_mirrored_finger_targets_produce_matching_flexion(self):
        left, right = self.make_hand("left"), self.make_hand("right")
        normalized = np.array([0., .2, .2, .3, .4, .5])
        left_points = synthetic_hand_points(left, normalized)
        right_points = synthetic_hand_points(right, normalized)
        left_robot = left.to_robot_points(left_points)
        right_robot = right.to_robot_points(right_points)
        np.testing.assert_allclose(left_robot[[9, 14, 19, 24]] * [1., -1., 1.], right_robot[[9, 14, 19, 24]], atol=2e-5)
        for _ in range(40):
            lq, rq = left.retarget(left_points), right.retarget(right_points)
        np.testing.assert_allclose(lq[2:], rq[2:], atol=.02)

    def test_dual_mode_and_mapping_identify_actual_solver(self):
        for method in METHODS:
            with self.subTest(method=method):
                dual = DualLinkerO6Retargeter(URDF_ROOT, method=method)
                left_points = synthetic_hand_points(dual.left, np.full(6, .2))
                right_points = synthetic_hand_points(dual.right, np.full(6, .3))
                left, right = dual.retarget(left_points, right_points)
                self.assertEqual((left.shape, right.shape), ((6,), (6,)))
                self.assertEqual(dual.method, method)
                self.assertIn(method, dual.mapping_name)
                dual.reset()

    def test_unknown_method_and_side_are_rejected(self):
        with self.assertRaises(ValueError):
            self.make_hand(method="geometric")
        with self.assertRaises(ValueError):
            LinkerO6HandRetargeter(URDF_ROOT / "left/linkerhand_o6_left.urdf", "both")


class TrackingFreshnessTest(unittest.TestCase):
    def test_tracking_age_boundary(self):
        self.assertTrue(is_tracking_fresh(True, 1., .25, now=1.25))
        self.assertFalse(is_tracking_fresh(True, 1., .25, now=1.251))
        self.assertFalse(is_tracking_fresh(True, 1., .25, now=.9))
        self.assertFalse(is_tracking_fresh(False, 1., .25, now=1.1))


class OfflineRecordingTest(unittest.TestCase):
    def load_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            return load_recording(path)

    def test_old_normalized_targets_cannot_be_used_as_new_tracking_inputs(self):
        with self.assertRaisesRegex(ValueError, "Old normalized targets"):
            self.load_rows([{"armed": True, "target_12": [0.] * 12}])

    def test_world_frame_points_are_not_mistaken_for_wrapper_local_points(self):
        row = {
            "left_hand_points": [[0., 0., 0.]] * 25,
            "right_hand_points": [[0., 0., 0.]] * 25,
            "hand_points_frame": "webxr_world_meters",
        }
        with self.assertRaisesRegex(ValueError, "unsupported hand_points_frame"):
            self.load_rows([row])

    def test_valid_paired_frames_are_deduplicated_and_unarmed_rows_skipped(self):
        points = np.arange(75, dtype=float).reshape(25, 3).tolist()
        row = {
            "armed": True, "monotonic_timestamp": 1.,
            "hand_points_frame": "televuer_unitree_hand_wrist_local_meters",
            "left_hand_points": points, "right_hand_points": points,
        }
        frames = self.load_rows([{"armed": False}, row, row])
        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0][0], points)


if __name__ == "__main__":
    unittest.main()
