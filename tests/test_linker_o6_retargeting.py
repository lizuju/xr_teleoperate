import os
from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT))

from teleop.robot_control.linker_o6_retargeting import (  # noqa: E402
    DualLinkerO6Retargeter,
    LinkerO6Calibration,
    LinkerO6HandRetargeter,
    is_tracking_fresh,
)


HOST_URDF_ROOT = Path("/home/hnh/unitree_r1_dev/linkerhand-urdf/O6")
DEFAULT_URDF_ROOT = HOST_URDF_ROOT if HOST_URDF_ROOT.exists() else STAGE_ROOT / "urdf" / "O6"
URDF_ROOT = Path(os.environ.get("LINKER_O6_URDF_ROOT", DEFAULT_URDF_ROOT))
CALIBRATION_PATH = STAGE_ROOT / "teleop" / "robot_control" / "linker_o6_visionpro_calibration.json"


def synthetic_hand(closed=False, thumb_adducted=False):
    points = np.zeros((25, 3), dtype=float)
    finger_bases = {
        5: (-0.03, 0.05, 0.0),
        10: (-0.01, 0.055, 0.0),
        15: (0.01, 0.05, 0.0),
        20: (0.03, 0.045, 0.0),
    }
    open_segments = np.array([[0.0, 0.02, 0.0]] * 4)
    closed_segments = np.array([
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
        [0.0, -0.02, 0.0],
        [0.0, 0.0, -0.02],
    ])
    for base_index, base in finger_bases.items():
        points[base_index] = base
        for offset, segment in enumerate(closed_segments if closed else open_segments, start=1):
            points[base_index + offset] = points[base_index + offset - 1] + segment

    points[1] = (-0.035, 0.015, 0.0)
    if closed:
        thumb_segments = np.array([
            [-0.015, 0.0, 0.0],
            [0.0, 0.0, 0.015],
            [0.015, 0.0, 0.0],
        ])
    elif thumb_adducted:
        thumb_segments = np.array([[0.0, 0.015, 0.0]] * 3)
    else:
        thumb_segments = np.array([[-0.015, 0.0, 0.0]] * 3)
    for offset, segment in enumerate(thumb_segments, start=1):
        points[1 + offset] = points[offset] + segment
    return points


class LinkerO6RetargetingTest(unittest.TestCase):
    def test_vendor_urdf_is_explicitly_reordered_to_hardware_order(self):
        retargeter = LinkerO6HandRetargeter(
            URDF_ROOT / "left" / "linkerhand_o6_left.urdf", "left"
        )

        self.assertEqual(
            retargeter.urdf_joint_order[:2],
            ["lh_thumb_cmc_yaw", "lh_thumb_cmc_pitch"],
        )
        self.assertEqual(
            retargeter.hardware_joint_order[:2],
            ("lh_thumb_cmc_pitch", "lh_thumb_cmc_yaw"),
        )
        self.assertEqual(retargeter.urdf_to_hardware, (1, 0, 2, 3, 4, 5))
        np.testing.assert_allclose(
            retargeter.hardware_upper,
            [0.58, 1.3, 1.6, 1.6, 1.6, 1.6],
        )

        right = LinkerO6HandRetargeter(
            URDF_ROOT / "right" / "linkerhand_o6_right.urdf", "right"
        )
        self.assertEqual(right.urdf_to_hardware, (1, 0, 2, 3, 4, 5))
        self.assertEqual(
            right.urdf_joint_order[:2],
            ["rh_thumb_cmc_yaw", "rh_thumb_cmc_pitch"],
        )
        self.assertAlmostEqual(right.hardware_upper[1], 1.36)
        np.testing.assert_allclose(right.hardware_axes[1], [0.0, 0.0, -1.0])

    def test_synthetic_open_and_closed_hands_produce_bounded_dual_targets(self):
        retargeter = DualLinkerO6Retargeter(URDF_ROOT)
        open_hand = synthetic_hand()
        closed_hand = synthetic_hand(closed=True)

        left_open, right_open = retargeter.retarget(open_hand, open_hand)
        left_closed, right_closed = retargeter.retarget(closed_hand, closed_hand)

        self.assertEqual(left_open.shape, (6,))
        self.assertEqual(right_open.shape, (6,))
        self.assertTrue(np.all((left_open >= 0.0) & (left_open <= 1.0)))
        self.assertTrue(np.all((left_closed >= 0.0) & (left_closed <= 1.0)))
        np.testing.assert_allclose(left_open, right_open)
        np.testing.assert_allclose(left_closed, right_closed)
        self.assertTrue(np.all(left_closed[[0, 2, 3, 4, 5]] > left_open[[0, 2, 3, 4, 5]]))

    def test_thumb_yaw_candidate_increases_toward_palm_direction(self):
        retargeter = LinkerO6HandRetargeter(
            URDF_ROOT / "right" / "linkerhand_o6_right.urdf", "right"
        )
        spread = retargeter.retarget(synthetic_hand())
        adducted = retargeter.retarget(synthetic_hand(thumb_adducted=True))

        self.assertGreater(adducted[1], spread[1])

    def test_tracking_freshness_uses_monotonic_age_and_ready_flag(self):
        self.assertTrue(is_tracking_fresh(True, 10.0, 0.25, now=10.2))
        self.assertFalse(is_tracking_fresh(True, 10.0, 0.25, now=10.3))
        self.assertFalse(is_tracking_fresh(False, 10.0, 0.25, now=10.1))
        self.assertFalse(is_tracking_fresh(True, 0.0, 0.25, now=10.1))
        self.assertFalse(is_tracking_fresh(True, 10.2, 0.25, now=10.1))

    def test_rejects_malformed_tracking_points(self):
        retargeter = LinkerO6HandRetargeter(
            URDF_ROOT / "left" / "linkerhand_o6_left.urdf", "left"
        )
        with self.assertRaises(ValueError):
            retargeter.retarget(np.zeros((24, 3)))
        malformed = np.zeros((25, 3))
        malformed[0, 0] = np.nan
        with self.assertRaises(ValueError):
            retargeter.retarget(malformed)

    def test_visionpro_calibration_is_side_specific_and_bounded(self):
        calibration = LinkerO6Calibration(CALIBRATION_PATH)

        left_low, right_low = calibration.apply(calibration.left_min, calibration.right_min)
        left_high, right_high = calibration.apply(calibration.left_max, calibration.right_max)
        np.testing.assert_allclose(left_low, np.zeros(6))
        np.testing.assert_allclose(right_low, np.zeros(6))
        np.testing.assert_allclose(left_high, np.ones(6))
        np.testing.assert_allclose(right_high, np.ones(6))

        left_mid, right_mid = calibration.apply(
            (calibration.left_min + calibration.left_max) / 2.0,
            (calibration.right_min + calibration.right_max) / 2.0,
        )
        np.testing.assert_allclose(left_mid, np.full(6, 0.5))
        np.testing.assert_allclose(right_mid, np.full(6, 0.5))
        self.assertEqual(calibration.name, "visionpro_giovanni_20260831_v1")


if __name__ == "__main__":
    unittest.main()
