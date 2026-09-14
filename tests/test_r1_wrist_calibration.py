import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
from r1_wrist_calibration import estimate


def pose(rotation, position):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result.tolist()


class WristCalibrationTest(unittest.TestCase):
    def test_estimates_side_specific_rotation_and_translation(self):
        vision = np.eye(3)
        robot = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        records = [{
            "event": "activation",
            "vision_left_reference": pose(vision, [0.1, 0.2, 0.3]),
            "robot_left_reference": pose(robot, [0.2, 0.4, 0.6]),
            "vision_right_reference": pose(robot, [0.0, 0.1, 0.2]),
            "robot_right_reference": pose(vision, [0.1, 0.3, 0.5]),
        }]
        result = estimate(records)
        np.testing.assert_allclose(result["sides"]["left"]["rotation_correction"], robot)
        np.testing.assert_allclose(result["sides"]["left"]["translation_offset_m"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(result["sides"]["right"]["rotation_correction"], robot.T)
        self.assertAlmostEqual(result["sides"]["left"]["rotation_correction_det"], 1.0)

    def test_rejects_missing_activation(self):
        with self.assertRaisesRegex(ValueError, "No activation"):
            estimate([{ "event": "sample" }])


if __name__ == "__main__":
    unittest.main()
