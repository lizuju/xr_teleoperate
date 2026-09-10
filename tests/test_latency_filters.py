import ast
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

FILTER_PATH = Path(__file__).resolve().parents[1] / "teleop/robot_control/dex-retargeting/src/dex_retargeting/optimizer_utils.py"
spec = importlib.util.spec_from_file_location("dex_filter_reference", FILTER_PATH)
filter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_module)
LPFilter = filter_module.LPFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teleop" / "robot_control"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teleop" / "utils"))
from r1_hand_tracking import R1WristHold
from one_euro_filter import OneEuroFilter


class LatencyFilterTest(unittest.TestCase):
    def test_retarget_config_is_passthrough_and_latest_frame_is_not_averaged(self):
        path = Path(__file__).resolve().parents[1] / "teleop/robot_control/linker_o6_retargeting.py"
        tree = ast.parse(path.read_text())
        values = [value.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                  for key, value in zip(node.keys, node.values)
                  if isinstance(key, ast.Constant) and key.value == "low_pass_alpha"]
        self.assertEqual(values, [1.0])
        smoothing = LPFilter(values[0])
        for value in (0.0, 0.1, 1.0, 0.2, 0.0):
            np.testing.assert_allclose(smoothing.next(np.full(6, value)), value, atol=1e-15)
        smoothing.reset()
        self.assertFalse(smoothing.is_init)

    def test_ideal_step_response_keeps_one_smoothing_stage(self):
        # Filter-only model at 30 Hz, not an end-to-end robot latency measurement.
        frames_to_90 = []
        for retarget_alpha in (0.5, 1.0):
            retarget = LPFilter(retarget_alpha)
            retarget.next(np.zeros(6))
            action = np.zeros(6)
            controller_alpha = -math.expm1(-(1 / 30) / 0.04)
            outputs = []
            for step in range(1, 16):
                target = retarget.next(np.ones(6))
                action += controller_alpha * (target - action)
                outputs.append(action.copy())
                if action[0] >= 0.9:
                    frames_to_90.append(step)
                    break
            self.assertLess(outputs[0][0], 1.0)
            self.assertTrue(np.all(np.diff(np.array(outputs), axis=0) >= 0.0))
        self.assertEqual(frames_to_90, [5, 3])

    def test_wrist_recovery_has_one_held_frame_without_bypassing_joint_filter(self):
        for angle in (math.pi / 2, math.pi):
            with self.subTest(angle=angle):
                held = np.eye(4)
                target = np.eye(4)
                c, s = math.cos(angle), math.sin(angle)
                target[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
                hold = R1WristHold(held)
                hold.hold()
                np.testing.assert_array_equal(hold.prepare(target, True), held)
                np.testing.assert_array_equal(hold.prepare(target, True), target)

                # Joint-space filtering remains separate from the wrist target handoff.
                smoothing = OneEuroFilter()
                actual_q = np.zeros(7)
                solved_q = np.full(7, angle)
                np.testing.assert_array_equal(smoothing.filter(solved_q, 1.0, actual_q), actual_q)
                filtered_q = smoothing.filter(solved_q, 1.0 + 1 / 30, actual_q)
                self.assertTrue(np.all(filtered_q > actual_q))
                self.assertTrue(np.all(filtered_q < solved_q))


if __name__ == "__main__":
    unittest.main()
