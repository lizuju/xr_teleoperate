import ast
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MAIN_PATH = Path(__file__).resolve().parents[1] / "teleop/teleop_hand_and_arm.py"


def load_helpers(names):
    """Execute the named module-level functions from the main script.

    The main program is not importable (it runs the whole application inside
    ``if __name__ == '__main__'``), so tests extract individual definitions by
    AST, the same way the other R1_A7 tests do.
    """
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    wanted = [node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
    return namespace


class WorkspaceSaturationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_helpers({"r1_workspace_saturation", "rotation_error_rad"})
        cls.saturation = staticmethod(cls.ns["r1_workspace_saturation"])

    @staticmethod
    def pose(position, rotation=None):
        result = np.eye(4)
        result[:3, 3] = position
        if rotation is not None:
            result[:3, :3] = rotation
        return result

    def test_reachable_targets_report_no_saturation(self):
        result = self.saturation(
            self.pose([0.30, 0.0, 0.0]), self.pose([-0.30, 0.0, 0.0]),
            self.pose([0.30, 0.0, 0.0]), self.pose([-0.30, 0.0, 0.0]),
            0.05, 0.15,
        )
        self.assertEqual(set(result), {"left", "right"})
        for side in ("left", "right"):
            self.assertFalse(result[side]["outside"])
            self.assertAlmostEqual(result[side]["position_m"], 0.0, places=12)
            self.assertAlmostEqual(result[side]["rotation_rad"], 0.0, places=12)

    def test_position_shortfall_marks_only_the_short_side(self):
        result = self.saturation(
            self.pose([0.50, 0.0, 0.0]), self.pose([-0.30, 0.0, 0.0]),
            self.pose([0.30, 0.0, 0.0]), self.pose([-0.30, 0.0, 0.0]),
            0.05, 0.15,
        )
        self.assertTrue(result["left"]["outside"])
        self.assertFalse(result["right"]["outside"])
        self.assertAlmostEqual(result["left"]["position_m"], 0.20, places=12)

    def test_tolerance_boundary_is_exclusive(self):
        solved = self.pose([0.0, 0.0, 0.0])
        for limit, offset in ((0.05, 0.05), (0.05, 0.0500001)):
            target = self.pose([offset, 0.0, 0.0])
            result = self.saturation(solved, solved, target, solved, limit, 0.15)
            self.assertEqual(bool(result["left"]["position_m"] > limit), result["left"]["outside"])

    def test_rotation_shortfall_alone_is_reported(self):
        angle = 0.4
        rotation = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = self.saturation(
            self.pose([0.0, 0.0, 0.0]), self.pose([0.0, 0.0, 0.0]),
            self.pose([0.0, 0.0, 0.0], rotation), self.pose([0.0, 0.0, 0.0]),
            0.05, 0.15,
        )
        self.assertTrue(result["left"]["outside"])
        self.assertFalse(result["right"]["outside"])
        self.assertAlmostEqual(result["left"]["position_m"], 0.0, places=12)
        self.assertAlmostEqual(result["left"]["rotation_rad"], angle, places=6)

    def test_large_rotation_error_stays_finite(self):
        # rotation_error_rad clips the cosine, so a 180 degree flip must not raise.
        flipped = np.diag([-1.0, -1.0, 1.0])
        result = self.saturation(
            self.pose([0.0, 0.0, 0.0], flipped), self.pose([0.0, 0.0, 0.0]),
            self.pose([0.0, 0.0, 0.0]), self.pose([0.0, 0.0, 0.0]),
            0.05, 0.15,
        )
        self.assertTrue(math.isfinite(result["left"]["rotation_rad"]))
        self.assertAlmostEqual(result["left"]["rotation_rad"], math.pi, places=4)


class HoldPublishedTargetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_helpers({"hold_r1_published_targets"})
        cls.hold = staticmethod(cls.ns["hold_r1_published_targets"])

    @staticmethod
    def controller(published):
        ctrl = Mock()
        ctrl.get_recording_snapshot.return_value = {"published": published}
        return ctrl

    def test_replays_the_last_published_command(self):
        published = {"arm_q": [0.1] * 14, "arm_tau": [0.2] * 14, "head_q": [0.3, 0.4]}
        ctrl = self.controller(published)
        self.hold(ctrl)
        ctrl.ctrl_dual_arm_and_head.assert_called_once()
        args = ctrl.ctrl_dual_arm_and_head.call_args.args
        np.testing.assert_allclose(args[0], published["arm_q"])
        np.testing.assert_allclose(args[1], published["arm_tau"])
        np.testing.assert_allclose(args[2], published["head_q"])
        ctrl.hold_targets.assert_not_called()
        ctrl.hold_waist.assert_called_once_with()

    def test_no_publication_yet_falls_back_to_the_controller_hold(self):
        # Regression: pausing before the first successful write used to raise
        # TypeError because published["arm_q"] was None.
        for published in (None, {"arm_q": None, "arm_tau": None, "head_q": None}):
            with self.subTest(published=published):
                ctrl = self.controller(published)
                self.hold(ctrl)
                ctrl.hold_targets.assert_called_once_with()
                ctrl.hold_waist.assert_called_once_with()
                ctrl.ctrl_dual_arm_and_head.assert_not_called()


if __name__ == "__main__":
    unittest.main()
