import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from teleop.utils.one_euro_filter import OneEuroFilter


class R1A7AdaptiveSmoothingTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "teleop/robot_control/robot_arm_ik.py"
        tree = ast.parse(path.read_text())
        r1_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "R1_A7_ArmIK")
        self.clock = Mock(return_value=1.0)
        self.dynamics = Mock(return_value=np.zeros(14))
        namespace = {
            "np": np, "time": SimpleNamespace(monotonic=self.clock),
            "pin": SimpleNamespace(rnea=self.dynamics), "logger_mp": Mock(),
        }
        # Execute the production solver method with an offline optimizer; no DDS imports.
        exec(compile(ast.Module(body=[r1_class], type_ignores=[]), str(path), "exec"), namespace)
        self.ik = namespace["R1_A7_ArmIK"].__new__(namespace["R1_A7_ArmIK"])
        self.ik.init_data = np.zeros(14)
        self.ik.smooth_filters = [OneEuroFilter(), OneEuroFilter()]
        self.ik.Visualization = False
        self.ik.var_q = object()
        self.ik.var_q_last = object()
        self.ik.param_tf_l = object()
        self.ik.param_tf_r = object()
        self.ik.reduced_robot = SimpleNamespace(model=SimpleNamespace(nv=14), data=object())
        self.target = np.full(14, 0.5)
        self.ik.opti = Mock()
        self.ik.opti.value.side_effect = lambda _: self.target.copy()
        self.ik.opti.debug.value.side_effect = lambda _: self.target.copy()

    def solve(self, t, measured=None):
        self.clock.return_value = t
        return self.ik.solve_ik(np.eye(4), np.eye(4), measured, raise_on_failure=True)[0]

    def test_activation_starts_at_measured_angles_then_smooths_and_recomputes_torque(self):
        measured = np.linspace(-0.2, 0.2, 14)
        np.testing.assert_array_equal(self.solve(1.0, measured), measured)
        output = self.solve(1.0 + 1 / 30.0, measured)
        self.assertTrue(np.all(output > measured))
        self.assertTrue(np.all(output < self.target))
        np.testing.assert_array_equal(self.dynamics.call_args.args[2], output)

    def test_resetting_one_arm_preserves_the_other_arms_history(self):
        measured = np.zeros(14)
        self.solve(1.0, measured)
        previous = self.solve(1.0 + 1 / 30.0, measured)
        self.ik.reset_smoothing(0)
        output = self.solve(1.0 + 2 / 30.0, measured)
        np.testing.assert_array_equal(output[:7], measured[:7])
        self.assertTrue(np.all(output[7:] > previous[7:]))

    def test_long_pause_reseeds_both_arms_from_measured_angles(self):
        self.solve(1.0, np.zeros(14))
        self.solve(1.0 + 1 / 30.0, np.zeros(14))
        measured = np.full(14, 0.1)
        np.testing.assert_array_equal(self.solve(2.0, measured), measured)

    def test_failed_ik_clears_filter_history_before_the_next_solution(self):
        self.solve(1.0, np.zeros(14))
        self.solve(1.0 + 1 / 30.0, np.zeros(14))
        self.ik.opti.solve.side_effect = RuntimeError("no solution")
        with self.assertRaisesRegex(RuntimeError, "failed to converge"):
            self.solve(1.0 + 2 / 30.0, np.zeros(14))
        self.ik.opti.solve.side_effect = None
        measured = np.full(14, 0.1)
        np.testing.assert_array_equal(self.solve(1.1, measured), measured)

    def test_fallback_does_not_store_the_failed_candidate_as_the_next_seed(self):
        measured = np.full(14, 0.1)
        self.solve(1.0, measured)
        self.ik.opti.solve.side_effect = RuntimeError("no solution")
        self.target[:] = 9.0
        output, _ = self.ik.solve_ik(np.eye(4), np.eye(4), measured)
        np.testing.assert_array_equal(output, measured)
        np.testing.assert_array_equal(self.ik.init_data, measured)


if __name__ == "__main__":
    unittest.main()
