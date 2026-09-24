import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


IK_PATH = Path(__file__).resolve().parents[1] / "teleop/robot_control/robot_arm_ik.py"


class FakeOpti:
    """Records set_value calls so the setter can be exercised without CasADi."""

    def __init__(self, nq):
        self.values = {}
        self.params = {name: object() for name in ("nominal", "posture", "limit")}

    def set_value(self, param, value):
        for name, token in self.params.items():
            if param is token:
                self.values[name] = np.asarray(value, dtype=float)
                return
        raise AssertionError("unknown parameter")


def load_setter(nq):
    """Extract R1_A7_ArmIK.set_redundancy_weights from the production source."""
    tree = ast.parse(IK_PATH.read_text(encoding="utf-8"))
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "R1_A7_ArmIK":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "set_redundancy_weights":
                    target = child
    if target is None:
        raise AssertionError("set_redundancy_weights not found in R1_A7_ArmIK")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[target], type_ignores=[]), str(IK_PATH), "exec"), namespace)
    instance = type("Stub", (), {})()
    instance.opti = FakeOpti(nq)
    instance._nominal_arm_q = np.zeros(nq)
    instance.posture_weight = 0.0
    instance.limit_weight = 0.0
    instance.param_nominal_q = instance.opti.params["nominal"]
    instance.param_posture_weight = instance.opti.params["posture"]
    instance.param_limit_weight = instance.opti.params["limit"]
    instance.set_redundancy_weights = namespace["set_redundancy_weights"].__get__(instance)
    return instance


class RedundancyWeightsTest(unittest.TestCase):
    def test_weights_and_nominal_are_pushed_to_the_solver(self):
        ik = load_setter(14)
        nominal = np.linspace(-0.3, 0.3, 14)
        ik.set_redundancy_weights(posture_weight=0.05, limit_weight=0.2, nominal_arm_q=nominal)
        self.assertEqual(ik.posture_weight, 0.05)
        self.assertEqual(ik.limit_weight, 0.2)
        np.testing.assert_allclose(ik.opti.values["posture"], [0.05])
        np.testing.assert_allclose(ik.opti.values["limit"], [0.2])
        np.testing.assert_allclose(ik.opti.values["nominal"], nominal)

    def test_omitted_arguments_leave_current_values_untouched(self):
        ik = load_setter(14)
        ik.set_redundancy_weights(posture_weight=0.03, limit_weight=0.4)
        ik.set_redundancy_weights(limit_weight=0.1)
        self.assertEqual(ik.posture_weight, 0.03)
        self.assertEqual(ik.limit_weight, 0.1)
        np.testing.assert_allclose(ik.opti.values["nominal"], np.zeros(14))

    def test_negative_or_non_finite_weights_are_rejected(self):
        for kwargs in ({"posture_weight": -0.1}, {"limit_weight": -1.0},
                       {"posture_weight": float("nan")}, {"limit_weight": float("inf")}):
            with self.subTest(**kwargs):
                ik = load_setter(14)
                with self.assertRaises(ValueError):
                    ik.set_redundancy_weights(**kwargs)

    def test_wrong_shaped_or_non_finite_nominal_is_rejected(self):
        ik = load_setter(14)
        for bad in (np.zeros(7), np.full(14, np.nan), np.zeros((2, 7))):
            with self.subTest(shape=np.shape(bad)):
                with self.assertRaises(ValueError):
                    ik.set_redundancy_weights(nominal_arm_q=bad)

    def test_rejected_call_does_not_change_state(self):
        ik = load_setter(14)
        ik.set_redundancy_weights(posture_weight=0.02)
        with self.assertRaises(ValueError):
            ik.set_redundancy_weights(limit_weight=-1.0)
        self.assertEqual(ik.posture_weight, 0.02)
        self.assertEqual(ik.limit_weight, 0.0)


class ObjectiveTermsTest(unittest.TestCase):
    def test_limit_and_posture_costs_are_distinct_from_smooth(self):
        # Regression guard: the barrier must be a range-normalised quartic offset,
        # and the posture cost must reference the nominal parameter, so none of the
        # three terms collapse into another.
        source = IK_PATH.read_text(encoding="utf-8")
        self.assertIn("normalised_offset ** 2", source)
        self.assertIn("self.joint_middle", source)
        self.assertIn("self.joint_half_range", source)
        self.assertIn("self.param_nominal_q", source)
        self.assertIn("self.param_limit_weight * self.limit_cost", source)
        self.assertIn("self.param_posture_weight * self.posture_cost", source)
        # The objective is built exactly once (opti.minimize may not be called twice
        # for one Opti instance), so the weights have to travel as parameters.
        self.assertEqual(source.count("self.opti.minimize("), 7)


if __name__ == "__main__":
    unittest.main()


class ElbowDriftGuardDefaultTest(unittest.TestCase):
    """The nominal-posture term is the only thing anchoring the 7-DoF null space.

    It was introduced with a default weight of 0.0, which silently disabled it:
    the flag existed, the IK term existed, and the elbow still drifted.
    """

    def setUp(self):
        from pathlib import Path
        self.source = (Path(__file__).resolve().parents[1]
                       / "teleop" / "teleop_hand_and_arm.py").read_text(encoding="utf-8")

    def test_the_nominal_posture_term_is_enabled_by_default(self):
        self.assertIn("--arm-posture-weight', type=float, default=0.01", self.source)

    def test_the_soft_limit_barrier_stays_on(self):
        self.assertIn("--arm-limit-softness', type=float, default=0.1", self.source)

    def test_the_effective_weights_are_always_logged(self):
        # A silent zero here is how the protection went missing the first time.
        self.assertIn("[R1 IK] redundancy terms", self.source)
        self.assertIn("elbow drift protection OFF", self.source)

    def test_both_weights_reach_the_solver(self):
        tree = ast.parse(self.source)
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "set_redundancy_weights")
        motor_q = np.arange(32, dtype=float) / 10
        arm_q = motor_q[15:29].copy()
        controller = Mock(get_current_dual_arm_q=Mock(return_value=arm_q))
        ik = load_setter(14)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "activation", "exec"), {
            "arm_ctrl": controller, "arm_ik": ik, "post_recenter_motor_q": motor_q,
            "args": SimpleNamespace(arm_posture_weight=0.01, arm_limit_softness=0.1),
        })
        np.testing.assert_array_equal(ik.opti.values["nominal"], arm_q)
        self.assertEqual(ik.posture_weight, 0.01)
        self.assertEqual(ik.limit_weight, 0.1)

    def test_the_wrappers_pass_both_weights(self):
        from pathlib import Path
        for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
            script = (Path(__file__).resolve().parents[1] / "teleop" / name).read_text(encoding="utf-8")
            self.assertIn('--arm-posture-weight "${ARM_POSTURE_WEIGHT:-0.01}"', script)
            self.assertIn('--arm-limit-softness "${ARM_LIMIT_SOFTNESS:-0.1}"', script)
