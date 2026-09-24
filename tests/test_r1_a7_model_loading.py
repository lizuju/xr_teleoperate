import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pinocchio as pin
    from pinocchio import casadi as cpin
except ImportError:
    pin = None


@unittest.skipIf(pin is None, "Requires Pinocchio with CasADi support")
class R1A7ModelLoadingTest(unittest.TestCase):
    def test_no_mesh_loading_preserves_joint_limits_fk_and_dynamics(self):
        spec = importlib.util.spec_from_file_location(
            "r1_model_loading", ROOT / "teleop/robot_control/robot_arm_ik.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        urdf = ROOT / "assets/r1/r1_a7.urdf"
        original = pin.RobotWrapper.BuildFromURDF(str(urdf), str(urdf.parent))
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(ROOT / "teleop")
        rng = np.random.default_rng(17)
        for waist in (0., .17, -.32):
            with self.subTest(waist_yaw=waist):
                with patch.object(pin.RobotWrapper, "BuildFromURDF", side_effect=AssertionError("IK loaded meshes")):
                    ik = module.R1_A7_ArmIK(waist_yaw=waist)
                self.assertEqual(ik.robot.visual_model.ngeoms, 0)
                self.assertEqual(ik.robot.collision_model.ngeoms, 0)
                reference = pin.neutral(original.model)
                joint = original.model.getJointId("waist_yaw_joint")
                reference[original.model.joints[joint].idx_q] = waist
                old = original.buildReducedRobot(
                    ["waist_yaw_joint", "head_pitch_joint", "head_yaw_joint"], reference,
                )
                for side, name in (("left", "L_ee"), ("right", "R_ee")):
                    old.model.addFrame(pin.Frame(
                        name, old.model.getJointId(f"{side}_wrist_yaw_joint"),
                        pin.SE3(np.eye(3), np.array([.05, 0., 0.])), pin.FrameType.OP_FRAME,
                    ))
                old.data = old.model.createData()
                self.assertEqual(list(old.model.names), list(ik.reduced_robot.model.names))
                for name in ("lowerPositionLimit", "upperPositionLimit", "velocityLimit", "effortLimit"):
                    np.testing.assert_array_equal(getattr(old.model, name), getattr(ik.reduced_robot.model, name))
                for _ in range(10):
                    q = rng.uniform(old.model.lowerPositionLimit * .5, old.model.upperPositionLimit * .5)
                    dq, ddq = rng.uniform(-.2, .2, (2, old.model.nv))
                    pin.framesForwardKinematics(old.model, old.data, q)
                    for name, pose in zip(("L_ee", "R_ee"), ik.forward_wrist_poses(q)):
                        np.testing.assert_allclose(pose, old.data.oMf[old.model.getFrameId(name)].homogeneous, atol=1e-12)
                    np.testing.assert_allclose(
                        pin.rnea(ik.reduced_robot.model, ik.reduced_robot.data, q, dq, ddq),
                        pin.rnea(old.model, old.data, q, dq, ddq), atol=1e-12,
                    )


if __name__ == "__main__":
    unittest.main()
