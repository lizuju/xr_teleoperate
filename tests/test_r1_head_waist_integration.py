import ast
from contextlib import nullcontext
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    pin = None


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "teleop" / "teleop_hand_and_arm.py"
sys.path.insert(0, str(ROOT / "teleop" / "robot_control"))
from r1_head_waist import compensate_wrist_for_waist


def assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def calls(node, name):
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, (ast.Name, ast.Attribute))
        and (child.func.id if isinstance(child.func, ast.Name) else child.func.attr) == name
        for child in ast.walk(node)
    )


def execute(nodes, namespace, loop=False):
    if loop:
        # Preserve the production loop's continue/break behavior without starting DDS.
        nodes = [ast.For(
            target=ast.Name(id="_iteration", ctx=ast.Store()),
            iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
            body=nodes + [ast.Assign(
                targets=[ast.Name(id="completed", ctx=ast.Store())], value=ast.Constant(True),
            )], orelse=[],
        )]
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, str(MAIN_PATH), "exec"), namespace)


def pose(yaw, xyz):
    result = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = xyz
    return result


class R1HeadWaistIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        self.loop = next(
            node for node in ast.walk(self.tree) if isinstance(node, ast.While)
            and any(assigns(child, "time_ik_start") for child in node.body)
        )
        start = next(i for i, node in enumerate(self.loop.body) if assigns(node, "tele_data"))
        end = next(i for i, node in enumerate(self.loop.body) if calls(node, "write_json_line"))
        self.control_nodes = self.loop.body[start:end + 1]
        self.left = pose(0.6, [0.4, 0.2, 0.8])
        self.right = pose(-0.4, [0.35, -0.22, 0.78])

    def context(self, freshness=(True, True)):
        tele_data = SimpleNamespace(
            head_pose=pose(0.7, [0, 0, 1]), left_wrist_pose=self.left,
            right_wrist_pose=self.right, motion_data_ready=True,
            motion_data_timestamp=0.99,
        )
        controller = Mock()
        controller.get_current_waist_yaw.return_value = 0.43
        controller.get_current_dual_arm_q.return_value = np.zeros(14)
        controller.get_current_dual_arm_dq.return_value = np.zeros(14)
        controller.get_current_head_q.return_value = np.array([0.1, 0.2])
        follower = Mock(total_yaw=0.7, following=True)
        follower.update.return_value = (np.array([0.2, 0.27]), 1.2)
        ik = Mock()
        ik.solve_ik.return_value = (np.zeros(14), np.zeros(14))
        ik.forward_wrist_poses.return_value = tuple(
            compensate_wrist_for_waist(p, 0.43, 0.17) for p in (self.left, self.right)
        )
        return {
            "np": np, "math": math, "STOP": False, "completed": False,
            "args": SimpleNamespace(
                waist_follow=True, ee=None, input_mode="hand", motion=False,
                tracking_timeout=0.25, frequency=30.0, arm_translation_scale=1.0,
            ),
            "r1_a7_anchored": True, "r1_a7_deferred_real": True,
            "r1_waist_yaw_reference": 0.17, "r1_head_pose_reference": np.eye(4),
            "r1_head_yaw_reference": np.eye(3), "r1_waist_to_root": np.eye(3),
            "r1_vision_left_reference": self.left, "r1_vision_right_reference": self.right,
            "r1_robot_left_reference": self.left, "r1_robot_right_reference": self.right,
            "last_fresh_tele_data": tele_data, "tracking_hold_active": False,
            "tv_wrapper": Mock(get_tele_data=Mock(return_value=tele_data)),
            "is_fresh_motion_data": Mock(side_effect=freshness),
            "logger_mp": Mock(), "arm_ctrl": controller, "arm_ik": ik,
            "waist_follower": follower,
            "relative_head_pitch_yaw": Mock(return_value=np.zeros(2)),
            "wrist_in_reference_head_yaw_frame": lambda wrist, *_: wrist.copy(),
            "anchored_wrist_target": lambda wrist, *_: wrist.copy(),
            "compensate_wrist_for_waist": compensate_wrist_for_waist,
            "xr_motion_data_ready": SimpleNamespace(get_lock=nullcontext, value=False),
            "time": SimpleNamespace(
                time=lambda: 1.0, monotonic=lambda: 1.0, sleep=Mock(),
                time_ns=lambda: 1000000000, monotonic_ns=lambda: 1000000000,
            ),
            "waist_diagnostic_next_time": 2.0, "arm_diagnostic_file": None,
            "arm_diagnostic_next_time": 0.0, "arm_diagnostic_sequence": 0,
            "loop_period_ms": 33.0, "write_json_line": Mock(),
            "rotation_error_rad": lambda a, b: float(np.arccos(np.clip(
                (np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0,
            ))),
        }

    def test_ik_uses_actual_waist_with_nonzero_reference_not_pending_command(self):
        ns = self.context()
        execute(self.control_nodes, ns, loop=True)
        self.assertTrue(ns["completed"])
        ik_call = ns["arm_ik"].solve_ik.call_args
        for observed, root_target in zip(ik_call.args[:2], (self.left, self.right)):
            expected = pose(0.17 - 0.43, [0, 0, 0]) @ root_target
            np.testing.assert_allclose(observed, expected, atol=1e-12)
            wrong = pose(0.17 - 1.2, [0, 0, 0]) @ root_target
            self.assertGreater(np.linalg.norm(observed - wrong), 0.1)
        self.assertTrue(ik_call.kwargs["raise_on_failure"])
        sent = ns["arm_ctrl"].ctrl_dual_arm_and_head.call_args
        self.assertEqual(sent.kwargs["waist_yaw_target"], 1.2)
        np.testing.assert_allclose(sent.args[2], [0.2, 0.27])
        self.assertEqual(ns["waist_follower"].update.call_args.args[2], 0.43)

    def test_stale_xr_does_not_solve_or_refresh_waist_command(self):
        ns = self.context(freshness=(False,))
        execute(self.control_nodes, ns, loop=True)
        self.assertFalse(ns["completed"])
        ns["arm_ctrl"].hold_waist.assert_called_once_with()
        ns["arm_ctrl"].ctrl_dual_arm_and_head.assert_not_called()
        ns["arm_ik"].solve_ik.assert_not_called()
        ns["waist_follower"].update.assert_not_called()

    def test_xr_expiring_during_ik_does_not_publish(self):
        ns = self.context(freshness=(True, False))
        execute(self.control_nodes, ns, loop=True)
        self.assertFalse(ns["completed"])
        ns["arm_ik"].solve_ik.assert_called_once()
        ns["arm_ctrl"].hold_waist.assert_called_once_with()
        ns["arm_ctrl"].ctrl_dual_arm_and_head.assert_not_called()
        self.assertTrue(ns["tracking_hold_active"])

    def test_tracking_resumption_resets_follower_from_actual_waist(self):
        ns = self.context()
        ns["tracking_hold_active"] = True
        execute(self.control_nodes, ns, loop=True)
        ns["waist_follower"].reset.assert_called_once_with(0.43, 1.0)
        self.assertFalse(ns["tracking_hold_active"])

    def test_diagnostics_convert_fixed_ik_fk_back_to_root_frame(self):
        ns = self.context()
        ns["arm_diagnostic_file"] = object()
        execute(self.control_nodes, ns, loop=True)
        payload = ns["write_json_line"].call_args.args[1]
        for side, target in (("left", self.left), ("right", self.right)):
            np.testing.assert_allclose(payload[f"actual_{side}_pose"], target, atol=1e-12)
            np.testing.assert_allclose(payload[f"solved_{side}_pose"], target, atol=1e-12)
            self.assertAlmostEqual(payload[f"{side}_actual_position_error_m"], 0.0)
            self.assertAlmostEqual(payload[f"{side}_solved_rotation_error_rad"], 0.0)
        self.assertEqual(payload["waist_yaw_actual_rad"], 0.43)
        self.assertEqual(payload["waist_yaw_target_rad"], 1.2)

    def test_recorded_body_uses_actual_state_and_command_action(self):
        ns = self.context()
        execute(self.control_nodes, ns, loop=True)
        body_record = next(
            node for node in ast.walk(self.tree) if isinstance(node, ast.If)
            and ast.unparse(node.test) == "args.waist_follow"
            and any(assigns(child, "current_body_state") for child in node.body)
        )
        execute([body_record], ns)
        np.testing.assert_allclose(ns["current_body_state"], [0.43, 0.1, 0.2])
        np.testing.assert_allclose(ns["current_body_action"], [1.2, 0.2, 0.27])

    def test_waist_option_rejects_non_r1_hand_only_dry_run_and_motion(self):
        guard = next(
            node for node in ast.walk(self.tree) if isinstance(node, ast.If)
            and isinstance(node.test, ast.BoolOp)
            and ast.unparse(node.test.values[0]) == "args.waist_follow"
            and calls(node, "error")
        )
        for override in ({"arm": "G1_29"}, {"hand_only": True}, {"dry_run": True}, {"motion": True}):
            with self.subTest(override=override):
                settings = dict(waist_follow=True, arm="R1_A7", hand_only=False, dry_run=False, motion=False)
                settings.update(override)
                parser = Mock(error=Mock(side_effect=ValueError))
                with self.assertRaises(ValueError):
                    execute([guard], {"args": SimpleNamespace(**settings), "parser": parser})
        for enabled in (True, False):
            parser = Mock()
            execute([guard], {"args": SimpleNamespace(
                waist_follow=enabled, arm="R1_A7", hand_only=False, dry_run=False, motion=False,
            ), "parser": parser})
            parser.error.assert_not_called()

    def test_waist_simulation_uses_deferred_activation_and_no_real_mode_switch(self):
        settings = SimpleNamespace(arm="R1_A7", sim=True, motion=False, hand_only=False)
        controller_branch = next(
            node for node in ast.walk(self.tree) if isinstance(node, ast.If)
            and ast.unparse(node.test) == "args.arm == 'R1_A7'"
            and calls(node, "R1_A7_ArmController")
        )
        constructor, ik_constructor, switcher = Mock(), Mock(), Mock()
        ns = {"args": settings, "r1_a7_anchored": True, "r1_a7_deferred_real": False,
              "R1_A7_ArmController": constructor, "R1_A7_ArmIK": ik_constructor,
              "MotionSwitcher": switcher, "logger_mp": Mock()}
        execute([controller_branch], ns)
        constructor.assert_called_once_with(motion_mode=False, simulation_mode=True, deferred_activation=True)
        ik_constructor.assert_not_called()
        switches = [
            node for node in ast.walk(self.tree) if isinstance(node, ast.If)
            and ast.unparse(node.test) in ("r1_a7_deferred_real", "not r1_a7_anchored")
            and calls(node, "Enter_Debug_Mode")
        ]
        self.assertEqual(len(switches), 2)
        execute(switches, ns)
        switcher.assert_not_called()

    @unittest.skipIf(pin is None, "Pinocchio is optional for the offline FK check")
    def test_full_urdf_fk_matches_fixed_waist_compensation(self):
        model = pin.buildModelFromUrdf(str(ROOT / "assets" / "r1" / "r1_a7.urdf"))
        waist_id = model.getJointId("waist_yaw_joint")
        waist_index = model.joints[waist_id].idx_q
        locked = [model.getJointId(name) for name in (
            "waist_yaw_joint", "head_pitch_joint", "head_yaw_joint",
        )]
        rng = np.random.default_rng(7)
        for reference, actual in ((0.17, 0.43), (-0.32, -0.85), (0.23, 1.3)):
            reference_q = pin.neutral(model)
            reference_q[waist_index] = reference
            reduced = pin.buildReducedModel(model, locked, reference_q)
            full_data, reduced_data = model.createData(), reduced.createData()
            for _ in range(10):
                reduced_q = rng.uniform(
                    reduced.lowerPositionLimit * 0.5, reduced.upperPositionLimit * 0.5,
                )
                actual_q = reference_q.copy()
                actual_q[waist_index] = actual
                for reduced_id in range(1, reduced.njoints):
                    full_joint = model.joints[model.getJointId(reduced.names[reduced_id])]
                    reduced_joint = reduced.joints[reduced_id]
                    actual_q[full_joint.idx_q:full_joint.idx_q + full_joint.nq] = reduced_q[
                        reduced_joint.idx_q:reduced_joint.idx_q + reduced_joint.nq
                    ]
                pin.framesForwardKinematics(model, full_data, actual_q)
                pin.framesForwardKinematics(reduced, reduced_data, reduced_q)
                for side in ("left", "right"):
                    frame_name = f"{side}_wrist_yaw_link"
                    fixed_pose = reduced_data.oMf[reduced.getFrameId(frame_name)].homogeneous.copy()
                    actual_pose = full_data.oMf[model.getFrameId(frame_name)].homogeneous.copy()
                    np.testing.assert_allclose(
                        compensate_wrist_for_waist(fixed_pose, reference, actual),
                        actual_pose, atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        compensate_wrist_for_waist(actual_pose, actual, reference),
                        fixed_pose, atol=1e-12,
                    )


if __name__ == "__main__":
    unittest.main()
