"""Does one stale hand freeze the other arm? Drive the real main loop."""
import ast
from contextlib import nullcontext
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.robot_control.r1_hand_tracking import R1WristHold, hand_tracking_freshness, hand_tracking_present

MAIN = Path(__file__).resolve().parents[1] / "teleop/teleop_hand_and_arm.py"


def pose(angle, position):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("XYZ", [0.3, angle, 0.2]).as_matrix()
    m[:3, 3] = position
    return m


class SideIndependenceTest(unittest.TestCase):
    """Hold the right hand stale and move the left: the left must keep following."""

    def test_fresh_side_keeps_following_while_other_side_is_stale(self):
        tree = ast.parse(MAIN.read_text(encoding="utf-8"))
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                    and any(isinstance(c, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "time_ik_start" for t in c.targets)
                        for c in n.body))
        def assigns_directly(stmt, name):
            # Only direct children: ast.walk would also match nested statements.
            return isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in stmt.targets)

        start = next(i for i, n in enumerate(loop.body) if assigns_directly(n, "tele_data"))
        end = next(i for i, n in enumerate(loop.body) if "write_json_line" in ast.unparse(n))
        control_nodes = loop.body[start:end + 1]

        # Right side lost (timestamp zeroed); left side still present.
        sample = SimpleNamespace(
            head_pose=pose(0.0, [0, 0, 1]),
            left_wrist_pose=pose(0.0, [0.40, 0.20, 0.80]),
            right_wrist_pose=pose(0.0, [0.35, -0.22, 0.78]),
            motion_data_ready=True, motion_data_timestamp=99.0,
            left_hand_timestamp=1.0, right_hand_timestamp=0.0,
        )
        controller = Mock()
        controller.get_current_waist_yaw.return_value = 0.0
        controller.get_current_dual_arm_q.return_value = np.zeros(14)
        controller.get_current_dual_arm_dq.return_value = np.zeros(14)
        controller.get_recording_snapshot.return_value = {
            "requested": {"arm_q": [0.0]*14, "arm_tau": [0.0]*14, "head_q": [0.1, 0.2]},
            "published": {"arm_q": [0.0]*14, "arm_tau": [0.0]*14, "head_q": [0.1, 0.2]},
        }
        ik = Mock()
        ik.solve_ik.return_value = (np.zeros(14), np.zeros(14))
        ik.forward_wrist_poses.return_value = (pose(0.0, [0.4, 0.2, 0.8]), pose(0.0, [0.35, -0.22, 0.78]))
        follower = Mock(total_yaw=0.0, following=False)
        follower.update.return_value = (np.array([0.2, 0.27]), 1.2)

        ns = {
            "np": np, "math": math, "STOP": False, "completed": False,
            "R1_PAUSE": None, "r1_frozen_generation": -1, "r1_head_q_offset": np.zeros(2),
            "args": SimpleNamespace(
                waist_follow=False, ee=None, input_mode="hand", motion=False,
                tracking_timeout=0.25, frequency=30.0, arm_translation_scale=1.0,
                arm_diagnostic_hz=10.0, workspace_position_tolerance_m=0.05,
                workspace_rotation_tolerance_rad=0.15,
                arm_limit_softness=0.0, arm_posture_weight=0.0, arm_velocity_limit=3.0,
            ),
            "r1_a7_anchored": True, "r1_a7_deferred_real": True,
            "r1_independent_hands": True,
            "tracking_hold_previous": False, "tracking_hold_events": 0,
            "workspace_diagnostic_next_time": 0.0, "workspace_warned_side": None,
            "workspace_saturation_events": 0,
            "r1_waist_yaw_reference": 0.0, "r1_head_pose_reference": np.eye(4),
            "r1_head_yaw_reference": np.eye(3), "r1_waist_to_root": np.eye(3),
            "r1_vision_left_reference": sample.left_wrist_pose.copy(),
            "r1_vision_right_reference": sample.right_wrist_pose.copy(),
            "r1_robot_left_reference": pose(0.0, [0.4, 0.2, 0.8]),
            "r1_robot_right_reference": pose(0.0, [0.35, -0.22, 0.78]),
            "wrist_holds": (R1WristHold(pose(0.0, [0.4, 0.2, 0.8])),
                            R1WristHold(pose(0.0, [0.35, -0.22, 0.78]))),
            "hand_tracking_freshness": hand_tracking_freshness,
            "hand_tracking_present": hand_tracking_present,
            "held_head_q_target": np.array([0.1, 0.2]),
            "last_fresh_tele_data": sample, "tracking_hold_active": False,
            "tv_wrapper": Mock(get_tele_data=Mock(return_value=sample)),
            "is_fresh_motion_data": Mock(return_value=True),
            "is_present_motion_data": Mock(return_value=True),
            "logger_mp": Mock(), "arm_ctrl": controller, "arm_ik": ik,
            "waist_follower": follower, "linker_o6_loop": None,
            "run_motion": True, "capture_mode": "following",
            "relative_head_pitch_yaw": Mock(return_value=np.zeros(2)),
            "wrist_in_reference_head_yaw_frame": lambda w, *_: w.copy(),
            "anchored_wrist_target": lambda w, *_: w.copy(),
            "r1_workspace_saturation": lambda *a, **k: {
                "left": {"position_m": 0.0, "rotation_rad": 0.0, "outside": False},
                "right": {"position_m": 0.0, "rotation_rad": 0.0, "outside": False}},
            "xr_motion_data_ready": SimpleNamespace(get_lock=nullcontext, value=False),
            "time": SimpleNamespace(time=lambda: 1.0, monotonic=lambda: 1.0, sleep=Mock(),
                                    time_ns=lambda: 1, monotonic_ns=lambda: 1),
            "waist_diagnostic_next_time": 2.0, "power_diagnostic_next_time": 2.0, "arm_diagnostic_file": None,
            "arm_diagnostic_next_time": 0.0, "arm_diagnostic_sequence": 0,
            "loop_period_ms": 33.0, "write_json_line": Mock(),
            "rotation_error_rad": lambda a, b: 0.0,
            "head_fresh": (True, True),
        }
        nodes = [ast.For(target=ast.Name(id="_i", ctx=ast.Store()),
                         iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
                         body=control_nodes + [ast.Assign(
                             targets=[ast.Name(id="completed", ctx=ast.Store())],
                             value=ast.Constant(True))], orelse=[])]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(MAIN), "exec"), ns)

        # Left hand fresh -> its target follows the sample; right hand stale -> held.
        left_target, right_target = ns["left_wrist_target"], ns["right_wrist_target"]
        np.testing.assert_allclose(left_target[:3, 3], sample.left_wrist_pose[:3, 3], atol=1e-12,
                                   err_msg="fresh left hand must keep following")
        np.testing.assert_allclose(right_target[:3, 3], [0.35, -0.22, 0.78], atol=1e-12,
                                   err_msg="stale right hand must hold its reference")
        # The IK must still be solving (one live side is enough to publish)
        ik.solve_ik.assert_called()
        print("\n  左目标 == 新鲜样本:", np.allclose(left_target[:3, 3], sample.left_wrist_pose[:3, 3]))
        print("  右目标 == 保持参考:", np.allclose(right_target[:3, 3], [0.35, -0.22, 0.78]))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ReadinessFlagTest(unittest.TestCase):
    """`motion_data_ready` must not require both hands."""

    def test_motion_data_ready_uses_any_side_not_all(self):
        source = (Path(__file__).resolve().parents[1]
                  / "teleop/televuer/src/televuer/televuer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        hand_move = next(n for n in ast.walk(tree)
                         if isinstance(n, ast.AsyncFunctionDef) and n.name == "on_hand_move")
        ready_guard = [n for n in ast.walk(hand_move)
                       if isinstance(n, ast.If)
                       and "motion_data_ready_shared.value = True" in ast.unparse(n)]
        self.assertEqual(len(ready_guard), 1, "expected exactly one ready latch in on_hand_move")
        condition = ast.unparse(ready_guard[0].test)
        self.assertIn("any(", condition)
        self.assertNotIn("all(", condition)

