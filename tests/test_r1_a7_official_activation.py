import ast
from enum import IntEnum
import math
from pathlib import Path
import threading
import time
import types
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ARM_PATH = REPO_ROOT / "teleop" / "robot_control" / "robot_arm.py"
IK_PATH = REPO_ROOT / "teleop" / "robot_control" / "robot_arm_ik.py"
MAIN_PATH = REPO_ROOT / "teleop" / "teleop_hand_and_arm.py"


class SilentLogger:
    def debug(self, message):
        pass

    def info(self, message):
        pass


class FakeMotorCmd:
    def __init__(self):
        self.mode = 0
        self.kp = 0.0
        self.kd = 0.0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0


class FakeLowCmd:
    def __init__(self):
        self.mode_pr = 0
        self.mode_machine = 0
        self.motor_cmd = [FakeMotorCmd() for _ in range(35)]
        self.crc = 0


class FakeSubscriber:
    def __init__(self, *args):
        pass

    def Init(self):
        pass

    def Read(self):
        return types.SimpleNamespace(
            mode_machine=7,
            motor_state=[
                types.SimpleNamespace(q=0.01 * index, dq=-0.001 * index)
                for index in range(35)
            ],
        )

    def Close(self):
        pass


class FakePublisher:
    instances = []

    def __init__(self, *args):
        self.writes = []
        type(self).instances.append(self)

    def Init(self):
        pass

    def Write(self, message):
        self.writes.append([command.q for command in message.motor_cmd])
        return True

    def Close(self):
        pass


class FakeCRC:
    def Crc(self, message):
        return 0


def wait_for_dds(predicate, name):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise TimeoutError(name)


def load_r1_controller_namespace():
    tree = ast.parse(ARM_PATH.read_text(encoding="utf-8"))
    selected_names = {
        "MotorState",
        "R1_A7_LowState",
        "DataBuffer",
        "R1_A7_JointArmIndex",
        "R1_A7_JointHeadIndex",
        "R1_A7_JointIndex",
        "R1_A7_ArmController",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in selected_names
    ]
    namespace = {
        "np": np,
        "threading": threading,
        "time": time,
        "IntEnum": IntEnum,
        "logger_mp": SilentLogger(),
        "ChannelPublisher": FakePublisher,
        "ChannelSubscriber": FakeSubscriber,
        "unitree_hg_msg_dds__LowCmd_": FakeLowCmd,
        "hg_LowCmd": object,
        "hg_LowState": object,
        "CRC": FakeCRC,
        "wait_for_dds": wait_for_dds,
        "R1_A7_Num_Motors": 35,
        "kTopicLowCommand_Debug": "rt/lowcmd",
        "kTopicLowState": "rt/lowstate",
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ARM_PATH), "exec"), namespace)
    return namespace


class R1A7OfficialActivationTest(unittest.TestCase):
    def setUp(self):
        FakePublisher.instances.clear()

    def test_deferred_controller_emits_nothing_until_activation_then_recenters_head_and_waist(self):
        namespace = load_r1_controller_namespace()
        controller = namespace["R1_A7_ArmController"](deferred_activation=True)
        self.assertEqual(FakePublisher.instances, [])
        self.assertFalse(controller.active)

        original_recenter = controller.ctrl_head_and_waist_go_home
        controller.ctrl_head_and_waist_go_home = lambda: original_recenter(duration=0.0)
        controller.activate()
        deadline = time.monotonic() + 1.0
        while not FakePublisher.instances[0].writes and time.monotonic() < deadline:
            time.sleep(0.001)

        self.assertTrue(FakePublisher.instances[0].writes)
        self.assertEqual(FakePublisher.instances[0].writes[0], [0.01 * index for index in range(35)])
        arm_indices = [member.value for member in namespace["R1_A7_JointArmIndex"]]
        np.testing.assert_allclose(controller.q_target, [0.01 * index for index in arm_indices])
        recentered = FakePublisher.instances[0].writes[1]
        for index in (12, 13, 29, 30):
            self.assertEqual(recentered[index], 0.0)
        np.testing.assert_allclose(
            [recentered[index] for index in arm_indices],
            [0.01 * index for index in arm_indices],
        )

        arm_target = np.array([0.02 * index for index in range(14)])
        controller.ctrl_dual_arm_and_head(
            arm_target,
            np.zeros(14),
            [-0.2, 0.3],
        )
        deadline = time.monotonic() + 1.0
        while (
            not any(
                write[29] == -0.2 and write[30] == 0.3
                for write in FakePublisher.instances[0].writes
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        commanded = FakePublisher.instances[0].writes[-1]
        self.assertEqual(commanded[29], -0.2)
        self.assertEqual(commanded[30], 0.3)
        np.testing.assert_allclose(controller.q_target, arm_target)

        with self.assertRaises(ValueError):
            controller.ctrl_dual_arm_and_head(
                np.ones(14),
                np.ones(14),
                [float("nan"), 0.0],
            )
        np.testing.assert_allclose(controller.q_target, arm_target)

        controller.ctrl_dual_arm_and_head(
            arm_target,
            np.zeros(14),
            [1.0, -3.0],
        )
        np.testing.assert_allclose(controller.head_q_target, [0.62832, -2.0071])

        controller.stop()
        write_count = len(FakePublisher.instances[0].writes)
        time.sleep(0.02)
        self.assertEqual(len(FakePublisher.instances[0].writes), write_count)
        self.assertFalse(controller.subscribe_thread.is_alive())
        self.assertFalse(controller.publish_thread.is_alive())

    def test_relative_head_pitch_yaw_tracks_robot_joint_axes(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "relative_head_pitch_yaw"
        )
        namespace = {"math": math, "np": np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)

        pitch = -0.2
        yaw = 0.3
        ry = np.array([
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ])
        rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        current = np.eye(4)
        current[:3, :3] = ry @ rz
        np.testing.assert_allclose(
            namespace["relative_head_pitch_yaw"](current, np.eye(4)),
            [pitch, yaw],
            atol=1e-12,
        )

        roll = 0.4
        current[:3, :3] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(roll), -math.sin(roll)],
            [0.0, math.sin(roll), math.cos(roll)],
        ])
        np.testing.assert_allclose(
            namespace["relative_head_pitch_yaw"](current, np.eye(4)),
            [0.0, 0.0],
            atol=1e-12,
        )

    def test_zero_vision_delta_is_exactly_the_live_robot_wrist_pose(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "anchored_wrist_target"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        vision_reference = np.eye(4)
        vision_reference[:3, 3] = [0.2, -0.1, 0.4]
        robot_reference = np.eye(4)
        robot_reference[:3, 3] = [0.55, 0.3, 0.75]

        target = namespace["anchored_wrist_target"](
            vision_reference.copy(),
            vision_reference,
            robot_reference,
            np.eye(3),
            1.0,
        )
        np.testing.assert_allclose(target, robot_reference)

    def test_vision_translation_is_rotated_from_waist_into_robot_root(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "anchored_wrist_target"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        vision_reference = np.eye(4)
        current_pose = np.eye(4)
        current_pose[0, 3] = 0.1
        robot_reference = np.eye(4)
        waist_to_root = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        target = namespace["anchored_wrist_target"](
            current_pose,
            vision_reference,
            robot_reference,
            waist_to_root,
            1.0,
        )
        np.testing.assert_allclose(target[:3, 3], [0.0, 0.1, 0.0], atol=1e-12)

    def test_vision_rotation_is_conjugated_into_robot_root(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "anchored_wrist_target"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        angle = 0.2
        vision_reference = np.eye(4)
        current_pose = np.eye(4)
        current_pose[:3, :3] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ])
        waist_to_root = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        target = namespace["anchored_wrist_target"](
            current_pose,
            vision_reference,
            np.eye(4),
            waist_to_root,
            1.0,
        )
        expected = waist_to_root @ current_pose[:3, :3] @ waist_to_root.T
        np.testing.assert_allclose(target[:3, :3], expected, atol=1e-12)

    def test_current_head_yaw_is_reexpressed_in_activation_heading(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"head_yaw_rotation", "wrist_in_reference_head_yaw_frame"}
        ]
        namespace = {"np": np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        current_head = np.eye(4)
        current_head[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        world_wrist = np.eye(4)
        world_wrist[:3, 3] = [0.5, 0.2, 0.7]
        waist_origin_offset = np.array([0.15, 0.0, 0.45])
        dynamic_wrist = np.eye(4)
        dynamic_wrist[:3, :3] = current_head[:3, :3].T @ world_wrist[:3, :3]
        dynamic_wrist[:3, 3] = (
            current_head[:3, :3].T @ (world_wrist[:3, 3] - current_head[:3, 3])
            + waist_origin_offset
        )

        fixed = namespace["wrist_in_reference_head_yaw_frame"](
            dynamic_wrist,
            current_head,
            np.eye(3),
        )
        np.testing.assert_allclose(fixed[:3, :3], world_wrist[:3, :3], atol=1e-12)
        np.testing.assert_allclose(
            fixed[:3, 3],
            world_wrist[:3, 3] + waist_origin_offset,
            atol=1e-12,
        )

    def test_fixed_world_wrist_is_independent_of_current_head_yaw(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"head_yaw_rotation", "wrist_in_reference_head_yaw_frame"}
        ]
        namespace = {"np": np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(MAIN_PATH), "exec"), namespace)

        def yaw(angle):
            return np.array([
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ])

        reference_head_yaw = yaw(0.4)
        current_head = np.eye(4)
        current_head[:3, :3] = yaw(-0.7)
        current_head[:3, 3] = [0.3, -0.2, 1.6]
        world_wrist = np.eye(4)
        world_wrist[:3, :3] = yaw(0.15)
        world_wrist[:3, 3] = [0.65, 0.1, 1.2]
        waist_origin_offset = np.array([0.15, 0.0, 0.45])
        dynamic_wrist = np.eye(4)
        dynamic_wrist[:3, :3] = current_head[:3, :3].T @ world_wrist[:3, :3]
        dynamic_wrist[:3, 3] = (
            current_head[:3, :3].T
            @ (world_wrist[:3, 3] - current_head[:3, 3])
            + waist_origin_offset
        )

        fixed = namespace["wrist_in_reference_head_yaw_frame"](
            dynamic_wrist,
            current_head,
            reference_head_yaw,
        )
        expected_rotation = reference_head_yaw.T @ world_wrist[:3, :3]
        expected_position = (
            reference_head_yaw.T @ (world_wrist[:3, 3] - current_head[:3, 3])
            + waist_origin_offset
        )
        np.testing.assert_allclose(fixed[:3, :3], expected_rotation, atol=1e-12)
        np.testing.assert_allclose(fixed[:3, 3], expected_position, atol=1e-12)

    def test_relative_rotation_left_multiplies_nonidentity_robot_reference(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "anchored_wrist_target"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        angle = 0.25
        relative_rotation = np.array([
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ])
        robot_rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        current_pose = np.eye(4)
        current_pose[:3, :3] = relative_rotation
        robot_reference = np.eye(4)
        robot_reference[:3, :3] = robot_rotation

        target = namespace["anchored_wrist_target"](
            current_pose,
            np.eye(4),
            robot_reference,
            np.eye(3),
            1.0,
        )
        np.testing.assert_allclose(
            target[:3, :3],
            relative_rotation @ robot_rotation,
            atol=1e-12,
        )

    def test_r1_ik_keeps_urdf_joint_bounds_and_biases_to_current_pose(self):
        source = IK_PATH.read_text(encoding="utf-8")
        r1_source = source[source.index("class R1_A7_ArmIK:") :]
        self.assertIn("self.opti.bounded(", r1_source)
        self.assertIn(
            "self.regularization_cost = casadi.sumsqr(self.var_q - self.var_q_last)",
            r1_source,
        )
        self.assertIn(
            "self.reduced_robot.data = self.reduced_robot.model.createData()",
            r1_source,
        )
        self.assertIn("if raise_on_failure:", r1_source)

    def test_r1_ready_state_cannot_swallow_the_first_r_request(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        ready_block = source[source.index("if r1_a7_anchored:\n            START = False") :]
        self.assertLess(
            ready_block.index("r1_arm_request_floor = ARM_REQUEST_GENERATION"),
            ready_block.index("READY = True"),
        )
        self.assertIn("R1_A7_DEFERRED_REAL_MODE and not READY", source)

    def test_recenter_precedes_ik_and_reference_capture(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        activation = source[source.index("r1_arm_request_floor = ARM_REQUEST_GENERATION") :]
        debug_index = activation.index("motion_switcher.Enter_Debug_Mode()")
        activate_index = activation.index("arm_ctrl.activate()")
        post_recenter_index = activation.index("post_recenter_motor_q =")
        waist_index = activation.index("r1_waist_yaw_reference =")
        ik_index = activation.index("arm_ik = R1_A7_ArmIK")
        fresh_index = activation.index("reference_tele_data = wait_for_new_fresh_motion_data")
        robot_reference_index = activation.index("r1_robot_left_reference,")
        vision_reference_index = activation.index("r1_vision_left_reference =")
        self.assertLess(debug_index, activate_index)
        self.assertLess(activate_index, post_recenter_index)
        self.assertLess(post_recenter_index, waist_index)
        self.assertLess(waist_index, ik_index)
        self.assertLess(ik_index, fresh_index)
        self.assertLess(fresh_index, robot_reference_index)
        self.assertLess(robot_reference_index, vision_reference_index)
        self.assertIn(
            "R1_A7_ArmIK(waist_yaw=r1_waist_yaw_reference)",
            activation,
        )

    def test_o6_waits_before_arm_activation_and_enables_after_reference_capture(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        activation = source[source.index("r1_arm_request_floor = ARM_REQUEST_GENERATION") :]
        self.assertLess(
            activation.index("hand_ctrl.wait_until_ready(timeout=3.0)"),
            activation.index("motion_switcher.Enter_Debug_Mode()"),
        )
        self.assertLess(
            activation.index("r1_head_yaw_reference ="),
            activation.index("hand_ctrl.activate()"),
        )
        self.assertLess(
            activation.index("hand_ctrl.activate()"),
            activation.index("START = True"),
        )

    def test_o6_command_is_not_published_before_arm_ik_and_command(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        tracking = source[source.index("# main loop. robot start to follow VR user's motion") :]
        solve = tracking.index("arm_ik.solve_ik(")
        hand_update = tracking.index("hand_ctrl.update(left_hand_target, right_hand_target)")
        arm_update = tracking.index("arm_ctrl.ctrl_dual_arm_and_head(sol_q, sol_tauff, head_q_target)")
        self.assertLess(
            solve,
            hand_update,
        )
        self.assertLess(
            hand_update,
            arm_update,
        )
        self.assertIn("if STOP:", tracking[solve:hand_update])
        self.assertIn("if STOP:", tracking[hand_update:arm_update])
        stale = tracking[tracking.index("if r1_a7_anchored:") :]
        stale_guard = stale[:stale.index('if args.ee == "linker_o6":')]
        self.assertIn("tele_data = last_fresh_tele_data", stale_guard)
        self.assertNotIn("hand_ctrl.stop()", stale_guard)
        self.assertNotIn("raise RuntimeError", stale_guard)

    def test_o6_is_stopped_before_arm_in_separate_cleanup_blocks(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        cleanup = source[source.index("finally:", source.index("if __name__ == '__main__':")) :]
        hand_stop = cleanup.index("hand_ctrl.stop()")
        arm_stop = cleanup.index("arm_ctrl.stop()")
        self.assertLess(hand_stop, arm_stop)
        self.assertIn("except Exception as e:", cleanup[hand_stop:arm_stop])

    def test_post_activation_wait_retries_cached_snapshot_until_new_fresh_sample(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"is_fresh_motion_data", "wait_for_new_fresh_motion_data"}
        ]
        namespace = {"time": time}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(MAIN_PATH), "exec"), namespace)

        floor = time.monotonic() - 0.01
        cached = types.SimpleNamespace(motion_data_ready=True, motion_data_timestamp=floor)
        fresh = types.SimpleNamespace(motion_data_ready=True, motion_data_timestamp=time.monotonic())

        class FakeWrapper:
            def __init__(self):
                self.samples = iter((cached, cached, fresh))

            def get_tele_data(self):
                return next(self.samples)

        result = namespace["wait_for_new_fresh_motion_data"](
            FakeWrapper(),
            floor,
            0.05,
        )
        self.assertIs(result, fresh)

    def test_post_activation_wait_rejects_duplicate_sample(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"is_fresh_motion_data", "wait_for_new_fresh_motion_data"}
        ]
        namespace = {"time": time}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(MAIN_PATH), "exec"), namespace)

        floor = time.monotonic()
        duplicate = types.SimpleNamespace(motion_data_ready=True, motion_data_timestamp=floor)

        class FakeWrapper:
            def get_tele_data(self):
                return duplicate

        result = namespace["wait_for_new_fresh_motion_data"](
            FakeWrapper(),
            floor,
            0.01,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
