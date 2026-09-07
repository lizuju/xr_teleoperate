import threading
import types
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from test_r1_a7_official_activation import REPO_ROOT, FakeLowCmd, load_r1_controller_namespace


class R1A7WaistControllerTest(unittest.TestCase):
    def setUp(self):
        namespace = load_r1_controller_namespace()
        self.now = 10.0
        namespace["time"] = types.SimpleNamespace(
            time=lambda: self.now,
            monotonic=lambda: self.now,
            sleep=lambda _: None,
        )
        self.controller = namespace["R1_A7_ArmController"].__new__(
            namespace["R1_A7_ArmController"]
        )
        controller = self.controller
        controller.ctrl_lock = threading.Lock()
        controller.q_target = np.zeros(14)
        controller.tauff_target = np.zeros(14)
        controller.head_q_target = np.zeros(2)
        controller.waist_yaw_target = None
        controller.waist_target_updated_at = None
        controller.waist_target_sequence = 0
        controller.waist_yaw_limit = 2.618
        controller.waist_velocity_limit = 0.35
        controller.waist_tracking_error_limit = np.deg2rad(5.0)
        controller.waist_target_timeout = 0.25
        controller.waist_hold_requested = False
        controller.control_dt = 0.004
        controller.simulation_mode = True
        controller.msg = FakeLowCmd()
        controller.crc = types.SimpleNamespace(Crc=lambda _: 0)
        controller.lowcmd_publisher = types.SimpleNamespace(Write=self.record_command)
        self.lowstate = types.SimpleNamespace(
            motor_state=[types.SimpleNamespace(q=0.0, dq=0.0) for _ in range(35)],
            monotonic_timestamp=self.now,
        )
        controller.lowstate_buffer = types.SimpleNamespace(GetData=lambda: self.lowstate)
        self.commands = []

    def record_command(self, message):
        self.commands.append([command.q for command in message.motor_cmd])
        self.controller.publish_running = False
        return True

    def tick(self):
        self.controller.publish_running = True
        self.controller._ctrl_motor_state()
        return self.commands[-1][13]

    def submit(self, waist_yaw):
        self.controller.ctrl_dual_arm_and_head(
            np.zeros(14), np.zeros(14), [0.1, -0.2], waist_yaw_target=waist_yaw
        )

    def test_existing_call_keeps_waist_hold_unchanged(self):
        self.controller.msg.motor_cmd[13].q = 0.12
        self.submit(None)
        self.assertEqual(self.tick(), 0.12)
        self.assertIsNone(self.controller.waist_target_updated_at)

    def test_getter_uses_joint_13_feedback(self):
        self.lowstate.motor_state[12].q = 0.7
        self.lowstate.motor_state[13].q = 0.14
        self.assertEqual(self.controller.get_current_waist_yaw(), 0.14)

    def test_target_clamps_to_urdf_limit_and_preserves_head(self):
        for sign in (-1.0, 1.0):
            with self.subTest(sign=sign):
                self.submit(sign * 100.0)
                self.assertAlmostEqual(
                    self.controller.waist_yaw_target, sign * 2.618
                )
                np.testing.assert_allclose(self.controller.head_q_target, [0.1, -0.2])

    def test_initialized_angle_limit_matches_robot_mechanical_model(self):
        namespace = load_r1_controller_namespace()
        controller = namespace["R1_A7_ArmController"](deferred_activation=True)
        try:
            urdf = ET.parse(REPO_ROOT / "assets" / "r1" / "r1_a7.urdf")
            limit = urdf.find(".//joint[@name='waist_yaw_joint']/limit")
            self.assertEqual(controller.waist_yaw_limit, float(limit.get("upper")))
            self.assertEqual(-controller.waist_yaw_limit, float(limit.get("lower")))
        finally:
            controller.stop()

    def test_nonfinite_target_does_not_partially_update_targets(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.controller.ctrl_dual_arm_and_head(
                        np.ones(14), np.ones(14), [0.1, 0.2], invalid
                    )
                np.testing.assert_array_equal(self.controller.q_target, np.zeros(14))
                np.testing.assert_array_equal(self.controller.head_q_target, np.zeros(2))
                self.assertIsNone(self.controller.waist_yaw_target)

    def test_rate_limit_integrates_command_even_in_simulation(self):
        self.lowstate.motor_state[13].q = 0.1
        self.controller.msg.motor_cmd[13].q = 0.1
        self.submit(0.3)
        self.assertAlmostEqual(self.tick(), 0.1014)
        self.now += 0.004
        self.assertAlmostEqual(self.tick(), 0.1028)
        self.submit(-0.3)
        self.assertAlmostEqual(self.tick(), 0.1014)

    def test_stalled_feedback_limits_accumulated_command_error(self):
        for _ in range(120):
            self.lowstate.monotonic_timestamp = self.now
            self.submit(0.3)
            self.tick()
            self.now += 0.004
        self.assertAlmostEqual(self.commands[-1][13], np.deg2rad(5.0))

    def test_disjoint_speed_and_tracking_constraints_hold_feedback(self):
        self.lowstate.motor_state[13].q = 0.3
        self.controller.msg.motor_cmd[13].q = 0.0
        self.submit(0.5)
        self.assertAlmostEqual(self.tick(), 0.3)
        self.assertIsNone(self.controller.waist_yaw_target)

    def test_normal_following_preserves_rate_and_mechanical_bounds(self):
        self.lowstate.motor_state[13].q = 2.6175
        self.controller.msg.motor_cmd[13].q = 2.6175
        self.submit(10.0)
        previous = self.controller.msg.motor_cmd[13].q
        for _ in range(10):
            command = self.tick()
            self.assertLessEqual(abs(command - previous), 0.35 * 0.004 + 1e-12)
            self.assertLessEqual(abs(command), 2.618)
            previous = command
            self.now += 0.004

    def test_small_error_reaches_target_without_overshoot(self):
        self.lowstate.motor_state[13].q = 0.1
        self.controller.msg.motor_cmd[13].q = 0.1
        self.submit(0.1005)
        self.assertAlmostEqual(self.tick(), 0.1005)

    def test_timeout_holds_feedback_clears_target_and_can_resume(self):
        self.lowstate.motor_state[13].q = 0.1
        self.submit(0.3)
        self.now += 0.25
        self.assertAlmostEqual(self.tick(), 0.1)
        self.assertIsNone(self.controller.waist_yaw_target)
        self.now += 1.0
        self.assertAlmostEqual(self.tick(), 0.1)
        self.lowstate.monotonic_timestamp = self.now
        self.submit(-0.3)
        self.assertAlmostEqual(self.tick(), 0.0986)

    def test_arm_head_only_updates_do_not_renew_waist_timeout(self):
        self.lowstate.motor_state[13].q = 0.1
        self.submit(0.3)
        self.now += 0.25
        self.submit(None)
        self.assertAlmostEqual(self.tick(), 0.1)
        self.assertIsNone(self.controller.waist_yaw_target)

    def test_invalid_feedback_holds_last_command_and_clears_target(self):
        self.controller.msg.motor_cmd[13].q = 0.07
        self.submit(0.3)
        self.lowstate.motor_state[13].q = np.nan
        with self.assertRaises(RuntimeError):
            self.controller.get_current_waist_yaw()
        self.assertAlmostEqual(self.tick(), 0.07)
        self.assertIsNone(self.controller.waist_yaw_target)

    def test_stale_feedback_holds_last_command_and_clears_target(self):
        self.controller.msg.motor_cmd[13].q = 0.07
        self.now += 0.251
        self.submit(0.3)
        with self.assertRaises(RuntimeError):
            self.controller.get_current_waist_yaw()
        self.assertAlmostEqual(self.tick(), 0.07)
        self.assertIsNone(self.controller.waist_yaw_target)

    def test_out_of_model_feedback_cannot_push_command_outside_mechanical_limit(self):
        self.controller.msg.motor_cmd[13].q = 2.6
        self.lowstate.motor_state[13].q = 2.8
        self.submit(2.618)
        with self.assertRaises(RuntimeError):
            self.controller.get_current_waist_yaw()
        self.assertAlmostEqual(self.tick(), 2.6)
        self.assertIsNone(self.controller.waist_yaw_target)

    def test_expired_snapshot_cannot_clear_new_target_with_same_timestamp(self):
        self.submit(0.3)
        self.controller.hold_waist()
        original_getter = self.controller.get_current_waist_yaw

        def get_feedback_and_receive_new_sample():
            self.submit(-0.3)
            self.controller.get_current_waist_yaw = original_getter
            return original_getter()

        self.controller.get_current_waist_yaw = get_feedback_and_receive_new_sample
        self.tick()
        self.assertEqual(self.controller.waist_yaw_target, -0.3)
        self.assertFalse(self.controller.waist_hold_requested)
        self.assertAlmostEqual(self.tick(), -0.0014)

    def test_explicit_hold_freezes_actual_feedback_next_cycle(self):
        self.lowstate.motor_state[13].q = 0.1
        self.submit(0.3)
        self.controller.hold_waist()
        self.assertAlmostEqual(self.tick(), 0.1)
        self.assertIsNone(self.controller.waist_yaw_target)
        self.assertFalse(self.controller.waist_hold_requested)
        self.submit(0.3)
        self.assertAlmostEqual(self.tick(), 0.1014)


if __name__ == "__main__":
    unittest.main()
