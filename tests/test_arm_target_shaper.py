import math
import unittest

import numpy as np
from unittest.mock import Mock

from teleop.robot_control.arm_target_shaper import ArmTargetShaper


ARMS = 14


class ArmTargetShaperTest(unittest.TestCase):
    def test_first_target_is_adopted_instead_of_ramped(self):
        shaper = ArmTargetShaper(velocity_limit=4.0, accel_limit=40.0, dof=ARMS)
        target = np.full(ARMS, 0.5)
        shaped = shaper.shape(target, now=100.0)
        np.testing.assert_allclose(shaped, target)

    def test_a_stale_sample_is_stretched_instead_of_jumping(self):
        # The measured failure: 84.5 deg of joint motion arriving in one 25 ms tick.
        shaper = ArmTargetShaper(velocity_limit=4.0, accel_limit=400.0, dof=ARMS)
        shaper.shape(np.zeros(ARMS), now=0.0)
        shaped = shaper.shape(np.full(ARMS, math.radians(84.5)), now=0.025)
        step = float(np.max(np.abs(shaped)))
        # Nothing may move faster than the limit, whatever the target does.
        self.assertLessEqual(step, 4.0 * 0.025 + 1e-9)

    def test_reference_converges_on_a_held_target(self):
        shaper = ArmTargetShaper(velocity_limit=2.0, accel_limit=200.0, dof=ARMS)
        target = np.full(ARMS, 1.0)
        shaper.shape(np.zeros(ARMS), now=0.0)
        now = 0.0
        for _ in range(200):
            now += 0.025
            shaped = shaper.shape(target, now=now)
        np.testing.assert_allclose(shaped, target, atol=1e-6)

    def test_speed_never_exceeds_the_limit(self):
        rng = np.random.default_rng(7)
        shaper = ArmTargetShaper(velocity_limit=3.0, accel_limit=1e9, dof=ARMS)
        shaper.shape(np.zeros(ARMS), now=0.0)
        previous = np.zeros(ARMS)
        now = 0.0
        for _ in range(400):
            now += 0.01
            target = rng.uniform(-2.0, 2.0, ARMS)
            shaped = shaper.shape(target, now=now)
            speed = np.max(np.abs(shaped - previous)) / 0.01
            self.assertLessEqual(speed, 3.0 + 1e-6)
            previous = shaped

    def test_zero_accel_limit_means_no_acceleration_limit(self):
        shaper = ArmTargetShaper(velocity_limit=8.0, accel_limit=0.0, dof=ARMS)
        shaper.shape(np.zeros(ARMS), now=0.0)
        # 8 rad/s reached immediately, not frozen at zero.
        shaped = shaper.shape(np.full(ARMS, 5.0), now=0.05)
        self.assertAlmostEqual(float(np.max(np.abs(shaped))), 8.0 * 0.05, places=9)

    def test_acceleration_limit_softens_the_start(self):
        hard = ArmTargetShaper(velocity_limit=8.0, accel_limit=0.0, dof=ARMS)
        soft = ArmTargetShaper(velocity_limit=8.0, accel_limit=1.0, dof=ARMS)
        target = np.full(ARMS, 5.0)
        hard.shape(np.zeros(ARMS), now=0.0)
        soft.shape(np.zeros(ARMS), now=0.0)
        hard_step = hard.shape(target, now=0.05)
        soft_step = soft.shape(target, now=0.05)
        self.assertLess(float(np.max(np.abs(soft_step))), float(np.max(np.abs(hard_step))))

    def test_disabled_shaper_passes_the_target_through(self):
        shaper = ArmTargetShaper(velocity_limit=0.0, accel_limit=40.0, dof=ARMS)
        self.assertFalse(shaper.enabled)
        shaper.shape(np.zeros(ARMS), now=0.0)
        target = np.full(ARMS, 9.0)
        np.testing.assert_allclose(shaper.shape(target, now=0.001), target)

    def test_a_stalled_loop_cannot_produce_an_unbounded_step(self):
        shaper = ArmTargetShaper(velocity_limit=4.0, accel_limit=1e9, dof=ARMS)
        shaper.shape(np.zeros(ARMS), now=0.0)
        shaped = shaper.shape(np.full(ARMS, 50.0), now=5.0)
        self.assertLessEqual(float(np.max(np.abs(shaped))), 4.0 * shaper.MAX_TICK_S + 1e-9)

    def test_reset_without_a_reference_keeps_the_reached_pose(self):
        shaper = ArmTargetShaper(velocity_limit=1.0, accel_limit=1e9, dof=ARMS)
        shaper.shape(np.zeros(ARMS), now=0.0)
        shaped = shaper.shape(np.full(ARMS, 10.0), now=0.1)
        reached = shaped.copy()
        self.assertGreater(float(np.max(reached)), 0.0)
        shaper.reset()
        after = shaper.shape(np.full(ARMS, 10.0), now=0.2)
        # It continues from where it was, it does not snap back or jump forward.
        self.assertLess(float(np.max(np.abs(after - reached))), 1.0 * 0.1 + 1e-9)

    def test_reset_with_a_reference_restarts_exactly_there(self):
        shaper = ArmTargetShaper(velocity_limit=1.0, accel_limit=1e9, dof=ARMS)
        shaper.shape(np.zeros(ARMS), now=0.0)
        shaper.reset(np.full(ARMS, -0.25))
        shaped = shaper.shape(np.full(ARMS, -0.25), now=0.01)
        np.testing.assert_allclose(shaped, np.full(ARMS, -0.25))

    def test_stats_track_how_often_the_target_outran_the_limit(self):
        shaper = ArmTargetShaper(velocity_limit=1.0, accel_limit=1e9, dof=ARMS)
        shaper.shape(np.zeros(ARMS), now=0.0)
        shaper.shape(np.full(ARMS, 0.001), now=0.01)      # inside the limit
        shaper.shape(np.full(ARMS, 2.0), now=0.02)        # far outside it
        stats = shaper.snapshot()
        self.assertEqual(stats["ticks"], 2)
        self.assertEqual(stats["limited_ticks"], 1)
        self.assertGreater(stats["max_residual_deg"], 0.0)
        self.assertLessEqual(stats["max_speed_rad_s"], 1.0 + 1e-9)

    def test_shape_rejects_bad_input(self):
        shaper = ArmTargetShaper(dof=ARMS)
        with self.assertRaises(ValueError):
            shaper.shape(np.zeros(ARMS - 1))
        with self.assertRaises(ValueError):
            shaper.shape(np.full(ARMS, np.nan))
        with self.assertRaises(ValueError):
            shaper.reset(np.zeros(ARMS + 1))

    def test_rejects_bad_limits(self):
        with self.assertRaises(ValueError):
            ArmTargetShaper(velocity_limit=-1.0)
        with self.assertRaises(ValueError):
            ArmTargetShaper(accel_limit=float("nan"))
        with self.assertRaises(ValueError):
            ArmTargetShaper(dof=0)


if __name__ == "__main__":
    unittest.main()


class ArmControllerShaperTest(unittest.TestCase):
    """The controller must publish a ramped reference, not the raw IK step."""

    def setUp(self):
        from test_r1_a7_official_activation import FakeCRC, FakeLowCmd, load_r1_controller_namespace

        self.now = 10.0
        self.namespace = load_r1_controller_namespace()
        self.controller = self.namespace["R1_A7_ArmController"](
            deferred_activation=True, target_velocity_limit=2.0, target_accel_limit=0.0,
        )
        self.controller.lowstate_subscriber.Close()
        self.namespace["time"] = __import__("types").SimpleNamespace(
            monotonic=lambda: self.now, sleep=lambda _: None)
        self.state = self.controller.lowstate_buffer.GetData()
        self.state.monotonic_timestamp = self.now
        self.state.sequence = 1
        for motor in self.state.motor_state:
            motor.q = 0.0
        self.controller.msg = FakeLowCmd()
        self.controller.crc = FakeCRC()
        self.controller.lowcmd_publisher = Mock()

    def tearDown(self):
        self.controller.stop()

    def submit(self, value):
        self.controller.ctrl_dual_arm_and_head(
            np.full(14, value), np.zeros(14), [0.0, 0.0])

    def test_a_large_ik_step_is_ramped_instead_of_published_whole(self):
        self.submit(0.0)                      # bootstrap adopts zero
        self.now += 0.025
        self.submit(1.0)                      # a full-radian IK step
        published = float(np.max(self.controller.q_target))
        self.assertLessEqual(published, 2.0 * 0.025 + 1e-9)
        self.assertGreater(published, 0.0)

    def test_snapshot_reports_the_limit_and_the_lag(self):
        self.submit(0.0)
        self.now += 0.025
        self.submit(1.0)
        snapshot = self.controller.get_target_shaper_snapshot()
        self.assertEqual(snapshot["velocity_limit_rad_s"], 2.0)
        self.assertEqual(snapshot["ticks"], 1)
        self.assertEqual(snapshot["limited_ticks"], 1)
        self.assertGreater(snapshot["max_residual_deg"], 45.0)

    def test_feedforward_follows_the_shaped_reference(self):
        self.submit(0.0)
        self.now += 0.025
        self.submit(1.0)
        speeds = np.abs(self.controller._dq_feedforward)
        self.assertLessEqual(float(np.max(speeds)), 2.0 + 1e-9)
        self.assertGreater(float(np.max(speeds)), 0.0)

    def test_reanchor_keeps_the_reached_pose(self):
        self.submit(0.0)
        self.now += 0.025
        self.submit(1.0)
        reached = self.controller.q_target.copy()
        self.controller.reset_target_shaper()
        np.testing.assert_allclose(self.controller.target_shaper.reference, reached)
        np.testing.assert_allclose(self.controller.target_shaper.velocity, np.zeros(14))

    def test_reference_getter_reports_what_the_servos_got(self):
        self.submit(0.0)
        self.now += 0.025
        self.submit(1.0)
        reference = self.controller.get_reference_q()
        self.assertEqual(reference.shape, (14,))
        self.assertLess(float(np.max(reference)), 1.0)

    def test_feedback_age_is_readable(self):
        self.controller.lowstate_buffer.SetData(self.state)
        self.state.monotonic_timestamp = self.now - 0.05
        self.assertAlmostEqual(self.controller.get_feedback_age(), 0.05, places=3)
