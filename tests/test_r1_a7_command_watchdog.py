import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from test_r1_a7_official_activation import (
    FakeCRC, FakeLowCmd, FakePublisher, load_r1_controller_namespace,
)


class R1A7CommandWatchdogTest(unittest.TestCase):
    def setUp(self):
        self.namespace = load_r1_controller_namespace()
        self.controller = self.namespace["R1_A7_ArmController"](
            deferred_activation=True, simulation_mode=True,
        )
        self.controller.lowstate_subscriber.Close()
        self.now = 10.0
        self.namespace["time"] = SimpleNamespace(
            monotonic=lambda: self.now, sleep=lambda _: None,
        )
        self.state = self.controller.lowstate_buffer.GetData()
        self.state.monotonic_timestamp = self.now
        self.controller.msg = FakeLowCmd()
        self.controller.crc = FakeCRC()
        self.writes = []
        self.controller.lowcmd_publisher = Mock(Write=self.write_once)

    def tearDown(self):
        self.controller.stop()

    def write_once(self, message):
        self.writes.append([command.q for command in message.motor_cmd])
        self.controller.publish_running = False
        return True

    def tick(self):
        self.controller.publish_running = True
        self.controller._ctrl_motor_state()

    def submit(self):
        self.controller.ctrl_dual_arm_and_head(
            np.full(14, 0.2), np.full(14, 0.3), [0.1, -0.2],
        )

    def test_every_feedback_getter_rejects_stale_feedback(self):
        self.now += self.controller.feedback_timeout + 0.001
        for getter in (
            "get_mode_machine", "get_current_motor_q", "get_current_dual_arm_q",
            "get_current_dual_arm_dq", "get_current_head_q", "get_current_waist_yaw",
        ):
            with self.subTest(getter=getter):
                with self.assertRaisesRegex(RuntimeError, "feedback is stale"):
                    getattr(self.controller, getter)()

    def test_default_arm_head_mode_stops_all_writes_when_feedback_disappears(self):
        self.submit()
        self.tick()
        self.now += 0.251
        self.tick()
        self.assertEqual(len(self.writes), 1)
        self.assertFalse(self.controller.publish_running)
        with self.assertRaisesRegex(RuntimeError, "feedback is stale"):
            self.controller.raise_if_failed()

    def test_live_feedback_cannot_keep_an_old_target_alive_during_main_loop_stall(self):
        self.submit()

        def publish_and_advance(message):
            self.writes.append([command.q for command in message.motor_cmd])
            self.now += 0.1
            self.state.monotonic_timestamp = self.now
            return True

        self.controller.lowcmd_publisher.Write = publish_and_advance
        self.tick()
        self.assertGreaterEqual(len(self.writes), 5)
        self.assertLessEqual(len(self.writes), 6)
        self.assertFalse(self.controller.publish_running)
        with self.assertRaisesRegex(RuntimeError, "arm/head target expired"):
            self.controller.raise_if_failed()
        np.testing.assert_allclose(self.controller.q_target, 0.2)
        self.controller.lowcmd_publisher.Close.assert_not_called()

    def test_startup_live_pose_hold_allows_ik_loading_before_first_target(self):
        self.now += 30.0
        self.state.monotonic_timestamp = self.now
        self.tick()
        self.assertEqual(len(self.writes), 1)
        self.assertIsNone(self.controller.target_updated_at)
        self.controller.raise_if_failed()

    def test_startup_hold_snapshot_cannot_bypass_a_first_target_that_arrives_then_expires(self):
        def accept_target_then_stall(message):
            self.submit()
            self.now += 0.5
            self.state.monotonic_timestamp = self.now
            return 0

        self.controller.crc.Crc = accept_target_then_stall
        self.tick()
        self.assertEqual(self.writes, [])
        self.assertFalse(self.controller.publish_running)
        with self.assertRaisesRegex(RuntimeError, "target expired"):
            self.controller.raise_if_failed()

    def test_explicit_tracking_hold_renews_target_without_renewing_waist_or_changing_pose(self):
        self.controller.ctrl_dual_arm_and_head(
            np.full(14, 0.2), np.full(14, 0.3), [0.1, -0.2], waist_yaw_target=0.4,
        )
        waist_timestamp = self.controller.waist_target_updated_at
        for _ in range(10):
            self.now += 0.2
            self.state.monotonic_timestamp = self.now
            self.controller.hold_targets()
        self.assertEqual(self.controller.target_updated_at, self.now)
        self.assertEqual(self.controller.waist_target_updated_at, waist_timestamp)
        np.testing.assert_allclose(self.controller.q_target, 0.2)
        np.testing.assert_allclose(self.controller.tauff_target, 0.3)
        np.testing.assert_allclose(self.controller.head_q_target, [0.1, -0.2])

    def test_late_target_or_hold_cannot_revive_expired_command_even_before_worker_runs(self):
        self.submit()
        self.now += 0.5
        self.state.monotonic_timestamp = self.now
        for update in (
            self.submit,
            lambda: self.controller.ctrl_dual_arm(np.zeros(14), np.zeros(14)),
            self.controller.hold_targets,
        ):
            with self.subTest(update=update):
                with self.assertRaisesRegex(RuntimeError, "target expired"):
                    update()
        self.tick()
        self.assertEqual(self.writes, [])

    def test_explicit_hold_still_requires_fresh_feedback(self):
        self.submit()
        self.now += 0.251
        with self.assertRaisesRegex(RuntimeError, "feedback is stale"):
            self.controller.hold_targets()
        self.assertEqual(self.controller.target_updated_at, 10.0)

    def test_feedback_is_checked_after_command_build_immediately_before_write(self):
        self.submit()

        def slow_crc(message):
            self.now += 0.251
            return 0

        self.controller.crc.Crc = slow_crc
        self.tick()
        self.assertEqual(self.writes, [])
        with self.assertRaisesRegex(RuntimeError, "feedback is stale"):
            self.controller.raise_if_failed()

    def test_write_failure_is_reported_to_main_and_cannot_be_resumed(self):
        self.submit()
        self.controller.lowcmd_publisher.Write = Mock(return_value=False)
        self.tick()
        self.controller.lowcmd_publisher.Write.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "Write failed"):
            self.controller.raise_if_failed()
        self.state.monotonic_timestamp = self.now
        with self.assertRaisesRegex(RuntimeError, "Write failed"):
            self.submit()
        self.tick()
        self.controller.lowcmd_publisher.Write.assert_called_once()

    def test_unexpected_worker_exception_is_reported_with_its_cause(self):
        cause = ValueError("transport disconnected")
        self.controller.lowcmd_publisher.Write = Mock(side_effect=cause)
        self.tick()
        with self.assertRaisesRegex(RuntimeError, "transport disconnected") as caught:
            self.controller.raise_if_failed()
        self.assertIs(caught.exception.__cause__, cause)


class R1A7CancelableActivationTest(unittest.TestCase):
    def setUp(self):
        FakePublisher.instances.clear()
        self.namespace = load_r1_controller_namespace()
        self.controller = self.namespace["R1_A7_ArmController"](deferred_activation=True)

    def tearDown(self):
        self.controller.stop()

    def test_cancel_before_activation_creates_no_command_writer(self):
        with self.assertRaises(InterruptedError):
            self.controller.activate(cancel_requested=lambda: True)
        self.assertEqual(FakePublisher.instances, [])

    def test_cancel_during_recenter_prevents_the_next_write_and_background_publisher(self):
        with self.assertRaises(InterruptedError):
            self.controller.activate(cancel_requested=lambda: (
                bool(FakePublisher.instances) and len(FakePublisher.instances[0].writes) >= 4
            ))
        self.assertEqual(len(FakePublisher.instances[0].writes), 4)
        last = FakePublisher.instances[0].writes[-1]
        self.assertGreater(last[13], 0.0)
        self.assertGreater(last[29], 0.0)
        self.assertIsNone(self.controller.publish_thread)
        self.assertFalse(self.controller.active)

    def test_cancel_during_activation_feedback_wait_prevents_initial_write(self):
        self.controller.lowstate_subscriber.Close()
        checks = [0]

        def cancelled():
            checks[0] += 1
            return checks[0] >= 3

        with self.assertRaises(InterruptedError):
            self.controller.activate(cancel_requested=cancelled)
        self.assertEqual(FakePublisher.instances[0].writes, [])

    def test_failed_recenter_write_aborts_without_starting_publisher(self):
        original = FakePublisher.Write
        self.addCleanup(setattr, FakePublisher, "Write", original)

        def fail_second_write(publisher, message):
            result = original(publisher, message)
            return False if len(publisher.writes) == 2 else result

        FakePublisher.Write = fail_second_write
        with self.assertRaisesRegex(RuntimeError, "Write failed"):
            self.controller.activate()
        self.assertEqual(len(FakePublisher.instances[0].writes), 2)
        self.assertIsNone(self.controller.publish_thread)
        self.assertFalse(self.controller.active)

    def test_feedback_loss_during_recenter_aborts_without_zeroing_remaining_motion(self):
        original = FakePublisher.Write
        self.addCleanup(setattr, FakePublisher, "Write", original)

        def lose_feedback(publisher, message):
            result = original(publisher, message)
            if len(publisher.writes) == 3:
                self.controller.lowstate_subscriber.Close()
                self.controller.lowstate_buffer.GetData().monotonic_timestamp = time.monotonic() - 1.0
            return result

        FakePublisher.Write = lose_feedback
        with self.assertRaisesRegex(RuntimeError, "feedback is stale"):
            self.controller.activate()
        self.assertEqual(len(FakePublisher.instances[0].writes), 3)
        self.assertGreater(FakePublisher.instances[0].writes[-1][13], 0.0)
        self.assertIsNone(self.controller.publish_thread)


if __name__ == "__main__":
    unittest.main()


class R1A7VelocityFeedforwardTest(unittest.TestCase):
    """The servo is fed the target's own velocity instead of dq=0.

    Measured 2026-09-15: with dq=0 the arm delivered only 56-75% of the commanded
    joint speed even at the best lag, which is what the operator felt as stutter.
    """

    def setUp(self):
        self.namespace = load_r1_controller_namespace()
        self.controller = self.namespace["R1_A7_ArmController"](
            deferred_activation=True, simulation_mode=True,
        )
        self.controller.lowstate_subscriber.Close()
        self.now = 10.0
        self.namespace["time"] = SimpleNamespace(
            monotonic=lambda: self.now, sleep=lambda _: None,
        )
        self.state = self.controller.lowstate_buffer.GetData()
        self.state.monotonic_timestamp = self.now
        self.controller.msg = FakeLowCmd()
        self.controller.crc = FakeCRC()
        self.published = []
        self.controller.lowcmd_publisher = Mock(Write=self.capture)

    def tearDown(self):
        self.controller.stop()

    def capture(self, message):
        self.published.append(
            (np.array([message.motor_cmd[i].q for i in range(15, 29)], dtype=float),
             np.array([message.motor_cmd[i].dq for i in range(15, 29)], dtype=float))
        )
        self.controller.publish_running = False
        return True

    def tick(self):
        self.controller.publish_running = True
        self.controller._ctrl_motor_state()

    def submit(self, q):
        self.controller.ctrl_dual_arm_and_head(np.full(14, q), np.zeros(14), [0.0, 0.0])

    def test_target_velocity_is_published_in_dq(self):
        self.controller.dq_feedforward_filter = 1.0            # no smoothing for the assertion
        self.submit(0.0)
        self.now += 0.025
        self.submit(0.05)                                      # 0.05 rad / 0.025 s = 2 rad/s
        self.tick()
        _, dq = self.published[-1]
        np.testing.assert_allclose(dq, np.full(14, 2.0), rtol=1e-6,
                                   err_msg="the commanded joint speed must reach the servo as dq")

    def test_feedforward_is_clamped_and_low_passed(self):
        self.controller.dq_feedforward_filter = 1.0
        self.controller.dq_feedforward_limit = 1.0
        self.submit(0.0)
        self.now += 0.025
        self.submit(0.25)                                      # 10 rad/s raw -> clamped to 1
        self.tick()
        _, dq = self.published[-1]
        np.testing.assert_allclose(dq, np.full(14, 1.0), rtol=1e-6)

    def test_feedforward_decays_to_zero_when_the_target_stops_arriving(self):
        self.controller.dq_feedforward_filter = 1.0
        self.submit(0.0)
        self.now += 0.025
        self.submit(0.05)
        self.tick()
        self.assertGreater(abs(self.published[-1][1][0]), 0.5)
        self.now += self.controller.dq_feedforward_timeout + 0.01
        for _ in range(40):
            self.now += 0.004
            self.state.monotonic_timestamp = self.now     # feedback keeps arriving
            self.tick()
        _, dq = self.published[-1]
        self.assertLess(abs(dq[0]), 0.02, "a hold must not keep the last commanded speed running")

    def test_feedforward_can_be_switched_off(self):
        """--arm-dq-feedforward off must restore the old dq=0 command exactly."""
        controller = self.namespace["R1_A7_ArmController"](
            deferred_activation=True, simulation_mode=True, dq_feedforward=False,
        )
        controller.lowstate_subscriber.Close()
        controller.msg = FakeLowCmd()
        controller.crc = FakeCRC()
        state = controller.lowstate_buffer.GetData()
        state.monotonic_timestamp = self.now
        published = []

        def capture(message):
            published.append(np.array([message.motor_cmd[i].dq for i in range(15, 29)]))
            controller.publish_running = False
            return True

        controller.lowcmd_publisher = Mock(Write=capture)
        try:
            self.assertFalse(controller.dq_feedforward_enabled)
            controller.ctrl_dual_arm_and_head(np.full(14, 0.1), np.zeros(14), [0.0, 0.0])
            controller.publish_running = True
            controller._ctrl_motor_state()
            np.testing.assert_allclose(published[-1], np.zeros(14))
        finally:
            controller.publish_running = False
            controller.stop()
