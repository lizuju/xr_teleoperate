import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from test_r1_a7_feedback_shutdown import load_controller, motor_state
from test_r1_a7_official_activation import FakeLowCmd


class R1A7FeedbackValidityTest(unittest.TestCase):
    def make_controller(self, *, simulation=False, initial=None):
        controller_class, namespace = load_controller()
        clock = SimpleNamespace(now=10.0)
        seed = motor_state() if initial is None else initial

        class Subscriber:
            controller = None

            def __init__(self, *args):
                self.closed = False

            def Init(self, handler, queue_len):
                self.handler = handler
                type(self).controller = handler.__self__
                handler(copy.deepcopy(seed))

            def Close(self):
                self.closed = True

        def wait_for_dds(predicate, name):
            if not predicate():
                raise TimeoutError(name)

        namespace.update({"ChannelSubscriber": Subscriber,
                          "time": SimpleNamespace(monotonic=lambda: clock.now),
                          "wait_for_dds": wait_for_dds})
        return controller_class, namespace, clock, Subscriber, simulation

    def open_controller(self, **kwargs):
        controller_class, namespace, clock, subscriber, simulation = self.make_controller(**kwargs)
        controller = controller_class(deferred_activation=True, simulation_mode=simulation)
        self.addCleanup(controller.stop)
        return controller, namespace, clock

    def test_empty_controlled_motor_blocks_never_become_zero_joint_feedback(self):
        controller, _, clock = self.open_controller()
        previous = controller.lowstate_buffer.GetData()
        count = len(controller.recording_history)
        for index in (13, 18, 25, 26, 27, 28, 29, 30):
            with self.subTest(index=index):
                missing = motor_state()
                motor = missing.motor_state[index]
                motor.q = motor.dq = motor.tau_est = motor.vol = 0.0
                motor.temperature = [0, 0]
                clock.now += 0.001
                controller._subscribe_motor_state(missing)
                self.assertIs(controller.lowstate_buffer.GetData(), previous)
                self.assertEqual(controller.lowstate_sequence, previous.sequence)
                self.assertEqual(len(controller.recording_history), count)
                self.assertEqual(previous.monotonic_timestamp, 10.0)
        self.assertEqual(controller.lowstate_rejected, {"empty_motor_block": 8})

    def test_alternating_complete_and_empty_frames_record_only_real_measurements(self):
        controller, _, clock = self.open_controller()
        initial_sequence = controller.lowstate_sequence
        for sequence in range(12):
            complete = motor_state()
            complete.tick = sequence + 1000
            complete.motor_state[28].q = -0.7366 + sequence * 0.0001
            clock.now += 0.002
            controller._subscribe_motor_state(complete)
            accepted = controller.lowstate_buffer.GetData()
            missing = copy.deepcopy(complete)
            motor = missing.motor_state[28]
            motor.q = motor.dq = motor.tau_est = motor.vol = 0.0
            motor.temperature = [0, 0]
            clock.now += 0.001
            controller._subscribe_motor_state(missing)
            self.assertIs(controller.lowstate_buffer.GetData(), accepted)
            self.assertEqual(accepted.tick, complete.tick)
            self.assertEqual(accepted.sequence, initial_sequence + sequence + 1)
        self.assertEqual(len(controller.recording_history), 13)
        recorded = list(controller.recording_history)[1:]
        np.testing.assert_allclose([state.motor_state[28].q for state in recorded],
                                   [-0.7366 + sequence * 0.0001 for sequence in range(12)])
        self.assertTrue(all(state.motor_state[28].q != 0.0 for state in recorded))
        self.assertEqual(controller.lowstate_rejected, {"empty_motor_block": 12})

    def test_actual_zero_neutral_pose_with_live_voltage_or_temperature_is_valid(self):
        for voltage, temperature in ((38.0, [30, 31]), (38.0, [0, 0]), (0.0, [30, 31])):
            with self.subTest(voltage=voltage, temperature=temperature):
                controller, _, clock = self.open_controller()
                neutral = motor_state()
                for index in (13, *range(15, 31)):
                    motor = neutral.motor_state[index]
                    motor.q = motor.dq = motor.tau_est = 0.0
                    motor.vol = voltage
                    motor.temperature = temperature
                clock.now += 0.004
                controller._subscribe_motor_state(neutral)
                self.assertEqual(controller.lowstate_sequence, 2)
                self.assertEqual(controller.lowstate_rejected, {})
                np.testing.assert_array_equal(controller.get_current_dual_arm_q(), np.zeros(14))

    def test_empty_unused_and_leg_telemetry_does_not_reject_complete_controlled_feedback(self):
        controller, _, clock = self.open_controller()
        feedback = motor_state()
        for index in (*range(13), 14, 31, 32, 33, 34):
            motor = feedback.motor_state[index]
            motor.q = motor.dq = motor.tau_est = motor.vol = 0.0
            motor.temperature = [0, 0]
        clock.now += 0.002
        controller._subscribe_motor_state(feedback)
        self.assertEqual(controller.lowstate_sequence, 2)
        self.assertEqual(controller.lowstate_rejected, {})

    def test_nonfinite_feedback_is_rejected_at_the_dds_boundary(self):
        for index in (0, 13, 18, 28, 30):
            for field in ("q", "dq", "tau_est", "vol"):
                for invalid in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(index=index, field=field, invalid=invalid):
                        controller, _, clock = self.open_controller()
                        previous = controller.lowstate_buffer.GetData()
                        feedback = motor_state()
                        setattr(feedback.motor_state[index], field, invalid)
                        clock.now += 0.005
                        controller._subscribe_motor_state(feedback)
                        self.assertIs(controller.lowstate_buffer.GetData(), previous)
                        self.assertEqual(controller.lowstate_sequence, 1)
                        self.assertEqual(len(controller.recording_history), 1)
                        self.assertEqual(controller.lowstate_rejected, {"nonfinite": 1})

    def test_simulation_accepts_empty_telemetry_but_still_requires_finite_values(self):
        controller, _, clock = self.open_controller(simulation=True)
        feedback = motor_state()
        for motor in feedback.motor_state:
            motor.q = motor.dq = motor.tau_est = motor.vol = 0.0
            motor.temperature = [0, 0]
        clock.now += 0.005
        controller._subscribe_motor_state(feedback)
        self.assertEqual(controller.lowstate_sequence, 2)
        self.assertEqual(controller.lowstate_rejected, {})
        feedback.motor_state[28].q = float("nan")
        controller._subscribe_motor_state(feedback)
        self.assertEqual(controller.lowstate_sequence, 2)
        self.assertEqual(controller.lowstate_rejected, {"nonfinite": 1})

    def test_missing_frames_do_not_extend_the_existing_feedback_watchdog(self):
        controller, namespace, clock = self.open_controller()
        namespace.update({"ChannelPublisher": Mock(), "CRC": Mock(), "unitree_hg_msg_dds__LowCmd_": FakeLowCmd,
                          "hg_LowCmd": object, "kTopicLowCommand_Debug": "rt/lowcmd"})
        previous = controller.lowstate_buffer.GetData()
        feedback = motor_state()
        motor = feedback.motor_state[28]
        motor.q = motor.dq = motor.tau_est = motor.vol = 0.0
        motor.temperature = [0, 0]
        for _ in range(26):
            clock.now += 0.01
            controller._subscribe_motor_state(feedback)
        self.assertIs(controller.lowstate_buffer.GetData(), previous)
        self.assertTrue(controller.lowstate_sub_ready)
        with self.assertRaisesRegex(RuntimeError, "feedback is stale"):
            controller._get_fresh_lowstate()
        with self.assertRaisesRegex(RuntimeError, "feedback is stale"):
            controller.activate(home_head_waist=False)
        namespace["ChannelPublisher"].return_value.Write.assert_not_called()

    def test_only_empty_initial_feedback_cannot_become_ready_or_publish(self):
        initial = motor_state()
        motor = initial.motor_state[28]
        motor.q = motor.dq = motor.tau_est = motor.vol = 0.0
        motor.temperature = [0, 0]
        controller_class, namespace, _, subscriber, _ = self.make_controller(initial=initial)
        with self.assertRaises(TimeoutError):
            controller_class(deferred_activation=True)
        controller = subscriber.controller
        self.addCleanup(controller.stop)
        self.assertFalse(controller.lowstate_sub_ready)
        self.assertEqual(controller.lowstate_sequence, 0)
        self.assertEqual(len(controller.recording_history), 0)
        self.assertIsNone(controller.lowstate_buffer.GetData())
        self.assertEqual(controller.lowstate_rejected, {"empty_motor_block": 1})
        namespace["ChannelPublisher"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
