import math
from types import SimpleNamespace
import unittest

from tools.check_r1_teleop import StreamWindow, validate_camera_config, validate_state


class StreamWindowTest(unittest.TestCase):
    def test_live_static_posture_is_valid_but_stopped_stream_is_not(self):
        window = StreamWindow(0.25)
        for tick in range(21):
            window.observe(tick * 0.05)
        self.assertIsNone(window.problem(1.05))
        self.assertIn("stale", window.problem(1.35))

    def test_gap_requires_a_new_continuous_window(self):
        window = StreamWindow(0.25)
        for index in range(21):
            window.observe(index * 0.05)
        window.observe(1.5)
        self.assertIsNotNone(window.problem(1.5))

    def test_invalid_feedback_cannot_inherit_old_good_window(self):
        window = StreamWindow(0.25)
        for index in range(21):
            window.observe(index * 0.05)
        window.observe(1.05, error="bad CRC")
        self.assertEqual(window.problem(1.05), "bad CRC")
        window.observe(1.1)
        self.assertIsNotNone(window.problem(1.1))

    def test_missing_feedback_fails(self):
        self.assertIn("no valid", StreamWindow(0.25).problem(1.0))


class FeedbackValidationTest(unittest.TestCase):
    def test_lowstate_rejects_bad_dimensions_and_nonfinite_velocity(self):
        message = SimpleNamespace(motor_state=[SimpleNamespace(q=0.0, dq=0.0) for _ in range(35)])
        validate_state(message)
        message.motor_state[12].dq = math.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_state(message)
        message.motor_state.pop()
        with self.assertRaisesRegex(ValueError, "35"):
            validate_state(message)

    def test_hand_axis_values_are_checked_without_optional_message_fields(self):
        message = SimpleNamespace(states=[SimpleNamespace(q=0.5) for _ in range(6)])
        validate_state(message, hand=True)
        message.states[0].q = 1.5
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            validate_state(message, hand=True)

    def test_monocular_or_wrong_size_config_cannot_pass(self):
        head = {
            "type": "rtp_h264_stereo", "image_shape": [448, 1088], "binocular": True,
            "enable_zmq": True, "zmq_port": 55555, "enable_webrtc": True,
            "webrtc_port": 60001, "left_rtp_port": 5002, "right_rtp_port": 5003,
        }
        validate_camera_config({"head_camera": head})
        head["binocular"] = False
        with self.assertRaisesRegex(ValueError, "binocular"):
            validate_camera_config({"head_camera": head})
        head["binocular"] = True
        head["image_shape"] = [448, 544]
        with self.assertRaisesRegex(ValueError, "image_shape"):
            validate_camera_config({"head_camera": head})


if __name__ == "__main__":
    unittest.main()
