import math
from types import SimpleNamespace
import unittest

from tools.check_r1_teleop import (
    StreamWindow,
    jpeg_decoder,
    jpeg_is_complete,
    unwrap_camera_config,
    validate_camera_config,
    validate_state,
    wrist_topics,
)


HEAD_CAMERA = {
    "type": "rtp_h264_stereo", "image_shape": [448, 1088], "binocular": True,
    "enable_zmq": True, "zmq_port": 55555, "enable_webrtc": True,
    "webrtc_port": 60001, "left_rtp_port": 5002, "right_rtp_port": 5003,
}


def wrist_camera(zmq_port=55556, webrtc_port=60002, zmq=True, webrtc=True):
    return {
        "type": "uvc", "image_shape": [480, 640], "binocular": False,
        "enable_zmq": zmq, "zmq_port": zmq_port,
        "enable_webrtc": webrtc, "webrtc_port": webrtc_port,
    }


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
        message = SimpleNamespace(motor_state=[SimpleNamespace(q=0.0, dq=0.0, tau_est=0.0) for _ in range(35)])
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


class JpegCompletenessTest(unittest.TestCase):
    """The wrist UVC modules emit most frames without their EOI marker."""

    def test_markers_are_detected(self):
        self.assertTrue(jpeg_is_complete(b"\xff\xd8payload\xff\xd9"))
        self.assertFalse(jpeg_is_complete(b"\xff\xd8payload"))
        self.assertFalse(jpeg_is_complete(b"payload\xff\xd9"))

    def test_lenient_decoder_is_used(self):
        # Either turbojpeg or the cv2 fallback, but always a callable that
        # returns None instead of raising on garbage.
        decode = jpeg_decoder()
        self.assertIsNone(decode(b"not a jpeg"))


class CameraConfigEnvelopeTest(unittest.TestCase):
    def test_teleimager2_envelope_is_unwrapped(self):
        response = {"webrtc": {"bitrate": {}}, "camera": {"head_camera": HEAD_CAMERA}}
        self.assertEqual(unwrap_camera_config(response), {"head_camera": HEAD_CAMERA})
        # A flat roster (or an already unwrapped dict) passes through unchanged.
        self.assertEqual(unwrap_camera_config({"head_camera": HEAD_CAMERA}),
                         {"head_camera": HEAD_CAMERA})
        self.assertIsNone(unwrap_camera_config(None))

    def test_envelope_validation_reaches_the_head_camera(self):
        response = {"webrtc": {}, "camera": {"head_camera": HEAD_CAMERA,
                                             "left_wrist_camera": wrist_camera()}}
        validate_camera_config(unwrap_camera_config(response))


class WristCameraConfigTest(unittest.TestCase):
    def test_enabled_wrist_cameras_must_use_the_panel_ports(self):
        config = {"head_camera": HEAD_CAMERA, "left_wrist_camera": wrist_camera()}
        validate_camera_config(config)
        config["left_wrist_camera"]["webrtc_port"] = 61002
        with self.assertRaisesRegex(ValueError, "left_wrist_camera.webrtc_port"):
            validate_camera_config(config)

    def test_both_local_wrist_drivers_are_accepted(self):
        # Production moved from the libusb driver (uvc) to the kernel one (v4l2).
        for driver in ("uvc", "v4l2"):
            config = {"head_camera": HEAD_CAMERA,
                      "left_wrist_camera": wrist_camera(),
                      "right_wrist_camera": wrist_camera(zmq_port=55557, webrtc_port=60003)}
            config["left_wrist_camera"]["type"] = driver
            config["right_wrist_camera"]["type"] = driver
            validate_camera_config(config)

    def test_unknown_wrist_driver_is_rejected(self):
        config = {"head_camera": HEAD_CAMERA, "left_wrist_camera": wrist_camera()}
        config["left_wrist_camera"]["type"] = "realsense"
        with self.assertRaisesRegex(ValueError, "left_wrist_camera.type"):
            validate_camera_config(config)

    def test_wrist_image_shape_must_match_the_panel_aspect(self):
        config = {"head_camera": HEAD_CAMERA,
                  "right_wrist_camera": wrist_camera(zmq_port=55557, webrtc_port=60003)}
        config["right_wrist_camera"]["image_shape"] = [720, 1280]
        with self.assertRaisesRegex(ValueError, "right_wrist_camera.image_shape"):
            validate_camera_config(config)

    def test_disabled_wrist_cameras_are_not_validated(self):
        disabled = {"enable_zmq": False, "enable_webrtc": False, "image_shape": [0, 0], "type": "uvc"}
        validate_camera_config({"head_camera": HEAD_CAMERA,
                                "left_wrist_camera": disabled, "right_wrist_camera": disabled})
        self.assertEqual(wrist_topics({"head_camera": HEAD_CAMERA,
                                       "left_wrist_camera": disabled, "right_wrist_camera": disabled}), [])

    def test_wrist_topics_reports_only_enabled_streams(self):
        config = {"head_camera": HEAD_CAMERA,
                  "left_wrist_camera": wrist_camera(),
                  "right_wrist_camera": wrist_camera(zmq_port=55557, webrtc_port=60003)}
        self.assertEqual(wrist_topics(config),
                         [("left_wrist_camera", 55556, 60002), ("right_wrist_camera", 55557, 60003)])
        del config["left_wrist_camera"]
        self.assertEqual(wrist_topics(config), [("right_wrist_camera", 55557, 60003)])

    def test_monocular_wrist_stream_needs_no_webrtc_port(self):
        config = {"head_camera": HEAD_CAMERA,
                  "left_wrist_camera": wrist_camera(webrtc=False, webrtc_port=61002)}
        validate_camera_config(config)
        self.assertEqual(wrist_topics(config), [("left_wrist_camera", 55556, 60002)])

if __name__ == "__main__":
    unittest.main()
