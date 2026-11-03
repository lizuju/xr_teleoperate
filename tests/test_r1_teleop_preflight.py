import math
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import check_r1_teleop
from tools.check_r1_teleop import (
    REQUIRE_CAMERA_CLOCK_ENV,
    StreamWindow,
    camera_clock_problem,
    check_camera_clock,
    clock_probe,
    clock_rate_ppm,
    jpeg_decoder,
    jpeg_is_complete,
    stream_check_failure,
    unwrap_camera_config,
    validate_camera_config,
    validate_state,
    wrist_topics,
)


HEAD_CAMERA = {
    "type": "rtp_jpeg_stereo", "image_shape": [544, 1280], "binocular": True,
    "enable_zmq": True, "zmq_port": 55555, "enable_webrtc": True,
    "webrtc_port": 60001, "left_rtp_port": 5004, "right_rtp_port": 5005, "external_publisher": True,
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
            "type": "rtp_jpeg_stereo", "image_shape": [544, 1280], "binocular": True,
            "enable_zmq": True, "zmq_port": 55555, "enable_webrtc": True,
            "webrtc_port": 60001, "left_rtp_port": 5004, "right_rtp_port": 5005, "external_publisher": True,
        }
        validate_camera_config({"head_camera": head})
        head["binocular"] = False
        with self.assertRaisesRegex(ValueError, "binocular"):
            validate_camera_config({"head_camera": head})
        head["binocular"] = True
        head["image_shape"] = [544, 640]
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

    def test_wrist_zmq_without_webrtc_is_the_recording_path(self):
        # Data collection reads 55556/55557. Wrist WebRTC 60002/60003 is unused
        # by the headset and can be off on PC2 without dropping color_2/color_3.
        config = {
            "head_camera": HEAD_CAMERA,
            "left_wrist_camera": wrist_camera(webrtc=False),
            "right_wrist_camera": wrist_camera(zmq_port=55557, webrtc_port=60003, webrtc=False),
        }
        validate_camera_config(config)
        self.assertEqual(
            wrist_topics(config),
            [("left_wrist_camera", 55556, 60002), ("right_wrist_camera", 55557, 60003)],
        )


class FakeClockRequest:
    def __init__(self, replies, answers=True):
        self.replies, self.answers = list(replies), answers
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def poll(self, timeout_ms):
        return self.answers

    def recv_json(self):
        return self.replies.pop(0)


class CameraClockTest(unittest.TestCase):
    def test_rate_is_the_camera_clock_against_ours(self):
        self.assertAlmostEqual(clock_rate_ppm((0, 5_000_000_000), (2_000_000_000, 7_000_020_000)), 10.0)
        self.assertAlmostEqual(clock_rate_ppm((0, 0), (2_000_000_000, 2_170_754_000)), 85377.0)

    def test_probe_keeps_the_round_trip_with_the_least_delay(self):
        request = FakeClockRequest([{"receive_ns": 900, "send_ns": 910}, {"receive_ns": 2000, "send_ns": 2010}])
        with patch("tools.check_r1_teleop.time.monotonic_ns", side_effect=[0, 500, 1000, 1010]):
            self.assertEqual(clock_probe(request, attempts=2), (1005.0, 2005.0))
        self.assertEqual(request.sent, [b"GET_CLOCK", b"GET_CLOCK"])

    def test_probe_fails_when_the_server_does_not_answer_or_report_its_clock(self):
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            clock_probe(FakeClockRequest([], answers=False))
        with self.assertRaisesRegex(RuntimeError, "did not report"):
            clock_probe(FakeClockRequest([{"cameras": {}}]))

    def test_only_a_slew_is_a_problem(self):
        for rate in (0.0, 30.0, -30.0, 500.0, -999.0):
            self.assertIsNone(camera_clock_problem(rate))
        for rate in (85377.0, -85377.0, 1500.0):
            self.assertIn("chrony", camera_clock_problem(rate))

    def check(self, last, required):
        output = io.StringIO()
        env = {REQUIRE_CAMERA_CLOCK_ENV: "1"} if required else {}
        with patch("tools.check_r1_teleop.clock_probe", side_effect=[(0, 0), last]), \
                patch("tools.check_r1_teleop.time.sleep"), patch.dict(os.environ, env), redirect_stdout(output):
            if not required:
                os.environ.pop(REQUIRE_CAMERA_CLOCK_ENV, None)
            check_camera_clock(object())
        return output.getvalue()

    def test_slew_only_warns_for_plain_teleop(self):
        self.assertIn("[WARN]", self.check((2_000_000_000, 2_170_754_000), required=False))

    def test_slew_refuses_recording_and_execution(self):
        with self.assertRaisesRegex(RuntimeError, "chrony"):
            self.check((2_000_000_000, 2_170_754_000), required=True)

    def test_normal_rate_passes_in_both_modes(self):
        for required in (False, True):
            self.assertIn("[OK]", self.check((2_000_000_000, 2_000_060_000), required))

    def test_recording_and_execution_launchers_require_the_camera_clock(self):
        teleop = Path(__file__).resolve().parents[1] / "teleop"
        needed = f'{REQUIRE_CAMERA_CLOCK_ENV}=1 bash "${{script_dir}}/run_r1_a7_vector.sh" --check-only'
        for name in ("run_r1_a7_capture.sh", "run_r1_a7_policy.sh"):
            self.assertIn(needed, (teleop / name).read_text(), name)
        self.assertNotIn(REQUIRE_CAMERA_CLOCK_ENV, (teleop / "run_r1_a7_vector.sh").read_text())


class HeadsetVideoProbeTest(unittest.TestCase):
    def run_main(self, failing_ports):
        probed = []

        def probe(_context, port, _label):
            probed.append(port)
            if port in failing_ports:
                raise RuntimeError(f"port {port} is down")

        output = io.StringIO()
        with patch.object(check_r1_teleop, "check_runtime_version"), \
                patch.object(check_r1_teleop, "check_local_and_https"), \
                patch.object(check_r1_teleop, "check_streams", return_value={check_r1_teleop.STEREO_JPEG_LABEL}), \
                patch.object(check_r1_teleop, "https_probe", probe), redirect_stdout(output):
            code = check_r1_teleop.main()
        return code, probed, output.getvalue()

    def test_headset_video_is_probed_when_the_scene_plane_is_down(self):
        code, probed, output = self.run_main({60001})
        self.assertEqual(code, 0)
        self.assertEqual(probed, [60001, 60004])
        self.assertIn("[OK] verified WebRTC HTTPS for the Vision Pro head video", output)

    def test_missing_headset_video_is_reported_with_the_scene_plane_up(self):
        code, _probed, output = self.run_main({60004})
        self.assertEqual(code, 0)
        self.assertIn("first-person view will have no picture", output)


class StreamCheckFailureTest(unittest.TestCase):
    def test_empty_problems_is_a_timeout_not_a_name_error(self):
        message = stream_check_failure({})
        self.assertIn("timed out after 8s", message)
        self.assertNotIn("NameError", message)

    def test_named_problems_are_listed(self):
        message = stream_check_failure({"stereo JPEG": "stale samples (0.9s old)"})
        self.assertIn("stereo JPEG: stale samples (0.9s old)", message)

if __name__ == "__main__":
    unittest.main()
