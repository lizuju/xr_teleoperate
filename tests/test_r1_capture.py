import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

import numpy as np

from teleop.utils.camera_calibration import camera_calibration_metadata, load_camera_calibration
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.r1_capture import R1Capture, capture_metadata


def calibration_document():
    """A minimal valid calibration covering all four cameras of this rig."""
    def intrinsics(width, height):
        return {"image_size": [width, height],
                "camera_matrix": [[float(width), 0.0, width / 2.0],
                                  [0.0, float(width), height / 2.0], [0.0, 0.0, 1.0]],
                "distortion_model": "plumb_bob",
                "distortion_coefficients": [0.01, -0.02, 0.0, 0.0, 0.003]}
    return {"schema": "r1_camera_calibration_v1",
            "cameras": {"head_left": intrinsics(544, 448), "head_right": intrinsics(544, 448),
                        "left_wrist": intrinsics(640, 480), "right_wrist": intrinsics(640, 480)}}


class Snapshot:
    def __init__(self, data):
        self.data = data

    def get_recording_samples(self, target_ns, since_sequence, end_ns):
        state = self.data["state"]
        packet = {key: state[key] for key in ("monotonic_ns", "sequence", "tick")}
        packet.update(state["imu"])
        return {"nearest": copy.deepcopy(state), "dropped": 0,
                "imu_packets": [packet] if since_sequence != state["sequence"] else []}

    def get_recording_states_at(self, target_ns, end_ns):
        return copy.deepcopy(self.data["state"])

    def get_recording_snapshot(self):
        return copy.deepcopy(self.data)


def build(now, image_shape=(16, 32)):
    """Fixtures shared by the head-only and the palm-camera capture tests."""
    arm = Snapshot({
        "state": {"monotonic_ns": now, "sequence": 7, "q": [0.1] * 14,
                  "dq": [0.2] * 14, "tau": [0.05] * 14,
                  "head_q": [0.3, 0.4], "waist_q": 0.5, "tick": 100,
                  "imu": {"quaternion": [1.0, 0.0, 0.0, 0.0], "rpy": [0.0]*3,
                          "gyroscope": [0.1]*3, "accelerometer": [0.0, 0.0, 9.81], "temperature": 30, "valid": True}},
        "requested": {"arm_q": [0.6] * 14, "arm_tau": [0.7] * 14,
                      "head_q": [0.8, 0.9], "waist_q": None, "monotonic_ns": now},
        "published": {"arm_q": [0.55] * 14, "arm_tau": [0.7] * 14,
                      "head_q": [0.75, 0.85], "waist_q": 0.5,
                      "monotonic_ns": now, "sequence": 9},
    })
    hand = Snapshot({
        "state": {side: {"q": [0.1] * 6, "monotonic_ns": now,
                         "sequence": 3, "mode": 1} for side in ("left", "right")},
        "requested": {"left_q": [0.9] * 6, "right_q": [0.8] * 6},
        "published": {side: {"q": [0.6] * 6, "monotonic_ns": now,
                             "sequence": 4, "mode": 1} for side in ("left", "right")},
    })
    inputs = {side: {"received_monotonic_ns": now - 10_000_000,
                     "points": np.ones((25, 3)).tolist()} for side in ("left", "right")}
    loop = SimpleNamespace(get_recording_sample=lambda: {
        "hand": hand.get_recording_snapshot(), "target_inputs": copy.deepcopy(inputs)})
    xr = SimpleNamespace(
        motion_data_ready=True, motion_data_timestamp=now / 1e9,
        left_hand_timestamp=now / 1e9, right_hand_timestamp=now / 1e9,
        left_hand_pos=np.zeros((25, 3)), right_hand_pos=np.zeros((25, 3)),
        left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4), head_pose=np.eye(4),
    )
    pixels = np.zeros((*image_shape, 3), dtype=np.uint8)
    pixels[:, image_shape[1] // 2:] = 200
    image = SimpleNamespace(bgr=pixels, sequence=10, received_monotonic_ns=now)
    return arm, hand, loop, xr, image


def palm(value, sequence, received, shape=(8, 16)):
    return SimpleNamespace(bgr=np.full((*shape, 3), value, dtype=np.uint8),
                           sequence=sequence, received_monotonic_ns=received)


def palm_at(value, sequence, offset_ms, shape=(8, 16)):
    """A palm frame that arrived ``offset_ms`` ago, so it is always in the past."""
    return palm(value, sequence, time.monotonic_ns() - int(offset_ms * 1e6), shape)


#: Sentinel for "leave this palm frame as the previous sample left it".
_UNCHANGED = object()


class CaptureTests(unittest.TestCase):
    def setUp(self):
        now = time.monotonic_ns()
        self.now = now
        self.arm, self.hand, loop, self.xr, self.image = build(now)
        self.inputs = {side: {"received_monotonic_ns": now - 10_000_000,
                              "points": np.ones((25, 3)).tolist()} for side in ("left", "right")}
        self.capture = R1Capture(self.arm, self.hand, loop, 0.25, (16, 32))

    def test_feedback_requested_and_published_are_distinct(self):
        frame = self.capture.frame(self.xr, self.image, "following")
        self.assertEqual(frame["states"]["left_arm"]["qpos"], [0.1] * 7)
        self.assertEqual(frame["actions"]["left_arm"]["qpos"], [0.6] * 7)
        self.assertEqual(frame["sample"]["commands"]["arm"]["published"]["arm_q"], [0.55] * 14)
        self.assertEqual(frame["states"]["body"]["qpos"], [0.5, 0.3, 0.4])
        self.assertEqual(frame["actions"]["body"]["qpos"], [0.5, 0.8, 0.9])
        self.assertEqual(frame["actions"]["left_ee"]["qpos"], [0.9] * 6)

    def test_images_split_without_swapping_and_repeats_are_explicit(self):
        first = self.capture.frame(self.xr, self.image, "following")
        self.assertTrue(np.all(first["colors"]["color_0"] == 0))
        self.assertTrue(np.all(first["colors"]["color_1"] == 200))
        self.assertFalse(first["sample"]["sources"]["image"]["repeated"])
        second = self.capture.frame(self.xr, self.image, "following")
        self.assertTrue(second["sample"]["sources"]["image"]["repeated"])
        self.capture.reset_episode()
        self.assertFalse(self.capture.frame(self.xr, self.image, "following")["sample"]["sources"]["image"]["repeated"])

    def test_colour_sequences_identify_the_head_frame(self):
        # The writer stores a colour key once per source frame and points later
        # samples at the same file; these sequences are how it knows they match.
        first = self.capture.frame(self.xr, self.image, "following")
        second = self.capture.frame(self.xr, self.image, "following")
        self.assertEqual(first["color_sequences"], {"color_0": 10, "color_1": 10})
        self.assertEqual(second["color_sequences"], first["color_sequences"])
        self.capture.reset_episode()
        self.assertEqual(self.capture.frame(self.xr, self.image, "following")["color_sequences"],
                         {"color_0": 10, "color_1": 10})

    def test_tracking_loss_and_pause_keep_data_without_faking_freshness(self):
        self.xr.motion_data_ready = False
        self.xr.left_hand_timestamp = 0
        for mode in ("paused", "tracking_hold"):
            frame = self.capture.frame(self.xr, self.image, mode)
            self.assertEqual(frame["sample"]["mode"], mode)
            self.assertFalse(frame["sample"]["sources"]["xr"]["fresh"])
            self.assertFalse(frame["sample"]["sources"]["left_hand_tracking"]["fresh"])
            self.assertEqual(frame["sample"]["sources"]["left_hand_tracking"]["received_monotonic_ns"], 0)

    def test_hand_command_input_keeps_independent_source_sample(self):
        frame = self.capture.frame(self.xr, self.image, "following")
        self.assertEqual(frame["sample"]["xr"]["left_hand_points"], np.zeros((25, 3)).tolist())
        self.assertEqual(frame["sample"]["hand_target_inputs"], self.inputs)
        self.assertLess(frame["sample"]["hand_target_inputs"]["left"]["received_monotonic_ns"],
                        frame["sample"]["sources"]["left_hand_tracking"]["received_monotonic_ns"])

    def test_lost_or_stale_image_is_not_silently_recorded(self):
        with self.assertRaisesRegex(RuntimeError, "lost"):
            self.capture.frame(self.xr, None, "following")
        self.image.received_monotonic_ns -= 600_000_000
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.capture.frame(self.xr, self.image, "following")

    def test_changed_dimensions_rejects_episode(self):
        self.image.bgr = self.image.bgr[:, :16]
        with self.assertRaisesRegex(RuntimeError, "dimensions"):
            self.capture.frame(self.xr, self.image, "following")

    def test_each_feedback_source_must_stay_fresh(self):
        sources = [self.arm.data["state"], *self.hand.data["state"].values()]
        for source in sources:
            with self.subTest(source=source):
                original = source["monotonic_ns"]
                source["monotonic_ns"] -= 300_000_000
                with self.assertRaisesRegex(RuntimeError, "feedback is stale"):
                    self.capture.frame(self.xr, self.image, "paused")
                source["monotonic_ns"] = original

    def test_odd_stereo_width_is_rejected_up_front(self):
        with self.assertRaisesRegex(ValueError, "even"):
            R1Capture(self.arm, self.hand, SimpleNamespace(get_recording_sample=lambda: {}),
                      0.25, (16, 33))

    def test_measured_torque_is_recorded_per_arm(self):
        frame = self.capture.frame(self.xr, self.image, "following")
        self.assertEqual(frame["states"]["left_arm"]["torque"], [0.05] * 7)
        self.assertEqual(frame["states"]["right_arm"]["torque"], [0.05] * 7)

    def test_torque_is_absent_rather_than_zero_when_the_source_has_none(self):
        del self.arm.data["state"]["tau"]
        frame = self.capture.frame(self.xr, self.image, "following")
        self.assertEqual(frame["states"]["left_arm"]["torque"], [])
        self.assertEqual(frame["states"]["right_arm"]["torque"], [])
        self.arm.data["state"]["tau"] = None
        frame = self.capture.frame(self.xr, self.image, "following")
        self.assertEqual(frame["states"]["left_arm"]["torque"], [])
        self.assertEqual(frame["states"]["right_arm"]["torque"], [])

    def test_capture_writer_offline_checker_round_trip(self):
        from tools.check_teleop_episode import validate_episode
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeWriter(directory, task_goal="offline fixture", image_size=(16, 16),
                                   rerun_log=False, metadata={"frequency": 30,
                                   "joint_names": {name: [str(i) for i in range(count)] for name, count in
                                                   (("left_arm", 7), ("right_arm", 7), ("left_ee", 6),
                                                    ("right_ee", 6), ("body", 3))}})
            writer.create_episode()
            writer.add_item(**self.capture.frame(self.xr, self.image, "following"))
            writer.add_item(**self.capture.frame(self.xr, self.image, "paused"))
            writer.save_episode(outcome="success")
            writer.close()
            episode = writer.episode_dir
            report = validate_episode(episode)
            self.assertTrue(report["valid"], report)
            self.assertEqual(report["modes"], {"following": 1, "paused": 1})
            self.assertEqual(report["images_checked"], 4)
            self.assertEqual(report["sources"]["image"]["repeated"], 1)
            self.assertEqual(report["measured_torque_frames"], 2)
            self.assertEqual(report["calibration_status"], None)
            # A head-only episode has one stream to pair, so nothing is skewed.
            self.assertEqual(report["camera_alignment"]["frames"], 2)
            self.assertEqual(report["unaligned_frames"], 0)
            lines = (episode / "frames.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(lines[0])["sample"]["commands"]["hands"]["requested"]["left_q"], [0.9] * 6)
            self.assertEqual(json.loads(lines[0])["sample"]["camera_alignment"]["offset_ms"], {"head": 0.0})


class PalmCaptureTests(unittest.TestCase):
    """One sample = one head frame plus the palm frames paired to it."""

    def setUp(self):
        start = time.monotonic_ns()
        self.arm, self.hand, loop, self.xr, self.image = build(start)
        self.loop = loop
        self.capture = R1Capture(self.arm, self.hand, loop, 0.25, (16, 32),
                                 wrist_image_shapes={"left": (8, 16), "right": (8, 16)})
        self.head_sequence = 100
        self.left = palm_at(40, 11, 5)
        self.right = palm_at(90, 12, 5)

    def sample(self, mode="following", left=_UNCHANGED, right=_UNCHANGED,
               head_offset_ms=0.0, capture=None):
        """Take one sample; the head frame is the anchor and lands last."""
        if left is not _UNCHANGED:
            self.left = left
        if right is not _UNCHANGED:
            self.right = right
        self.head_sequence += 1
        head = SimpleNamespace(bgr=self.image.bgr, sequence=self.head_sequence,
                               received_monotonic_ns=time.monotonic_ns() - int(head_offset_ms * 1e6))
        return (capture or self.capture).frame(self.xr, head, mode,
                                               {"left": self.left, "right": self.right})

    def sources(self, frame):
        return frame["sample"]["sources"]

    def test_palm_cameras_ride_the_same_sample_as_the_head(self):
        frame = self.sample()
        self.assertEqual(sorted(frame["colors"]), ["color_0", "color_1", "color_2", "color_3"])
        self.assertTrue(np.all(frame["colors"]["color_2"] == 40))
        self.assertTrue(np.all(frame["colors"]["color_3"] == 90))
        self.assertEqual(frame["colors"]["color_2"].shape, (8, 16, 3))
        self.assertEqual(frame["colors"]["color_1"].shape, (16, 16, 3))
        for side, sequence in (("left", 11), ("right", 12)):
            source = self.sources(frame)[f"{side}_wrist_image"]
            self.assertTrue(source["fresh"])
            self.assertEqual(source["sequence"], sequence)
            self.assertFalse(source["repeated"])

    def test_each_palm_camera_tracks_its_own_repeats(self):
        self.sample()
        second = self.sample()
        self.assertTrue(self.sources(second)["left_wrist_image"]["repeated"])
        self.assertTrue(self.sources(second)["right_wrist_image"]["repeated"])
        third = self.sample(left=palm_at(40, 13, 5))
        self.assertFalse(self.sources(third)["left_wrist_image"]["repeated"])
        self.assertTrue(self.sources(third)["right_wrist_image"]["repeated"])
        self.capture.reset_episode()
        self.assertFalse(self.sources(self.sample())["right_wrist_image"]["repeated"])

    def test_colour_sequences_carry_each_cameras_own_sequence(self):
        first = self.sample()
        self.assertEqual(first["color_sequences"]["color_2"], 11)
        self.assertEqual(first["color_sequences"]["color_3"], 12)
        self.assertEqual(first["color_sequences"]["color_0"], first["color_sequences"]["color_1"])
        # Pairing can reuse the head too; pixels and source sequence must agree.
        second = self.sample()
        self.assertEqual(second["color_sequences"]["color_2"], 11)
        self.assertEqual(second["color_sequences"]["color_3"], 12)
        self.assertEqual(second["color_sequences"]["color_0"],
                         second["sample"]["sources"]["image"]["sequence"])
        self.assertEqual(self.sample(left=palm_at(40, 13, 5))["color_sequences"]["color_2"], 13)

    def test_an_absent_palm_frame_carries_no_sequence(self):
        # A null colour key is never written, so it must not claim a sequence.
        capture = R1Capture(self.arm, self.hand, self.loop, 0.25, (16, 32),
                            wrist_image_shapes={"left": (8, 16), "right": (8, 16)},
                            wrist_timeout=0.02)
        head = SimpleNamespace(bgr=self.image.bgr, sequence=100,
                               received_monotonic_ns=time.monotonic_ns())
        frames = {"left": palm_at(40, 11, 0), "right": palm_at(90, 12, 0)}
        capture.frame(self.xr, head, "following", frames)
        time.sleep(0.05)
        stale = capture.frame(self.xr, head, "following", frames)
        self.assertIsNone(stale["colors"]["color_3"])
        self.assertNotIn("color_3", stale["color_sequences"])

    def test_swapping_the_palm_frames_does_not_swap_the_colour_keys(self):
        frame = self.sample(left=self.right, right=self.left)
        self.assertTrue(np.all(frame["colors"]["color_2"] == 90))
        self.assertTrue(np.all(frame["colors"]["color_3"] == 40))

    def test_missing_palm_entry_reuses_the_last_frame_within_the_window(self):
        first = self.sample()
        frame = self.sample(right=None)
        # The stream went quiet for one sample. Its last frame is still the one
        # nearest the head, so it is reused and flagged rather than dropped.
        self.assertIsNotNone(frame["colors"]["color_3"])
        self.assertTrue(np.all(frame["colors"]["color_3"] == 90))
        self.assertTrue(np.all(frame["colors"]["color_2"] == 40))
        source = self.sources(frame)["right_wrist_image"]
        self.assertTrue(source["fresh"])
        self.assertTrue(source["repeated"])
        self.assertIsInstance(first["colors"]["color_3"], np.ndarray)

    def test_palm_camera_goes_absent_once_its_last_frame_is_stale(self):
        capture = R1Capture(self.arm, self.hand, self.loop, 0.25, (16, 32),
                            wrist_image_shapes={"left": (8, 16), "right": (8, 16)},
                            wrist_timeout=0.02)
        head = SimpleNamespace(bgr=self.image.bgr, sequence=100,
                               received_monotonic_ns=time.monotonic_ns())
        frames = {"left": palm_at(40, 11, 0), "right": palm_at(90, 12, 0)}
        capture.frame(self.xr, head, "following", frames)
        time.sleep(0.05)
        frame = capture.frame(self.xr, head, "following", frames)
        self.assertIsNone(frame["colors"]["color_3"])
        source = self.sources(frame)["right_wrist_image"]
        self.assertFalse(source["fresh"])
        self.assertGreater(source["age_ms"], 20.0)
        # The offset is still reported so the reason stays visible.
        self.assertLess(abs(source["offset_ms"]), 2.0)

    def test_a_palm_camera_that_never_delivers_fails_the_episode(self):
        with self.assertRaisesRegex(RuntimeError, "left wrist image never arrived"):
            self.sample(left=None)

    def test_a_palm_camera_with_no_usable_packet_fails_the_episode(self):
        with self.assertRaisesRegex(RuntimeError, "right wrist image never arrived"):
            self.sample(right=palm(90, 0, 0))

    def test_a_palm_camera_that_stops_after_delivering_reuses_its_last_frame(self):
        self.sample()
        frame = self.sample(left=None, right=None)
        self.assertIsNotNone(frame["colors"]["color_2"])
        self.assertTrue(self.sources(frame)["left_wrist_image"]["repeated"])
        self.assertTrue(self.sources(frame)["right_wrist_image"]["repeated"])

    def test_the_frame_nearest_the_head_wins_over_the_newest(self):
        # The palm delivered one frame 8 ms before the head frame and another
        # 10 ms after it. Pairing must take the nearer one, not the newest.
        older = palm_at(77, 11, 20)
        newer = palm_at(88, 12, 2)
        self.capture.observe(None, {"left": older, "right": older})
        frame = self.sample(left=newer, right=newer, head_offset_ms=12)
        self.assertTrue(np.all(frame["colors"]["color_2"] == 77))
        self.assertTrue(np.all(frame["colors"]["color_3"] == 77))
        source = self.sources(frame)["left_wrist_image"]
        self.assertEqual(source["sequence"], 11)
        self.assertAlmostEqual(source["offset_ms"], -8.0, delta=1.0)

    def test_alignment_block_reports_the_actual_spread(self):
        # Head lands 6 ms ago; the left palm 8 ms before it, the right palm 6 ms
        # after it. Both palms straddle the head, as they do on the live rig.
        frame = self.sample(left=palm_at(40, 11, 14), right=palm_at(90, 12, 0),
                            head_offset_ms=6)
        alignment = frame["sample"]["camera_alignment"]
        # The instant is whichever stream gives the tightest three-way match; on
        # this fixture that is the head frame.
        self.assertEqual(alignment["anchor"], "head")
        self.assertAlmostEqual(alignment["offset_ms"][alignment["anchor"]], 0.0)
        self.assertEqual(alignment["anchor_monotonic_ns"],
                         self.sources(frame)["image"]["received_monotonic_ns"])
        self.assertEqual(alignment["tolerance_ms"], 25.0)
        self.assertTrue(alignment["aligned"])
        self.assertAlmostEqual(alignment["offset_ms"]["head"], 0.0)
        self.assertAlmostEqual(alignment["offset_ms"]["left_wrist"], -8.0, delta=1.5)
        self.assertAlmostEqual(alignment["offset_ms"]["right_wrist"], 6.0, delta=1.5)
        self.assertAlmostEqual(alignment["skew_ms"], 14.0, delta=2.0)
        self.assertAlmostEqual(self.sources(frame)["right_wrist_image"]["offset_ms"],
                               6.0, delta=1.5)

    def test_a_skew_over_the_tolerance_is_flagged_not_hidden(self):
        tight = R1Capture(self.arm, self.hand, self.loop, 0.25, (16, 32),
                          wrist_image_shapes={"left": (8, 16), "right": (8, 16)},
                          sync_tolerance_ms=5.0)
        frame = self.sample(left=palm_at(40, 11, 16), right=palm_at(90, 12, 0),
                            head_offset_ms=6, capture=tight)
        alignment = frame["sample"]["camera_alignment"]
        self.assertGreater(alignment["skew_ms"], 5.0)
        self.assertEqual(alignment["tolerance_ms"], 5.0)
        self.assertFalse(alignment["aligned"])
        # Flagged, but still recorded: the decision to drop it belongs downstream.
        self.assertIsNotNone(frame["colors"]["color_2"])

    def test_reset_episode_clears_the_pairing_history(self):
        self.sample()
        self.assertNotEqual(self.capture.sync.last_used, {})
        self.capture.reset_episode()
        self.assertEqual(self.capture.sync.last_used, {})

    def test_changed_palm_dimensions_reject_the_episode(self):
        with self.assertRaisesRegex(RuntimeError, "left wrist dimensions changed"):
            self.sample(left=palm_at(40, 11, 5, shape=(16, 16)))

    def test_no_palm_shapes_keeps_the_two_colour_schema(self):
        plain = R1Capture(self.arm, self.hand, self.loop, 0.25, (16, 32))
        frame = self.sample(capture=plain)
        self.assertEqual(sorted(frame["colors"]), ["color_0", "color_1"])
        self.assertNotIn("left_wrist_image", frame["sample"]["sources"])
        # With one stream there is nothing to pair against, so the spread is 0.
        alignment = frame["sample"]["camera_alignment"]
        self.assertEqual(alignment["offset_ms"], {"head": 0.0})
        self.assertEqual(alignment["skew_ms"], 0.0)
        self.assertTrue(alignment["aligned"])

    def offline_metadata(self, calibration=None):
        return {
            "frequency": 30,
            "images": {"color_0": {"camera": "head_left", "width": 16, "height": 16},
                       "color_1": {"camera": "head_right", "width": 16, "height": 16},
                       "color_2": {"camera": "left_wrist", "width": 16, "height": 8},
                       "color_3": {"camera": "right_wrist", "width": 16, "height": 8}},
            "joint_names": {name: [str(i) for i in range(count)] for name, count in
                            (("left_arm", 7), ("right_arm", 7), ("left_ee", 6),
                             ("right_ee", 6), ("body", 3))},
            "camera_calibration": camera_calibration_metadata(calibration, [
                "head_left", "head_right", "left_wrist", "right_wrist"]),
        }

    def test_offline_checker_accepts_the_four_camera_episode(self):
        from tools.check_teleop_episode import validate_episode
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeWriter(directory, task_goal="palm fixture", image_size=(16, 16),
                                   rerun_log=False, metadata=self.offline_metadata())
            writer.create_episode()
            writer.add_item(**self.sample())
            self.right = None
            writer.add_item(**self.sample())
            writer.save_episode(outcome="success")
            writer.close()
            episode = writer.episode_dir
            report = validate_episode(episode)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["images_checked"], 8)
            self.assertEqual(report["absent_image_frames"], 0)
            self.assertEqual(report["calibration_status"], "uncalibrated")
            self.assertEqual(report["camera_alignment"]["frames"], 2)
            self.assertEqual(report["unaligned_frames"], 0)
            # Both palms are paired against the same head frame, so the spread
            # is only the sub-millisecond difference between the two arrivals.
            self.assertLess(report["camera_alignment"]["skew_ms_max"], 25.0)
            self.assertEqual(sorted(report["sources"]), sorted([
                "image", "xr", "left_hand_tracking", "right_hand_tracking", "robot",
                "left_hand_feedback", "right_hand_feedback",
                "left_wrist_image", "right_wrist_image"]))
            self.assertEqual(report["sources"]["right_wrist_image"]["repeated"], 1)
            self.assertEqual(report["usable_following_frames"], 0)
            frames = [json.loads(line) for line in (episode / "frames.jsonl").read_text().splitlines()]
            # The second sample reuses the right palm frame rather than dropping
            # the view, and says so.
            self.assertEqual((episode / frames[1]["colors"]["color_3"]).read_bytes(),
                             (episode / frames[0]["colors"]["color_3"]).read_bytes())
            self.assertTrue(frames[1]["sample"]["sources"]["right_wrist_image"]["repeated"])
            self.assertIsInstance(frames[0]["colors"]["color_2"], str)

    def test_offline_checker_carries_the_calibration_of_a_calibrated_episode(self):
        from tools.check_teleop_episode import validate_episode
        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"
            calibration_path.write_text(json.dumps(calibration_document()), encoding="utf-8")
            calibration = load_camera_calibration(calibration_path)
            writer = EpisodeWriter(directory, task_goal="calibrated fixture", image_size=(16, 16),
                                   rerun_log=False, metadata=self.offline_metadata(calibration))
            writer.create_episode()
            writer.add_item(**self.sample())
            writer.save_episode(outcome="success")
            writer.close()
            episode = writer.episode_dir
            report = validate_episode(episode)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["calibration_status"], "calibrated")
            manifest = json.loads((episode / "episode.json").read_text())
            block = manifest["info"]["camera_calibration"]
            self.assertEqual(block["status"], "calibrated")
            self.assertEqual(len(block["sha256"]), 64)
            self.assertEqual(sorted(block["cameras"]), ["head_left", "head_right",
                                                        "left_wrist", "right_wrist"])

    def test_checker_rejects_a_null_image_whose_source_claims_freshness(self):
        from tools.check_teleop_episode import validate_episode
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeWriter(directory, task_goal="inconsistent fixture", image_size=(16, 16),
                                   rerun_log=False, metadata=self.offline_metadata())
            writer.create_episode()
            writer.add_item(**self.sample())
            writer.save_episode(outcome="success")
            writer.close()
            episode = writer.episode_dir
            frames_path = episode / "frames.jsonl"
            frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
            frames[0]["colors"]["color_2"] = None
            frames_path.write_text("".join(json.dumps(frame) + "\n" for frame in frames), encoding="utf-8")
            report = validate_episode(episode)
            self.assertFalse(report["valid"])
            self.assertTrue(any("null image needs" in message for message in report["errors"]),
                            report["errors"])


class HandTelemetryUnitsTest(unittest.TestCase):
    """The O6 telemetry is easy to misread as a measured force.

    It is inferred from motor current and comes off a one-byte device register,
    so the episode has to say so: measured on a recorded session, 103848 torque
    values held exactly 255 distinct levels on the 1/255 grid, saturating at 1.0,
    and the device is the quantizer -- PC2 leaves q_raw/dq_raw at zero, so no
    more precision can be recovered downstream.
    """

    @classmethod
    def setUpClass(cls):
        # capture_metadata hashes the hand URDFs, so point at a real one.
        urdf = Path(__file__).resolve().parents[1] / "assets" / "r1" / "r1_a7.urdf"
        hand = SimpleNamespace(
            urdf_path=urdf,
            hardware_joint_order=[f"joint_{index}" for index in range(6)],
            hardware_lower=np.zeros(6), hardware_upper=np.ones(6),
        )
        cls.retargeter = SimpleNamespace(
            method="vector", mapping_name="o6",
            left=hand, right=hand,
        )
        cls.camera_config = {
            "head_camera": {"binocular": True, "image_shape": [448, 1088], "fps": 30,
                            "enable_zmq": True},
            "left_wrist_camera": {"enable_zmq": True, "image_shape": [480, 640], "fps": 30},
            "right_wrist_camera": {"enable_zmq": True, "image_shape": [480, 640], "fps": 30},
        }
        cls.args = SimpleNamespace(frequency=40.0, camera_sync_tolerance_ms=25.0)
        cls.units = capture_metadata(cls.args, cls.camera_config, cls.retargeter)["units"]

    def test_torque_is_not_advertised_as_a_measured_force(self):
        torque = self.units["hand_torque"]
        self.assertIn("Not N*m", torque)
        self.assertIn("not a measured contact force", torque)
        self.assertIn("motor current", torque)

    def test_torque_carries_the_device_quantisation(self):
        torque = self.units["hand_torque"]
        self.assertIn("256 levels", torque)
        self.assertIn("1/255", torque)
        self.assertIn("saturating at", torque)
        # The device is the quantizer, so a consumer cannot recover precision.
        self.assertIn("q_raw", torque)

    def test_torque_states_what_it_is_usable_for(self):
        self.assertIn("contact and grasp events", self.units["hand_torque"])
        self.assertIn("not as a force regressor", self.units["hand_torque"])

    def test_hand_position_reports_the_same_register_width(self):
        self.assertIn("256 levels", self.units["hand_qpos"])
        self.assertEqual(self.units["hand_points"], "m")


if __name__ == "__main__":
    unittest.main()
