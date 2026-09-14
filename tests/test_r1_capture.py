import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.r1_capture import R1Capture


class Snapshot:
    def __init__(self, data):
        self.data = data

    def get_recording_snapshot(self):
        return copy.deepcopy(self.data)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        now = time.monotonic_ns()
        self.arm = Snapshot({
            "state": {"monotonic_ns": now, "sequence": 7, "q": [0.1] * 14,
                      "dq": [0.2] * 14, "head_q": [0.3, 0.4], "waist_q": 0.5},
            "requested": {"arm_q": [0.6] * 14, "arm_tau": [0.7] * 14,
                          "head_q": [0.8, 0.9], "waist_q": None, "monotonic_ns": now},
            "published": {"arm_q": [0.55] * 14, "arm_tau": [0.7] * 14,
                          "head_q": [0.75, 0.85], "waist_q": 0.5,
                          "monotonic_ns": now, "sequence": 9},
        })
        self.hand = Snapshot({
            "state": {side: {"q": [0.1] * 6, "monotonic_ns": now,
                             "sequence": 3, "mode": 1} for side in ("left", "right")},
            "requested": {"left_q": [0.9] * 6, "right_q": [0.8] * 6},
            "published": {side: {"q": [0.6] * 6, "monotonic_ns": now,
                                 "sequence": 4, "mode": 1} for side in ("left", "right")},
        })
        self.inputs = {side: {"received_monotonic_ns": now - 10_000_000,
                              "points": np.ones((25, 3)).tolist()} for side in ("left", "right")}
        loop = SimpleNamespace(get_recording_sample=lambda: {
            "hand": self.hand.get_recording_snapshot(), "target_inputs": copy.deepcopy(self.inputs)})
        self.capture = R1Capture(self.arm, self.hand, loop, 0.25, (16, 32))
        self.xr = SimpleNamespace(
            motion_data_ready=True, motion_data_timestamp=now / 1e9,
            left_hand_timestamp=now / 1e9, right_hand_timestamp=now / 1e9,
            left_hand_pos=np.zeros((25, 3)), right_hand_pos=np.zeros((25, 3)),
            left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4), head_pose=np.eye(4),
        )
        pixels = np.zeros((16, 32, 3), dtype=np.uint8)
        pixels[:, 16:] = 200
        self.image = SimpleNamespace(bgr=pixels, sequence=10, received_monotonic_ns=now)

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
            episode = Path(directory) / "episode_0000"
            report = validate_episode(episode)
            self.assertTrue(report["valid"], report)
            self.assertEqual(report["modes"], {"following": 1, "paused": 1})
            self.assertEqual(report["images_checked"], 4)
            self.assertEqual(report["sources"]["image"]["repeated"], 1)
            lines = (episode / "frames.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(lines[0])["sample"]["commands"]["hands"]["requested"]["left_q"], [0.9] * 6)


if __name__ == "__main__":
    unittest.main()
