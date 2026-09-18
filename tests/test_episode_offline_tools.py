import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_teleop_episode import SOURCE_NAMES, validate_episode
from replay_teleop_episode import replay_episode


class EpisodeFixture(unittest.TestCase):
    """A small valid episode plus the helpers every checker test needs.

    Kept separate from the cases themselves so a second suite can reuse the
    fixture without inheriting -- and re-running -- the first suite's tests.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode = self.root / "episode_0001"
        (self.episode / "colors").mkdir(parents=True)
        self.manifest = {"schema": "xr_teleop_episode_v2", "status": "complete", "episode_id": 1,
                         "frame_count": 3, "frames": "frames.jsonl", "outcome": "success", "text": {},
                         "info": {"image": {"width": 64, "height": 48, "fps": 15}, "frequency": 30,
                                  "joint_names": {"left_arm": ["joint0", "joint1"]}}}
        self.frames = []
        for index, seconds in enumerate((0.0, 0.2, 0.5)):
            colors = {}
            for eye in range(2):
                relative = f"colors/{index:06d}_color_{eye}.png"
                pixels = np.zeros((48, 64, 3), dtype=np.uint8)
                pixels[:, :, index] = 220 if eye == 0 else 120
                self.assertTrue(cv2.imwrite(str(self.episode / relative), pixels))
                colors[f"color_{eye}"] = relative
            mono = 10_000_000_000 + round(seconds * 1e9)
            self.frames.append({"idx": index, "colors": colors,
                                "states": {"left_arm": {"qpos": [0.1, 0.2], "qvel": [], "torque": []}},
                                "actions": {"left_arm": {"qpos": [0.2, 0.3], "qvel": [], "torque": []}},
                                "sample": {"timestamp_ns": 1_000_000_000_000 + round(seconds * 1e9),
                                           "monotonic_ns": mono, "mode": "paused" if index == 1 else "following",
                                           "sources": {name: {"sequence": index, "received_monotonic_ns": mono,
                                                              "age_ms": 0.0, "fresh": True}
                                                       for name in SOURCE_NAMES},
                                           "xr": {"left_hand_points": np.zeros((25, 3)).tolist()},
                                           "commands": {"arm": {"published": None}}}})
            self.frames[-1]["sample"]["sources"]["image"]["repeated"] = False
        self.write()

    def write(self):
        (self.episode / "episode.json").write_text(json.dumps(self.manifest))
        (self.episode / "frames.jsonl").write_text("".join(json.dumps(frame) + "\n" for frame in self.frames))

    def assert_invalid(self, text):
        result = validate_episode(self.episode)
        self.assertFalse(result["valid"], result)
        self.assertFalse(result["trainable"])
        self.assertIn(text, "\n".join(result["errors"]))

    def set_r1_active_commands(self):
        self.manifest["info"].update(robot="R1_A7", end_effector="linker_o6")
        for frame in self.frames:
            sample = frame["sample"]
            sample["mode"] = "following"
            sample["commands"] = {
                "arm": {"published": {"arm_q": [0.1] * 14, "arm_tau": [0.0] * 14,
                                      "head_q": [0.0, 0.0], "waist_q": 0.0,
                                      "monotonic_ns": sample["monotonic_ns"], "sequence": frame["idx"] + 1}},
                "hands": {"published": {side: {"q": [0.2] * 6, "mode": 1,
                                                "monotonic_ns": sample["monotonic_ns"], "sequence": frame["idx"] + 1}
                                          for side in ("left", "right")}},
            }

class EpisodeOfflineToolsTests(EpisodeFixture):
    def test_valid_statistics_and_explicit_video_repetition(self):
        image = self.frames[1]["sample"]["sources"]["image"]
        image.update(sequence=0, repeated=True, age_ms=200.0,
                     received_monotonic_ns=self.frames[0]["sample"]["monotonic_ns"])
        self.frames[1]["sample"]["mode"] = "tracking_hold"
        self.frames[1]["sample"]["sources"]["left_hand_tracking"]["fresh"] = False
        self.write()
        result = validate_episode(self.episode)
        self.assertTrue(result["valid"], result)
        self.assertTrue(result["trainable"])
        self.assertEqual(result["images_checked"], 6)
        self.assertEqual(result["sources"]["image"]["repeated"], 1)
        self.assertEqual(result["sources"]["image"]["sequence_repeats"], 1)
        self.assertEqual(result["tracking_invalid_frames"], 1)
        self.assertEqual(result["usable_following_frames"], 2)
        self.assertTrue(result["requires_frame_filtering"])
        self.assertTrue(any("do not train on the full episode" in warning for warning in result["warnings"]))
        self.assertEqual(result["sampling"]["max_gap_ms"], 300.0)

    def test_missing_image(self):
        (self.episode / self.frames[1]["colors"]["color_1"]).unlink()
        self.assert_invalid("image does not exist")

    def test_undecodable_image(self):
        (self.episode / self.frames[1]["colors"]["color_1"]).write_bytes(b"not an image")
        self.assert_invalid("cannot be decoded")

    def test_broken_json_tail(self):
        with (self.episode / "frames.jsonl").open("a") as stream:
            stream.write('{"idx":3,"sample":')
        self.assert_invalid("line 4: invalid JSON")

    def test_size_mismatch(self):
        cv2.imwrite(str(self.episode / self.frames[1]["colors"]["color_1"]), np.zeros((24, 64, 3), dtype=np.uint8))
        self.assert_invalid("differs from")

    def test_non_finite_state_action_and_raw_input(self):
        original = copy.deepcopy(self.frames)
        for location in ("state", "action", "xr"):
            with self.subTest(location=location):
                self.frames = copy.deepcopy(original)
                if location == "xr":
                    self.frames[1]["sample"]["xr"]["left_hand_points"][0][0] = float("nan")
                else:
                    self.frames[1][location + "s"]["left_arm"]["qpos"][0] = float("inf")
                self.write()
                self.assert_invalid("non-finite")

    def test_time_and_index_regression(self):
        self.frames[1]["idx"] = 7
        self.frames[1]["sample"]["monotonic_ns"] = self.frames[0]["sample"]["monotonic_ns"] - 1
        self.frames[1]["sample"]["timestamp_ns"] = self.frames[0]["sample"]["timestamp_ns"] - 1
        self.write()
        self.assert_invalid("timestamp moved backwards")
        self.assert_invalid(".idx: expected 1")

    def test_path_cannot_escape_episode(self):
        self.frames[1]["colors"]["color_1"] = "../outside.png"
        self.write()
        self.assert_invalid("escapes the episode")

    def test_non_complete_and_discarded_are_not_training_data(self):
        for status, outcome in (("incomplete", "success"), ("recording", "unspecified"), ("complete", "discarded")):
            with self.subTest(status=status, outcome=outcome):
                self.manifest.update(status=status, outcome=outcome)
                self.write()
                result = validate_episode(self.episode)
                self.assertTrue(result["valid"], result)
                self.assertFalse(result["trainable"])
                self.assertTrue(result["excluded_reasons"])
                process = subprocess.run([sys.executable, str(ROOT / "tools/check_teleop_episode.py"), str(self.episode)],
                                         capture_output=True, text=True)
                self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
                self.assertFalse(json.loads(process.stdout)["trainable"])

    def test_invalid_cli_prints_json_and_exits_nonzero(self):
        self.frames[1]["actions"]["left_arm"]["qpos"] = None
        self.write()
        process = subprocess.run([sys.executable, str(ROOT / "tools/check_teleop_episode.py"), str(self.episode)],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        self.assertFalse(json.loads(process.stdout)["valid"])

    def test_missing_action_labels_are_invalid(self):
        self.frames[1]["actions"] = {}
        self.write()
        self.assert_invalid("expected non-empty joint states")

    def test_command_dimension_and_source_age_disagreement(self):
        self.frames[1]["sample"]["commands"]["arm"]["requested"] = {
            "arm_q": [0] * 13, "arm_tau": [0] * 14, "head_q": [0, 0],
        }
        self.frames[1]["sample"]["sources"]["image"]["age_ms"] = 100
        self.write()
        self.assert_invalid("expected 14 numeric values")
        self.assert_invalid("age_ms disagrees")

    def test_all_paused_or_all_stale_episode_is_not_trainable(self):
        original = copy.deepcopy(self.frames)
        for condition in ("paused", "stale"):
            with self.subTest(condition=condition):
                self.frames = copy.deepcopy(original)
                for frame in self.frames:
                    if condition == "paused":
                        frame["sample"]["mode"] = "paused"
                    else:
                        frame["sample"]["sources"]["right_hand_tracking"]["fresh"] = False
                self.write()
                result = validate_episode(self.episode)
                self.assertTrue(result["valid"], result)
                self.assertFalse(result["trainable"])
                self.assertEqual(result["usable_following_frames"], 0)
                self.assertTrue(any("no following frames" in reason for reason in result["excluded_reasons"]))

    def test_r1_fresh_feedback_with_released_hand_commands_is_not_trainable(self):
        for released_side in ("left", "right"):
            with self.subTest(released_side=released_side):
                self.set_r1_active_commands()
                for frame in self.frames:
                    frame["sample"]["commands"]["hands"]["published"][released_side]["mode"] = 0
                self.write()
                result = validate_episode(self.episode)
                self.assertTrue(result["valid"], result)
                self.assertFalse(result["trainable"])
                self.assertEqual(result["command_inactive_frames"], 3)
                self.assertEqual(result["usable_following_frames"], 0)
                self.assertTrue(any("released" in reason for reason in result["excluded_reasons"]))

    def test_r1_unpublished_arm_or_hand_is_not_trainable(self):
        for missing in ("arm", "left", "right"):
            with self.subTest(missing=missing):
                self.set_r1_active_commands()
                for frame in self.frames:
                    commands = frame["sample"]["commands"]
                    if missing == "arm":
                        commands["arm"]["published"] = None
                    else:
                        commands["hands"]["published"][missing] = None
                self.write()
                result = validate_episode(self.episode)
                self.assertTrue(result["valid"], result)
                self.assertFalse(result["trainable"])
                self.assertEqual(result["command_inactive_frames"], 3)
                self.assertEqual(result["usable_following_frames"], 0)

    def test_r1_stale_commands_are_excluded_but_250ms_publications_remain_usable(self):
        self.set_r1_active_commands()
        self.frames[0]["sample"]["commands"]["arm"]["published"]["monotonic_ns"] -= 250_000_001
        self.frames[1]["sample"]["commands"]["hands"]["published"]["right"]["monotonic_ns"] -= 250_000_001
        commands = self.frames[2]["sample"]["commands"]
        for command in (commands["arm"]["published"], *commands["hands"]["published"].values()):
            command["monotonic_ns"] -= 250_000_000
        self.write()
        result = validate_episode(self.episode)
        self.assertTrue(result["valid"], result)
        self.assertTrue(result["trainable"])
        self.assertEqual(result["command_inactive_frames"], 2)
        self.assertEqual(result["usable_following_frames"], 1)
        self.assertTrue(result["requires_frame_filtering"])
        self.assertTrue(any("Exclude 2 R1/O6 frames" in warning for warning in result["warnings"]))

    def test_replay_respects_timestamps_and_does_not_show_future_frames(self):
        output = self.root / "replay.mp4"
        result = replay_episode(self.episode, output, fps=10)
        self.assertIn("replay", result, result)
        self.assertEqual(result["replay"]["frames"], 6)
        self.assertAlmostEqual(result["replay"]["video_duration_seconds"], 0.6)
        reader = cv2.VideoCapture(str(output))
        dominant_colors = []
        right_means = []
        try:
            while True:
                ok, image = reader.read()
                if not ok:
                    break
                self.assertEqual(image.shape, (148, 128, 3))
                dominant_colors.append(int(np.argmax(image[8:40, 8:56].mean(axis=(0, 1)))))
                right_means.append(float(image[8:40, 72:120].max(axis=2).mean()))
        finally:
            reader.release()
        self.assertEqual(dominant_colors, [0, 0, 1, 1, 1, 2])
        self.assertTrue(all(100 < value < 140 for value in right_means), right_means)

    def test_replay_refuses_invalid_data_and_huge_gaps_without_output(self):
        output = self.root / "replay.mp4"
        self.frames[2]["sample"]["monotonic_ns"] += 3_600_000_000_000
        self.frames[2]["sample"]["timestamp_ns"] += 3_600_000_000_000
        for source in self.frames[2]["sample"]["sources"].values():
            source["received_monotonic_ns"] += 3_600_000_000_000
        self.write()
        result = replay_episode(self.episode, output)
        self.assertIn("--max-gap-seconds", result["error"])
        self.assertFalse(output.exists())
        self.frames[1]["actions"]["left_arm"]["qpos"] = [float("nan"), 0]
        self.write()
        result = replay_episode(self.episode, output)
        self.assertFalse(result["validation"]["valid"])
        self.assertNotIn("replay", result)
        self.assertFalse(output.exists())


class EpisodeImageReuseTests(EpisodeFixture):
    """A repeated camera frame is stored once and referenced again.

    The writer has done that since 2026-09-17, so the offline checker has to
    accept a colour path several samples share -- and to catch the case where
    the repeated flag and the stored path disagree, because then a consumer
    following either signal alone reads the wrong picture. Episodes recorded
    before the change wrote a new file every time, so there the flag means "the
    camera frame is unchanged" and a mismatch is only worth a warning.
    """

    def drop_frame_one_copies(self):
        for eye in range(2):
            (self.episode / f"colors/000001_color_{eye}.png").unlink()

    def test_a_deduped_episode_accepts_a_shared_path(self):
        self.manifest["info"]["image_storage"] = {"written_images": 4, "reused_images": 2}
        for eye in range(2):
            key = f"color_{eye}"
            self.frames[1]["colors"][key] = self.frames[0]["colors"][key]
        self.frames[1]["sample"]["sources"]["image"]["repeated"] = True
        self.drop_frame_one_copies()
        self.write()
        result = validate_episode(self.episode)
        self.assertNotIn("flagged", "\n".join(result["errors"]))
        self.assertEqual(result["repeated_flag_mismatches"], 0)
        self.assertEqual(result["unreferenced_image_files"], 0)
        reuse = result["image_reuse"]["color_0"]
        self.assertEqual((reuse["referenced"], reuse["files"], reuse["reused_references"]), (3, 2, 1))

    def test_a_repeated_flag_that_names_a_new_file_is_an_error(self):
        self.manifest["info"]["image_storage"] = {"written_images": 6, "reused_images": 0}
        self.frames[1]["sample"]["sources"]["image"]["repeated"] = True
        self.write()
        result = validate_episode(self.episode)
        self.assertIn("flagged repeated but points at a new file", "\n".join(result["errors"]))

    def test_a_fresh_flag_that_reuses_a_file_is_an_error(self):
        self.manifest["info"]["image_storage"] = {"written_images": 4, "reused_images": 2}
        for eye in range(2):
            key = f"color_{eye}"
            self.frames[1]["colors"][key] = self.frames[0]["colors"][key]
        self.drop_frame_one_copies()
        self.write()
        result = validate_episode(self.episode)
        self.assertIn("flagged fresh but reuses", "\n".join(result["errors"]))

    def test_an_episode_without_the_dedup_block_only_warns(self):
        self.frames[1]["sample"]["sources"]["image"]["repeated"] = True
        self.write()
        result = validate_episode(self.episode)
        self.assertNotIn("flagged", "\n".join(result["errors"]))
        self.assertEqual(result["repeated_flag_mismatches"], 2)
        self.assertTrue(any("predates the 2026-09-17 dedup" in text for text in result["warnings"]))

    def test_a_colour_file_no_sample_references_is_reported(self):
        self.assertTrue(cv2.imwrite(str(self.episode / "colors/999999_color_0.png"),
                                    np.zeros((48, 64, 3), dtype=np.uint8)))
        self.write()
        result = validate_episode(self.episode)
        self.assertEqual(result["unreferenced_image_files"], 1)
        self.assertTrue(any("no sample references" in text for text in result["warnings"]))

    def test_a_manifest_count_that_disagrees_with_the_files_is_reported(self):
        self.manifest["info"]["image_storage"] = {"written_images": 99, "reused_images": 0}
        self.write()
        result = validate_episode(self.episode)
        self.assertTrue(any("written_images: declared 99" in text for text in result["warnings"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
