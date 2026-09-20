import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop.utils.rerun_visualizer import (
    RerunEpisodeReader,
    RerunLogger,
    detect_episode_format,
    episode_directory,
    resolve_rerun_viewer,
    summarize_episode,
)


REAL_EPISODES = (
    Path("/home/hnh/unitree_r1_dev/teleop-recordings/c_check/episode_0000"),
    Path("/home/hnh/unitree_r1_dev/teleop-recordings/dedup_check/episode_0000"),
    Path("/home/hnh/unitree_r1_dev/teleop-recordings/door_pick/episode_0000"),
)


class FakeScalar:
    def __init__(self, scalar):
        self.scalar = scalar


class FakeImage:
    def __init__(self, image, **kwargs):
        self.image = image
        self.kwargs = kwargs


class FakeTextLog:
    def __init__(self, text, **kwargs):
        self.text = text
        self.kwargs = kwargs


class FakeRR:
    def __init__(self):
        self.logs = []
        self.sequences = []
        self.nanos = []
        self.inited = None
        self.spawned = False
        self.saved = None
        self.blueprint = None
        self.Scalar = FakeScalar
        self.Image = FakeImage
        self.TextLog = FakeTextLog
        self.blueprint_module = None

    def init(self, application_id, **kwargs):
        self.inited = application_id

    def spawn(self, **kwargs):
        self.spawned = True

    def save(self, path, **kwargs):
        self.saved = str(path)

    def disconnect(self):
        pass

    def set_time_sequence(self, name, value):
        self.sequences.append((name, value))

    def set_time_nanos(self, name, value):
        self.nanos.append((name, value))

    def log(self, path, entity):
        self.logs.append((path, entity))

    def send_blueprint(self, layout):
        self.blueprint = layout


class EpisodeRerunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode = self.root / "episode_0001"
        (self.episode / "colors").mkdir(parents=True)
        self.manifest = {
            "schema": "xr_teleop_episode_v2",
            "status": "complete",
            "episode_id": 1,
            "frame_count": 2,
            "frames": "frames.jsonl",
            "outcome": "success",
            "text": {},
            "info": {
                "image": {"width": 16, "height": 12, "fps": 10},
                "images": {
                    "color_0": {"width": 16, "height": 12},
                    "color_1": {"width": 16, "height": 12},
                    "color_2": {"width": 20, "height": 16},
                    "color_3": {"width": 20, "height": 16},
                },
                "joint_names": {"body": ["waist_yaw", "head_pitch", "head_yaw"]},
            },
        }
        self.frames = []
        first_colors = {}
        for key, size in (("color_0", (12, 16)), ("color_1", (12, 16)),
                          ("color_2", (16, 20)), ("color_3", (16, 20))):
            relative = f"colors/000000_{key}.jpg"
            pixels = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            pixels[:] = (10, 80, 160) if key.endswith("0") else (200, 30, 30)
            self.assertTrue(cv2.imwrite(str(self.episode / relative), pixels))
            first_colors[key] = relative
        self.frames.append(self._frame(
            0, first_colors, mode="following",
            sources={"left_hand_tracking": {"fresh": True, "age_ms": 12.5},
                     "right_hand_tracking": {"fresh": True, "age_ms": 11.0}},
            aligned=True,
        ))
        self.frames.append(self._frame(
            1, dict(first_colors), mode="tracking_hold",
            sources={"left_hand_tracking": {"fresh": True, "age_ms": 20.0},
                     "right_hand_tracking": {"fresh": False, "age_ms": None}},
            aligned=False,
            timestamp_ns=1_000_000_050,
            monotonic_ns=20_050_000,
        ))
        self.write()

    def _frame(self, index, colors, mode, sources, aligned, timestamp_ns=1_000_000_000, monotonic_ns=20_000_000):
        return {
            "idx": index,
            "colors": colors,
            "states": {
                "left_arm": {"qpos": [0.1, 0.2], "qvel": [9.9], "torque": [8.8]},
                "right_arm": {"qpos": [0.3, 0.4]},
                "left_ee": {"qpos": [0.01] * 6},
                "right_ee": {"qpos": [0.02] * 6},
                "body": {"qpos": [0.05, -0.1, 0.2]},
            },
            "actions": {
                "left_arm": {"qpos": [0.11, 0.21]},
                "right_arm": {"qpos": [0.31, 0.41]},
                "left_ee": {"qpos": [0.03] * 6},
                "right_ee": {"qpos": [0.04] * 6},
                "body": {"qpos": [0.06, -0.11, 0.21]},
            },
            "sample": {
                "timestamp_ns": timestamp_ns,
                "monotonic_ns": monotonic_ns,
                "mode": mode,
                "sources": sources,
                "camera_alignment": {"aligned": aligned},
                "xr": {"left_hand_points": [[0.0, 0.0, 0.0]] * 25},
            },
        }

    def write(self):
        (self.episode / "episode.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.episode / "frames.jsonl").write_text(
            "".join(json.dumps(frame) + "\n" for frame in self.frames), encoding="utf-8"
        )

    def test_v2_reader_does_not_look_for_data_json(self):
        self.assertFalse((self.episode / "data.json").exists())
        fmt, manifest = detect_episode_format(self.episode)
        self.assertEqual(fmt, "v2")
        self.assertEqual(manifest["schema"], "xr_teleop_episode_v2")
        reader = RerunEpisodeReader(task_dir=str(self.root))
        items = reader.return_episode_data(1)
        self.assertEqual(len(items), 2)
        self.assertEqual(sorted(items[0]["colors"]), ["color_0", "color_1", "color_2", "color_3"])
        self.assertEqual(items[0]["colors"]["color_0"].shape, (12, 16, 3))
        self.assertEqual(items[0]["colors"]["color_2"].shape, (16, 20, 3))
        self.assertEqual(items[1]["sample"]["mode"], "tracking_hold")
        self.assertEqual(items[0]["states"]["body"]["qpos"][0], 0.05)
        self.assertEqual(items[0]["states"]["left_ee"]["qpos"][0], 0.01)

    def test_v2_reuses_repeated_jpeg_without_a_second_file(self):
        items = list(RerunEpisodeReader().iter_episode_data(self.episode))
        self.assertIs(items[0]["colors"]["color_0"], items[1]["colors"]["color_0"])

    def test_v1_data_json_still_loads(self):
        v1 = self.root / "episode_0002"
        (v1 / "colors").mkdir(parents=True)
        relative = "colors/000000_color_0.jpg"
        pixels = np.full((12, 16, 3), 70, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(v1 / relative), pixels))
        payload = {
            "data": [{
                "idx": 0,
                "colors": {"color_0": relative},
                "states": {"left_arm": {"qpos": [0.5]}},
                "actions": {"left_arm": {"qpos": [0.6]}},
            }]
        }
        (v1 / "data.json").write_text(json.dumps(payload), encoding="utf-8")
        self.assertFalse((v1 / "episode.json").exists())
        reader = RerunEpisodeReader(task_dir=str(self.root), json_file="data.json")
        items = reader.return_episode_data(2)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["colors"]["color_0"].shape, (12, 16, 3))
        self.assertEqual(items[0]["states"]["left_arm"]["qpos"], [0.5])
        summary = summarize_episode(v1)
        self.assertEqual(summary["format"], "v1")
        self.assertTrue(summary["data_json"])
        self.assertFalse(summary["episode_json"])

    def test_missing_episode_mentions_both_formats(self):
        empty = self.root / "episode_0099"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError) as caught:
            detect_episode_format(empty)
        message = str(caught.exception)
        self.assertIn("episode.json", message)
        self.assertIn("data.json", message)
        self.assertNotEqual(message, "Episode 99 data.json not found.")

    def test_logger_emits_qa_streams_and_skips_unused_channels(self):
        fake = FakeRR()
        with mock.patch("teleop.utils.rerun_visualizer.rr", fake), \
             mock.patch("teleop.utils.rerun_visualizer.rrb", mock.Mock()):
            logger = RerunLogger(prefix="offline/", IdxRangeBoundary=None, spawn=False)
            self.assertFalse(fake.spawned)
            items = list(RerunEpisodeReader().iter_episode_data(self.episode, max_frames=1))
            logger.log_item_data(items[0])
        paths = [path for path, _ in fake.logs]
        joined = "\n".join(paths)
        self.assertIn("offline/colors/color_0", paths)
        self.assertIn("offline/colors/color_2", paths)
        self.assertIn("offline/left_arm/states/qpos/0", paths)
        self.assertIn("offline/waist/states/qpos/waist_yaw", paths)
        self.assertIn("offline/head/states/qpos/head_pitch", paths)
        self.assertIn("offline/left_ee/states/qpos/0", paths)
        self.assertIn("offline/sample/tracking_hold", paths)
        self.assertIn("offline/sample/left_hand_tracking/fresh", paths)
        self.assertTrue(any(path == "offline/sample/mode" for path in paths))
        self.assertIn(("idx", 0), fake.sequences)
        self.assertTrue(any(name == "timestamp" for name, _ in fake.nanos))
        self.assertNotIn("qvel", joined)
        self.assertNotIn("torque", joined)
        self.assertNotIn("hand_points", joined)
        self.assertNotIn("qvel", joined)
        hold = next(entity.scalar for path, entity in fake.logs if path.endswith("sample/tracking_hold"))
        self.assertEqual(hold, 0.0)

    def test_logger_marks_tracking_hold_and_does_not_reload_path_strings(self):
        fake = FakeRR()
        with mock.patch("teleop.utils.rerun_visualizer.rr", fake), \
             mock.patch("teleop.utils.rerun_visualizer.rrb", mock.Mock()):
            logger = RerunLogger(prefix="", IdxRangeBoundary=None, spawn=False)
            logger.log_item_data({
                "idx": 7,
                "colors": {"color_0": "colors/000000_color_0.jpg"},
                "states": {},
                "actions": {},
                "sample": {"mode": "tracking_hold", "timestamp_ns": 3, "monotonic_ns": 4,
                           "sources": {"right_hand_tracking": {"fresh": False}}},
            })
        paths = [path for path, _ in fake.logs]
        self.assertNotIn("colors/color_0", paths)
        hold = next(entity.scalar for path, entity in fake.logs if path == "sample/tracking_hold")
        self.assertEqual(hold, 1.0)
        self.assertEqual(fake.nanos, [("timestamp", 3), ("monotonic", 4)])

    def test_summary_cli_reads_v2_without_data_json(self):
        process = subprocess.run(
            [sys.executable, str(ROOT / "tools/rerun_teleop_episode.py"), str(self.episode), "--summary"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        payload = json.loads(process.stdout)
        self.assertEqual(payload["format"], "v2")
        self.assertEqual(payload["schema"], "xr_teleop_episode_v2")
        self.assertFalse(payload["data_json"])
        self.assertTrue(payload["episode_json"])
        self.assertEqual(payload["color_keys"], ["color_0", "color_1", "color_2", "color_3"])
        self.assertTrue(payload["has_hands"])
        self.assertEqual(payload["tracking_hold_frames"], 1)
        self.assertEqual(episode_directory(self.root), self.episode)

    @unittest.skipUnless(any(path.is_dir() for path in REAL_EPISODES), "no production v2 episode on this host")
    def test_real_v2_episode_opens_without_data_json(self):
        episode = next(path for path in REAL_EPISODES if path.is_dir())
        self.assertFalse((episode / "data.json").exists())
        summary = summarize_episode(episode)
        self.assertEqual(summary["format"], "v2")
        self.assertEqual(summary["schema"], "xr_teleop_episode_v2")
        self.assertFalse(summary["data_json"])
        self.assertGreater(summary["frame_count"], 0)
        self.assertIn("color_0", summary["color_keys"])
        frames = list(RerunEpisodeReader().iter_episode_data(episode, max_frames=2))
        self.assertEqual(len(frames), 2)
        self.assertIn("color_0", frames[0]["colors"])
        self.assertEqual(len(frames[0]["colors"]["color_0"].shape), 3)
        self.assertIn("left_arm", frames[0]["states"])
        self.assertIn("mode", frames[0]["sample"])

    def test_resolve_rerun_viewer_finds_binary_next_to_python(self):
        fake_bin = self.root / "fake-venv" / "bin"
        fake_bin.mkdir(parents=True)
        fake_python = fake_bin / "python"
        fake_rerun = fake_bin / "rerun"
        fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_rerun.write_text("#!/bin/sh\necho viewer\n", encoding="utf-8")
        fake_python.chmod(stat.S_IRWXU)
        fake_rerun.chmod(stat.S_IRWXU)
        found, env = resolve_rerun_viewer(
            python_executable=fake_python,
            environ={"PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(found.resolve(), fake_rerun.resolve())
        self.assertEqual(env["PATH"].split(os.pathsep)[0], str(fake_bin.resolve()))
        self.assertNotIn(str(fake_bin), "/usr/bin:/bin")

    def test_resolve_rerun_viewer_ignores_nonexecutable_neighbor(self):
        fake_bin = self.root / "noexec-venv" / "bin"
        fake_bin.mkdir(parents=True)
        fake_python = fake_bin / "python"
        fake_rerun = fake_bin / "rerun"
        fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_rerun.write_text("not executable\n", encoding="utf-8")
        fake_python.chmod(stat.S_IRWXU)
        fake_rerun.chmod(stat.S_IRUSR | stat.S_IWUSR)
        with mock.patch("teleop.utils.rerun_visualizer._bundled_rerun_cli", return_value=None):
            found, env = resolve_rerun_viewer(
                python_executable=fake_python,
                environ={"PATH": "/usr/bin:/bin"},
            )
        self.assertIsNone(found)
        self.assertEqual(env["PATH"].split(os.pathsep)[0], str(fake_bin.resolve()))

    def test_logger_prepares_spawn_path_before_rr_spawn(self):
        fake = FakeRR()
        with mock.patch("teleop.utils.rerun_visualizer.rr", fake), \
             mock.patch("teleop.utils.rerun_visualizer.rrb", mock.Mock()), \
             mock.patch(
                 "teleop.utils.rerun_visualizer.prepare_rerun_spawn_path",
                 return_value=Path("/fake/rerun"),
             ) as prepared:
            logger = RerunLogger(prefix="", IdxRangeBoundary=None, spawn=True)
        prepared.assert_called_once()
        self.assertTrue(fake.spawned)
        self.assertIsNotNone(logger)

    def test_summary_cli_works_without_display(self):
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        process = subprocess.run(
            [sys.executable, str(ROOT / "tools/rerun_teleop_episode.py"), str(self.episode), "--summary"],
            capture_output=True, text=True, check=False, env=env,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        payload = json.loads(process.stdout)
        self.assertEqual(payload["format"], "v2")

    def test_cli_spawn_without_display_explains_save(self):
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        process = subprocess.run(
            [sys.executable, str(ROOT / "tools/rerun_teleop_episode.py"), str(self.episode)],
            capture_output=True, text=True, check=False, env=env,
        )
        self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
        text = process.stdout + process.stderr
        self.assertIn("--save", text)
        self.assertIn("--summary", text)
        self.assertNotIn("Failed to find Rerun Viewer executable in PATH", text)


if __name__ == "__main__":
    unittest.main()
