import ast
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from teleop.robot_control.r1_pause import R1PauseState
from teleop.utils.episode_writer import EpisodeWriter


MAIN_PATH = Path(__file__).resolve().parents[1] / "teleop/teleop_hand_and_arm.py"


class ContinuousCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(MAIN_PATH.read_text())
        on_press = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "on_press")
        recording = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                         and ast.unparse(node.test) == "args.record and RECORD_TOGGLE"
                         and any(isinstance(child, ast.Call)
                                 and isinstance(child.func, ast.Attribute)
                                 and child.func.attr == "create_episode"
                                 for child in ast.walk(node)))
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While)
                    and recording in node.body)
        previous = loop.body[loop.body.index(recording) - 1]
        cls.key_code = compile(ast.Module(body=[on_press], type_ignores=[]), str(MAIN_PATH), "exec")
        cls.tick_code = compile(ast.Module(body=[previous, recording], type_ignores=[]),
                                str(MAIN_PATH), "exec")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.writer = EpisodeWriter(self.directory, rerun_log=False)
        self.pause = R1PauseState()
        self.release_events = []
        self.ns = {
            "START": True, "STOP": False, "READY": True,
            "RECORD_ENABLED": True, "RECORD_RUNNING": False,
            "RECORD_TOGGLE": False, "RECORD_OUTCOME": "unspecified",
            "DRY_RUN_MODE": False, "ARM_REQUEST_GENERATION": 1,
            "R1_A7_DEFERRED_REAL_MODE": True, "R1_PAUSE": self.pause,
            "logger_mp": Mock(), "recorder": self.writer, "r1_capture": Mock(),
            "args": SimpleNamespace(record=True, sim=False, record_max_tracking_age_ms=100.0),
            "tv_wrapper": object(), "tracking_diagnostics": Mock(return_value={}),
            "recording_blocked_by_tracking": Mock(return_value=None),
        }
        exec(self.key_code, self.ns)

    def tearDown(self):
        for event in self.release_events:
            event.set()
        try:
            self.writer.close()
        finally:
            self.temporary.cleanup()

    def key(self, key):
        self.ns["on_press"](key)

    def tick(self):
        exec(self.tick_code, self.ns)

    def wait_ready(self):
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.writer.is_ready():
                return
            time.sleep(0.005)
        self.fail("episode writer did not finish saving")

    def start(self):
        self.key("s")
        self.tick()
        self.assertTrue(self.ns["RECORD_RUNNING"])
        self.assertFalse(self.ns["RECORD_TOGGLE"])

    def test_three_episodes_keep_unique_directories_images_and_all_outcomes(self):
        saved_hashes = {}
        for index, (key, outcome) in enumerate((("y", "success"), ("n", "failure"), ("x", "discarded"))):
            self.start()
            self.writer.add_item({"color_0": np.full((8, 8, 3), 40 + index, dtype=np.uint8)},
                                 states={"left_arm": {"qpos": [index]}},
                                 sample={"timestamp_ns": index + 1})
            self.key(key)
            self.tick()
            self.assertFalse(self.ns["RECORD_RUNNING"])
            self.wait_ready()
            episodes = sorted(self.directory.glob("episode_*"))
            self.assertEqual(len(episodes), index + 1)
            manifest = json.loads((episodes[-1] / "episode.json").read_text())
            self.assertEqual((manifest["status"], manifest["outcome"], manifest["frame_count"]),
                             ("complete", outcome, 1))
            frame = json.loads((episodes[-1] / "frames.jsonl").read_text())
            self.assertEqual(frame["idx"], 0)
            self.assertEqual(frame["states"]["left_arm"]["qpos"], [index])
            self.assertTrue((episodes[-1] / frame["colors"]["color_0"]).is_file())
            for path, digest in saved_hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            saved_hashes.update({p: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in episodes[-1].rglob("*") if p.is_file()})
        self.assertEqual(self.ns["r1_capture"].reset_episode.call_count, 3)

    def test_s_during_async_save_waits_and_starts_exactly_once(self):
        entered, release = threading.Event(), threading.Event()
        self.release_events.append(release)
        finish = self.writer._finish_episode

        def wait_before_finish():
            entered.set()
            if not release.wait(3.0):
                raise TimeoutError("test did not release episode save")
            finish()

        with patch.object(self.writer, "_finish_episode", side_effect=wait_before_finish):
            self.start()
            self.writer.add_item({})
            self.key("y")
            self.tick()
            self.assertTrue(entered.wait(2.0))
            self.assertFalse(self.writer.is_ready())
            self.key("s")
            for _ in range(3):
                self.tick()
                self.assertTrue(self.ns["RECORD_TOGGLE"])
                self.assertFalse(self.ns["RECORD_RUNNING"])
            release.set()
            self.wait_ready()
            self.tick()
            self.assertTrue(self.ns["RECORD_RUNNING"])
            self.assertFalse(self.ns["RECORD_TOGGLE"])
            self.tick()
            self.assertEqual(self.ns["r1_capture"].reset_episode.call_count, 2)

    def test_paused_s_waits_for_stable_realign_then_starts_next_tick(self):
        self.key("p")
        with patch("teleop.robot_control.r1_pause.time.monotonic", return_value=1.0):
            self.key("s")
        self.assertTrue(self.pause.paused)
        self.assertEqual(self.pause.resume_after, 1.0)
        for index in range(1, 6):
            now = 1.0 + index * 0.1
            self.tick()
            self.assertFalse(self.ns["RECORD_RUNNING"])
            self.assertTrue(self.ns["RECORD_TOGGLE"])
            sample = SimpleNamespace(motion_data_ready=True,
                                     left_hand_timestamp=now, right_hand_timestamp=now)
            token = self.pause.poll_resume(sample, 0.25, now)
            if index < 5:
                self.assertIsNone(token)
        self.assertIsNotNone(token)
        self.assertTrue(self.pause.complete_resume(token))
        self.assertFalse(self.ns["RECORD_RUNNING"])
        self.tick()
        self.assertTrue(self.ns["RECORD_RUNNING"])
        self.assertFalse(self.ns["RECORD_TOGGLE"])

    def test_repeated_s_does_not_restart_pause_stability_window(self):
        self.key("p")
        with patch("teleop.robot_control.r1_pause.time.monotonic", return_value=1.0):
            self.key("s")
        sample = SimpleNamespace(motion_data_ready=True, left_hand_timestamp=1.1,
                                 right_hand_timestamp=1.1)
        self.pause.poll_resume(sample, 0.25, 1.1)
        with patch("teleop.robot_control.r1_pause.time.monotonic", return_value=1.2):
            self.key("s")
        self.assertEqual(self.pause.resume_after, 1.0)
        self.assertEqual(self.pause.stable_since, 1.1)
        self.assertEqual(self.pause.samples, 1)
        self.assertTrue(self.ns["RECORD_TOGGLE"])

    def test_p_cancels_pending_start_and_resume(self):
        self.key("p")
        self.key("s")
        self.assertTrue(self.ns["RECORD_TOGGLE"])
        self.key("p")
        self.tick()
        self.assertFalse(self.ns["RECORD_TOGGLE"])
        self.assertFalse(self.ns["RECORD_RUNNING"])
        self.assertIsNone(self.pause.resume_after)
        self.assertTrue(self.writer.is_ready())

    def test_q_cancels_pending_start(self):
        self.key("p")
        self.key("s")
        self.key("q")
        self.assertTrue(self.ns["STOP"])
        self.assertFalse(self.ns["START"])
        self.assertFalse(self.ns["RECORD_TOGGLE"])
        self.assertTrue(self.writer.is_ready())

    def test_y_then_p_keeps_pending_success_save(self):
        self.start()
        self.writer.add_item({})
        self.key("y")
        self.key("p")
        self.assertTrue(self.ns["RECORD_TOGGLE"])
        self.assertEqual(self.ns["RECORD_OUTCOME"], "success")
        self.tick()
        self.wait_ready()
        path = next(self.directory.glob("episode_*/episode.json"))
        self.assertEqual(json.loads(path.read_text())["outcome"], "success")
        self.assertTrue(self.pause.paused)

    def test_nonrecord_s_does_not_request_resume(self):
        self.ns["RECORD_ENABLED"] = False
        self.ns["args"].record = False
        self.key("p")
        self.key("s")
        self.tick()
        self.assertIsNone(self.pause.resume_after)
        self.assertFalse(self.ns["RECORD_TOGGLE"])
        self.assertFalse(self.ns["RECORD_RUNNING"])

    def test_ready_flag_does_not_block_new_episode_after_activation(self):
        self.ns["READY"] = False
        self.start()

    def test_s_arriving_inside_save_is_not_cleared_by_save_completion(self):
        self.start()
        save = self.writer.save_episode

        def save_with_next_request(*args, **kwargs):
            self.key("s")
            save(*args, **kwargs)

        self.key("y")
        with patch.object(self.writer, "save_episode", side_effect=save_with_next_request):
            self.tick()
        self.assertTrue(self.ns["RECORD_TOGGLE"])
        self.assertFalse(self.ns["RECORD_RUNNING"])
        self.wait_ready()
        self.tick()
        self.assertTrue(self.ns["RECORD_RUNNING"])

    def test_tracking_gate_still_refuses_an_unusable_new_episode(self):
        self.ns["recording_blocked_by_tracking"].return_value = "right hand stale"
        self.key("s")
        self.tick()
        self.assertFalse(self.ns["RECORD_RUNNING"])
        self.assertFalse(self.ns["RECORD_TOGGLE"])
        self.assertTrue(self.writer.is_ready())

    def test_writer_error_is_not_hidden_while_start_is_pending(self):
        self.key("p")
        self.key("s")
        with patch.object(self.writer, "raise_if_failed", side_effect=RuntimeError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                self.tick()


if __name__ == "__main__":
    unittest.main()
