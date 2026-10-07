import ast
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from teleop.robot_control.r1_pause import R1PauseState
from teleop.utils import teleop_status
from teleop.utils.teleop_status import TeleopStatusPublisher


MAIN_PATH = Path(__file__).resolve().parents[1] / "teleop/teleop_hand_and_arm.py"


class TeleopStatusTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "status.json"
        self.publisher = TeleopStatusPublisher(self.path)

    def tearDown(self):
        self.publisher.close()
        self.directory.cleanup()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if self.path.exists():
                snapshot = json.loads(self.path.read_text())
                if predicate(snapshot):
                    return snapshot
            time.sleep(0.005)
        self.fail("status was not published")

    def test_main_loop_timestamp_does_not_renew_without_submit(self):
        with patch.object(teleop_status.time, "monotonic_ns", return_value=1_000_000_000):
            self.publisher.submit("following")
        first = self.wait_for(lambda value: value["motion"] == "following")
        time.sleep(0.25)
        self.assertEqual(json.loads(self.path.read_text()), first)
        self.assertEqual(first["sample_monotonic_ns"], 1_000_000_000)
        self.assertEqual(first["sequence"], 1)

    def test_ten_hz_limit_keeps_state_transitions_immediate(self):
        with patch.object(teleop_status.time, "monotonic_ns", return_value=1_000_000_000):
            self.publisher.submit("waiting")
        self.wait_for(lambda value: value["sequence"] == 1)
        with patch.object(teleop_status.time, "monotonic_ns", return_value=1_099_000_000):
            self.publisher.submit("waiting")
        self.assertEqual(self.publisher._sequence, 1)
        with patch.object(teleop_status.time, "monotonic_ns", return_value=1_100_000_000):
            self.publisher.submit("waiting")
        self.wait_for(lambda value: value["sequence"] == 2)
        with patch.object(teleop_status.time, "monotonic_ns", return_value=1_101_000_000):
            self.publisher.submit("paused")
        result = self.wait_for(lambda value: value["motion"] == "paused")
        self.assertEqual(result["sequence"], 3)

    def test_async_io_coalesces_latest_without_blocking_submit(self):
        entered, release = threading.Event(), threading.Event()
        replace = teleop_status.os.replace

        def blocked_replace(source, destination):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release status writer")
            replace(source, destination)

        try:
            with patch.object(teleop_status.os, "replace", side_effect=blocked_replace):
                self.publisher.submit("following")
                self.assertTrue(entered.wait(1))
                self.publisher.submit("paused")
                self.publisher.submit("tracking_hold")
                self.assertFalse(self.path.exists())
                release.set()
                snapshot = self.wait_for(lambda value: value["motion"] == "tracking_hold")
                self.assertEqual(snapshot["sequence"], 3)
        finally:
            release.set()

    def test_close_publishes_terminal_state_and_new_run_has_new_identity(self):
        self.publisher.submit("following")
        first = self.wait_for(lambda value: value["motion"] == "following")
        self.publisher.close(error="feedback stopped")
        stopped = json.loads(self.path.read_text())
        self.assertEqual(stopped["motion"], "failed")
        self.assertEqual(stopped["error"], "feedback stopped")
        self.assertGreater(stopped["sequence"], first["sequence"])
        replacement = TeleopStatusPublisher(self.path)
        try:
            replacement.submit("waiting")
            new = self.wait_for(lambda value: value["run_id"] != first["run_id"])
            self.assertEqual(new["sequence"], 1)
        finally:
            replacement.close()

    def test_recording_and_pause_are_independent(self):
        recording = {"state": "recording", "episode": {"id": 2, "name": "episode_0002"},
                     "frames_accepted": 4, "last_saved": None, "error": None}
        self.publisher.submit("paused", Mock(status_snapshot=Mock(return_value=recording)), True)
        snapshot = self.wait_for(lambda value: value["motion"] == "paused")
        self.assertEqual(snapshot["recording"], recording)

    def test_write_failure_leaves_old_timestamp_then_recovers(self):
        self.publisher.submit("waiting")
        original = self.wait_for(lambda value: value["motion"] == "waiting")
        attempted = threading.Event()

        def fail_replace(*_):
            attempted.set()
            raise OSError("disk unavailable")

        with patch.object(teleop_status.os, "replace", side_effect=fail_replace):
            self.publisher.submit("following")
            self.assertTrue(attempted.wait(1))
            self.assertEqual(json.loads(self.path.read_text()), original)
        self.publisher.submit("paused")
        self.wait_for(lambda value: value["motion"] == "paused")

    def test_main_status_uses_final_control_state_not_start_request(self):
        tree = ast.parse(MAIN_PATH.read_text())
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and any(isinstance(child, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == "motion_status"
                                     for target in child.targets) for child in node.body))
        code = compile(ast.Module(body=[block], type_ignores=[]), str(MAIN_PATH), "exec")
        publisher = Mock()
        state = {"native_status": publisher, "STOP": False, "START": True,
                 "capture_mode": "following", "run_motion": False,
                 "R1_PAUSE": SimpleNamespace(paused=False, input_hold=False), "recorder": None,
                 "visionpro_source": SimpleNamespace(get_hold_reason=lambda **_: None),
                 "args": SimpleNamespace(record=False)}
        exec(code, state)
        self.assertEqual(publisher.submit.call_args.args[0], "tracking_hold")
        state["run_motion"] = True
        state["R1_PAUSE"].paused = True
        exec(code, state)
        self.assertEqual(publisher.submit.call_args.args[0], "paused")
        state["STOP"] = True
        exec(code, state)
        self.assertEqual(publisher.submit.call_args.args[0], "stopped")

    def test_reason_updates_publish_immediately_and_clear_outside_hold(self):
        with patch.object(teleop_status.time, "monotonic_ns", return_value=1_000_000_000):
            self.publisher.submit("tracking_hold", hold_reason="receive_timeout")
            self.wait_for(lambda value: value.get("hold_reason") == "receive_timeout")
            self.publisher.submit("tracking_hold", hold_reason="video_source_stale")
            value = self.wait_for(lambda value: value.get("hold_reason") == "video_source_stale")
            self.assertEqual(value["sequence"], 2)
            self.publisher.submit("following", hold_reason="receive_timeout")
            value = self.wait_for(lambda value: value["motion"] == "following")
            self.assertNotIn("hold_reason", value)

    def test_main_distinguishes_latched_tracking_pause_from_operator_pause(self):
        tree = ast.parse(MAIN_PATH.read_text())
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and any(isinstance(child, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == "motion_status"
                                     for target in child.targets) for child in node.body))
        code = compile(ast.Module(body=[block], type_ignores=[]), str(MAIN_PATH), "exec")
        publisher = Mock()
        source = Mock()
        source.get_hold_reason.return_value = "video_clock_expired"
        pause = R1PauseState()
        pause.pause(input_lost=True)
        state = {"native_status": publisher, "STOP": False, "capture_mode": "paused",
                 "run_motion": False, "R1_PAUSE": pause,
                 "recorder": None, "args": SimpleNamespace(record=False), "visionpro_source": source}
        exec(code, state)
        self.assertEqual(publisher.submit.call_args.args[0], "tracking_hold")
        self.assertEqual(publisher.submit.call_args.kwargs["hold_reason"], "video_clock_expired")
        source.get_hold_reason.assert_called_with(latched_only=True)
        pause.pause()
        exec(code, state)
        self.assertEqual(publisher.submit.call_args.args[0], "paused")
        self.assertIsNone(publisher.submit.call_args.kwargs["hold_reason"])

    def test_main_publisher_initialization_is_only_native_real_path(self):
        tree = ast.parse(MAIN_PATH.read_text())
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and any(isinstance(child, ast.ImportFrom)
                             and child.module == "teleop.utils.teleop_status" for child in node.body))
        code = compile(ast.Module(body=[block], type_ignores=[]), str(MAIN_PATH), "exec")
        with patch.object(teleop_status, "TeleopStatusPublisher") as factory:
            for source, real in (("webxr", True), ("visionpro", False)):
                state = {"args": SimpleNamespace(tracking_source=source, record=False),
                         "r1_a7_deferred_real": real, "native_status": None}
                exec(code, state)
                self.assertIsNone(state["native_status"])
            factory.assert_not_called()
            exec(code, {"args": SimpleNamespace(tracking_source="visionpro", record=True),
                        "r1_a7_deferred_real": True})
            factory.return_value.submit.assert_called_once_with("waiting", record_enabled=True)


class EpisodeStatusTest(unittest.TestCase):
    def setUp(self):
        from teleop.utils.episode_writer import EpisodeWriter
        self.directory = tempfile.TemporaryDirectory()
        self.writer = EpisodeWriter(self.directory.name, rerun_log=False)
        self.release_events = []

    def tearDown(self):
        for event in self.release_events:
            event.set()
        try:
            self.writer.close()
        except RuntimeError:
            pass
        self.directory.cleanup()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = self.writer.status_snapshot()
            if predicate(status):
                return status
            time.sleep(0.005)
        self.fail("episode status did not reach expected state")

    def save_first(self):
        self.writer.create_episode()
        self.writer.add_item({})
        self.writer.save_episode("success")
        return self.wait_for(lambda status: status["last_saved"] is not None)["last_saved"]

    def test_saved_state_waits_for_complete_manifest_and_matches_disk(self):
        entered, release = threading.Event(), threading.Event()
        self.release_events.append(release)
        write_manifest = self.writer._write_manifest

        def blocked_complete(status):
            if status == "complete":
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release manifest")
            write_manifest(status)

        with patch.object(self.writer, "_write_manifest", side_effect=blocked_complete):
            self.assertEqual(self.writer.status_snapshot()["state"], "idle")
            self.writer.create_episode()
            self.writer.add_item({})
            self.wait_for(lambda status: status["episode"] is not None)
            self.writer.save_episode("discarded")
            self.assertTrue(entered.wait(1))
            pending = self.writer.status_snapshot()
            self.assertEqual(pending["state"], "saving")
            self.assertEqual(pending["frames_accepted"], 1)
            self.assertIsNone(pending["last_saved"])
            release.set()
            saved = self.wait_for(lambda status: status["state"] == "idle")["last_saved"]
        manifest = json.loads((Path(self.directory.name) / saved["name"] / "episode.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual((saved["id"], saved["outcome"], saved["saved_at"], saved["frames"]),
                         (manifest["episode_id"], manifest["outcome"], manifest["saved_at"], manifest["frame_count"]))

    def test_next_episode_preserves_last_saved_without_old_current_identity(self):
        first = self.save_first()
        entered, release = threading.Event(), threading.Event()
        self.release_events.append(release)
        start = self.writer._start_episode

        def blocked_start():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release next episode")
            start()

        with patch.object(self.writer, "_start_episode", side_effect=blocked_start):
            self.writer.create_episode()
            self.assertTrue(entered.wait(1))
            current = self.writer.status_snapshot()
            self.assertEqual(current["state"], "recording")
            self.assertIsNone(current["episode"])
            self.assertEqual(current["last_saved"], first)
            current["last_saved"]["outcome"] = "failure"
            self.assertEqual(self.writer.status_snapshot()["last_saved"], first)
            release.set()
            self.wait_for(lambda status: status["episode"] is not None)

    def test_failed_second_save_cannot_overwrite_last_saved(self):
        first = self.save_first()
        write_manifest = self.writer._write_manifest

        def fail_complete(status):
            if status == "complete":
                raise OSError("manifest disk write failed")
            write_manifest(status)

        with patch.object(self.writer, "_write_manifest", side_effect=fail_complete):
            self.writer.create_episode()
            self.writer.add_item({})
            self.writer.save_episode("failure")
            status = self.wait_for(lambda value: value["state"] == "failed")
        self.assertIn("manifest disk write failed", status["error"])
        self.assertEqual(status["last_saved"], first)


if __name__ == "__main__":
    unittest.main()
