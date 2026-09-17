import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from teleop.utils import episode_writer
from teleop.utils.episode_writer import EpisodeWriter


class EpisodeWriterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "recordings"
        self.writers = []
        self.release_events = []

    def tearDown(self):
        for event in self.release_events:
            event.set()
        for writer in self.writers:
            try:
                writer.close()
            except RuntimeError:
                pass
            writer.worker_thread.join(timeout=1.0)
        self.temporary.cleanup()

    def make_writer(self, **kwargs):
        writer = EpisodeWriter(self.directory, rerun_log=False, **kwargs)
        self.writers.append(writer)
        return writer

    def wait_until(self, predicate):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        self.fail("writer did not reach the expected state")

    def hold_worker_start(self, writer):
        entered, release = threading.Event(), threading.Event()
        self.release_events.append(release)
        start = writer._start_episode

        def blocked():
            entered.set()
            release.wait(3.0)
            start()

        patcher = mock.patch.object(writer, "_start_episode", side_effect=blocked)
        patcher.start()
        self.addCleanup(patcher.stop)
        return entered, release

    def manifest(self, index=0):
        path = self.directory / f"episode_{index:04d}" / "episode.json"
        return json.loads(path.read_text())

    def frames(self, index=0):
        path = self.directory / f"episode_{index:04d}" / "frames.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_normal_multiple_episodes_real_jpeg_and_metadata(self):
        writer = self.make_writer(image_size=(544, 448), metadata={"retargeting_method": "vector"})
        pixels = np.full((448, 544, 3), 80, dtype=np.uint8)
        sample = {"timestamp_ns": 100, "monotonic_ns": 50, "mode": "teleoperation"}
        self.assertTrue(writer.is_ready())
        self.assertTrue(writer.create_episode())
        self.assertFalse(writer.create_episode())
        writer.add_item({"color_0": pixels, "color_1": pixels}, states={"left_arm": {"qpos": [0.1]}}, sample=sample)
        writer.save_episode(outcome="success")
        self.wait_until(writer.is_ready)
        first = self.manifest()
        self.assertEqual(first["schema"], "xr_teleop_episode_v2")
        self.assertEqual((first["status"], first["outcome"], first["frame_count"]), ("complete", "success", 1))
        self.assertEqual(first["info"]["image"], {"width": 544, "height": 448, "fps": 30})
        self.assertEqual(first["info"]["retargeting_method"], "vector")
        row = self.frames()[0]
        self.assertEqual(row["sample"], sample)
        self.assertEqual(row["states"]["left_arm"]["qpos"], [0.1])
        for image_path in row["colors"].values():
            decoded = cv2.imread(str(self.directory / "episode_0000" / image_path))
            self.assertEqual(decoded.shape, (448, 544, 3))
        self.assertFalse((self.directory / "episode_0000" / "data.json").exists())
        self.assertFalse((self.directory / "episode_0000" / ".episode.json.tmp").exists())
        self.assertTrue(writer.create_episode())
        writer.add_item({})
        writer.save_episode(outcome="discarded")
        writer.close()
        self.assertEqual(self.manifest(1)["outcome"], "discarded")
        self.assertEqual(self.frames(1)[0]["idx"], 0)
        self.assertEqual(self.manifest(), first)

    def test_depth_is_declared_only_when_depth_frames_are_recorded(self):
        writer = self.make_writer(image_size=(544, 448))
        writer.create_episode()
        writer.add_item({"color_0": np.zeros((448, 544, 3), dtype=np.uint8)})
        writer.save_episode(outcome="success")
        self.wait_until(writer.is_ready)
        episode = self.directory / "episode_0000"
        # Advertising depth while every sample carries depths=null makes a
        # consumer believe a modality exists that was never recorded.
        self.assertIsNone(self.manifest()["info"]["depth"])
        self.assertFalse((episode / "depths").exists())

        with_depth = self.make_writer(image_size=(544, 448), depth_size=(544, 448))
        with_depth.create_episode()
        with_depth.add_item({"color_0": np.zeros((448, 544, 3), dtype=np.uint8)},
                            depths={"depth_0": np.zeros((448, 544), dtype=np.uint16)})
        with_depth.save_episode(outcome="success")
        self.wait_until(with_depth.is_ready)
        self.assertEqual(self.manifest(1)["info"]["depth"], {"width": 544, "height": 448, "fps": 30})
        self.assertTrue((self.directory / "episode_0001" / "depths").is_dir())

    def test_null_colour_keeps_the_key_without_writing_a_file(self):
        writer = self.make_writer(image_size=(16, 16))
        pixels = np.full((16, 16, 3), 90, dtype=np.uint8)
        writer.create_episode()
        writer.add_item({"color_0": pixels, "color_1": pixels, "color_2": None})
        writer.save_episode(outcome="success")
        self.wait_until(writer.is_ready)
        row = self.frames()[0]
        self.assertIsNone(row["colors"]["color_2"])
        self.assertTrue(row["colors"]["color_0"].endswith("_color_0.jpg"))
        self.assertFalse((self.directory / "episode_0000" / "colors" / "000000_color_2.jpg").exists())
        self.assertEqual(sorted(path.name for path in (self.directory / "episode_0000" / "colors").iterdir()),
                         ["000000_color_0.jpg", "000000_color_1.jpg"])

    def colour_files(self, index=0):
        return sorted(path.name for path in (self.directory / f"episode_{index:04d}" / "colors").iterdir())

    def test_a_repeated_camera_sequence_is_stored_once_and_referenced_again(self):
        # The 40 Hz sample loop sees a fresh head frame about a quarter of the
        # time; writing the same JPEG again each tick measured 62% of an episode.
        writer = self.make_writer(image_size=(16, 16))
        pixels = np.full((16, 16, 3), 70, dtype=np.uint8)
        writer.create_episode()
        for index in range(4):
            writer.add_item({"color_0": pixels, "color_1": pixels},
                            sample={"timestamp_ns": index * 25_000_000},
                            color_sequences={"color_0": 11, "color_1": 11})
        writer.save_episode(outcome="success")
        self.wait_until(writer.is_ready)
        rows = self.frames()
        self.assertEqual(self.colour_files(), ["000000_color_0.jpg", "000000_color_1.jpg"])
        for row in rows:
            self.assertEqual(row["colors"]["color_0"], "colors/000000_color_0.jpg")
            self.assertEqual(row["colors"]["color_1"], "colors/000000_color_1.jpg")
        info = self.manifest()["info"]
        self.assertEqual(info["image_storage"]["written_images"], 2)
        self.assertEqual(info["image_storage"]["reused_images"], 6)

    def test_a_new_camera_sequence_writes_a_new_file(self):
        writer = self.make_writer(image_size=(16, 16))
        writer.create_episode()
        for index, sequence in enumerate((11, 11, 12)):
            writer.add_item({"color_0": np.full((16, 16, 3), sequence, dtype=np.uint8)},
                            sample={"timestamp_ns": index * 25_000_000},
                            color_sequences={"color_0": sequence})
        writer.save_episode(outcome="success")
        self.wait_until(writer.is_ready)
        self.assertEqual(self.colour_files(), ["000000_color_0.jpg", "000002_color_0.jpg"])
        rows = self.frames()
        self.assertEqual(rows[1]["colors"]["color_0"], rows[0]["colors"]["color_0"])
        self.assertNotEqual(rows[2]["colors"]["color_0"], rows[0]["colors"]["color_0"])

    def test_colours_without_a_sequence_are_always_written(self):
        # A caller that predates color_sequences must keep the old behaviour
        # rather than have unrelated frames collapse onto one file.
        writer = self.make_writer(image_size=(16, 16))
        pixels = np.full((16, 16, 3), 70, dtype=np.uint8)
        writer.create_episode()
        for index in range(3):
            writer.add_item({"color_0": pixels}, sample={"timestamp_ns": index * 25_000_000})
        writer.save_episode(outcome="success")
        self.wait_until(writer.is_ready)
        self.assertEqual(len(self.colour_files()), 3)

    def test_the_reuse_cache_does_not_leak_into_the_next_episode(self):
        pixels = np.full((16, 16, 3), 70, dtype=np.uint8)
        writer = self.make_writer(image_size=(16, 16))
        for index in range(2):
            writer.create_episode()
            writer.add_item({"color_0": pixels}, sample={"timestamp_ns": index * 25_000_000},
                            color_sequences={"color_0": 11})
            writer.save_episode(outcome="success")
            self.wait_until(writer.is_ready)
        # Episode 1 reuses sequence 11, but that file belongs to episode 0, so it
        # has to be written again under its own directory.
        self.assertEqual(self.colour_files(0), ["000000_color_0.jpg"])
        self.assertEqual(self.colour_files(1), ["000000_color_0.jpg"])
        self.assertTrue((self.directory / "episode_0001" / "colors" / "000000_color_0.jpg").is_file())

    def test_manifest_reports_the_measured_rate_and_keeps_the_declared_one(self):
        writer = self.make_writer(
            image_size=(16, 16), frequency=40,
            metadata={"images": {"color_0": {"fps": 30, "width": 16, "height": 16}}},
        )
        pixels = np.full((16, 16, 3), 30, dtype=np.uint8)
        writer.create_episode()
        # Three samples over two seconds holding one camera frame.
        for timestamp in (0, 1_000_000_000, 2_000_000_000):
            writer.add_item({"color_0": pixels}, sample={"timestamp_ns": timestamp},
                            color_sequences={"color_0": 5})
        writer.save_episode(outcome="success")
        self.wait_until(writer.is_ready)
        info = self.manifest()["info"]
        self.assertAlmostEqual(info["image"]["fps"], 0.5)
        self.assertEqual(info["image"]["declared_fps"], 40)
        self.assertAlmostEqual(info["images"]["color_0"]["fps"], 0.5)
        self.assertEqual(info["images"]["color_0"]["declared_fps"], 30)
        self.assertEqual(info["images"]["color_0"]["unique_frames"], 1)
        self.assertEqual(info["images"]["color_0"]["frames"], 3)

    def test_a_single_sample_episode_leaves_the_declared_rate_alone(self):
        # No elapsed time means no measurable rate; guessing one would be worse
        # than repeating what the camera config asked for.
        writer = self.make_writer(image_size=(16, 16), frequency=40)
        writer.create_episode()
        writer.add_item({"color_0": np.zeros((16, 16, 3), dtype=np.uint8)},
                        sample={"timestamp_ns": 7})
        writer.save_episode(outcome="success")
        self.wait_until(writer.is_ready)
        self.assertEqual(self.manifest()["info"]["image"]["fps"], 40)

    def test_create_and_save_do_not_wait_for_filesystem(self):
        writer = self.make_writer()
        entered, release = self.hold_worker_start(writer)
        self.assertTrue(writer.create_episode())
        self.assertTrue(entered.wait(1.0))
        self.assertFalse(self.directory.exists())
        writer.add_item({})
        writer.save_episode("failure")
        self.assertFalse(writer.is_ready())
        self.assertFalse(self.directory.exists())
        release.set()
        writer.close()
        self.assertEqual(self.manifest()["outcome"], "failure")

    def test_queued_frames_own_image_and_state_snapshots(self):
        writer = self.make_writer()
        entered, release = self.hold_worker_start(writer)
        writer.create_episode()
        self.assertTrue(entered.wait(1.0))
        image = np.full((4, 6, 3), 60, dtype=np.uint8)
        state = {"left_arm": {"qpos": [0.1]}}
        sample = {"sources": {"xr": {"sequence": 3}}}
        writer.add_item({"color_0": image}, states=state, sample=sample)
        image[:] = 240
        state["left_arm"]["qpos"][0] = 0.9
        sample["sources"]["xr"]["sequence"] = 99
        release.set()
        writer.close()
        row = self.frames()[0]
        saved = cv2.imread(str(self.directory / "episode_0000" / row["colors"]["color_0"]))
        self.assertEqual(int(saved[0, 0, 0]), 60)
        self.assertEqual(row["states"]["left_arm"]["qpos"], [0.1])
        self.assertEqual(row["sample"]["sources"]["xr"]["sequence"], 3)

    def test_full_queue_immediately_reports_failure_and_marks_incomplete(self):
        writer = self.make_writer(queue_capacity=1)
        entered, release = self.hold_worker_start(writer)
        writer.create_episode()
        self.assertTrue(entered.wait(1.0))
        writer.add_item({})
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            writer.add_item({})
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            writer.raise_if_failed()
        release.set()
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            writer.close()
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "incomplete")
        self.assertIn("queue is full", manifest["error"])
        self.assertEqual(manifest["frame_count"], 0)

    def test_imwrite_false_does_not_commit_broken_image_reference(self):
        writer = self.make_writer()
        with mock.patch.object(episode_writer.cv2, "imwrite", return_value=False):
            writer.create_episode()
            writer.add_item({"color_0": np.zeros((4, 6, 3), dtype=np.uint8)})
            self.wait_until(lambda: not writer.worker_thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "Failed to save colors/"):
            writer.close()
        self.assertEqual(self.frames(), [])
        self.assertEqual(self.manifest()["status"], "incomplete")
        self.assertEqual(self.manifest()["frame_count"], 0)

    def test_final_manifest_disk_failure_propagates_without_hanging(self):
        writer = self.make_writer()
        write_manifest = writer._write_manifest

        def fail_complete(status):
            if status == "complete":
                raise OSError("disk full during final manifest")
            write_manifest(status)

        with mock.patch.object(writer, "_write_manifest", side_effect=fail_complete):
            writer.create_episode()
            writer.add_item({})
            with self.assertRaisesRegex(RuntimeError, "disk full during final manifest"):
                writer.close()
        self.assertFalse(writer.worker_thread.is_alive())
        self.assertEqual(self.manifest()["status"], "incomplete")
        self.assertEqual(self.manifest()["frame_count"], 1)
        self.assertEqual(len(self.frames()), 1)

    def test_new_episode_creation_failure_preserves_previous_episode(self):
        writer = self.make_writer()
        writer.create_episode()
        writer.add_item({})
        writer.save_episode("success")
        self.wait_until(writer.is_ready)
        original_manifest = self.manifest()
        mkdir = Path.mkdir

        def deny_task_directory(path, *args, **kwargs):
            if path == self.directory:
                raise PermissionError("task directory is now read-only")
            return mkdir(path, *args, **kwargs)

        with mock.patch.object(Path, "mkdir", new=deny_task_directory):
            self.assertTrue(writer.create_episode())
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                writer.close()
        self.assertEqual(self.manifest(), original_manifest)
        self.assertFalse((self.directory / "episode_0001").exists())

    def test_close_timeout_is_bounded_and_worker_cannot_later_claim_success(self):
        writer = self.make_writer()
        entered, release = self.hold_worker_start(writer)
        writer.create_episode()
        self.assertTrue(entered.wait(1.0))
        with mock.patch.object(episode_writer, "CLOSE_TIMEOUT", 0.03):
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                writer.close()
            self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(writer.worker_thread.daemon)
        release.set()
        self.wait_until(lambda: not writer.worker_thread.is_alive())
        self.assertEqual(self.manifest()["status"], "incomplete")
        self.assertIn("did not stop", self.manifest()["error"])

    def test_close_finishes_active_episode_once_and_is_idempotent(self):
        writer = self.make_writer()
        writer.create_episode()
        writer.add_item({}, sample={"timestamp_ns": 1})
        writer.close()
        first = self.manifest()
        writer.close()
        self.assertEqual(self.manifest(), first)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["outcome"], "unspecified")
        self.assertFalse(writer.is_ready())

    def test_main_thread_abort_preserves_frames_but_marks_incomplete(self):
        writer = self.make_writer()
        writer.create_episode()
        writer.add_item({}, sample={"timestamp_ns": 1})
        self.wait_until(lambda: writer._frame_count == 1)
        writer.abort(RuntimeError("camera frame is stale before enqueue"))
        with self.assertRaisesRegex(RuntimeError, "camera frame is stale"):
            writer.raise_if_failed()
        with self.assertRaisesRegex(RuntimeError, "camera frame is stale"):
            writer.close()
        self.assertEqual(self.manifest()["status"], "incomplete")
        self.assertEqual(self.manifest()["frame_count"], 1)
        self.assertEqual(len(self.frames()), 1)

    def test_abort_between_episodes_does_not_relabel_completed_data(self):
        writer = self.make_writer()
        writer.create_episode()
        writer.add_item({})
        writer.save_episode("success")
        self.wait_until(writer.is_ready)
        first = self.manifest()
        writer.abort("control fault while not recording")
        writer.close()
        self.assertEqual(self.manifest(), first)

    def test_abort_pending_creation_only_marks_the_new_episode(self):
        writer = self.make_writer()
        writer.create_episode()
        writer.add_item({})
        writer.save_episode("success")
        self.wait_until(writer.is_ready)
        first = self.manifest()
        entered, release = self.hold_worker_start(writer)
        writer.create_episode()
        self.assertTrue(entered.wait(1.0))
        writer.abort("tracking failed during episode startup")
        release.set()
        with self.assertRaisesRegex(RuntimeError, "tracking failed"):
            writer.close()
        self.assertEqual(self.manifest(), first)
        self.assertEqual(self.manifest(1)["status"], "incomplete")
        self.assertEqual(self.manifest(1)["frame_count"], 0)


if __name__ == "__main__":
    unittest.main()
