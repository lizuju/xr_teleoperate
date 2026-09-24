import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_teleop_episode import (SOURCE_NAMES, TRAINING_GROUPS, TRAINING_CAMERAS,
                                 training_frame_rejections, validate_episode)
from export_r1_training_dataset import export_dataset
from teleop.utils.act_dataset import R1ACTDataset, load_act_data
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.r1_capture import R1Capture
from test_r1_capture import build


class TrainingExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "recordings"
        self.source.mkdir()
        self.output = self.root / "dataset"

    def episode(self, number=0, count=8, outcome="success", base_value=0.1):
        directory = self.source / f"episode_{number:04d}"
        (directory / "colors").mkdir(parents=True)
        info = {"robot": "R1_A7", "end_effector": "linker_o6", "frequency": 40,
                "image": {"width": 16, "height": 12},
                "joint_names": {g: [str(i) for i in range(n)] for g, n in TRAINING_GROUPS.items()},
                "images": {k: {"width": 16, "height": 12, "fps": 40} for k in TRAINING_CAMERAS}}
        manifest = {"schema": "xr_teleop_episode_v2", "status": "complete", "episode_id": number,
                    "frame_count": count, "frames": "frames.jsonl", "outcome": outcome,
                    "info": info, "text": {"goal": "pick cup"}}
        rows = []
        for i in range(count):
            now = 10_000_000_000 + i * 25_000_000
            sources = {n: {"received_monotonic_ns": now, "age_ms": 0.0, "fresh": True, "sequence": i+1}
                       for n in (*SOURCE_NAMES, "left_wrist_image", "right_wrist_image")}
            colors = {}
            for k in TRAINING_CAMERAS:
                name = f"colors/{i:04d}_{k}.png"
                cv2.imwrite(str(directory / name), np.full((12, 16, 3), [20, 60, 200], np.uint8))
                colors[k] = name
            for n in ("image", "left_wrist_image", "right_wrist_image"):
                sources[n].update(repeated=False, offset_ms=0.0)
            arm = {"arm_q": [base_value + i*.001]*14, "arm_tau": [0.0]*14,
                   "head_q": [0.0]*2, "waist_q": 0.0, "monotonic_ns": now, "sequence": i+1}
            rows.append({"idx": i, "colors": colors,
                         "states": {g: {"qpos": [base_value + i*.001]*n} for g,n in TRAINING_GROUPS.items()},
                         "actions": {g: {"qpos": [base_value + .2 + i*.001]*n} for g,n in TRAINING_GROUPS.items()},
                         "sample": {"timestamp_ns": now + 1_000_000_000_000, "monotonic_ns": now,
                                    "mode": "following", "sources": sources, "xr": {},
                                    "commands": {"arm": {"requested": arm, "published": copy.deepcopy(arm)},
                                                 "hands": {"published": {s: {"q": [.3]*6, "mode": 1,
                                                                             "sequence": i+1, "monotonic_ns": now}
                                                                          for s in ("left", "right")}}},
                                    "camera_alignment": {"anchor": "head", "anchor_monotonic_ns": now,
                                                         "offset_ms": {"head": 0.0, "left_wrist": 0.0, "right_wrist": 0.0},
                                                         "skew_ms": 0.0, "tolerance_ms": 25.0, "aligned": True}}})
        self.write(directory, manifest, rows)
        return directory, manifest, rows

    def write(self, directory, manifest, rows):
        (directory / "episode.json").write_text(json.dumps(manifest))
        (directory / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_export_tensor_semantics_rgb_no_imu_and_no_action_shift(self):
        directory, manifest, rows = self.episode()
        result = validate_episode(directory)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["training_no_imu"]["eligible_frames"], 8)
        dataset = export_dataset(self.source, self.output, min_frames=2)
        self.assertFalse(dataset["imu_used"])
        self.assertEqual(dataset["state_dim"], 29)
        self.assertEqual(dataset["splits"]["val"], [])
        with h5py.File(self.output / "episode_0.hdf5") as h:
            keys = []
            h.visit(keys.append)
            self.assertFalse(any("imu" in key for key in keys))
            np.testing.assert_allclose(h["action"][0], [.3]*29)
            np.testing.assert_allclose(h["observations/qpos"][0], [.1]*29)
            np.testing.assert_array_equal(h["observations/images/head_left"][0, 0, 0], [200, 60, 20])
        loader = R1ACTDataset(self.output, chunk_size=4)
        images, qpos, actions, padding = loader[6]
        self.assertEqual(tuple(images.shape), (4, 3, 240, 320))
        self.assertEqual(tuple(qpos.shape), (29,))
        self.assertEqual(padding.tolist(), [False, False, True, True])
        recovered = actions[0].numpy() * loader.stats["action_std"] + loader.stats["action_mean"]
        np.testing.assert_allclose(recovered, [.306]*29)
        self.assertTrue(np.all(actions[2:].numpy() == 0))
        train, val, _, _ = load_act_data(self.output, batch_size=2, chunk_size=3)
        self.assertIsNone(val)
        self.assertEqual(tuple(next(iter(train))[2].shape), (2, 3, 29))

    def test_filter_splits_boundaries_outcomes_and_original_episode_holdout(self):
        directory, manifest, rows = self.episode(count=10)
        rows[3]["sample"]["mode"] = "tracking_hold"
        rows[6]["sample"]["commands"]["hands"]["published"]["left"]["mode"] = 0
        self.write(directory, manifest, rows)
        other, meta, samples = self.episode(1, base_value=.8)
        meta["text"]["goal"] = "pick cup, second trial"
        self.write(other, meta, samples)
        self.episode(2, outcome="failure")
        self.episode(3, outcome="discarded")
        self.episode(4, outcome="unspecified")
        dataset = export_dataset(self.source, self.output, min_frames=2)
        segments = [e for e in dataset["episodes"] if e["source_id"] == 0]
        self.assertEqual([(e["start_idx"],e["stop_idx"]) for e in segments], [(0,3),(4,6),(7,10)])
        train_sources = {dataset["episodes"][i]["source_id"] for i in dataset["splits"]["train"]}
        val_sources = {dataset["episodes"][i]["source_id"] for i in dataset["splits"]["val"]}
        self.assertFalse(train_sources & val_sources)
        self.assertEqual(train_sources | val_sources, {0,1})
        train_values = []
        for i in dataset["splits"]["train"]:
            with h5py.File(self.output / f"episode_{i}.hdf5") as h:
                train_values.extend(h["observations/qpos"][:])
        np.testing.assert_allclose(dataset["normalization"]["qpos_mean"], np.mean(train_values, axis=0), rtol=1e-5)
        for i in range(2,5):
            self.assertEqual(dataset["sources"][i]["excluded_reason"], "not_success")

    def test_missing_images_invalid_sources_and_no_overwrite(self):
        directory, manifest, rows = self.episode()
        (directory / rows[0]["colors"]["color_0"]).unlink()
        with self.assertRaisesRegex(ValueError, "No eligible"):
            export_dataset(self.source, self.output, min_frames=2)
        self.assertEqual(json.loads((self.output / "dataset.json").read_text())["status"], "failed")
        with self.assertRaisesRegex(ValueError, "already exists"):
            export_dataset(self.source, self.output)
        with self.assertRaisesRegex(ValueError, "not complete"):
            R1ACTDataset(self.output)

    def test_imu_invalid_or_dropped_does_not_gate_vision_joint_policy(self):
        _, manifest, rows = self.episode()
        row = rows[0]
        row["sample"]["sensor_alignment"] = {"clock_valid": True, "stereo_aligned": True,
                                               "feedback_aligned": True, "imu_valid": False,
                                               "imu_dropped_packets": 12, "usable": False,
                                               "usable_with_imu": False}
        row["sample"]["imu"] = {"quaternion": [0]*4}
        self.assertEqual(training_frame_rejections(row, manifest["info"]), [])
        row["sample"]["sensor_alignment"]["clock_valid"] = False
        self.assertIn("sensor_unaligned", training_frame_rejections(row, manifest["info"]))

    def test_time_gaps_split_and_stale_tracking_rejected(self):
        directory, manifest, rows = self.episode(count=8)
        rows[3]["sample"]["sources"]["xr"].update(
            received_monotonic_ns=rows[3]["sample"]["monotonic_ns"]-150_000_000, age_ms=150.0)
        self.write(directory, manifest, rows)
        report = validate_episode(directory)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["training_no_imu"]["excluded_counts"]["tracking_stale"], 1)
        rows.pop(3)
        for i, r in enumerate(rows):
            r["idx"] = i
        manifest["frame_count"] = len(rows)
        self.write(directory, manifest, rows)
        report = validate_episode(directory)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["training_no_imu"]["sampling_gap_boundaries"], 1)
        self.assertEqual(report["training_no_imu"]["segments"], [{"start_idx":0,"stop_idx":3},{"start_idx":3,"stop_idx":7}])

    def test_new_sensor_recording_with_zero_imu_exports_real_joint_and_image_inputs(self):
        _, manifest, _ = self.episode()
        info = manifest["info"]
        info["sensor_sync"] = {"schema": "r1_sensor_sync_v1"}
        recording = self.root / "new_recording"
        writer = EpisodeWriter(recording, metadata=info, rerun_log=False, quality_report=True)
        writer.create_episode()
        for i in range(8):
            now = 10_000_000_000 + i * 25_000_000
            arm, hand, loop, xr, head = build(now, image_shape=(12,32))
            arm.data["state"]["sequence"] = i+1
            arm.data["state"]["imu"].update(quaternion=[0.0]*4, valid=False)
            timing = {"clock_valid": True, "clock_id": "pc2", "mapped_monotonic_ns": now,
                      "clock_measured_monotonic_ns": now, "clock_uncertainty_ns": 1000,
                      "stereo_skew_ns": 0}
            head.timing = timing
            head.sequence = i+1
            wrist = SimpleNamespace(bgr=np.zeros((12,16,3),np.uint8), sequence=i+1,
                                    received_monotonic_ns=now, timing=timing)
            capture = R1Capture(arm, hand, loop, .5, (12,32),
                                wrist_image_shapes={"left": (12,16), "right": (12,16)})
            with patch('teleop.utils.r1_capture.time.monotonic_ns', return_value=now), \
                    patch('teleop.utils.r1_capture.time.time_ns', return_value=now+1_000_000_000_000):
                frame = capture.frame(xr, head, "following", {"left":wrist,"right":wrist})
            self.assertFalse(frame["sample"]["sensor_alignment"]["usable_with_imu"])
            writer.add_item(**frame)
        writer.save_episode("success")
        writer.close()
        writer._quality_process.wait(timeout=20)
        episode = next(recording.glob("episode_*"))
        report = json.loads((episode/"quality.json").read_text())
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["imu_invalid_frames"], 8)
        self.assertEqual(report["training_no_imu"]["eligible_frames"], 8)
        result = export_dataset(recording, self.output, min_frames=2)
        self.assertEqual(len(result["episodes"]),1)
        self.assertEqual(result["sources"][0]["timing_basis"],"mapped_source")

    def test_worker_drains_multiple_episodes_and_summary_preserves_labels(self):
        first, _, _ = self.episode()
        second, _, _ = self.episode(1, outcome="discarded")
        result = subprocess.run([sys.executable, str(ROOT / "tools/check_teleop_episode.py"), "--worker"],
                                input=json.dumps(str(first))+"\n"+json.dumps(str(second))+"\n",
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IMU 不参与训练", result.stdout)
        self.assertEqual(json.loads((first / "quality.json").read_text())["training_no_imu"]["eligible_frames"], 8)
        self.assertFalse(json.loads((second / "quality.json").read_text())["training_no_imu"]["demonstration_eligible"])

    def test_slow_quality_check_does_not_block_next_recording_and_survives_close(self):
        checker = self.root / "slow_checker.py"
        checker.write_text("import sys,json,time\nfrom pathlib import Path\nfor line in sys.stdin:\n time.sleep(.6)\n (Path(json.loads(line))/'quality.json').write_text('{}')\n")
        actual_popen = subprocess.Popen
        def slow_popen(command, **kwargs):
            return actual_popen([sys.executable, str(checker)], **kwargs)
        writer = EpisodeWriter(self.root / "writer", rerun_log=False, quality_report=True)
        self.addCleanup(writer.close)
        with patch('teleop.utils.episode_writer.subprocess.Popen', side_effect=slow_popen):
            for _ in range(2):
                self.assertTrue(writer.create_episode())
                writer.add_item({})
                writer.save_episode("success")
                deadline = time.monotonic() + .4
                while not writer.is_ready() and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertTrue(writer.is_ready())
            paths = list((self.root / "writer").glob("episode_*"))
            self.assertFalse(any((p/"quality.json").exists() for p in paths))
            writer.close()
            writer._quality_process.wait(timeout=5)
            self.assertTrue(all((p/"quality.json").exists() for p in paths))


if __name__ == "__main__":
    unittest.main()
