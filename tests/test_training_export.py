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
from check_teleop_episode import (DEFAULT_MIN_SEGMENT_FRAMES, SOURCE_NAMES, TRAINING_GROUPS, TRAINING_CAMERAS,
                                 quality_summary, training_action, training_frame_rejections, training_timing, validate_episode)
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
            anchor = now - 50_000_000
            sources = {n: {"received_monotonic_ns": now, "age_ms": 0.0, "fresh": True, "sequence": i+1}
                       for n in (*SOURCE_NAMES, "left_wrist_image", "right_wrist_image")}
            colors = {}
            for k in TRAINING_CAMERAS:
                name = f"colors/{i:04d}_{k}.png"
                cv2.imwrite(str(directory / name), np.full((12, 16, 3), [20, 60, 200], np.uint8))
                colors[k] = name
            for n in ("image", "left_wrist_image", "right_wrist_image"):
                sources[n].update(repeated=False, offset_ms=0.0)
                sources[n]["timing"] = {
                    "source_monotonic_ns": anchor-1_000_000_000, "mapped_monotonic_ns": anchor,
                    "clock_offset_ns": 1_000_000_000, "clock_id": "pc2",
                    "clock_valid": True, "clock_measured_monotonic_ns": now-100_000_000,
                    "clock_uncertainty_ns": 1000, "stereo_skew_ns": 2_000_000 if n == "image" else None,
                    "timestamp_kind": "pc2_rtp_decoded_receive" if n == "image" else "pc2_v4l2_dequeue"}
            arm = {"arm_q": [base_value + i*.001]*14, "arm_tau": [0.0]*14,
                   "head_q": [0.0]*2, "waist_q": 0.0, "monotonic_ns": now-2_000_000, "sequence": i+1}
            imu = {"monotonic_ns": now, "sequence": i+1, "tick": i,
                   "quaternion": [1.0, 0.0, 0.0, 0.0], "gyroscope": [0.0]*3,
                   "accelerometer": [0.0]*3, "rpy": [0.0]*3, "temperature": 30, "valid": True}
            aligned_value = base_value - .05 + i*.001
            rows.append({"idx": i, "colors": colors,
                         "states": {g: {"qpos": [base_value + i*.001]*n} for g,n in TRAINING_GROUPS.items()},
                         "actions": {g: {"qpos": [base_value + .2 + i*.001]*n} for g,n in TRAINING_GROUPS.items()},
                         "sample": {"timestamp_ns": now + 1_000_000_000_000, "monotonic_ns": now,
                                    "mode": "following", "sources": sources, "xr": {},
                                    "imu": imu, "imu_packets": [imu],
                                    "aligned_states": {
                                        "robot": {"q": [aligned_value]*14, "head_q": [aligned_value]*2,
                                                  "waist_q": aligned_value, "monotonic_ns": anchor+1_000_000,
                                                  "imu": imu},
                                        "hands": {s: {"q": [aligned_value]*6,
                                                      "monotonic_ns": anchor+offset*1_000_000}
                                                  for s,offset in (("left",2),("right",-3))}},
                                    "sensor_alignment": {
                                        "schema": "r1_sensor_sync_v1", "target_monotonic_ns": anchor,
                                        "clock_valid": True, "stereo_aligned": True, "feedback_aligned": True,
                                        "imu_valid": True, "imu_dropped_packets": 0, "usable": True,
                                        "usable_with_imu": True,
                                        "feedback_offset_ms": {"robot": 1.0, "left_hand_feedback": 2.0,
                                                               "right_hand_feedback": -3.0}},
                                    "commands": {"arm": {"requested": arm, "published": copy.deepcopy(arm)},
                                                 "hands": {"requested": {"left_q": [.3]*6, "right_q": [.3]*6,
                                                                         "monotonic_ns": now-1_000_000, "sequence": i+1},
                                                           "published": {s: {"q": [.3]*6, "mode": 1,
                                                                             "sequence": i+1, "request_sequence": i+1,
                                                                             "monotonic_ns": now}
                                                                          for s in ("left", "right")}}},
                                    "camera_alignment": {"anchor": "head", "anchor_monotonic_ns": anchor,
                                                         "timestamp_basis": "mapped_source",
                                                         "offset_ms": {"head": 0.0, "left_wrist": 0.0, "right_wrist": 0.0},
                                                         "skew_ms": 0.0, "tolerance_ms": 25.0, "aligned": True}}})
        self.write(directory, manifest, rows)
        return directory, manifest, rows

    def write(self, directory, manifest, rows):
        (directory / "episode.json").write_text(json.dumps(manifest))
        (directory / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def add_action_alignment(self, rows):
        for row in rows:
            sample = row["sample"]
            arm, hands = copy.deepcopy(sample["commands"]["arm"]), copy.deepcopy(sample["commands"]["hands"])
            arm["published"]["request_sequence"] = arm["requested"]["sequence"]
            sample["action_alignment"] = {
                "schema": "r1_action_alignment_v1", "aligned": True,
                "anchor_monotonic_ns": hands["requested"]["monotonic_ns"],
                "arm": arm, "hands": hands,
                "tracking_fresh": {"left": True, "right": True},
                "hand_target_inputs": {side: {"received_monotonic_ns": sample["monotonic_ns"] - 10_000_000,
                                              "points": [[.1, .2, .3] for _ in range(25)]}
                                       for side in ("left", "right")}}

    def test_aligned_actions_export_selected_real_requests_and_not_latest_raw_snapshot(self):
        directory, manifest, rows = self.episode()
        self.add_action_alignment(rows)
        for row in rows:
            row["sample"]["commands"]["arm"]["requested"].update(
                arm_q=[.9] * 14, monotonic_ns=row["sample"]["monotonic_ns"] - 30_000_000)
            row["actions"]["left_arm"]["qpos"] = [.9] * 7
        self.write(directory, manifest, rows)
        report = validate_episode(directory, min_segment_frames=2)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["training_no_imu"]["eligible_frames"], 8)
        self.assertEqual(training_timing(rows[0]["sample"])["request_skew_ms"], 1.0)
        self.assertEqual(training_action(rows[0])["left_arm"], [.1] * 7)
        dataset = export_dataset(self.source, self.output, min_frames=2)
        self.assertEqual(dataset["sources"][0]["exported_frames"], 8)
        with h5py.File(self.output / "episode_0.hdf5") as h:
            np.testing.assert_allclose(h["action"][0], [.1] * 14 + [.3] * 12 + [0.] * 3)
            self.assertEqual(h["source/arm_request_ns"][0], 9_998_000_000)
            self.assertEqual(h["source/hand_request_ns"][0], 9_999_000_000)
            self.assertEqual(h["source/action_anchor_ns"][0], 9_999_000_000)
            self.assertTrue(h["source/action_alignment_used"][:].all())
            self.assertEqual(h["source/arm_request_sequence"][0], 1)
            np.testing.assert_array_equal(h["source/hand_input_ns"][0], [9_990_000_000] * 2)
            np.testing.assert_array_equal(h["source/hand_input_age_ms"][0], [10.0] * 2)
            self.assertEqual(list(h["source/hand_input_ns"].attrs["sides"]), ["left", "right"])
            self.assertIn("already include arm target shaping", h.attrs["action_semantics"])
        saved = json.loads((directory / "frames.jsonl").read_text().splitlines()[0])
        self.assertEqual(saved["actions"]["left_arm"]["qpos"], [.9] * 7)
        self.assertEqual(saved["sample"]["commands"]["arm"]["requested"]["monotonic_ns"], 9_970_000_000)

    def test_aligned_action_boundaries_reject_without_falling_back_to_raw_commands(self):
        _, manifest, rows = self.episode(count=1)
        self.add_action_alignment(rows)
        original = rows[0]
        mutations = {
            "unaligned": ("actions_unaligned", lambda a: a.update(aligned=False)),
            "unknown_schema": ("actions_unaligned", lambda a: a.update(schema="unknown")),
            "missing_arm": ("invalid_aligned_actions", lambda a: a.update(arm=None)),
            "missing_publication": ("arm_request_unmatched", lambda a: a["arm"].update(published=None)),
            "wrong_arm_publication": ("arm_request_unmatched", lambda a: a["arm"]["published"].update(request_sequence=0)),
            "early_arm_publication": ("arm_request_unmatched", lambda a: a["arm"]["published"].update(monotonic_ns=9_997_000_000)),
            "wrong_anchor": ("actions_unaligned", lambda a: a.update(anchor_monotonic_ns=9_998_000_000)),
            "late_arm_request": ("actions_unaligned", lambda a: a["arm"]["requested"].update(monotonic_ns=9_999_500_000)),
            "stale_source_input": ("hand_input_stale", lambda a: a["hand_target_inputs"]["left"].update(received_monotonic_ns=9_899_000_000)),
            "input_after_request": ("hand_input_stale", lambda a: a["hand_target_inputs"]["left"].update(received_monotonic_ns=9_999_500_000)),
            "missing_source_inputs": ("hand_input_stale", lambda a: a.pop("hand_target_inputs")),
            "tracking_invalid": ("actions_unaligned", lambda a: a["tracking_fresh"].update(left=False)),
            "short_selected_arm": ("invalid_aligned_actions", lambda a: a["arm"]["requested"].update(arm_q=[.1] * 13)),
        }
        for name, (reason, mutate) in mutations.items():
            with self.subTest(name=name):
                row = copy.deepcopy(original)
                mutate(row["sample"]["action_alignment"])
                self.assertIn(reason, training_frame_rejections(row, manifest["info"]))

    def test_selected_shape_errors_are_file_invalid_and_preserved(self):
        directory, manifest, rows = self.episode()
        self.add_action_alignment(rows)
        rows[0]["sample"]["action_alignment"]["hands"]["requested"]["left_q"] = [.3] * 5
        self.write(directory, manifest, rows)
        report = validate_episode(directory, min_segment_frames=2)
        self.assertFalse(report["valid"])
        self.assertTrue(any("action_alignment.hands.requested.left_q" in error for error in report["errors"]))
        self.assertFalse(report["training_no_imu"]["demonstration_eligible"])

    def test_legacy_rows_with_recorded_hand_inputs_check_the_real_input_age(self):
        _, manifest, rows = self.episode(count=1)
        row = rows[0]
        self.assertEqual(training_frame_rejections(row, manifest["info"]), [])
        row["sample"]["hand_target_inputs"] = {
            side: {"received_monotonic_ns": 9_990_000_000} for side in ("left", "right")}
        self.assertEqual(training_frame_rejections(row, manifest["info"]), [])
        row["sample"]["hand_target_inputs"]["left"]["received_monotonic_ns"] = 9_850_000_000
        self.assertIn("hand_input_stale", training_frame_rejections(row, manifest["info"]))
        self.assertEqual(training_timing(row["sample"])["left_hand_input_age_ms"], 150.0)

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
            np.testing.assert_allclose(h["observations/qpos"][0], [.05]*29)
            self.assertEqual(h["source/observation_ns"][0], 9_950_000_000)
            np.testing.assert_array_equal(h["source/hand_input_ns"][0], [0, 0])
            np.testing.assert_array_equal(h["source/hand_input_age_ms"][0], [-1.0, -1.0])
            self.assertEqual(h["source/hand_input_ns"].attrs["missing_value"], 0)
            self.assertEqual(h["source/feedback_ns/robot"][0], 9_951_000_000)
            self.assertEqual(h["source/hand_request_ns"][0], 9_999_000_000)
            self.assertEqual(h["source/hand_request_sequence"][0], 1)
            self.assertEqual(h["source/observation_skew_ms"][0], 5.0)
            self.assertEqual(h["source/observation_age_ms"][0], 53.0)
            self.assertEqual(h["source/arm_observation_delay_ms"][0], 48.0)
            self.assertEqual(h["source/hands_observation_delay_ms"][0], 49.0)
            self.assertEqual(h["source/camera_timing/image/clock_id"].asstr()[0], "pc2")
            self.assertEqual(h["source/camera_timing/image/stereo_skew_ns"][0], 2_000_000)
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

    def test_alignment_and_command_times_are_checked_from_actual_stamps(self):
        _, manifest, rows = self.episode(count=1)
        original = rows[0]
        self.assertEqual(training_frame_rejections(original, manifest["info"]), [])
        mutations = {
            "missing_wrist_clock": ("camera_clock_invalid", lambda s: s["sources"]["left_wrist_image"].pop("timing")),
            "request_skew": ("request_skew", lambda s: s["commands"]["hands"]["requested"].update(monotonic_ns=s["monotonic_ns"]-40_000_000)),
            "missing_request_stamp": ("request_stale", lambda s: s["commands"]["hands"]["requested"].pop("monotonic_ns")),
            "unpublished_request": ("hand_request_unmatched", lambda s: s["commands"]["hands"]["published"]["left"].update(request_sequence=0)),
            "publication_before_request": ("hand_request_unmatched", lambda s: s["commands"]["hands"]["published"]["left"].update(monotonic_ns=s["monotonic_ns"]-10_000_000)),
            "wide_feedback": ("observation_skew", lambda s: s["aligned_states"]["robot"].update(monotonic_ns=s["monotonic_ns"]-90_000_000)),
            "missing_aligned_hand": ("invalid_aligned_states", lambda s: s["aligned_states"]["hands"].pop("left")),
            "nan_aligned_hand": ("invalid_aligned_states", lambda s: s["aligned_states"]["hands"]["left"].update(q=[float("nan")]*6)),
            "future_second_eye": ("observation_skew", lambda s: s["sources"]["image"]["timing"].update(stereo_skew_ns=60_000_000)),
            "missing_stereo_time": ("observation_skew", lambda s: s["sources"]["image"]["timing"].pop("stereo_skew_ns")),
            "legacy_basis": ("sensor_unaligned", lambda s: s["camera_alignment"].update(timestamp_basis="host_receive")),
        }
        for name, (reason, mutate) in mutations.items():
            with self.subTest(name=name):
                row = copy.deepcopy(original)
                mutate(row["sample"])
                self.assertIn(reason, training_frame_rejections(row, manifest["info"]))

    def test_fresh_receive_cannot_hide_old_observation_and_report_uses_actual_basis(self):
        directory, manifest, rows = self.episode(count=1)
        sample = rows[0]["sample"]
        for name in ("image", "left_wrist_image", "right_wrist_image"):
            for field in ("source_monotonic_ns", "mapped_monotonic_ns"):
                sample["sources"][name]["timing"][field] -= 200_000_000
        sample["camera_alignment"]["anchor_monotonic_ns"] -= 200_000_000
        sample["sensor_alignment"]["target_monotonic_ns"] -= 200_000_000
        for state in (sample["aligned_states"]["robot"], *sample["aligned_states"]["hands"].values()):
            state["monotonic_ns"] -= 200_000_000
        self.assertEqual(training_timing(sample)["observation_age_ms"], 253.0)
        self.assertIn("observation_age", training_frame_rejections(rows[0], manifest["info"]))
        self.write(directory, manifest, rows)
        report = validate_episode(directory)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["training_no_imu"]["eligible_frames"], 0)
        self.assertEqual(report["training_no_imu"]["timing_basis"], "mapped_source")
        self.assertEqual(report["training_no_imu"]["timing"]["observation_age_ms"]["p95"], 253.0)

    def test_loader_rejects_previous_observation_semantics(self):
        self.episode()
        dataset = export_dataset(self.source, self.output, min_frames=2)
        dataset["schema"] = "r1_act_hdf5_v1"
        (self.output / "dataset.json").write_text(json.dumps(dataset))
        with self.assertRaises(ValueError):
            R1ACTDataset(self.output)

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
            r["sample"]["imu"]["sequence"] = i+1
            r["sample"]["sources"]["robot"]["sequence"] = i+1
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
                      "source_monotonic_ns": now-1_000_000_000, "clock_offset_ns": 1_000_000_000,
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
        self.assertIn("IMU 不参与训练", result.stderr)
        results = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([item["episode"] for item in results], [first.name, second.name])
        self.assertTrue(all(item["state"] == "complete" for item in results))
        self.assertTrue(results[0]["file_valid"])
        self.assertEqual(results[0]["min_segment_frames"], DEFAULT_MIN_SEGMENT_FRAMES)
        self.assertEqual(results[0]["exportable_frames"], 0)
        self.assertEqual(results[1]["outcome"], "discarded")
        self.assertEqual(json.loads((first / "quality.json").read_text())["training_no_imu"]["eligible_frames"], 8)
        self.assertFalse(json.loads((second / "quality.json").read_text())["training_no_imu"]["demonstration_eligible"])

    def test_worker_survives_closed_result_pipe_and_checks_all_episodes(self):
        first, _, _ = self.episode()
        second, _, _ = self.episode(1, outcome="failure")
        process = subprocess.Popen([sys.executable, str(ROOT / "tools/check_teleop_episode.py"), "--worker"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        process.stdout.close()
        process.stdin.write(json.dumps(str(first)) + "\n" + json.dumps(str(second)) + "\n")
        process.stdin.close()
        process.wait(timeout=20)
        stderr = process.stderr.read()
        process.stderr.close()
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(json.loads((first / "quality.json").read_text())["outcome"], "success")
        self.assertEqual(json.loads((second / "quality.json").read_text())["outcome"], "failure")

    def test_worker_report_failure_returns_episode_error_and_continues(self):
        first, _, _ = self.episode()
        (first / "quality.json").mkdir()
        second, _, _ = self.episode(1)
        result = subprocess.run([sys.executable, str(ROOT / "tools/check_teleop_episode.py"), "--worker"],
                                input=json.dumps(str(first)) + "\n" + json.dumps(str(second)) + "\n",
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        results = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(results[0]["episode"], first.name)
        self.assertEqual(results[0]["state"], "failed")
        self.assertTrue(results[0]["error"])
        self.assertEqual(results[1]["state"], "complete")
        self.assertTrue((second / "quality.json").is_file())

    def test_report_length_counts_match_exporter_and_custom_minimum(self):
        directory, manifest, rows = self.episode(count=85)
        rows[40]["sample"]["mode"] = "tracking_hold"
        self.write(directory, manifest, rows)
        default = validate_episode(directory)["training_no_imu"]
        self.assertEqual(default["min_segment_frames"], 40)
        self.assertEqual(default["eligible_frames"], 84)
        self.assertEqual(default["longest_segment_frames"], 44)
        self.assertEqual((default["exportable_segments"], default["exportable_frames"],
                          default["short_segment_frames"]), (2, 84, 0))
        report = validate_episode(directory, min_segment_frames=42)
        expected = report["training_no_imu"]
        self.assertEqual((expected["exportable_segments"], expected["exportable_frames"],
                          expected["short_segment_frames"]), (1, 44, 40))
        dataset = export_dataset(self.source, self.output, min_frames=42)
        exported = json.loads((self.output / dataset["sources"][0]["quality_report"]).read_text())
        self.assertEqual(exported["training_no_imu"], expected)
        self.assertEqual(dataset["sources"][0]["exported_frames"], expected["exportable_frames"])
        self.assertEqual(dataset["sources"][0]["short_segment_frames"], expected["short_segment_frames"])
        self.assertEqual(len(dataset["episodes"]), expected["exportable_segments"])

    def test_short_success_reports_no_exportable_segment(self):
        directory, _, _ = self.episode(count=8)
        report = validate_episode(directory)
        training = report["training_no_imu"]
        self.assertEqual(training["longest_segment_frames"], 8)
        self.assertEqual(training["short_segment_frames"], 8)
        self.assertFalse(training["demonstration_eligible"])
        self.assertIn("没有达到 40 帧", quality_summary(report))
        with self.assertRaisesRegex(ValueError, "No eligible segments"):
            export_dataset(self.source, self.output)

    def test_labels_and_incomplete_files_override_length_qualification(self):
        directory, manifest, rows = self.episode(count=40)
        for outcome in ("success", "failure", "discarded", "unspecified"):
            with self.subTest(outcome=outcome):
                manifest["outcome"] = outcome
                self.write(directory, manifest, rows)
                report = validate_episode(directory)
                self.assertEqual(report["training_no_imu"]["exportable_frames"], 40)
                self.assertEqual(report["training_no_imu"]["demonstration_eligible"], outcome == "success")
                if outcome != "success":
                    self.assertIn("当前标签不进入", quality_summary(report))
        manifest.update(status="incomplete", outcome="success")
        self.write(directory, manifest, rows)
        report = validate_episode(directory)
        self.assertFalse(report["training_no_imu"]["demonstration_eligible"])
        self.assertIn("文件不完整", quality_summary(report))

    def test_quality_minimum_rejects_invalid_lengths(self):
        directory, _, _ = self.episode()
        for value in (0, 1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_episode(directory, min_segment_frames=value)

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
