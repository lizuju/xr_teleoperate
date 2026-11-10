"""Recording, checking and exporting a left Linker O6 with a right Dex1-1 gripper."""
import copy
import json
from pathlib import Path
import sys
import time
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
from check_teleop_episode import end_effectors, training_groups, validate_episode
from export_r1_training_dataset import export_dataset
from teleop.robot_control.robot_hand_dex1_r1 import Dex1PinchRetargeter, O6Dex1Retargeter
from teleop.utils.camera_calibration import camera_calibration_metadata, load_camera_calibration, without_cameras
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.r1_capture import R1Capture, capture_metadata
import test_r1_capture
import test_training_export

MIXED = {"left": "linker_o6", "right": "dex1_1"}
MIXED_NAME = "linker_o6_left+dex1_1_right"


def o6_hand():
    return SimpleNamespace(urdf_path=ROOT / "assets" / "r1" / "r1_a7.urdf",
                           hardware_joint_order=[f"joint_{index}" for index in range(6)],
                           hardware_lower=np.zeros(6), hardware_upper=np.ones(6), method="vector")


def to_mixed(manifest, rows):
    """Turn a two-O6 synthetic episode into left O6 + right Dex1-1."""
    info = manifest["info"]
    info.update(end_effector=MIXED_NAME, end_effectors=dict(MIXED))
    info["joint_names"]["right_ee"] = ["dex1_1_gripper"]
    for row in rows:
        for field in ("states", "actions"):
            row[field]["right_ee"]["qpos"] = row[field]["right_ee"]["qpos"][:1]
        sample = row["sample"]
        sample["aligned_states"]["hands"]["right"]["q"] = sample["aligned_states"]["hands"]["right"]["q"][:1]
        blocks = [sample["commands"]["hands"]]
        if "action_alignment" in sample:
            blocks.append(sample["action_alignment"]["hands"])
        for hands in blocks:
            hands["requested"]["right_q"] = [2.0]
            hands["published"]["right"]["q"] = [1.9]


class MixedMetadataTest(unittest.TestCase):
    def test_metadata_names_each_wrist_device(self):
        retargeter = O6Dex1Retargeter(o6_hand(), Dex1PinchRetargeter(closed_q=2.0), gripper_side="right")
        camera_config = {"head_camera": {"binocular": True, "image_shape": [448, 1088], "fps": 30, "enable_zmq": True},
                         "left_wrist_camera": {"enable_zmq": True, "image_shape": [480, 640], "fps": 30},
                         "right_wrist_camera": {"enable_zmq": True, "image_shape": [480, 640], "fps": 30}}
        args = SimpleNamespace(frequency=40.0, camera_sync_tolerance_ms=25.0)
        metadata = capture_metadata(args, camera_config, retargeter)
        self.assertEqual(metadata["end_effector"], MIXED_NAME)
        self.assertEqual(metadata["end_effectors"], MIXED)
        self.assertEqual(metadata["gripper_position_limits"], {"right": {"min_q_rad": 2.0, "max_q_rad": 5.4}})
        self.assertEqual(metadata["joint_names"]["right_ee"], ["dex1_1_gripper"])
        self.assertEqual(len(metadata["joint_names"]["left_ee"]), 6)
        self.assertEqual(set(metadata["models"]), {"arm", "left_hand"})
        self.assertEqual(set(metadata["hand_axis_normalization"]), {"left"})
        self.assertIn("rad", metadata["units"]["end_effectors"]["right"]["qpos"])
        self.assertIn("normalized", metadata["units"]["end_effectors"]["left"]["qpos"])
        self.assertEqual(end_effectors(metadata | {"robot": "R1_A7"}), MIXED)
        self.assertEqual(training_groups(metadata)["right_ee"], 1)

    def test_two_o6_metadata_keeps_its_old_name(self):
        hand = o6_hand()
        retargeter = SimpleNamespace(method="vector", mapping_name="o6", left=hand, right=hand)
        camera_config = {"head_camera": {"binocular": True, "image_shape": [448, 1088], "fps": 30, "enable_zmq": True}}
        metadata = capture_metadata(SimpleNamespace(frequency=40.0), camera_config, retargeter)
        self.assertEqual(metadata["end_effector"], "linker_o6")
        self.assertEqual(metadata["gripper_position_limits"], {})
        self.assertEqual(training_groups(metadata), {"left_arm": 7, "right_arm": 7, "left_ee": 6,
                                                     "right_ee": 6, "body": 3})

    def test_unknown_devices_are_unsupported(self):
        self.assertIsNone(end_effectors({"robot": "R1_A7", "end_effectors": {"left": "linker_o6", "right": "inspire"}}))
        self.assertIsNone(end_effectors({"robot": "G1_29", "end_effector": "linker_o6"}))

    def test_replaced_right_wrist_camera_loses_its_calibration(self):
        calibration = load_camera_calibration(ROOT / "assets" / "r1" / "camera_calibration.json")
        reduced = without_cameras(calibration, ["right_wrist"], "gripper camera swap")
        self.assertNotIn("right_wrist", reduced["cameras"])
        self.assertIn("left_wrist", reduced["cameras"])
        self.assertFalse(any(entry["camera"] == "right_wrist" for entry in reduced["hand_eye"].values()))
        self.assertIn("right_wrist", calibration["cameras"])
        block = camera_calibration_metadata(reduced, ["head_left", "head_right", "left_wrist", "right_wrist"])
        self.assertEqual(block["status"], "partial")
        self.assertEqual(block["missing_cameras"], ["right_wrist"])
        self.assertTrue(any("gripper camera swap" in warning for warning in block["warnings"]))
        self.assertEqual(block["sha256"], calibration["sha256"])


class MixedCaptureRoundTripTest(unittest.TestCase):
    def test_capture_writer_and_checker_accept_a_one_axis_right_hand(self):
        now = time.monotonic_ns()
        arm, hand, loop, xr, image = test_r1_capture.build(now)
        hand.data["state"]["right"] = {"q": [2.0], "monotonic_ns": now, "sequence": 3,
                                       "qvel": [0.0], "torque": [0.1], "temperature": [], "errors": []}
        hand.data["requested"]["right_q"] = [2.1]
        hand.data["published"]["right"] = {"q": [1.92], "mode": 1, "kp": 5.0, "kd": 0.05,
                                           "monotonic_ns": now, "sequence": 4, "request_sequence": 4}
        capture = R1Capture(arm, hand, loop, 0.5, (16, 32))
        names = {"left_arm": 7, "right_arm": 7, "left_ee": 6, "right_ee": 1, "body": 3}
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeWriter(directory, task_goal="gripper fixture", image_size=(16, 16), rerun_log=False,
                                   metadata={"frequency": 30, "robot": "R1_A7", "end_effector": MIXED_NAME,
                                             "end_effectors": dict(MIXED),
                                             "joint_names": {name: [str(i) for i in range(count)]
                                                             for name, count in names.items()}})
            writer.create_episode()
            writer.add_item(**capture.frame(xr, image, "following"))
            writer.save_episode(outcome="success")
            writer.close()
            report = validate_episode(writer.episode_dir)
            self.assertTrue(report["valid"], report["errors"])
            row = json.loads((writer.episode_dir / "frames.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["states"]["right_ee"]["qpos"], [2.0])
            self.assertEqual(row["states"]["right_ee"]["torque"], [0.1])
            self.assertEqual(row["actions"]["right_ee"]["qpos"], [2.1])
            self.assertEqual(len(row["states"]["left_ee"]["qpos"]), 6)


class MixedExportTest(unittest.TestCase):
    def setUp(self):
        self.helper = test_training_export.TrainingExportTests("test_aligned_actions_export_selected_real_requests_and_not_latest_raw_snapshot")
        self.helper.setUp()
        self.addCleanup(self.helper.temp.cleanup)

    def mixed_episode(self, number=0):
        directory, manifest, rows = self.helper.episode(number)
        self.helper.add_action_alignment(rows)
        to_mixed(manifest, rows)
        self.helper.write(directory, manifest, rows)
        return directory

    def test_checker_trains_from_the_one_axis_gripper(self):
        directory = self.mixed_episode()
        report = validate_episode(directory, min_segment_frames=2)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["training_no_imu"]["eligible_frames"], 8)

    def test_wrong_gripper_width_is_an_error(self):
        directory, manifest, rows = self.helper.episode(0)
        self.helper.add_action_alignment(rows)
        to_mixed(manifest, rows)
        rows[0]["sample"]["commands"]["hands"]["requested"]["right_q"] = [2.0] * 6
        self.helper.write(directory, manifest, rows)
        report = validate_episode(directory, min_segment_frames=2)
        self.assertFalse(report["valid"])
        self.assertTrue(any("right_q" in error for error in report["errors"]), report["errors"])

    def test_export_sizes_the_dataset_from_the_devices(self):
        self.mixed_episode()
        dataset = export_dataset(self.helper.source, self.helper.output, min_frames=2)
        self.assertEqual(dataset["state_dim"], 24)
        self.assertEqual(dataset["joint_groups"]["right_ee"], 1)
        self.assertEqual(dataset["end_effectors"], MIXED)
        self.assertEqual(dataset["units"][14:20], ["normalized_0_to_1"] * 6)
        self.assertEqual(dataset["units"][20], "rad")
        with h5py.File(self.helper.output / "episode_0.hdf5") as h:
            self.assertEqual(h["action"].shape[1], 24)
            self.assertEqual(h["observations/qpos"].shape[1], 24)
            np.testing.assert_allclose(h["action"][0][20], 2.0)

    def test_two_o6_and_gripper_episodes_do_not_mix(self):
        directory, manifest, rows = self.helper.episode(0)
        self.helper.add_action_alignment(rows)
        self.helper.write(directory, manifest, rows)
        self.mixed_episode(1)
        with self.assertRaisesRegex(ValueError, "Mixed frequency or joint order"):
            export_dataset(self.helper.source, self.helper.output, min_frames=2)


if __name__ == "__main__":
    unittest.main()
