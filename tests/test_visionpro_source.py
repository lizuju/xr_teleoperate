from copy import deepcopy
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils import visionpro_source


def packet(sample=50.0, received=100.0):
    result = {
        "tracking_protocol_version": 1,
        "received_monotonic": received,
        "sample_time": sample,
        "prediction_seconds": 0.0,
        "head": np.eye(4).tolist(),
    }
    result["head"][1][3] = 1.6
    for key in ("head", "left", "right"):
        result[f"{key}_time"] = sample
        result[f"{key}_valid"] = True
    for side, x in (("left", -0.3), ("right", 0.3)):
        wrist = np.eye(4)
        wrist[:3, 3] = [x, 1.25, -0.6]
        joints = np.tile(np.eye(4), (25, 1, 1))
        joints[:, 0, 3] = np.arange(25) * 0.005
        joints[:, 2, 3] = np.sin(np.arange(25)) * 0.02
        result[f"{side}_wrist"] = wrist.tolist()
        result[f"{side}_joints"] = joints.tolist()
    return result


class VisionProMotionSourceTest(unittest.TestCase):
    def setUp(self):
        self.clock = mock.patch.object(visionpro_source.time, "monotonic", return_value=100.0).start()
        self.addCleanup(mock.patch.stopall)
        self.source = visionpro_source.VisionProMotionSource(host=None, python=None, timeout=0.25)

    def test_initial_snapshot_is_invalid_and_has_rigid_defaults(self):
        sample = self.source.get_hand_motion_snapshot()
        self.assertFalse(sample["motion_data_ready"])
        self.assertEqual(sample["motion_data_timestamp"], 0.0)
        for side in ("left", "right"):
            self.assertEqual(sample[f"{side}_hand_timestamp"], 0.0)
            self.assertAlmostEqual(np.linalg.det(sample[f"{side}_arm_pose"]), 1.0)
        self.assertFalse(self.source.consume_realign_required())

    def test_native_axes_match_webxr_for_both_hands(self):
        data = packet()
        rotation = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        for side in ("left", "right"):
            wrist = np.asarray(data[f"{side}_wrist"])
            wrist[:3, :3] = rotation
            data[f"{side}_wrist"] = wrist.tolist()
        original = deepcopy(data)
        self.source.accept_packet(data)
        sample = self.source.get_hand_motion_snapshot(include_orientations=True)
        np.testing.assert_allclose(sample["left_arm_pose"][:3, :3], [
            [0, 1, 0], [0, 0, -1], [-1, 0, 0],
        ])
        np.testing.assert_allclose(sample["right_arm_pose"][:3, :3], [
            [0, -1, 0], [0, 0, 1], [-1, 0, 0],
        ])
        for side, x in (("left", -0.3), ("right", 0.3)):
            np.testing.assert_allclose(sample[f"{side}_hand_positions"][0], [x, 1.25, -0.6])
            np.testing.assert_allclose(sample[f"{side}_hand_positions"][24], [
                x, 1.37, -0.6 + np.sin(24) * 0.02,
            ])
            np.testing.assert_allclose(
                sample[f"{side}_hand_orientations"][24], sample[f"{side}_arm_pose"][:3, :3],
            )
        np.testing.assert_allclose(self.source.head_pose, data["head"])
        self.assertEqual(data, original)

    def test_anchor_age_is_preserved_in_host_time(self):
        data = packet()
        data["left_time"] -= 0.1
        self.source.accept_packet(data)
        sample = self.source.get_hand_motion_snapshot()
        self.assertAlmostEqual(sample["left_hand_timestamp"], 99.9)
        self.assertEqual(sample["right_hand_timestamp"], 100.0)
        self.assertAlmostEqual(sample["motion_data_timestamp"], 99.9)
        self.assertAlmostEqual(self.source.get_tracking_diagnostics()["left_age_ms"], 100.0)

    def test_prediction_does_not_create_a_future_host_timestamp(self):
        data = packet()
        data["prediction_seconds"] = 0.03
        for key in ("head", "left", "right"):
            data[f"{key}_time"] += 0.03
        self.source.accept_packet(data)
        sample = self.source.get_hand_motion_snapshot()
        self.assertTrue(sample["motion_data_ready"])
        self.assertEqual(sample["motion_data_timestamp"], 100.0)

    def test_duplicate_anchors_never_refresh_motion_age_or_sequence(self):
        self.source.accept_packet(packet())
        initial = self.source.get_hand_motion_snapshot()
        for now in (100.1, 100.2, 100.3):
            self.clock.return_value = now
            data = packet(sample=50.0 + now - 100.0, received=now)
            for key in ("head", "left", "right"):
                data[f"{key}_time"] = 50.0
            self.source.accept_packet(data)
        sample = self.source.get_hand_motion_snapshot()
        self.assertFalse(sample["motion_data_ready"])
        self.assertEqual(sample["motion_sample_seq"], initial["motion_sample_seq"])
        self.assertEqual(sample["left_hand_timestamp"], 0.0)
        self.assertTrue(self.source.consume_realign_required())

    def test_one_missing_hand_preserves_other_and_cached_poses(self):
        self.source.accept_packet(packet())
        before = self.source.get_hand_motion_snapshot()
        self.clock.return_value = 100.05
        data = packet(sample=50.05, received=100.05)
        data["right_valid"] = False
        data["right_wrist"] = None
        data["right_joints"] = None
        data["left_wrist"][0][3] -= 0.05
        self.source.accept_packet(data)
        sample = self.source.get_hand_motion_snapshot()
        self.assertTrue(sample["motion_data_ready"])
        self.assertEqual(sample["left_hand_timestamp"], 100.05)
        self.assertEqual(sample["right_hand_timestamp"], 0.0)
        self.assertEqual(sample["motion_data_timestamp"], 0.0)
        np.testing.assert_array_equal(sample["right_arm_pose"], before["right_arm_pose"])
        np.testing.assert_array_equal(sample["right_hand_positions"], before["right_hand_positions"])
        self.assertAlmostEqual(sample["left_hand_positions"][0, 0], -0.35)
        self.assertFalse(self.source.consume_realign_required())

    def test_one_missing_hand_on_first_packet_does_not_block_other(self):
        data = packet()
        data["left_valid"] = False
        data["left_wrist"] = None
        data["left_joints"] = None
        self.source.accept_packet(data)
        sample = self.source.get_hand_motion_snapshot()
        self.assertTrue(sample["motion_data_ready"])
        self.assertEqual(sample["left_hand_timestamp"], 0.0)
        self.assertEqual(sample["right_hand_timestamp"], 100.0)
        self.assertAlmostEqual(np.linalg.det(sample["left_arm_pose"]), 1.0)

    def test_head_loss_invalidates_both_hands_and_latches_realignment(self):
        self.source.accept_packet(packet())
        self.clock.return_value = 100.05
        data = packet(sample=50.05, received=100.05)
        data["head_valid"] = False
        data["head"] = None
        self.source.accept_packet(data)
        sample = self.source.get_hand_motion_snapshot()
        self.assertFalse(sample["motion_data_ready"])
        self.assertEqual(sample["left_hand_timestamp"], 0.0)
        self.assertEqual(sample["right_hand_timestamp"], 0.0)
        self.assertFalse(self.source.get_tracking_diagnostics()["head_tracking"])
        self.assertTrue(self.source.consume_realign_required())
        self.assertFalse(self.source.consume_realign_required())
        self.clock.return_value = 100.1
        self.source.accept_packet(packet(sample=50.1, received=100.1))
        self.assertTrue(self.source.get_hand_motion_snapshot()["motion_data_ready"])
        self.assertFalse(self.source.consume_realign_required())

    def test_head_loss_event_survives_recovery_until_consumed(self):
        self.source.accept_packet(packet())
        self.clock.return_value = 100.05
        data = packet(sample=50.05, received=100.05)
        data["head_valid"] = False
        self.source.accept_packet(data)
        self.clock.return_value = 100.1
        self.source.accept_packet(packet(sample=50.1, received=100.1))
        self.assertTrue(self.source.consume_realign_required())
        self.assertFalse(self.source.consume_realign_required())

    def test_transport_timeout_invalidates_both_consumer_paths(self):
        self.source.accept_packet(packet())
        self.clock.return_value = 100.251
        for getter in (self.source.get_hand_motion_snapshot, self.source.pop_hand_motion_sample):
            sample = getter()
            self.assertFalse(sample["motion_data_ready"])
            self.assertEqual(sample["left_hand_timestamp"], 0.0)
            self.assertEqual(sample["right_hand_timestamp"], 0.0)
        self.assertTrue(self.source.consume_realign_required())

    def test_gap_before_new_packet_still_requires_realignment(self):
        self.source.accept_packet(packet())
        self.clock.return_value = 100.5
        self.source.accept_packet(packet(sample=50.5, received=100.5))
        self.assertTrue(self.source.get_hand_motion_snapshot()["motion_data_ready"])
        self.assertTrue(self.source.consume_realign_required())

    def test_delayed_sidecar_packet_cannot_become_fresh_when_read(self):
        self.clock.return_value = 101.0
        self.source.accept_packet(packet())
        sample = self.source.get_hand_motion_snapshot()
        self.assertFalse(sample["motion_data_ready"])
        self.assertEqual(sample["motion_data_timestamp"], 0.0)

    def test_invalid_protocol_clock_and_prediction_are_rejected(self):
        cases = (
            ("tracking_protocol_version", 0), ("received_monotonic", float("nan")),
            ("received_monotonic", 0.0), ("received_monotonic", 100.1),
            ("sample_time", float("inf")), ("sample_time", 0.0),
            ("prediction_seconds", -0.01), ("prediction_seconds", 0.11),
            ("left_valid", 1), ("right_time", float("nan")),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                source = visionpro_source.VisionProMotionSource(None, None)
                data = packet()
                data[key] = value
                with self.assertRaises(ValueError):
                    source.accept_packet(data)
                self.assertFalse(source.get_hand_motion_snapshot()["motion_data_ready"])

    def test_nonfinite_nonrigid_and_wrong_count_joint_poses_are_rejected(self):
        cases = []
        for value in (float("nan"), 2.0):
            data = packet()
            data["right_joints"][4][0][0] = value
            cases.append(data)
        for count in (24, 27):
            data = packet()
            data["right_joints"] = (data["right_joints"] * 2)[:count]
            cases.append(data)
        data = packet()
        data["head"][3][0] = 0.1
        cases.append(data)
        data = packet()
        data["left_wrist"][0][0] = -1.0
        cases.append(data)
        for index, data in enumerate(cases):
            with self.subTest(index=index):
                source = visionpro_source.VisionProMotionSource(None, None)
                with self.assertRaises(ValueError):
                    source.accept_packet(data)

    def test_hand_size_limit_matches_legacy_input_boundary(self):
        data = packet()
        data["right_joints"][24][0][3] = 0.5
        with self.assertRaises(ValueError):
            self.source.accept_packet(data)

    def test_rejected_packet_cannot_refresh_old_pose_or_poison_retry(self):
        self.source.accept_packet(packet())
        before = self.source.get_hand_motion_snapshot()
        before_head = self.source.head_pose
        self.clock.return_value = 100.05
        data = packet(sample=50.05, received=100.05)
        data["head"][0][3] = 0.1
        data["left_wrist"][0][3] = -0.4
        data["right_joints"][5][0][0] = float("nan")
        with self.assertRaises(ValueError):
            self.source.accept_packet(data)
        rejected = self.source.get_hand_motion_snapshot()
        self.assertEqual(rejected["left_hand_timestamp"], before["left_hand_timestamp"])
        np.testing.assert_array_equal(self.source.head_pose, before_head)
        np.testing.assert_array_equal(rejected["left_arm_pose"], before["left_arm_pose"])
        data["right_joints"][5][0][0] = 1.0
        self.source.accept_packet(data)
        after = self.source.get_hand_motion_snapshot()
        self.assertEqual(after["left_hand_timestamp"], 100.05)
        self.assertAlmostEqual(after["left_arm_pose"][0, 3], -0.4)
        self.assertAlmostEqual(self.source.head_pose[0, 3], 0.1)

    def test_sample_or_anchor_clock_regression_is_rejected(self):
        for key in ("sample_time", "head_time", "left_time", "right_time"):
            with self.subTest(key=key):
                source = visionpro_source.VisionProMotionSource(None, None)
                source.accept_packet(packet())
                data = packet(sample=50.05, received=100.05)
                data[key] = 49.99
                self.clock.return_value = 100.05
                with self.assertRaisesRegex(ValueError, "clock"):
                    source.accept_packet(data)

    def test_stale_anchor_and_excessive_prediction_invalidate_only_that_hand(self):
        for anchor_time in (49.7, 50.2):
            with self.subTest(anchor_time=anchor_time):
                source = visionpro_source.VisionProMotionSource(None, None)
                data = packet()
                data["left_time"] = anchor_time
                source.accept_packet(data)
                sample = source.get_hand_motion_snapshot()
                self.assertEqual(sample["left_hand_timestamp"], 0.0)
                self.assertEqual(sample["right_hand_timestamp"], 100.0)

    def test_snapshot_reads_return_copies_and_close_invalidates(self):
        self.source.accept_packet(packet())
        sample = self.source.get_hand_motion_snapshot()
        sample["left_arm_pose"][:] = 0.0
        head = self.source.head_pose
        head[:] = 0.0
        self.assertAlmostEqual(np.linalg.det(self.source.get_hand_motion_snapshot()["left_arm_pose"]), 1.0)
        self.assertAlmostEqual(np.linalg.det(self.source.head_pose), 1.0)
        self.source.close()
        self.assertFalse(self.source.get_hand_motion_snapshot()["motion_data_ready"])
        self.assertTrue(self.source.consume_realign_required())

    def test_coalesced_head_loss_latches_even_when_latest_pose_is_valid(self):
        self.source.accept_packet(packet())
        self.clock.return_value = 100.05
        latest = packet(50.05, 100.05)
        latest["tracking_lost"] = ["head"]
        latest["frames_coalesced"] = 20
        self.source.accept_packet(latest)
        self.assertTrue(self.source.get_hand_motion_snapshot()["motion_data_ready"])
        self.assertTrue(self.source.consume_realign_required())
        self.assertEqual(self.source.get_tracking_diagnostics()["frames_coalesced"], 20)

    def test_coalesced_hand_loss_is_visible_once_even_after_newer_packets(self):
        self.source.accept_packet(packet())
        self.clock.return_value = 100.05
        latest = packet(50.05, 100.05)
        latest["tracking_lost"] = ["left"]
        self.source.accept_packet(latest)
        self.clock.return_value = 100.1
        self.source.accept_packet(packet(50.1, 100.1))
        first = self.source.get_hand_motion_snapshot()
        self.assertEqual(first["left_hand_timestamp"], 0.)
        self.assertEqual(first["right_hand_timestamp"], 100.1)
        self.assertFalse(self.source.consume_realign_required())
        self.assertEqual(self.source.get_hand_motion_snapshot()["left_hand_timestamp"], 100.1)

    def test_hand_thread_cannot_consume_loss_before_arm_thread_sees_it(self):
        self.source.accept_packet(packet())
        latest = packet()
        latest["tracking_lost"] = ["left"]
        self.source.accept_packet(latest)
        hand_reads = []
        def read_hand():
            hand_reads.extend(self.source.get_hand_motion_snapshot() for _ in range(2))
        thread = threading.Thread(target=read_hand)
        thread.start()
        thread.join(timeout=1.)
        self.assertFalse(thread.is_alive())
        arm_reads = [self.source.get_hand_motion_snapshot() for _ in range(2)]
        for reads in (hand_reads, arm_reads):
            self.assertEqual(reads[0]["left_hand_timestamp"], 0.)
            self.assertEqual(reads[0]["right_hand_timestamp"], 100.)
            self.assertEqual(reads[1]["left_hand_timestamp"], 100.)

    def test_reconnected_stream_resets_clocks_but_requires_explicit_realignment(self):
        initial = packet()
        initial["stream_id"] = 1
        self.source.accept_packet(initial)
        self.clock.return_value = 100.1
        recovered = packet(10., 100.1)
        recovered["stream_id"] = 2
        self.source.accept_packet(recovered)
        self.assertTrue(self.source.get_hand_motion_snapshot()["motion_data_ready"])
        self.assertTrue(self.source.needs_realign)
        self.assertTrue(self.source.needs_realign)
        self.assertTrue(self.source.consume_realign_required())
        self.assertFalse(self.source.needs_realign)

    def test_bridge_best_offset_prevents_delayed_newest_packet_becoming_fresh(self):
        latest = packet(50., 100.)
        latest["clock_offset"] = 49.7
        self.source.accept_packet(latest)
        self.assertFalse(self.source.get_hand_motion_snapshot()["motion_data_ready"])


if __name__ == "__main__":
    unittest.main()
