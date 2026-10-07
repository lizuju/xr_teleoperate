from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
HAS_GRPC = importlib.util.find_spec("grpc") is not None
if HAS_GRPC:
    from visionpro_bridge import LatestTracking
    from visionpro_protocol.handtracking_pb2 import HandUpdate


def message(sample=50., **changes):
    fields = dict(tracking_protocol_version=1, sample_time=sample,
                  head_time=sample, left_time=sample, right_time=sample,
                  head_valid=True, left_valid=True, right_valid=True)
    fields.update(changes)
    return HandUpdate(**fields)


@unittest.skipUnless(HAS_GRPC, "Run with the isolated Vision Pro receiver Python")
class LatestTrackingTest(unittest.TestCase):
    def setUp(self):
        self.pending = LatestTracking(.25)
        self.pending.begin_stream(1)

    def test_burst_keeps_only_latest_pose_and_original_receive_clock(self):
        for index in range(1000):
            self.pending.offer(message(50. + index * .001), 100. + index * .001)
        latest, packet = self.pending.take()
        self.assertAlmostEqual(latest.sample_time, 50.999)
        self.assertAlmostEqual(packet["received_monotonic"], 100.999)
        self.assertEqual(packet["frames_coalesced"], 999)
        self.assertEqual(packet["tracking_lost"], [])
        self.assertIsNone(self.pending.latest)

    def test_head_loss_inside_burst_survives_new_valid_pose(self):
        self.pending.offer(message(), 100.)
        self.pending.offer(message(50.01, head_valid=False), 100.01)
        self.pending.offer(message(50.02), 100.02)
        latest, packet = self.pending.take()
        self.assertTrue(latest.head_valid)
        self.assertEqual(packet["tracking_lost"], ["head"])
        self.pending.offer(message(50.03), 100.03)
        self.assertEqual(self.pending.take()[1]["tracking_lost"], [])

    def test_single_hand_loss_does_not_invalidate_other_hand(self):
        self.pending.offer(message(50., left_valid=False), 100.)
        self.pending.offer(message(50.01), 100.01)
        self.assertEqual(self.pending.take()[1]["tracking_lost"], ["left"])

    def test_gap_and_frozen_anchor_cannot_be_hidden_by_recovery(self):
        self.pending.offer(message(), 100.)
        self.pending.take()
        self.pending.offer(message(50.3), 100.3)
        self.assertIn("head", self.pending.take()[1]["tracking_lost"])
        for index in range(1, 5):
            sample = 50.3 + index * .1
            self.pending.offer(message(sample, head_time=50.3), sample + 50.)
        self.pending.offer(message(50.71), 100.71)
        self.assertIn("head", self.pending.take()[1]["tracking_lost"])

    def test_best_clock_offset_survives_dropped_packets(self):
        self.pending.offer(message(), 100.)
        self.pending.offer(message(50.1), 100.2)
        _, packet = self.pending.take()
        self.assertEqual(packet["clock_offset"], 50.)
        self.assertEqual(packet["received_monotonic"], 100.2)

    def test_network_age_and_bridge_wait_are_distinct_and_peaks_survive_coalescing(self):
        self.pending.offer(message(50.), 100.)
        self.pending.offer(message(50.01), 100.71)
        self.pending.offer(message(50.72), 100.72)
        with patch("visionpro_bridge.time.monotonic", return_value=100.92):
            _, packet = self.pending.take()
        transport = packet["transport"]
        self.assertAlmostEqual(transport["upstream_age_ms"], 0.)
        self.assertAlmostEqual(transport["upstream_age_max_ms"], 700.)
        self.assertAlmostEqual(transport["receive_gap_max_ms"], 710.)
        self.assertAlmostEqual(transport["bridge_pending_ms"], 200.)
        self.assertEqual(transport["upstream_over_100ms"], 1)
        self.assertEqual(transport["packets_received"], 3)
        self.assertGreater(transport["bytes_received"], 0)
        self.pending.begin_stream(2)
        self.assertEqual(self.pending.transport["upstream_age_max_ms"], 0.)
        self.assertAlmostEqual(transport["upstream_age_max_ms"], 700.)

    def test_protocol_and_clock_errors_are_checked_before_coalescing(self):
        self.pending.offer(message(), 100.)
        for changes in (dict(tracking_protocol_version=0), dict(sample_time=float("nan")),
                        dict(prediction_seconds=.2), dict(head_time=float("nan")),
                        dict(sample_time=49.9), dict(left_time=49.9)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.pending.offer(message(**changes), 100.01)

    def test_disconnect_discards_pending_pose_and_precedes_reconnected_pose(self):
        self.pending.offer(message(), 100.)
        self.pending.disconnect("connection ended")
        self.pending.begin_stream(2)
        self.pending.offer(message(10.), 100.01)
        latest, packet = self.pending.take()
        self.assertIsNone(latest)
        self.assertEqual(packet, {"disconnected": "connection ended"})
        latest, packet = self.pending.take()
        self.assertEqual(latest.sample_time, 10.)
        self.assertEqual(packet["stream_id"], 2)
        self.assertEqual(packet["clock_offset"], 90.01)
        self.assertIn("head", packet["tracking_lost"])

    def test_fatal_error_discards_pending_pose_and_finishes(self):
        self.pending.offer(message(), 100.)
        self.pending.disconnect("invalid protocol", fatal=True)
        self.assertEqual(self.pending.take(), (None, {"error": "invalid protocol"}))
        self.assertEqual(self.pending.take(), (None, None))

    def test_source_loss_counts_preserve_invalid_frames_skipped_before_network(self):
        self.pending.offer(message(diagnostics_version=1, head_anchor_valid=True,
                                   head_loss_seq=5, left_loss_seq=9), 100.)
        self.assertEqual(self.pending.take()[1]["tracking_lost"], [])
        self.pending.offer(message(50.01, diagnostics_version=1, head_anchor_valid=True,
                                   head_loss_seq=7, left_loss_seq=10,
                                   head_loss_reason="video_source_stale"), 100.01)
        latest, data = self.pending.take()
        self.assertTrue(latest.head_valid)
        self.assertEqual(data["tracking_lost"], ["head", "left"])
        self.assertEqual(data["tracking_loss_reason"], "video_source_stale")
        self.pending.offer(message(50.02, diagnostics_version=1, head_anchor_valid=True,
                                   head_loss_seq=7, left_loss_seq=10), 100.02)
        self.assertEqual(self.pending.take()[1]["tracking_lost"], [])

    def test_diagnostics_first_packet_is_baseline_and_legacy_defaults_are_ignored(self):
        self.pending.offer(message(), 100.)
        self.pending.offer(message(50.01, diagnostics_version=1, head_anchor_valid=True,
                                   head_loss_seq=88), 100.01)
        self.assertEqual(self.pending.take()[1]["tracking_lost"], [])
        self.pending.offer(message(50.02), 100.02)
        self.pending.offer(message(50.03, diagnostics_version=1, head_anchor_valid=True,
                                   head_loss_seq=89, head_loss_reason="head_untracked"), 100.03)
        data = self.pending.take()[1]
        self.assertEqual(data["tracking_lost"], ["head"])
        self.assertEqual(data["tracking_loss_reason"], "head_untracked")

    def test_source_counter_regression_is_rejected_but_reconnection_resets_baseline(self):
        self.pending.offer(message(diagnostics_version=1, head_anchor_valid=True, left_loss_seq=4), 100.)
        self.pending.take()
        with self.assertRaisesRegex(ValueError, "counter moved backwards"):
            self.pending.offer(message(50.01, diagnostics_version=1, head_anchor_valid=True,
                                       left_loss_seq=3), 100.01)
        self.pending.disconnect("reconnecting")
        self.pending.begin_stream(2)
        self.pending.offer(message(10., diagnostics_version=1, head_anchor_valid=True), 100.02)
        self.pending.take()
        data = self.pending.take()[1]
        self.assertNotIn("left", data["tracking_lost"])

    def test_coalescing_preserves_original_invalid_reason_after_recovery(self):
        self.pending.offer(message(), 100.)
        self.pending.take()
        self.pending.offer(message(50.01, head_valid=False, diagnostics_version=1,
                                   head_anchor_valid=True, video_required=True,
                                   video_ready=False, video_reason="clock_expired"), 100.01)
        self.pending.offer(message(50.02, diagnostics_version=1, head_anchor_valid=True,
                                   video_required=True, video_ready=True), 100.02)
        data = self.pending.take()[1]
        self.assertEqual(data["tracking_loss_reason"], "video_clock_expired")
        self.pending.offer(message(50.03), 100.03)
        latest = self.pending.take()[1]
        self.assertEqual(latest["tracking_loss_reason"], "video_clock_expired")
        self.assertEqual(latest["tracking_loss_seq"], data["tracking_loss_seq"])
        self.assertFalse(latest["tracking_lost"])

    def test_timeout_and_source_age_take_priority_over_video_or_head_diagnostics(self):
        for mode, expected in (("gap", "receive_timeout"), ("old_anchor", "input_stale"),
                               ("delayed", "input_stale")):
            with self.subTest(mode=mode):
                pending = LatestTracking(.25)
                pending.offer(message(), 100.)
                pending.take()
                if mode == "gap":
                    sample, received, anchor = 50.3, 100.3, 50.3
                elif mode == "old_anchor":
                    sample, received, anchor = 50.1, 100.1, 49.7
                else:
                    sample, received, anchor = 50.01, 100.1, 50.01
                    pending.offset = 49.7
                pending.offer(message(sample, head_time=anchor, diagnostics_version=1,
                                      head_anchor_valid=False, head_loss_seq=1,
                                      head_loss_reason="video_source_stale"), received)
                self.assertEqual(pending.take()[1]["tracking_loss_reason"], expected)

    def test_raw_head_loss_takes_priority_over_video_failure(self):
        self.pending.offer(message(head_valid=False, diagnostics_version=1,
                                   head_anchor_valid=False, video_required=True,
                                   video_reason="source_stale"), 100.)
        self.assertEqual(self.pending.take()[1]["tracking_loss_reason"], "head_untracked")


if __name__ == "__main__":
    unittest.main()
