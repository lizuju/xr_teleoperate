from pathlib import Path
import importlib.util
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
