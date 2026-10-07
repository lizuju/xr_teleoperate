from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from unittest.mock import patch

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from visionpro_bridge import LatestTracking, receive
from visionpro_protocol.handtracking_pb2 import HandUpdate


def message(sample, **changes):
    fields = dict(tracking_protocol_version=1, sample_time=sample,
                  head_time=sample, left_time=sample, right_time=sample,
                  head_valid=True, left_valid=True, right_valid=True,
                  diagnostics_version=1, head_anchor_valid=True,
                  video_required=True, video_ready=True)
    fields.update(changes)
    return HandUpdate(**fields)


class RecoveryClockTest(unittest.TestCase):
    def test_runtime_timeout_is_unchanged_and_recovery_has_separate_bound(self):
        for timeout, limit in ((.25, 1.), (.5, 1.), (.75, 1.5)):
            with self.subTest(timeout=timeout), patch("visionpro_bridge.time.monotonic", return_value=100.):
                pending = LatestTracking(timeout)
                self.assertEqual(pending.timeout, timeout)
                self.assertIsNone(pending.recovery_reason(100. + limit - .001))
                self.assertEqual(pending.recovery_reason(100. + limit), "source_silence")

    def test_coalesced_old_samples_do_not_renew_recovery_timer(self):
        pending = LatestTracking(.5)
        pending.offer(message(50.), 100.)
        for i in range(1, 11):
            pending.offer(message(50. + i * .01), 100.6 + i * .05)
        self.assertEqual(pending.last_fresh_received, 100.)
        self.assertEqual(pending.recovery_reason(101.1), "source_stale")
        self.assertGreater(pending.coalesced, 0)

    def test_fresh_invalid_tracking_and_video_never_trigger_transport_recovery(self):
        pending = LatestTracking(.5)
        for i in range(31):
            received = 100. + i * .1
            pending.offer(message(50. + i * .1, head_valid=False, left_valid=False,
                                  right_valid=False, head_anchor_valid=False,
                                  video_ready=False, video_reason="source_stale"), received)
            self.assertIsNone(pending.recovery_reason(received))
        self.assertEqual(pending.reconnect_count, 0)

    def test_duplicate_source_samples_do_not_renew_timer(self):
        pending = LatestTracking(.5)
        pending.offer(message(50.), 100.)
        for i in range(1, 12):
            pending.offer(message(50.), 100. + i * .1)
        self.assertEqual(pending.last_fresh_received, 100.)
        self.assertEqual(pending.recovery_reason(101.1), "source_stale")

    def test_fresh_recovery_before_limit_does_not_reconnect(self):
        pending = LatestTracking(.5)
        pending.offer(message(50.), 100.)
        pending.offer(message(50.1), 100.8)
        pending.offer(message(50.9), 100.9)
        self.assertEqual(pending.last_fresh_received, 100.9)
        self.assertIsNone(pending.recovery_reason(101.1))

    def test_reconnect_preserves_minimum_offset_and_rejects_clock_regression(self):
        pending = LatestTracking(.5)
        pending.offer(message(50.), 100.)
        pending.disconnect("stale", reconnect_reason="source_stale")
        with patch("visionpro_bridge.time.monotonic", return_value=103.):
            pending.begin_stream(2, pending.offset, pending.sample)
        pending.offer(message(51.), 103.)
        self.assertEqual(pending.offset, 50.)
        self.assertFalse(pending.live["head"])
        self.assertEqual(pending.last_fresh_received, 103.)
        with self.assertRaisesRegex(ValueError, "restart the receiver"):
            pending.offer(message(1.), 103.01)

    def test_disconnect_wins_over_racing_old_offer_and_precedes_new_stream(self):
        pending = LatestTracking(.5)
        pending.offer(message(50.), 100.)
        offered = threading.Event()
        def old_offer():
            offered.set()
            pending.offer(message(50.01), 100.01)
        with pending.condition:
            worker = threading.Thread(target=old_offer)
            worker.start()
            self.assertTrue(offered.wait(1.))
            pending.disconnect("stale", reconnect_reason="source_stale")
        worker.join(timeout=1.)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(pending.latest)
        pending.begin_stream(2, pending.offset, pending.sample)
        pending.offer(message(50.02), 100.02)
        latest, event = pending.take()
        self.assertIsNone(latest)
        self.assertEqual(event["transport"], {"reconnect_count": 1, "reconnect_reason": "source_stale"})
        self.assertEqual(event["tracking_loss_reason"], "input_stale")
        latest, packet = pending.take()
        self.assertEqual(latest.sample_time, 50.02)
        self.assertEqual(packet["stream_id"], 2)
        self.assertIn("head", packet["tracking_lost"])
        self.assertEqual(packet["transport"]["reconnect_count"], 1)

    def test_begin_stream_discards_previous_pending_frame(self):
        pending = LatestTracking(.5)
        pending.offer(message(50.), 100.)
        pending.begin_stream(2, pending.offset, pending.sample)
        self.assertIsNone(pending.latest)


class RecoveryGrpcTest(unittest.TestCase):
    def setUp(self):
        self.server = grpc.server(ThreadPoolExecutor(max_workers=2))
        self.pending = LatestTracking(.5)
        self.stopping = threading.Event()
        self.connections = 0
        self.peers = []
        self.channel_options = []
        self.receiver = None

    def tearDown(self):
        self.stopping.set()
        if self.receiver is not None:
            self.receiver.join(timeout=2.)
            self.assertFalse(self.receiver.is_alive())
        self.server.stop(0).wait(2.)
        self.assertFalse(any(t.name == "visionpro-stream-watchdog" and t.is_alive()
                             for t in threading.enumerate()))

    def start(self, mode):
        def stream(request, context):
            self.connections += 1
            connection = self.connections
            self.peers.append(context.peer())
            initial = time.monotonic()
            while context.is_active():
                now = time.monotonic()
                if mode == "fatal":
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid protocol")
                if connection == 1 and now - initial > .1:
                    if mode == "silence":
                        time.sleep(.01)
                        continue
                    if mode == "rpc_end":
                        return
                    else:
                        sample = now
                else:
                    sample = now
                if mode == "stale" and connection == 1 and now - initial > .1:
                    # Remain monotonic while lag grows, rather than a clock rollback.
                    sample = initial + .1 + (now - initial - .1) * .001
                yield message(sample, head_valid=mode != "invalid", left_valid=mode != "invalid",
                              right_valid=mode != "invalid", video_ready=mode != "invalid")
                time.sleep(.01)

        self.server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "handtracking.HandTrackingService", {"StreamHandUpdates": grpc.unary_stream_rpc_method_handler(
                stream, request_deserializer=HandUpdate.FromString,
                response_serializer=HandUpdate.SerializeToString)}),))
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        factory = grpc.insecure_channel
        def channel(target, options):
            self.channel_options.append(options)
            return factory(target, options=options)
        self.patcher = patch("visionpro_bridge.grpc.insecure_channel", side_effect=channel)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.receiver = threading.Thread(target=receive, args=("127.0.0.1", port, self.pending, self.stopping))
        self.receiver.start()

    def wait_for(self, predicate, timeout=5.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.pending.condition:
                if predicate():
                    return
            time.sleep(.01)
        self.fail("receiver did not reach expected state")

    def test_silent_blocked_rpc_is_cancelled_and_new_channel_recovers(self):
        self.start("silence")
        self.wait_for(lambda: self.connections >= 2 and self.pending.latest is not None)
        latest, event = self.pending.take()
        self.assertIsNone(latest)
        self.assertEqual(event["transport"]["reconnect_reason"], "source_silence")
        latest, packet = self.pending.take()
        self.assertEqual(packet["stream_id"], 2)
        self.assertIn("head", packet["tracking_lost"])
        self.assertNotEqual(self.peers[0], self.peers[1])
        self.assertTrue(all(("grpc.use_local_subchannel_pool", 1) in options
                            for options in self.channel_options))

    def test_continuous_stale_samples_trigger_recovery(self):
        self.start("stale")
        self.wait_for(lambda: self.connections >= 2 and self.pending.latest is not None)
        _, event = self.pending.take()
        self.assertEqual(event["transport"]["reconnect_reason"], "source_stale")
        self.assertEqual(event["transport"]["reconnect_count"], 1)
        _, packet = self.pending.take()
        self.assertGreaterEqual(packet["stream_id"], 2)
        self.assertLess(packet["transport"]["upstream_age_ms"], 500.)

    def test_fresh_invalid_inputs_keep_connection_and_stop_cleans_watchdog(self):
        self.start("invalid")
        self.wait_for(lambda: self.pending.transport["packets_received"] >= 130)
        self.assertEqual(self.connections, 1)
        self.assertEqual(self.pending.reconnect_count, 0)

    def test_caller_stop_is_not_reported_as_recovery(self):
        self.start("silence")
        self.wait_for(lambda: self.pending.transport["packets_received"] >= 2)
        self.stopping.set()
        self.receiver.join(timeout=2.)
        self.assertFalse(self.receiver.is_alive())
        self.assertEqual(self.pending.reconnect_count, 0)

    def test_fatal_rpc_does_not_reconnect_or_leak_watchdog(self):
        self.start("fatal")
        self.receiver.join(timeout=2.)
        self.assertFalse(self.receiver.is_alive())
        _, event = self.pending.take()
        self.assertIn("invalid protocol", event["error"])
        self.assertEqual(self.pending.reconnect_count, 0)
        self.assertEqual(self.connections, 1)

    def test_stop_during_channel_ready_does_not_wait_eight_seconds(self):
        with socket.socket() as unavailable:
            unavailable.bind(("127.0.0.1", 0))
            port = unavailable.getsockname()[1]
            self.receiver = threading.Thread(target=receive,
                                             args=("127.0.0.1", port, self.pending, self.stopping))
            self.receiver.start()
            time.sleep(.15)
            self.stopping.set()
            self.receiver.join(timeout=.5)
        self.assertFalse(self.receiver.is_alive())
        self.assertEqual(self.pending.reconnect_count, 0)

    def test_stop_during_retry_backoff_does_not_create_new_channel(self):
        self.start("rpc_end")
        self.wait_for(lambda: self.pending.reconnect_count == 1)
        self.stopping.set()
        self.receiver.join(timeout=.5)
        self.assertFalse(self.receiver.is_alive())
        self.assertEqual(self.connections, 1)

    def test_peer_ending_rpc_has_a_distinct_reason(self):
        self.start("rpc_end")
        self.wait_for(lambda: self.connections >= 2 and self.pending.latest is not None)
        _, event = self.pending.take()
        self.assertEqual(event["transport"]["reconnect_reason"], "stream_ended")


if __name__ == "__main__":
    unittest.main()
