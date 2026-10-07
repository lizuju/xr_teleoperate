import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time

import grpc

from visionpro_protocol import handtracking_pb2


def matrix(message):
    return [[getattr(message, f"m{row}{column}") for column in range(4)] for row in range(4)]


class LatestTracking:
    def __init__(self, timeout):
        self.timeout = timeout
        self.condition = threading.Condition()
        self.latest = None
        self.event = None
        self.finished = False
        self.lost = set()
        self.head_loss_reason = None
        self.tracking_loss_seq = {key: 0 for key in ("head", "left", "right")}
        self.last_head_loss_reason = None
        self.coalesced = 0
        self.reconnect_count = 0
        self.last_reconnect_reason = None
        self.begin_stream(0)

    def begin_stream(self, stream_id, clock_offset=None, sample_floor=0.):
        with self.condition:
            self.stream_id = stream_id
            self.latest = None
            self.sample = sample_floor
            self.received = 0.
            self.offset = clock_offset
            self.accepting = True
            self.last_fresh_received = time.monotonic()
            self.anchors = {key: 0. for key in ("head", "left", "right")}
            self.timestamps = dict(self.anchors)
            self.live = {key: False for key in self.anchors}
            self.source_loss_seq = None
            self.transport = {"packets_received": 0, "bytes_received": 0,
                              "receive_gap_max_ms": 0., "upstream_age_max_ms": 0.,
                              "source_interval_max_ms": 0.,
                              "receive_gaps_over_100ms": 0, "upstream_over_100ms": 0}

    def offer(self, message, received):
        if message.tracking_protocol_version != 1:
            raise ValueError("This Tracking Streamer lacks validity/timestamps. Install the supplied patched visionOS client; the unmodified App Store version cannot enable robot following.")
        sample, prediction = message.sample_time, message.prediction_seconds
        if (not all(math.isfinite(v) for v in (sample, prediction, received))
                or sample <= 0. or received <= 0. or not 0. <= prediction <= .1):
            raise ValueError("Invalid tracking clock or prediction offset")
        with self.condition:
            if not self.accepting:
                return
            if sample < self.sample:
                raise ValueError("Vision Pro tracking clock moved backwards; restart the receiver to establish a new clock epoch")
            offset = received - sample
            offset = offset if self.offset is None else min(offset, self.offset)
            receive_gap = (received - self.received) * 1000. if self.received else 0.
            source_interval = (sample - self.sample) * 1000. if self.sample else 0.
            # Relative to the best delivery in this receiver session, not absolute one-way latency.
            upstream_age = max(0., (received - sample - offset) * 1000.)
            self.transport.update(receive_gap_ms=receive_gap, source_interval_ms=source_interval,
                                  upstream_age_ms=upstream_age)
            self.transport["packets_received"] += 1
            self.transport["bytes_received"] += message.ByteSize()
            self.transport["receive_gap_max_ms"] = max(self.transport["receive_gap_max_ms"], receive_gap)
            self.transport["source_interval_max_ms"] = max(self.transport["source_interval_max_ms"], source_interval)
            self.transport["upstream_age_max_ms"] = max(self.transport["upstream_age_max_ms"], upstream_age)
            self.transport["receive_gaps_over_100ms"] += receive_gap > 100.
            self.transport["upstream_over_100ms"] += upstream_age > 100.
            if sample > self.sample and upstream_age <= self.timeout * 1000.:
                self.last_fresh_received = received
            lost = set()
            head_reason = None
            if self.received and received - self.received > self.timeout:
                lost.add("head")
                head_reason = "receive_timeout"
            source_loss_seq = None
            source_head_loss_reason = None
            if message.diagnostics_version == 1:
                source_loss_seq = {key: getattr(message, f"{key}_loss_seq") for key in self.anchors}
                if self.source_loss_seq is not None:
                    for key, sequence in source_loss_seq.items():
                        if sequence < self.source_loss_seq[key]:
                            raise ValueError(f"{key} loss counter moved backwards")
                        if sequence > self.source_loss_seq[key]:
                            lost.add(key)
                            if key == "head":
                                source_head_loss_reason = message.head_loss_reason or "tracking_interrupted"
            for key in self.anchors:
                anchor = getattr(message, f"{key}_time")
                if not math.isfinite(anchor):
                    raise ValueError("Invalid tracking anchor timestamp")
                age = sample - anchor
                valid = (getattr(message, f"{key}_valid") and anchor > 0.
                         and -prediction - .01 <= age <= self.timeout)
                if valid and anchor < self.anchors[key]:
                    raise ValueError(f"{key} tracking clock moved backwards")
                if self.live[key] and received - self.timestamps[key] > self.timeout:
                    lost.add(key)
                    if key == "head" and head_reason is None:
                        head_reason = "input_stale"
                if valid and anchor > self.anchors[key]:
                    self.anchors[key] = anchor
                    self.timestamps[key] = min(received, anchor + offset)
                live = valid and 0. <= received - self.timestamps[key] <= self.timeout
                if not live:
                    lost.add(key)
                    if key == "head" and head_reason is None:
                        if (anchor > 0. and age > self.timeout) or (valid and received - self.timestamps[key] > self.timeout):
                            head_reason = "input_stale"
                        elif message.diagnostics_version == 1 and not message.head_anchor_valid:
                            head_reason = "head_untracked"
                        elif message.diagnostics_version == 1 and message.video_required and not message.video_ready:
                            head_reason = "video_" + (message.video_reason or "unavailable")
                        else:
                            head_reason = "tracking_interrupted"
                self.live[key] = live
            self.lost.update(lost)
            if self.head_loss_reason is None:
                self.head_loss_reason = head_reason or source_head_loss_reason
            if source_loss_seq is not None:
                self.source_loss_seq = source_loss_seq
            self.offset, self.sample, self.received = offset, sample, received
            if self.latest is not None:
                self.coalesced += 1
            self.latest = (message, received)
            self.condition.notify()

    def disconnect(self, reason, fatal=False, reconnect_reason=None):
        with self.condition:
            self.accepting = False
            self.latest = None
            self.lost.add("head")
            if self.head_loss_reason is None:
                self.head_loss_reason = "connection_lost"
            self.event = {"error" if fatal else "disconnected": reason}
            if reconnect_reason is not None:
                self.reconnect_count += 1
                self.last_reconnect_reason = reconnect_reason
                self.event["transport"] = self.recovery_diagnostics()
                self.event["tracking_loss_reason"] = (
                    "receive_timeout" if reconnect_reason == "source_silence" else
                    "input_stale" if reconnect_reason == "source_stale" else "connection_lost")
            self.finished = fatal
            self.condition.notify()

    def recovery_diagnostics(self):
        return {"reconnect_count": self.reconnect_count,
                "reconnect_reason": self.last_reconnect_reason}

    def recovery_reason(self, now):
        with self.condition:
            if not self.accepting or now - self.last_fresh_received < max(1., 2. * self.timeout):
                return None
            return "source_silence" if not self.received or now - self.received > self.timeout else "source_stale"

    def take(self):
        with self.condition:
            self.condition.wait_for(lambda: self.event is not None or self.latest is not None or self.finished)
            if self.event is not None:
                event, self.event = self.event, None
                return None, event
            if self.latest is None:
                return None, None
            message, received = self.latest
            # Cumulative events survive a newer snapshot replacing an unread one.
            for key in self.lost:
                self.tracking_loss_seq[key] += 1
            if self.head_loss_reason is not None:
                self.last_head_loss_reason = self.head_loss_reason
            packet = {"received_monotonic": received, "stream_id": self.stream_id,
                      "clock_offset": self.offset, "tracking_lost": sorted(self.lost),
                      "tracking_loss_seq": dict(self.tracking_loss_seq),
                      "frames_coalesced": self.coalesced,
                      "transport": dict(self.transport, **self.recovery_diagnostics(),
                                        bridge_pending_ms=max(0., (time.monotonic()-received)*1000.))}
            if self.last_head_loss_reason is not None:
                packet["tracking_loss_reason"] = self.last_head_loss_reason
            self.latest = None
            self.lost.clear()
            self.head_loss_reason = None
            return message, packet


def receive(host, port, pending, stopping=None):
    stopping = stopping or threading.Event()
    stream_id = 0
    clock_offset = None
    sample_floor = 0.
    while not stopping.is_set():
        watchdog = None
        watch_done = threading.Event()
        recovered = threading.Event()
        call = None
        try:
            with grpc.insecure_channel(f"{host}:{port}", options=(("grpc.use_local_subchannel_pool", 1),)) as channel:
                ready = grpc.channel_ready_future(channel)
                deadline = time.monotonic() + 8.
                try:
                    while not stopping.is_set():
                        try:
                            ready.result(timeout=min(.1, max(0., deadline-time.monotonic())))
                            break
                        except grpc.FutureTimeoutError:
                            if time.monotonic() >= deadline:
                                raise
                    if stopping.is_set():
                        return
                finally:
                    ready.cancel()
                stream = channel.unary_stream(
                    "/handtracking.HandTrackingService/StreamHandUpdates",
                    request_serializer=handtracking_pb2.HandUpdate.SerializeToString,
                    response_deserializer=handtracking_pb2.HandUpdate.FromString,
                )
                request = handtracking_pb2.HandUpdate(tracking_protocol_version=1)
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                    route.connect((host, port))
                    address = [int(part) for part in route.getsockname()[0].split(".")]
                request.Head.m00 = 888.
                request.Head.m01, request.Head.m02, request.Head.m03, request.Head.m10 = address
                request.Head.m30 = 25000.
                stream_id += 1
                pending.begin_stream(stream_id, clock_offset, sample_floor)
                call = stream(request)

                def watch():
                    while not watch_done.wait(.05):
                        if stopping.is_set():
                            call.cancel()
                            return
                        # Serialize the recovery decision with offer(); an old packet
                        # cannot repopulate the queue after its connection is abandoned.
                        with pending.condition:
                            reason = pending.recovery_reason(time.monotonic())
                            if reason is not None:
                                recovered.set()
                                pending.disconnect("Tracking Streamer has no fresh source samples; rebuilding connection",
                                                   reconnect_reason=reason)
                                call.cancel()
                                return

                watchdog = threading.Thread(target=watch, name="visionpro-stream-watchdog", daemon=True)
                watchdog.start()
                try:
                    for message in call:
                        pending.offer(message, time.monotonic())
                finally:
                    watch_done.set()
                    watchdog.join()
                    with pending.condition:
                        clock_offset = pending.offset
                        sample_floor = pending.sample
                if not stopping.is_set() and not recovered.is_set():
                    pending.disconnect("Tracking Streamer connection ended; reconnecting",
                                       reconnect_reason="stream_ended")
        except ValueError as exc:
            pending.disconnect(str(exc), fatal=True)
            return
        except grpc.RpcError as exc:
            if stopping.is_set():
                return
            if not recovered.is_set():
                retryable = exc.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED,
                                          grpc.StatusCode.DEADLINE_EXCEEDED)
                pending.disconnect(str(exc), fatal=not retryable,
                                   reconnect_reason=f"rpc_{exc.code().name.lower()}" if retryable else None)
                if not retryable:
                    return
        except (grpc.FutureTimeoutError, OSError) as exc:
            if stopping.is_set():
                return
            pending.disconnect(str(exc) or type(exc).__name__, reconnect_reason="connect_failed")
        finally:
            watch_done.set()
            if call is not None:
                call.cancel()
            if watchdog is not None:
                watchdog.join()
        stopping.wait(1.)


def main():
    parser = argparse.ArgumentParser(description="Tracking Streamer receiver; no robot connection or command publisher")
    parser.add_argument("--host", required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--timeout", type=float, default=.25)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0.:
        parser.error("--timeout must be positive and finite")
    pending = LatestTracking(args.timeout)
    stopping = threading.Event()
    receiver = threading.Thread(target=receive, args=(args.host, args.port, pending, stopping),
                                name="visionpro-receiver", daemon=True)
    receiver.start()
    previous_write_ms = 0.
    write_max_ms = 0.
    publish_seq = 0
    destination = args.snapshot_dir / "snapshot.json"
    staging = args.snapshot_dir / "snapshot.tmp"
    try:
        while True:
            message, packet = pending.take()
            prepared_at = time.monotonic()
            if packet is None:
                return 1
            if message is not None:
                packet["head"] = matrix(message.Head)
                for field in ("tracking_protocol_version", "sample_time", "head_time", "left_time", "right_time",
                              "head_valid", "left_valid", "right_valid", "prediction_seconds"):
                    packet[field] = getattr(message, field)
                if message.diagnostics_version == 1:
                    for field in ("diagnostics_version", "head_anchor_valid", "video_required", "video_ready",
                                  "video_reason", "head_loss_seq", "left_loss_seq", "right_loss_seq",
                                  "last_write_ms", "packets_sent", "head_loss_reason"):
                        packet[field] = getattr(message, field)
                for side in ("left", "right"):
                    hand = getattr(message, f"{side}_hand")
                    packet[f"{side}_wrist"] = matrix(hand.wristMatrix)
                    packet[f"{side}_joints"] = [matrix(joint) for joint in hand.skeleton.jointMatrices[:25]]
                packet["transport"].update(bridge_prepare_ms=(time.monotonic()-prepared_at)*1000.,
                                           previous_snapshot_write_ms=previous_write_ms,
                                           snapshot_write_max_ms=write_max_ms)
                packet["bridge_emit_monotonic"] = time.monotonic()
            publish_seq += 1
            packet["bridge_publish_seq"] = publish_seq
            encoded = json.dumps(packet, separators=(",", ":"))
            write_started = time.monotonic()
            staging.write_text(encoded)
            os.replace(staging, destination)
            previous_write_ms = (time.monotonic()-write_started)*1000.
            write_max_ms = max(write_max_ms, previous_write_ms)
    finally:
        stopping.set()
        receiver.join(timeout=2.)


if __name__ == "__main__":
    sys.exit(main())
