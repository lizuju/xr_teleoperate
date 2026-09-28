import argparse
import json
import math
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
        self.coalesced = 0
        self.begin_stream(0)

    def begin_stream(self, stream_id):
        with self.condition:
            self.stream_id = stream_id
            self.sample = 0.
            self.received = 0.
            self.offset = None
            self.anchors = {key: 0. for key in ("head", "left", "right")}
            self.timestamps = dict(self.anchors)
            self.live = {key: False for key in self.anchors}
            self.source_loss_seq = None

    def offer(self, message, received):
        if message.tracking_protocol_version != 1:
            raise ValueError("This Tracking Streamer lacks validity/timestamps. Install the supplied patched visionOS client; the unmodified App Store version cannot enable robot following.")
        sample, prediction = message.sample_time, message.prediction_seconds
        if (not all(math.isfinite(v) for v in (sample, prediction, received))
                or sample <= 0. or received <= 0. or not 0. <= prediction <= .1):
            raise ValueError("Invalid tracking clock or prediction offset")
        with self.condition:
            if sample < self.sample:
                raise ValueError("Vision Pro tracking clock moved backwards")
            offset = received - sample
            offset = offset if self.offset is None else min(offset, self.offset)
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

    def disconnect(self, reason, fatal=False):
        with self.condition:
            self.latest = None
            self.lost.add("head")
            if self.head_loss_reason is None:
                self.head_loss_reason = "connection_lost"
            self.event = {"error" if fatal else "disconnected": reason}
            self.finished = fatal
            self.condition.notify()

    def take(self):
        with self.condition:
            self.condition.wait_for(lambda: self.event is not None or self.latest is not None or self.finished)
            if self.event is not None:
                event, self.event = self.event, None
                return None, event
            if self.latest is None:
                return None, None
            message, received = self.latest
            packet = {"received_monotonic": received, "stream_id": self.stream_id,
                      "clock_offset": self.offset, "tracking_lost": sorted(self.lost),
                      "frames_coalesced": self.coalesced}
            if self.head_loss_reason is not None:
                packet["tracking_loss_reason"] = self.head_loss_reason
            self.latest = None
            self.lost.clear()
            self.head_loss_reason = None
            return message, packet


def receive(host, port, pending):
    stream_id = 0
    while True:
        try:
            with grpc.insecure_channel(f"{host}:{port}") as channel:
                grpc.channel_ready_future(channel).result(timeout=8.)
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
                pending.begin_stream(stream_id)
                for message in stream(request):
                    pending.offer(message, time.monotonic())
                pending.disconnect("Tracking Streamer connection ended; reconnecting")
        except ValueError as exc:
            pending.disconnect(str(exc), fatal=True)
            return
        except grpc.RpcError as exc:
            retryable = exc.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED,
                                      grpc.StatusCode.DEADLINE_EXCEEDED)
            pending.disconnect(str(exc), fatal=not retryable)
            if not retryable:
                return
        except (grpc.FutureTimeoutError, OSError) as exc:
            pending.disconnect(str(exc) or type(exc).__name__)
        time.sleep(1.)


def main():
    parser = argparse.ArgumentParser(description="Tracking Streamer receiver; no robot connection or command publisher")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--timeout", type=float, default=.25)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0.:
        parser.error("--timeout must be positive and finite")
    pending = LatestTracking(args.timeout)
    threading.Thread(target=receive, args=(args.host, args.port, pending), daemon=True).start()
    while True:
        message, packet = pending.take()
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
        print(json.dumps(packet, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    sys.exit(main())
