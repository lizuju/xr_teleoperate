#!/usr/bin/python3
"""Receive the raw head cameras' RTP JPEG streams and write each JPEG to stdout.

Runs under the system python3, whose PyGObject/GStreamer bindings are the distro's own.
Each record is HEADER followed by the JPEG bytes. Nothing is decoded here, so a
viewer that is not connected costs only the packet handling.
"""
import argparse
import os
import signal
import struct
import sys
import threading
import time

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

HEADER = struct.Struct("!4sBdI")
CAPS = "application/x-rtp,media=video,clock-rate=90000,encoding-name=JPEG,payload=96"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-port", type=int, default=5004)
    parser.add_argument("--right-port", type=int, default=5005)
    args = parser.parse_args()
    Gst.init(None)
    branches = [
        f'udpsrc port={port} do-timestamp=true buffer-size=4194304 caps="{CAPS}" '
        f"! rtpjpegdepay ! appsink name=eye{eye} emit-signals=true max-buffers=1 drop=true sync=false async=false"
        for eye, port in enumerate((args.left_port, args.right_port))
    ]
    pipeline = Gst.parse_launch(" ".join(branches))
    condition = threading.Condition()
    latest = [None, None]
    stopped = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stopped.set())

    def sample_ready(sink, eye):
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        running_pts = sample.get_segment().to_running_time(Gst.Format.TIME, buffer.pts)
        if running_pts == Gst.CLOCK_TIME_NONE:
            return Gst.FlowReturn.OK
        running_now = pipeline.get_clock().get_time() - pipeline.get_base_time()
        timestamp = time.monotonic() - max(0, running_now - running_pts) / Gst.SECOND
        data = buffer.extract_dup(0, buffer.get_size())
        with condition:
            # The slot keeps only the newest frame if the pipe falls behind.
            latest[eye] = HEADER.pack(b"R1HD", eye, timestamp, len(data)) + data
            condition.notify()
        return Gst.FlowReturn.OK

    for eye in range(2):
        pipeline.get_by_name(f"eye{eye}").connect("new-sample", sample_ready, eye)
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("raw head pipeline failed to start")
    bus = pipeline.get_bus()
    try:
        while not stopped.is_set():
            message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message is not None:
                raise RuntimeError(f"raw head pipeline stopped: {message.type}")
            with condition:
                packets, latest[:] = latest[:], [None, None]
                if not any(packets):
                    condition.wait(0.050)
            for packet in packets:
                if packet is not None:
                    remaining = memoryview(packet)
                    while remaining:
                        remaining = remaining[os.write(sys.stdout.fileno(), remaining):]
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
    except Exception as error:
        print(f"[HeadHD capture] {error}", file=sys.stderr, flush=True)
        sys.exit(1)
