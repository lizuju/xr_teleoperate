"""Publish the raw head cameras for recording, training and the Vision Pro.

The only receiver of UDP 5004/5005. ZMQ JPEG at half the raw size feeds recording and
policies; WebRTC at 0.75 of it feeds the headset. A separate process from r1-teleimager
on purpose: a head camera fault here does not stop the wrist cameras. The app undistorts
and rectifies on its GPU from head-optics.json, whose image_size must match
VIEWER_EYE_SCALE. The roster on port 60000 is still served by r1-teleimager.
"""
import argparse
import concurrent.futures
import fractions
import os
import signal
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from turbojpeg import TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE

from teleimager import server as teleimager
from teleimager.gst_stereo_capture import StereoFrames

ZMQ_PORT = 55555
# 60004 for the app; 60001 for the older Safari view and the preflight check.
WEBRTC_PORTS = (60001, 60004)
JPEG_QUALITY = 90
# Scaled by the JPEG decoder: 1280x1088 becomes 640x544 and 960x816. The recording image
# must not depend on whether a headset is watching, or a policy would see different
# pixels from the ones it trained on, so it is always decoded on its own.
RECORDING_EYE_SCALE = (1, 2)
VIEWER_EYE_SCALE = (3, 4)
ENCODER_THREADS = 2
# Only the headset stream crosses Wi-Fi, so only it is thinned. Recording keeps every frame.
VIEWER_FPS = 30
GOP_LENGTH = 2 * VIEWER_FPS
# Fixed on purpose: the receiver's bandwidth estimate starts low and can pin this
# stream near its minimum on a link that carries it fine. Lower it for a weak network.
BITRATE_BPS = 10_000_000
# Half a 30 fps frame period: anything further apart is a different frame pair.
MAX_PAIR_SKEW_S = 0.017
HEADER = struct.Struct("!4sBdI")
STATS_INTERVAL_S = 5.0


def encode_frame(self, frame, force_keyframe):
    """libx264 for Jetson, at a fixed bitrate and the real VIEWER_FPS."""
    if self.codec is not None and (frame.width != self.codec.width or frame.height != self.codec.height):
        self.codec = None
    if self.codec is None:
        self.codec = teleimager.av.CodecContext.create("libx264", "w")
        self.codec.width = frame.width
        self.codec.height = frame.height
        self.codec.bit_rate = BITRATE_BPS
        self.codec.pix_fmt = "yuv420p"
        self.codec.framerate = fractions.Fraction(VIEWER_FPS, 1)
        self.codec.time_base = fractions.Fraction(1, VIEWER_FPS)
        self.codec.options = {"preset": "ultrafast", "tune": "zerolatency",
                              "threads": str(ENCODER_THREADS), "g": str(GOP_LENGTH)}
        self.frame_count = 0
        force_keyframe = True
        print(f"[HeadHD] encoder {frame.width}x{frame.height} at {BITRATE_BPS / 1e6:.1f} Mbps", flush=True)
    if self.frame_count % GOP_LENGTH == 0:
        force_keyframe = True
    self.frame_count += 1
    frame.pict_type = (teleimager.av.video.frame.PictureType.I if force_keyframe
                       else teleimager.av.video.frame.PictureType.NONE)
    for packet in self.codec.encode(frame):
        data = bytes(packet)
        if data:
            yield from self._split_bitstream(data)


class RawCapture:
    def __init__(self):
        self.frames = StereoFrames()
        self.frames.max_pair_skew = MAX_PAIR_SKEW_S
        self.process = subprocess.Popen(
            ["/usr/bin/python3", "-u", str(Path(__file__).with_name("head_hd_capture.py"))],
            stdout=subprocess.PIPE, bufsize=0)
        threading.Thread(target=self._read, daemon=True).start()

    def _read_exact(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.process.stdout.read(size - len(data))
            if not chunk:
                raise EOFError("raw head capture helper exited")
            data.extend(chunk)
        return bytes(data)

    def _read(self):
        try:
            while True:
                magic, eye, timestamp, size = HEADER.unpack(self._read_exact(HEADER.size))
                if magic != b"R1HD" or eye > 1 or size > 16 * 1024 * 1024:
                    raise ValueError("invalid record from raw head capture helper")
                self.frames.put(eye, timestamp, 0, 0, 0, self._read_exact(size))
        except Exception as error:
            with self.frames.condition:
                self.frames.error = str(error)
                self.frames.condition.notify_all()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()


def decode_eye(jpeg, scale):
    # The camera sends whole JPEGs, and rtpjpegdepay puts its own generated header in
    # front. Decoding from that first header gives a flat gray image without any error.
    start = jpeg.find(b"\xff\xd8", 2)
    return teleimager._turbojpeg.decode(jpeg[start:] if start > 0 else jpeg, scaling_factor=scale,
                                        flags=TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zmq-port", type=int, default=ZMQ_PORT)
    parser.add_argument("--webrtc-ports", type=int, nargs="+", default=list(WEBRTC_PORTS))
    args = parser.parse_args()
    teleimager._apply_webrtc_config({"webrtc": {"gop_length": GOP_LENGTH}})
    teleimager.h264.H264Encoder._encode_frame = encode_frame
    publishers = []
    for port in args.webrtc_ports:
        publisher = teleimager.WebRTC_PublisherThread(port, "0.0.0.0", "h264")
        publisher.start()
        if not publisher.wait_for_start(timeout=5.0):
            sys.exit(f"[HeadHD] WebRTC port {port} failed to start")
        publishers.append(publisher)
    zmq_publisher = teleimager.ZMQ_PublisherManager.get_instance()
    capture = RawCapture()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    stop = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stop.set())
    clock = teleimager.clock_id()
    epoch = str(uuid.uuid4())
    sequence = 0
    decoded, decode_s, encode_s, stats_at = 0, 0.0, 0.0, time.monotonic()
    viewer_period = 1.0 / VIEWER_FPS
    next_viewer_at, viewer_sent = 0.0, 0
    print(f"[HeadHD] zmq {args.zmq_port}, webrtc {args.webrtc_ports}", flush=True)
    exit_code = 0
    try:
        while not stop.is_set():
            now = time.monotonic()
            if now - stats_at >= STATS_INTERVAL_S:
                print(f"[HeadHD] viewers={sum(len(p._pcs) for p in publishers)} published={decoded / (now - stats_at):.1f} fps "
                      f"viewer={viewer_sent / (now - stats_at):.1f} fps "
                      f"decode={1e3 * decode_s / max(1, decoded):.1f} ms jpeg={1e3 * encode_s / max(1, decoded):.1f} ms", flush=True)
                decoded, decode_s, encode_s, stats_at = 0, 0.0, 0.0, now
                viewer_sent = 0
            try:
                pair = capture.frames.take(timeout=1.0)
            except TimeoutError:
                continue  # No source right now; this service just waits for one.
            started = time.monotonic()
            watching = [publisher for publisher in publishers if publisher._pcs]
            # Allowance for receive jitter, so a frame just early is not skipped.
            if watching and started + viewer_period / 3 >= next_viewer_at:
                next_viewer_at = max(next_viewer_at, started - viewer_period) + viewer_period
                viewer_sent += 1
            else:
                watching = []
            recording_eyes = [pool.submit(decode_eye, eye[4], RECORDING_EYE_SCALE) for eye in pair]
            viewer_eyes = [pool.submit(decode_eye, eye[4], VIEWER_EYE_SCALE) for eye in pair] if watching else []
            frame = np.concatenate([eye.result() for eye in recording_eyes], axis=1)
            viewer_frame = np.concatenate([eye.result() for eye in viewer_eyes], axis=1) if watching else None
            decoded_at = time.monotonic()
            decode_s += decoded_at - started
            decoded += 1
            sequence += 1
            eye_ns = [int(eye[0] * 1e9) for eye in pair]
            metadata = {
                "source_monotonic_ns": min(eye_ns), "source_sequence": sequence, "clock_id": clock,
                # The app accepts only this kind: the time PC2 received the frame over RTP.
                "source_epoch": epoch, "timestamp_kind": "pc2_rtp_decoded_receive",
            }
            zmq_publisher.publish(teleimager.timestamp_jpeg(
                teleimager._turbojpeg.encode(frame, quality=JPEG_QUALITY),
                {**metadata, "eye_monotonic_ns": eye_ns, "stereo_skew_ns": max(eye_ns) - min(eye_ns)}), args.zmq_port)
            encode_s += time.monotonic() - decoded_at
            for publisher in watching:
                publisher.send((viewer_frame, metadata))
    except Exception as error:
        print(f"[HeadHD] {error}", file=sys.stderr, flush=True)
        exit_code = 1
    capture.close()
    # logging_mp leaves a child process and publisher threads that would hold shutdown open.
    os._exit(exit_code)


if __name__ == "__main__":
    main()
