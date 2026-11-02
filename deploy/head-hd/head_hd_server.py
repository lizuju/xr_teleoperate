"""Publish the raw head cameras over WebRTC for the Vision Pro first-person view.

A separate process from r1-teleimager on purpose: recording and training keep using
the rectified 544x448 stream there, and a fault here cannot stop that service.
Frames are decoded only while a viewer is connected. The app undistorts and rectifies
on its GPU from head-optics.json, whose image_size must match OUTPUT_EYE_SIZE.
"""
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

import cv2
import numpy as np
from turbojpeg import TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE

from teleimager import server as teleimager
from teleimager.gst_stereo_capture import StereoFrames

WEBRTC_PORT = 60004
OUTPUT_EYE_SIZE = (960, 816)  # 0.75 of the raw 1280x1088
ENCODER_THREADS = 2
GOP_LENGTH = 60
# Fixed on purpose: the receiver's bandwidth estimate starts low and can pin this
# stream near its minimum on a link that carries it fine. Lower it for a weak network.
BITRATE_BPS = 12_000_000
# Half a 30 fps frame period: anything further apart is a different frame pair.
MAX_PAIR_SKEW_S = 0.017
HEADER = struct.Struct("!4sBdI")
STATS_INTERVAL_S = 5.0


def encode_frame(self, frame, force_keyframe):
    """libx264 for Jetson, at a fixed bitrate and the source's real 30 fps."""
    if self.codec is not None and (frame.width != self.codec.width or frame.height != self.codec.height):
        self.codec = None
    if self.codec is None:
        self.codec = teleimager.av.CodecContext.create("libx264", "w")
        self.codec.width = frame.width
        self.codec.height = frame.height
        self.codec.bit_rate = BITRATE_BPS
        self.codec.pix_fmt = "yuv420p"
        self.codec.framerate = fractions.Fraction(30, 1)
        self.codec.time_base = fractions.Fraction(1, 30)
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


def decode_eye(jpeg):
    # The camera sends whole JPEGs, and rtpjpegdepay puts its own generated header in
    # front. Decoding from that first header gives a flat gray image without any error.
    start = jpeg.find(b"\xff\xd8", 2)
    image = teleimager._turbojpeg.decode(jpeg[start:] if start > 0 else jpeg,
                                         flags=TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE)
    return cv2.resize(image, OUTPUT_EYE_SIZE, interpolation=cv2.INTER_AREA)


def main():
    teleimager._apply_webrtc_config({"webrtc": {"gop_length": GOP_LENGTH}})
    teleimager.h264.H264Encoder._encode_frame = encode_frame
    publisher = teleimager.WebRTC_PublisherThread(WEBRTC_PORT, "0.0.0.0", "h264")
    publisher.start()
    if not publisher.wait_for_start(timeout=5.0):
        sys.exit(f"[HeadHD] WebRTC port {WEBRTC_PORT} failed to start")
    capture = RawCapture()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    stop = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stop.set())
    clock = teleimager.clock_id()
    epoch = str(uuid.uuid4())
    sequence = 0
    decoded, decode_s, stats_at = 0, 0.0, time.monotonic()
    print(f"[HeadHD] listening on {WEBRTC_PORT}, eye {OUTPUT_EYE_SIZE[0]}x{OUTPUT_EYE_SIZE[1]}", flush=True)
    exit_code = 0
    try:
        while not stop.is_set():
            now = time.monotonic()
            if now - stats_at >= STATS_INTERVAL_S:
                print(f"[HeadHD] viewers={len(publisher._pcs)} decoded={decoded / (now - stats_at):.1f} fps "
                      f"decode={1e3 * decode_s / max(1, decoded):.1f} ms/pair", flush=True)
                decoded, decode_s, stats_at = 0, 0.0, now
            try:
                pair = capture.frames.take(timeout=1.0)
            except TimeoutError:
                continue  # No source right now; this service just waits for one.
            if not publisher._pcs:
                continue
            started = time.monotonic()
            eyes = list(pool.map(decode_eye, (pair[0][4], pair[1][4])))
            decode_s += time.monotonic() - started
            decoded += 1
            sequence += 1
            eye_ns = [int(eye[0] * 1e9) for eye in pair]
            publisher.send((np.concatenate(eyes, axis=1), {
                "source_monotonic_ns": min(eye_ns), "source_sequence": sequence, "clock_id": clock,
                # The app accepts only this kind: the time PC2 received the frame over RTP.
                "source_epoch": epoch, "timestamp_kind": "pc2_rtp_decoded_receive",
            }))
    except Exception as error:
        print(f"[HeadHD] {error}", file=sys.stderr, flush=True)
        exit_code = 1
    capture.close()
    # logging_mp leaves a child process and publisher threads that would hold shutdown open.
    os._exit(exit_code)


if __name__ == "__main__":
    main()
