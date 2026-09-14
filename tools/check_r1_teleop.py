#!/usr/bin/env python3
import math
from pathlib import Path
import socket
import ssl
import sys
import threading
import time
import urllib.request


class StreamWindow:
    def __init__(self, max_age):
        self.max_age = max_age
        self.count = 0
        self.first = None
        self.last = None
        self.error = None

    def observe(self, now, error=None):
        if error is not None:
            self.count = 0
            self.first = self.last = None
            self.error = error
            return
        if self.last is None or now - self.last > self.max_age:
            self.first = now
            self.count = 0
        self.count += 1
        self.last = now
        self.error = None

    def problem(self, now):
        if self.error is not None:
            return self.error
        if self.last is None:
            return "no valid samples received"
        age = now - self.last
        if not 0.0 <= age <= self.max_age:
            return f"stale samples ({age:.3f}s old)"
        if self.count < 3 or self.last - self.first < 1.0:
            return "need at least 1s of continuous valid samples"
        return None


def validate_state(message, hand=False):
    motors = message.states if hand else message.motor_state
    if len(motors) != (6 if hand else 35):
        raise ValueError("expected 6 O6 axes" if hand else "expected 35 R1_A7 motor states")
    values = [motor.q for motor in motors]
    if not hand:
        values += [motor.dq for motor in motors]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite motor feedback")
    if hand:
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("O6 feedback outside [0, 1]")


def validate_camera_config(config):
    head = config.get("head_camera", {})
    expected = {
        "type": "rtp_h264_stereo",
        "image_shape": [448, 1088],
        "binocular": True,
        "enable_zmq": True,
        "zmq_port": 55555,
        "enable_webrtc": True,
        "webrtc_port": 60001,
        "left_rtp_port": 5002,
        "right_rtp_port": 5003,
    }
    for name, value in expected.items():
        if head.get(name) != value:
            raise ValueError(f"head_camera.{name}: expected {value!r}, got {head.get(name)!r}")


def check_local_and_https():
    if "eno1" not in {name for _, name in socket.if_nameindex()}:
        raise RuntimeError("DDS interface eno1 is missing; run this on the Ubuntu Host")
    dev_root = Path(__file__).resolve().parents[2]
    for side in ("left", "right"):
        path = dev_root / "linkerhand-urdf" / "O6" / side / f"linkerhand_o6_{side}.urdf"
        if not path.is_file():
            raise RuntimeError(f"O6 Vector hand model missing: {path}")
    tls_dir = Path.home() / ".config" / "xr_teleoperate"
    try:
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(tls_dir / "cert.pem", tls_dir / "key.pem")
        context = ssl.create_default_context(cafile=str(tls_dir / "rootCA.pem"))
    except (OSError, ssl.SSLError) as error:
        raise RuntimeError(f"XR TLS certificate/key/root CA invalid in {tls_dir}: {error}") from error
    try:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", 8012))
    except OSError as error:
        raise RuntimeError(f"XR port 8012 unavailable; stop the existing XR process: {error}") from error
    url = "https://192.168.124.147:60001/"
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context)
        )
        with opener.open(url, timeout=3.0) as response:
            response.read(1)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"WebRTC HTTPS unavailable at {url}; check 60001 forwarding and r1-teleimager: {error}") from error
    print("[OK] eno1, O6 models, XR TLS files, port 8012 and verified WebRTC HTTPS", flush=True)


def check_streams():
    import cv2
    import numpy as np
    import zmq
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.utils.crc import CRC

    context = zmq.Context()
    subscribers = []
    try:
        with context.socket(zmq.REQ) as request:
            request.setsockopt(zmq.LINGER, 0)
            request.setsockopt(zmq.SNDTIMEO, 3000)
            request.connect("tcp://192.168.124.147:60000")
            request.send(b"GET_DATA")
            if not request.poll(3000):
                raise RuntimeError("camera configuration timed out; check 60000 forwarding and r1-teleimager")
            validate_camera_config(request.recv_json())
        print("[OK] stereo camera configuration (1088x448, RTP 5002/5003)", flush=True)
        windows = {
            "rt/lowstate (robot feedback on eno1)": StreamWindow(0.25),
            "rt/linker/left/state (left O6 bridge)": StreamWindow(0.25),
            "rt/linker/right/state (right O6 bridge)": StreamWindow(0.25),
            "stereo JPEG (192.168.124.147:55555)": StreamWindow(0.5),
        }
        lock = threading.Lock()
        crc = CRC()

        def callback(name, hand):
            def receive(message):
                now = time.monotonic()
                try:
                    validate_state(message, hand)
                    if not hand and crc.Crc(message) != int(message.crc):
                        raise ValueError("R1 lowstate CRC mismatch")
                    error = None
                except (ValueError, AttributeError, TypeError, OverflowError) as failure:
                    error = str(failure)
                with lock:
                    windows[name].observe(now, error)
            return receive

        ChannelFactoryInitialize(0, "eno1")
        for index, name in enumerate(list(windows)[:3]):
            subscriber = ChannelSubscriber(name.split()[0], LowState_ if index == 0 else MotorStates_)
            subscribers.append(subscriber)
            subscriber.Init(callback(name, index != 0), 1)
        with context.socket(zmq.SUB) as frames:
            frames.setsockopt(zmq.LINGER, 0)
            frames.setsockopt(zmq.CONFLATE, 1)
            frames.setsockopt(zmq.SUBSCRIBE, b"")
            frames.connect("tcp://192.168.124.147:55555")
            deadline = time.monotonic() + 8.0
            print("[CHECK] waiting for continuous robot, O6 and stereo feedback (up to 8s)", flush=True)
            while time.monotonic() < deadline:
                if frames.poll(50):
                    frame = cv2.imdecode(np.frombuffer(frames.recv(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    error = None if frame is not None and frame.shape == (448, 1088, 3) else "JPEG missing or not 448x1088x3"
                    with lock:
                        windows["stereo JPEG (192.168.124.147:55555)"].observe(time.monotonic(), error=error)
                with lock:
                    now = time.monotonic()
                    problems = {name: window.problem(now) for name, window in windows.items() if window.problem(now)}
                    if not problems:
                        for name, window in windows.items():
                            rate = (window.count - 1) / (window.last - window.first)
                            print(f"[OK] {name}: {rate:.1f} Hz, age {(now - window.last) * 1000:.0f}ms", flush=True)
                        return
            raise RuntimeError("stream checks failed:\n" + "\n".join(f"  {name}: {problem}" for name, problem in problems.items()))
    finally:
        for subscriber in subscribers:
            subscriber.Close()
        context.term()


def main():
    print("[PREFLIGHT] read-only checks; DDS state subscriptions only", flush=True)
    try:
        check_local_and_https()
        check_streams()
    except KeyboardInterrupt:
        print("[STOP] preflight cancelled; teleoperation was not started", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print("[READY] preflight passed; Vector startup prerequisites are available", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
