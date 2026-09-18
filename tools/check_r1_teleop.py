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


def unwrap_camera_config(response):
    """teleimager 2.x answers 60000 with {"webrtc": {...}, "camera": {topic: {...}}};
    teleimager.client.ZMQ_Requester unwraps ["camera"] the same way."""
    if isinstance(response, dict) and isinstance(response.get("camera"), dict):
        return response["camera"]
    return response


def jpeg_decoder():
    """Decode a teleimager JPEG payload the way the runtime does.

    Both teleimager's TeleImageClient and the uvc library decode with turbojpeg,
    which accepts the frames the two wrist UVC modules emit without their
    trailing EOI marker (measured on 2026-09-15: ~85% of wrist frames arrive
    1-2 KB short, costing at most the last 8 of 480 rows; the head stereo JPEGs
    are always complete). cv2.imdecode rejects those frames outright, so using
    it here would report a healthy stream as broken.
    """
    try:
        from turbojpeg import TurboJPEG
        decoder = TurboJPEG()

        def decode(payload):
            try:
                return decoder.decode(payload)
            except Exception:
                return None
    except Exception:
        import cv2
        import numpy as np

        def decode(payload):
            return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    return decode


def jpeg_is_complete(payload):
    """True when the payload carries both JPEG markers (SOI ... EOI)."""
    return payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")


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

    # Wrist cameras feed the Vision Pro HUD panels. They may be disabled on PC2
    # (both transports off) in which case there is nothing to check, but an
    # enabled stream must be on the ports the panel URLs and recording use.
    for topic, zmq_port, webrtc_port in (
        ("left_wrist_camera", 55556, 60002),
        ("right_wrist_camera", 55557, 60003),
    ):
        wrist = config.get(topic)
        if not wrist:
            continue
        if not (wrist.get("enable_zmq") or wrist.get("enable_webrtc")):
            continue
        # Either local driver is valid: `uvc` (libuvc) and `v4l2` (kernel driver)
        # both serve the same 640x480 stream; production moved to v4l2 on
        # 2026-09-15 because libuvc fought the kernel driver over the device.
        if wrist.get("type") not in ("uvc", "v4l2"):
            raise ValueError(f"{topic}.type: expected 'uvc' or 'v4l2', got {wrist.get('type')!r}")
        expected = {"image_shape": [480, 640]}
        if wrist.get("enable_zmq"):
            expected.update({"zmq_port": zmq_port})
        if wrist.get("enable_webrtc"):
            expected.update({"webrtc_port": webrtc_port})
        for name, value in expected.items():
            if wrist.get(name) != value:
                raise ValueError(f"{topic}.{name}: expected {value!r}, got {wrist.get(name)!r}")


def wrist_topics(config):
    """[(topic, zmq_port, webrtc_port)] for every wrist stream that is enabled."""
    selected = []
    for topic, zmq_port, webrtc_port in (
        ("left_wrist_camera", 55556, 60002),
        ("right_wrist_camera", 55557, 60003),
    ):
        wrist = config.get(topic) or {}
        if wrist.get("enable_zmq") or wrist.get("enable_webrtc"):
            selected.append((topic, zmq_port, webrtc_port))
    return selected


#: Label of the head stereo JPEG window, shared by the stream check and main().
STEREO_JPEG_LABEL = "stereo JPEG (192.168.124.147:55555)"


def stream_check_failure(problems):
    """Format the 8s stream-check timeout so `problems` is never unbound.

    The map is created before the wait loop. If the deadline is already in the
    past (clock jump) the loop body never runs and `problems` stays empty; that
    used to NameError when the raise interpolated an unassigned name.
    """
    if not problems:
        return "stream checks timed out after 8s before a diagnosis was collected"
    return "stream checks failed:\n" + "\n".join(
        f"  {name}: {problem}" for name, problem in problems.items()
    )


def https_probe(context, port, label):
    url = f"https://192.168.124.147:{port}/"
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context)
        )
        with opener.open(url, timeout=3.0) as response:
            response.read(1)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"WebRTC HTTPS unavailable for {label} at {url}; check the {port} forwarding and r1-teleimager: {error}") from error


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
    print("[OK] eno1, O6 models, XR TLS files, port 8012", flush=True)
    return context


def check_streams():
    import zmq
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.utils.crc import CRC

    context = zmq.Context()
    subscribers = []
    frame_sockets = []
    try:
        with context.socket(zmq.REQ) as request:
            request.setsockopt(zmq.LINGER, 0)
            request.setsockopt(zmq.SNDTIMEO, 3000)
            request.connect("tcp://192.168.124.147:60000")
            request.send(b"GET_DATA")
            if not request.poll(3000):
                raise RuntimeError("camera configuration timed out; check 60000 forwarding and r1-teleimager")
            camera_config = unwrap_camera_config(request.recv_json())
            validate_camera_config(camera_config)
        print("[OK] stereo camera configuration (1088x448, RTP 5002/5003)", flush=True)
        wrist = wrist_topics(camera_config)
        if wrist:
            print("[OK] wrist camera configuration: " + ", ".join(
                f"{topic} (zmq {zmq_port}{', webrtc ' + str(webrtc_port) if (camera_config.get(topic) or {}).get('enable_webrtc') else ''})"
                for topic, zmq_port, webrtc_port in wrist), flush=True)
        else:
            print("[SKIP] wrist cameras are disabled on PC2; the Vision Pro HUD panels stay hidden", flush=True)
        windows = {
            "rt/lowstate (robot feedback on eno1)": StreamWindow(0.25),
            "rt/linker/left/state (left O6 bridge)": StreamWindow(0.25),
            "rt/linker/right/state (right O6 bridge)": StreamWindow(0.25),
            STEREO_JPEG_LABEL: StreamWindow(0.5),
        }
        # (label, port, expected shape) for every JPEG feed the XR scene consumes.
        jpeg_feeds = [(STEREO_JPEG_LABEL, 55555, (448, 1088, 3))]
        for topic, zmq_port, _ in wrist:
            if (camera_config.get(topic) or {}).get("enable_zmq"):
                label = f"{topic.removesuffix('_camera').replace('_', ' ')} JPEG (192.168.124.147:{zmq_port})"
                windows[label] = StreamWindow(0.5)
                jpeg_feeds.append((label, zmq_port, (480, 640, 3)))
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
        socket_feeds = []
        for label, port, shape in jpeg_feeds:
            socket_ = context.socket(zmq.SUB)
            socket_.setsockopt(zmq.LINGER, 0)
            socket_.setsockopt(zmq.CONFLATE, 1)
            socket_.setsockopt(zmq.SUBSCRIBE, b"")
            socket_.connect(f"tcp://192.168.124.147:{port}")
            socket_feeds.append((label, socket_, shape))
            frame_sockets.append(socket_)
        try:
            decode_jpeg = jpeg_decoder()
            truncated = {label: [0, 0] for label, _, _ in jpeg_feeds}
            deadline = time.monotonic() + 8.0
            print("[CHECK] waiting for continuous robot, O6 and camera feedback (up to 8s)", flush=True)
            problems = {}
            while time.monotonic() < deadline:
                for label, socket_, shape in socket_feeds:
                    if socket_.poll(50):
                        payload = socket_.recv()
                        frame = decode_jpeg(payload)
                        expected = f"{shape[1]}x{shape[0]}"
                        error = None if frame is not None and frame.shape == shape else f"JPEG missing or not {expected}x3"
                        if error is None:
                            truncated[label][1] += 1
                            if not jpeg_is_complete(payload):
                                truncated[label][0] += 1
                        with lock:
                            windows[label].observe(time.monotonic(), error=error)
                with lock:
                    now = time.monotonic()
                    problems = {name: window.problem(now) for name, window in windows.items() if window.problem(now)}
                    if not problems:
                        for name, window in windows.items():
                            rate = (window.count - 1) / (window.last - window.first)
                            print(f"[OK] {name}: {rate:.1f} Hz, age {(now - window.last) * 1000:.0f}ms", flush=True)
                        for label, (short, total) in truncated.items():
                            if short:
                                print(f"[WARN] {label}: {short}/{total} frames arrive without the JPEG EOI marker "
                                      f"(wrist UVC module quirk; turbojpeg decodes them, losing at most the last MCU row)",
                                      flush=True)
                        return set(windows)
            raise RuntimeError(stream_check_failure(problems))
        finally:
            for socket_ in frame_sockets:
                socket_.close()
    finally:
        for subscriber in subscribers:
            subscriber.Close()
        context.term()


def main():
    print("[PREFLIGHT] read-only checks; DDS state subscriptions only", flush=True)
    try:
        context = check_local_and_https()
        # The teleop reads the head image over ZMQ and paints it into the XR scene from
        # shared memory; the WebRTC plane is only negotiated on headsets that support a
        # second element, and this one does not. So a dead 60001 upstream is a warning
        # while the ZMQ feed is healthy, and a hard failure only when it is not.
        healthy = check_streams()
        try:
            https_probe(context, 60001, "head stereo")
            print("[OK] verified WebRTC HTTPS for the head stereo plane", flush=True)
        except RuntimeError as error:
            if STEREO_JPEG_LABEL in healthy:
                print(f"[WARN] {error}; the head image still arrives over ZMQ, so only the "
                      f"optional WebRTC scene element is unavailable", flush=True)
            else:
                raise
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
