#!/usr/bin/env python3
"""Capture eye-in-hand samples: chessboard image + robot joint q.

Targets:
  left_wrist / right_wrist  — ZMQ 55556/55557, frame left/right_wrist_yaw_link
  head                      — ZMQ 55555 head stereo left eye, frame head_yaw_link

On SPACE (or s), saves PNG + JSON with motor_q / pinocchio_q / link pose.
Does not start teleop, does not publish lowcmd, does not Enter_Debug_Mode.
Joint read is subscribe-only on rt/lowstate (or --shadow-json / --q-json).

q / Ctrl+C quit capture. That is NOT an e-stop. The robot is not commanded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (REPO, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from r1_hand_eye_common import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    DEFAULT_PATTERN,
    DEFAULT_SQUARE_M,
    DEFAULT_URDF,
    HandEyeError,
    R1A7FK,
    TARGETS,
    capture_day_dirs,
    motor_q_to_pinocchio_q,
    pinocchio_q_to_named,
    rt_from_se3,
    target_spec,
)

try:
    from calibrate_r1_wrist_capture import (  # noqa: E402
        FrameStability,
        detect_corners,
        inner_corner_count,
        next_image_index,
        require_wrist_frame,
    )
except ImportError:  # pragma: no cover - production always has wrist capture
    raise

try:
    from calibrate_r1_head_stereo_capture import (  # noqa: E402
        HEAD_FRAME_SIZE,
        HEAD_SPLIT_X,
        split_head_stereo,
    )
except ImportError:  # pragma: no cover
    HEAD_FRAME_SIZE = (1088, 448)
    HEAD_SPLIT_X = 544

    def split_head_stereo(frame, split_x=HEAD_SPLIT_X, expected_size=HEAD_FRAME_SIZE):
        height, width = frame.shape[:2]
        if (width, height) != expected_size:
            raise HandEyeError(f"head frame {width}x{height} != {expected_size}")
        return np.ascontiguousarray(frame[:, :split_x]), np.ascontiguousarray(frame[:, split_x:])


QUIT_NOTE = "q is NOT e-stop; capture stopped. The robot was not commanded."
WEBRTC_PORTS = (60001, 60002, 60003)


class CaptureError(HandEyeError):
    pass


def require_display():
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise SystemExit(
        "No DISPLAY/WAYLAND_DISPLAY. Run on the Ubuntu desktop graphical session, "
        "not a raw SSH tty. q is not e-stop."
    )


def overlay_label(n_corners, next_index, stable, expected, has_q):
    if n_corners == expected and has_q:
        tag = "OK" if stable else "OK hold"
        return f"{tag}  corners={n_corners}  q=yes  next={next_index:03d}"
    if n_corners == expected and not has_q:
        return f"corners={n_corners}  q=MISSING  next={next_index:03d}"
    return f"corners={n_corners}/{expected}  next={next_index:03d}"


def annotate_frame(bgr, pattern, corners, ok, next_index, stable, has_q, expected=None):
    expected = inner_corner_count(pattern) if expected is None else expected
    vis = bgr.copy()
    if ok and corners is not None:
        cv2.drawChessboardCorners(vis, tuple(pattern), corners, True)
    n_corners = 0 if not ok or corners is None else int(np.asarray(corners).reshape(-1, 2).shape[0])
    if n_corners == expected and stable and has_q:
        color = (0, 255, 0)
    elif n_corners == expected:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)
    text = overlay_label(n_corners, next_index, stable, expected, has_q)
    cv2.putText(vis, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(vis, "SPACE/s=save   q=quit (NOT e-stop)", (12, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def save_sample(folder, index, image, payload):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"{index:03d}.png"
    meta = folder / f"{index:03d}.json"
    if png.exists() or meta.exists():
        raise FileExistsError(f"refusing to overwrite existing sample {index:03d}")
    temporary_png = folder / f".{index:03d}.tmp.png"
    temporary_json = folder / f".{index:03d}.tmp.json"
    try:
        if not cv2.imwrite(str(temporary_png), image):
            raise IOError(f"cv2.imwrite failed for {temporary_png}")
        temporary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_png, png)
        os.replace(temporary_json, meta)
    finally:
        temporary_png.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
    return png, meta


class StaticQSource:
    """Fixed joint vector for offline / unit tests. Never touches DDS."""

    def __init__(self, full_q):
        self.full_q = np.asarray(full_q, dtype=float).reshape(-1)
        if self.full_q.size < 31:
            raise CaptureError("static motor_q must have length >= 31")

    def get_motor_q(self):
        return self.full_q.copy()

    def close(self):
        return None


class ShadowJsonQSource:
    """Read motor_q from r1_a7_lowstate_shadow JSON (subscribe-only sidecar)."""

    def __init__(self, path, max_age_s=0.5):
        self.path = Path(path).expanduser()
        self.max_age_s = float(max_age_s)

    def get_motor_q(self):
        if not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        full_q = payload.get("full_q")
        if not isinstance(full_q, list) or len(full_q) < 31:
            return None
        sample_ns = payload.get("sample_monotonic_ns") or payload.get("published_monotonic_ns")
        if sample_ns is not None:
            age = (time.monotonic_ns() - int(sample_ns)) / 1e9
            if age > self.max_age_s:
                return None
        return np.asarray(full_q, dtype=float)

    def close(self):
        return None


class LowstateSubscribeQSource:
    """Subscribe-only rt/lowstate reader. Never creates a lowcmd publisher."""

    def __init__(self, interface="eno1", startup_timeout=5.0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC

        self._lock = threading.Lock()
        self._latest = None
        self._ready = threading.Event()
        self._crc = CRC()
        self._subscriber = None

        def handle(message):
            if self._crc.Crc(message) != int(message.crc):
                return
            full_q = [float(message.motor_state[index].q) for index in range(35)]
            if not all(np.isfinite(value) for value in full_q):
                return
            with self._lock:
                self._latest = np.asarray(full_q, dtype=float)
            self._ready.set()

        ChannelFactoryInitialize(0, interface)
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(handle, 10)
        if not self._ready.wait(float(startup_timeout)):
            raise CaptureError(
                f"no rt/lowstate on {interface} within {startup_timeout:.1f}s "
                "(subscribe-only; no lowcmd was created)"
            )

    def get_motor_q(self):
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def close(self):
        if self._subscriber is not None:
            try:
                self._subscriber.Close()
            except Exception:
                pass
            self._subscriber = None


def open_q_source(args):
    if args.q_json:
        payload = json.loads(Path(args.q_json).expanduser().read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "full_q" in payload:
            return StaticQSource(payload["full_q"])
        if isinstance(payload, list):
            return StaticQSource(payload)
        raise CaptureError("--q-json must be a list or an object with full_q")
    if args.shadow_json:
        return ShadowJsonQSource(args.shadow_json, max_age_s=args.shadow_max_age)
    if args.no_lowstate:
        raise CaptureError("refusing to capture without a joint source; omit --no-lowstate")
    return LowstateSubscribeQSource(interface=args.interface, startup_timeout=args.lowstate_timeout)


def prepare_view_image(target, frame):
    spec = target_spec(target)
    if spec["kind"] == "wrist":
        return require_wrist_frame(frame, expected_size=spec["image_size"])
    left, _right = split_head_stereo(frame)
    height, width = left.shape[:2]
    expected_w, expected_h = spec["image_size"]
    if width != expected_w or height != expected_h:
        raise CaptureError(f"head left eye is {width}x{height}, expected {expected_w}x{expected_h}")
    return np.ascontiguousarray(left)


def build_sample_payload(spec, full_q, pin_q, T_link, pattern, square, host, port, topic):
    R, t = rt_from_se3(T_link)
    named = pinocchio_q_to_named(pin_q)
    return {
        "schema": "r1_hand_eye_sample_v1",
        "camera": spec["camera"],
        "frame": spec["frame"],
        "pattern": [int(pattern[0]), int(pattern[1])],
        "square_size_m": float(square),
        "image_size": [int(spec["image_size"][0]), int(spec["image_size"][1])],
        "host": host,
        "port": int(port),
        "topic": topic,
        "motor_q": [float(x) for x in np.asarray(full_q, dtype=float).reshape(-1)],
        "pinocchio_q": named["pinocchio_q"],
        "joints": {
            "waist_yaw": named["waist_yaw"],
            "head_pitch": named["head_pitch"],
            "head_yaw": named["head_yaw"],
            "left_arm": named["left_arm"],
            "right_arm": named["right_arm"],
        },
        "T_link_in_root": {
            "rotation": R.tolist(),
            "translation_m": t.tolist(),
        },
        "monotonic_s": time.monotonic(),
        "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "eye-in-hand sample; robot was not commanded by this tool",
    }


def iter_client_frames(client, idle_s=0.02, sleep=time.sleep):
    while True:
        frame = client.get_frame()
        if frame is None or getattr(frame, "bgr", None) is None:
            sleep(idle_s)
            continue
        yield frame.bgr


def run_capture_loop(frames, sample_dir, target, q_source, fk, pattern=DEFAULT_PATTERN,
                     square=DEFAULT_SQUARE_M, stable_s=0.3, host="", port=0, topic="",
                     clock=time.monotonic, wait_key=None, imshow=None, detect=detect_corners,
                     log=print, window_name="hand_eye"):
    """Consume frames; on SPACE/s save image+q when board is OK and stable."""
    spec = target_spec(target)
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    index = next_image_index(sample_dir)
    saved = 0
    gate = FrameStability(stable_s=stable_s)
    expected = inner_corner_count(pattern)
    log(f"resuming at {index:03d}; existing samples will not be overwritten")
    if wait_key is None:
        wait_key = lambda delay: cv2.waitKey(delay) & 0xFF
    if imshow is None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        imshow = cv2.imshow

    for frame in frames:
        try:
            image = prepare_view_image(target, frame)
        except Exception as error:
            log(f"frame rejected: {error}")
            key = wait_key(1) & 0xFF
            if key in (ord("q"), 27):
                log(QUIT_NOTE)
                break
            continue
        ok, corners = detect(image, pattern)
        now = clock()
        stable = gate.update(image, ok, now)
        full_q = q_source.get_motor_q()
        has_q = full_q is not None
        next_index = next_image_index(sample_dir)
        vis = annotate_frame(image, pattern, corners, ok, next_index, stable, has_q, expected)
        imshow(window_name, vis)
        key = wait_key(1) & 0xFF
        if key in (ord("q"), 27):
            log(QUIT_NOTE)
            break
        if key not in (ord(" "), ord("s"), ord("S")):
            continue
        n_corners = 0 if not ok or corners is None else int(np.asarray(corners).reshape(-1, 2).shape[0])
        if n_corners != expected:
            log(f"not saved: detect FAIL need {expected} corners (got {n_corners})")
            continue
        if not stable or gate.last_good is None:
            log(f"not saved: hold still for {int(round(stable_s * 1000))} ms")
            continue
        if full_q is None:
            log("not saved: no joint q (lowstate/shadow missing)")
            continue
        try:
            pin_q = motor_q_to_pinocchio_q(full_q)
            T_link = fk.link_pose(pin_q, spec["frame"])
        except HandEyeError as error:
            log(f"not saved: FK failed: {error}")
            continue
        save_index = next_image_index(sample_dir)
        payload = build_sample_payload(
            spec, full_q, pin_q, T_link, pattern, square, host, port, topic)
        try:
            png, meta = save_sample(sample_dir, save_index, gate.last_good, payload)
        except FileExistsError as error:
            log(str(error))
            continue
        saved += 1
        log(f"saved OK {png.name} + {meta.name}  total={saved}  "
            f"t=[{T_link[0, 3]:.3f},{T_link[1, 3]:.3f},{T_link[2, 3]:.3f}]")
    return saved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture R1 hand-eye samples (image + joint q). Does not move the robot.")
    parser.add_argument("--target", required=True, choices=sorted(TARGETS),
                        help="left_wrist | right_wrist | head (three separate sessions)")
    parser.add_argument("--host", default="192.168.124.147")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--pattern", nargs=2, type=int, default=list(DEFAULT_PATTERN),
                        metavar=("COLS", "ROWS"))
    parser.add_argument("--square", type=float, default=DEFAULT_SQUARE_M)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--day", default=None)
    parser.add_argument("--stable-ms", type=int, default=300)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--interface", default="eno1", help="DDS NIC for subscribe-only lowstate")
    parser.add_argument("--lowstate-timeout", type=float, default=5.0)
    parser.add_argument("--shadow-json", type=Path, default=None,
                        help="Optional: read q from r1_a7_lowstate_shadow JSON instead of DDS")
    parser.add_argument("--shadow-max-age", type=float, default=0.5)
    parser.add_argument("--q-json", type=Path, default=None,
                        help="Offline fixed motor_q JSON (tests / dry runs; no DDS)")
    parser.add_argument("--no-lowstate", action="store_true",
                        help="Refuse live capture without --q-json/--shadow-json")
    args = parser.parse_args(argv)
    spec = TARGETS[args.target]
    if args.port is None:
        args.port = spec["port"]
    if args.topic is None:
        args.topic = spec["topic"]
    if args.port in WEBRTC_PORTS:
        parser.error("do not use WebRTC ports; wrist JPEG is 55556/55557, head stereo JPEG is 55555")
    if args.pattern[0] <= 0 or args.pattern[1] <= 0:
        parser.error("--pattern must be positive inner-corner counts")
    if args.square <= 0:
        parser.error("--square must be positive metres")
    return args


def open_image_client(host, port, topic):
    from teleimager.client import TeleImageClient
    return TeleImageClient(topic, host, port, request_bgr=True)


def main(argv=None):
    args = parse_args(argv)
    require_display()
    spec = target_spec(args.target)
    pattern = tuple(args.pattern)
    day_dir, sample_dir = capture_day_dirs(args.out_root, args.target, day=args.day)
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"R1 hand-eye capture target={args.target} camera={spec['camera']} frame={spec['frame']}")
    print(f"  host={args.host} port={args.port} topic={args.topic}")
    print(f"  board inner corners {pattern[0]}x{pattern[1]}, square={args.square:.3f} m, landscape")
    print(f"  writing {sample_dir}")
    print(f"  resuming at {next_image_index(sample_dir):03d}; will not overwrite")
    print("  YOU move the arm/head by hand (or carefully with existing teleop already running).")
    print("  This tool does NOT command motion. Do not start teleop from this script.")
    print("  Aim >=20 diverse OK samples (hold some out at fit). SPACE/s = save, q = quit (NOT e-stop).")
    fk = R1A7FK(urdf_path=args.urdf)
    q_source = None
    client = None
    saved = 0
    try:
        q_source = open_q_source(args)
        try:
            client = open_image_client(args.host, args.port, args.topic)
        except Exception as error:
            raise SystemExit(
                f"could not subscribe to {args.topic} at {args.host}:{args.port}: {error}\n"
                f"preflight: export XDG_RUNTIME_DIR=/run/user/$(id -u); "
                f"systemctl --user is-active {spec['service']}\n"
                "do not start teleop; do not restart services unsupervised."
            ) from error
        saved = run_capture_loop(
            iter_client_frames(client), sample_dir, args.target, q_source, fk,
            pattern=pattern, square=args.square, stable_s=args.stable_ms / 1000.0,
            host=args.host, port=args.port, topic=args.topic, window_name=spec["camera"])
    except KeyboardInterrupt:
        print("\ninterrupted (Ctrl+C). " + QUIT_NOTE)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if q_source is not None:
            try:
                q_source.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
    print(f"saved {saved} samples under {day_dir}")
    print(QUIT_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
