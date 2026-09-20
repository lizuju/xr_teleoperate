#!/usr/bin/env python3
"""Capture monocular wrist chessboard frames for R1 plumb_bob calibration.

Reads ZMQ ``left_wrist_camera`` (55556) or ``right_wrist_camera`` (55557) via
TeleImageClient. Saves a PNG only when all 54 inner corners are present and the
board has been held still. One wrist at a time.

Does not start teleoperation, does not command the robot, and does not use
WebRTC 60002/60003.

q quits capture. That is not an e-stop.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import cv2
import numpy as np


DEFAULT_HOST = "192.168.124.147"
DEFAULT_PATTERN = (9, 6)
DEFAULT_SQUARE_M = 0.030
DEFAULT_OUT_ROOT = Path("~/r1-cam-calib")
WRIST_FRAME_SIZE = (640, 480)  # width, height of the recording JPEG
WEBRTC_PORTS = (60002, 60003)
QUIT_NOTE = "q is NOT e-stop; capture stopped. The robot was not commanded."

SIDES = {
    "left": {
        "port": 55556,
        "topic": "left_wrist_camera",
        "camera": "left_wrist",
        "service": "r1-camera-forward-55556.service",
    },
    "right": {
        "port": 55557,
        "topic": "right_wrist_camera",
        "camera": "right_wrist",
        "service": "r1-camera-forward-55557.service",
    },
}


class CaptureError(RuntimeError):
    """Raised when the live stream cannot be used for chessboard capture."""


class FrameShapeError(CaptureError):
    """Raised when the ZMQ frame is not the 640x480 wrist layout."""


def inner_corner_count(pattern):
    return int(pattern[0]) * int(pattern[1])


def side_spec(side):
    try:
        return SIDES[side]
    except KeyError as error:
        raise CaptureError(f"side must be left or right, got {side!r}") from error


def require_wrist_frame(frame, expected_size=WRIST_FRAME_SIZE):
    """Return a contiguous BGR copy after checking 640x480."""
    if frame is None:
        raise FrameShapeError("no wrist frame")
    if getattr(frame, "ndim", 0) != 3:
        raise FrameShapeError("wrist frame must be an HxWxC BGR image")
    height, width = frame.shape[:2]
    expected_width, expected_height = expected_size
    if width != expected_width or height != expected_height:
        hint = (
            "Use ZMQ 55556 topic left_wrist_camera or 55557 topic right_wrist_camera; "
            "do not use head 55555 or WebRTC 60002/60003."
        )
        if width == 1088 and height == 448:
            hint = "this looks like concatenated head stereo (55555), not a wrist camera. " + hint
        elif width == 544 and height == 448:
            hint = "this looks like one head eye (544x448), not a wrist camera. " + hint
        raise FrameShapeError(
            f"wrist frame is {width}x{height}, expected {expected_width}x{expected_height}. {hint}"
        )
    return np.ascontiguousarray(frame)


def detect_corners(bgr, pattern=DEFAULT_PATTERN):
    """Return (ok, corners) only when every inner corner is present."""
    expected = inner_corner_count(pattern)
    if bgr is None or bgr.size == 0:
        return False, None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(
        gray, tuple(pattern), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            gray, tuple(pattern),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    if not ok or corners is None:
        return False, None
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    if len(corners) != expected:
        return False, None
    return True, corners


def corner_count(ok, corners):
    if not ok or corners is None:
        return 0
    return int(np.asarray(corners).reshape(-1, 2).shape[0])


def overlay_label(n_corners, next_index, stable, expected=54):
    if n_corners == expected:
        tag = "OK" if stable else "OK hold"
        return f"{tag}  corners={n_corners}  next={next_index:03d}"
    return f"corners={n_corners}/{expected}  next={next_index:03d}"


def annotate_frame(bgr, pattern, corners, ok, next_index, stable, expected=None):
    expected = inner_corner_count(pattern) if expected is None else expected
    vis = bgr.copy()
    if ok and corners is not None:
        cv2.drawChessboardCorners(vis, tuple(pattern), corners, True)
    n_corners = corner_count(ok, corners)
    if n_corners == expected and stable:
        color = (0, 255, 0)
    elif n_corners == expected:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)
    text = overlay_label(n_corners, next_index, stable, expected=expected)
    cv2.putText(vis, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(vis, "SPACE=save   q=quit (NOT e-stop)", (12, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


class FrameStability:
    """Accept a save only after the board stays OK for ``stable_s`` seconds."""

    def __init__(self, stable_s=0.3):
        self.stable_s = float(stable_s)
        self.ok_since = None
        self.last_good = None

    def update(self, image, ok, now):
        if ok:
            if self.ok_since is None:
                self.ok_since = now
            self.last_good = image.copy()
            return (now - self.ok_since) >= self.stable_s
        self.ok_since = None
        return False


def next_image_index(folder):
    existing = []
    folder = Path(folder)
    if folder.is_dir():
        for path in folder.glob("*.png"):
            if path.stem.isdigit():
                existing.append(int(path.stem))
    return (max(existing) + 1) if existing else 0


def save_image(folder, index, image):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{index:03d}.png"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing {path}")
    temporary = folder / f".{index:03d}.tmp.png"
    try:
        if not cv2.imwrite(str(temporary), image):
            raise IOError(f"cv2.imwrite failed for {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def capture_day_dirs(out_root, side, day=None):
    spec = side_spec(side)
    root = Path(out_root).expanduser()
    day = day or time.strftime("%Y%m%d")
    day_dir = root / day
    image_dir = day_dir / "captures" / spec["camera"]
    return day_dir, image_dir


def require_display():
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise SystemExit(
        "No DISPLAY/WAYLAND_DISPLAY. Run this on the Ubuntu desktop graphical session, "
        "not a raw SSH tty. q is not e-stop."
    )


def iter_client_frames(client, idle_s=0.02, sleep=time.sleep):
    while True:
        frame = client.get_frame()
        if frame is None or getattr(frame, "bgr", None) is None:
            sleep(idle_s)
            continue
        yield frame.bgr


def run_capture_loop(frames, image_dir, pattern=DEFAULT_PATTERN, stable_s=0.3,
                     expected_size=WRIST_FRAME_SIZE, clock=time.monotonic,
                     wait_key=None, imshow=None, detect=detect_corners, log=print,
                     window_name="wrist"):
    """Consume BGR frames, overlay detection, save PNGs on SPACE.

    ``wait_key`` and ``imshow`` are injectable so tests do not need a GUI.
    Returns the number of images written in this session.
    """
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    index = next_image_index(image_dir)
    saved = 0
    gate = FrameStability(stable_s=stable_s)
    expected = inner_corner_count(pattern)
    warned_shape = False
    log(f"resuming at {index:03d}.png; existing captures will not be overwritten")
    if wait_key is None:
        wait_key = lambda delay: cv2.waitKey(delay) & 0xFF
    if imshow is None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        imshow = cv2.imshow

    for frame in frames:
        try:
            image = require_wrist_frame(frame, expected_size=expected_size)
        except FrameShapeError as error:
            if not warned_shape:
                log(str(error))
                warned_shape = True
            key = wait_key(1) & 0xFF
            if key in (ord("q"), 27):
                log(QUIT_NOTE)
                break
            continue
        ok, corners = detect(image, pattern)
        now = clock()
        stable = gate.update(image, ok, now)
        next_index = next_image_index(image_dir)
        vis = annotate_frame(
            image, pattern, corners, ok, next_index, stable, expected=expected)
        imshow(window_name, vis)
        key = wait_key(1) & 0xFF
        if key in (ord("q"), 27):
            log(QUIT_NOTE)
            break
        if key != ord(" "):
            continue
        n_corners = corner_count(ok, corners)
        if not stable or gate.last_good is None:
            if n_corners != expected:
                log(f"not saved: need {expected} corners (got {n_corners})")
            else:
                log(f"not saved: hold still for {int(round(stable_s * 1000))} ms (motion-blur guard)")
            continue
        save_index = next_image_index(image_dir)
        try:
            path = save_image(image_dir, save_index, gate.last_good)
        except FileExistsError as error:
            log(str(error))
            continue
        saved += 1
        log(f"saved {path.name}  {path}  total={saved}")
    return saved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture R1 wrist chessboard frames. Does not start teleop or move the robot.")
    parser.add_argument("--side", required=True, choices=sorted(SIDES),
                        help="Calibrate one wrist at a time (left then right)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="TeleImageClient ZMQ host")
    parser.add_argument("--port", type=int, default=None,
                        help="ZMQ port (default: 55556 left / 55557 right; not 60002/60003)")
    parser.add_argument("--topic", default=None,
                        help="ZMQ topic (default: left_wrist_camera / right_wrist_camera)")
    parser.add_argument("--pattern", nargs=2, type=int, default=list(DEFAULT_PATTERN), metavar=("COLS", "ROWS"),
                        help="OpenCV inner corners, not square count")
    parser.add_argument("--square", type=float, default=DEFAULT_SQUARE_M, help="Square size in metres")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--stable-ms", type=int, default=300,
                        help="Require OK this long before SPACE is accepted")
    parser.add_argument("--day", default=None, help="YYYYMMDD capture folder (default: today)")
    args = parser.parse_args(argv)
    spec = SIDES[args.side]
    if args.port is None:
        args.port = spec["port"]
    if args.topic is None:
        args.topic = spec["topic"]
    if args.port in WEBRTC_PORTS:
        parser.error("do not use WebRTC ports 60002/60003; wrist JPEG is ZMQ 55556/55557")
    if args.pattern[0] <= 0 or args.pattern[1] <= 0:
        parser.error("--pattern must be two positive inner-corner counts")
    if args.square <= 0:
        parser.error("--square must be a positive length in metres")
    if args.stable_ms < 0:
        parser.error("--stable-ms must be >= 0")
    return args


def open_wrist_client(host, port, topic):
    from teleimager.client import TeleImageClient
    return TeleImageClient(topic, host, port, request_bgr=True)


def main(argv=None):
    args = parse_args(argv)
    require_display()
    spec = SIDES[args.side]
    pattern = tuple(args.pattern)
    day_dir, image_dir = capture_day_dirs(args.out_root, args.side, day=args.day)
    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"R1 {spec['camera']} capture")
    print(f"  host={args.host} port={args.port} topic={args.topic}")
    print(f"  board inner corners {pattern[0]}x{pattern[1]} = {inner_corner_count(pattern)}, "
          f"square={args.square:.3f} m, landscape")
    resume_at = next_image_index(image_dir)
    print(f"  writing {image_dir}")
    print(f"  resuming at {resume_at:03d}.png; existing PNGs are kept (will not overwrite)")
    print("  SPACE = save (only when overlay says OK and the board is held still)")
    print("  q = quit capture (NOT e-stop; the robot is not commanded)")
    print("  need >=40 OK (30 train + 10 holdout). Hold 0.25-0.7 m in FRONT of this wrist camera.")
    print("  robot stays still. do not start teleop. do not command arm motion.")
    print("  one wrist at a time. do not use 60002/60003.")
    client = None
    saved = 0
    try:
        try:
            client = open_wrist_client(args.host, args.port, args.topic)
        except Exception as error:
            raise SystemExit(
                f"could not subscribe to {args.topic} at {args.host}:{args.port}: {error}\n"
                "preflight: export XDG_RUNTIME_DIR=/run/user/$(id -u); "
                f"systemctl --user is-active {spec['service']}\n"
                "If inactive, check the unit; do not restart services unsupervised. "
                "do not start teleop."
            ) from error
        saved = run_capture_loop(
            iter_client_frames(client), image_dir,
            pattern=pattern, stable_s=args.stable_ms / 1000.0,
            window_name=spec["camera"])
    except KeyboardInterrupt:
        print("\ninterrupted (Ctrl+C). " + QUIT_NOTE)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
    print(f"saved {saved} images under {day_dir}")
    print(QUIT_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
