#!/usr/bin/env python3
"""Capture paired head-stereo chessboard frames for R1 fisheye calibration.

Reads ZMQ ``head_camera`` (default 192.168.124.147:55555) via TeleImageClient,
splits one 1088x448 frame into left [:, :544] and right [:, 544:], and writes a
pair only when BOTH eyes have 54 inner corners and the board has been held
still. Does not start teleoperation, does not command the robot, and does not
touch wrist cameras or WebRTC 60002/60003.

q quits capture. That is not an e-stop.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


DEFAULT_HOST = "192.168.124.147"
DEFAULT_PORT = 55555
DEFAULT_TOPIC = "head_camera"
DEFAULT_PATTERN = (9, 6)
DEFAULT_SQUARE_M = 0.030
DEFAULT_OUT_ROOT = Path("~/r1-cam-calib")
HEAD_FRAME_SIZE = (1088, 448)  # width, height of the concatenated stereo JPEG
HEAD_SPLIT_X = 544
HEAD_EYE_SIZE = (544, 448)
WINDOW_NAME = "head_stereo"
QUIT_NOTE = "q is NOT e-stop; capture stopped. The robot was not commanded."


class CaptureError(RuntimeError):
    """Raised when the live stream cannot be used for chessboard capture."""


class FrameShapeError(CaptureError):
    """Raised when the ZMQ frame is not the 1088x448 head stereo layout."""


def inner_corner_count(pattern):
    return int(pattern[0]) * int(pattern[1])


def split_head_stereo(frame, split_x=HEAD_SPLIT_X, expected_size=HEAD_FRAME_SIZE):
    """Split one concatenated head frame into left and right 544x448 images."""
    if frame is None:
        raise FrameShapeError("no head stereo frame")
    if getattr(frame, "ndim", 0) != 3:
        raise FrameShapeError("head stereo frame must be an HxWxC BGR image")
    height, width = frame.shape[:2]
    expected_width, expected_height = expected_size
    if width != expected_width or height != expected_height:
        raise FrameShapeError(
            f"head stereo frame is {width}x{height}, expected {expected_width}x{expected_height}. "
            "Use ZMQ 55555 topic head_camera; do not use wrist 55556/55557 or WebRTC 60002/60003."
        )
    if not 0 < split_x < width:
        raise FrameShapeError(f"split_x={split_x} is not inside width={width}")
    left = np.ascontiguousarray(frame[:, :split_x])
    right = np.ascontiguousarray(frame[:, split_x:])
    if left.shape[1] != HEAD_EYE_SIZE[0] or right.shape[1] != HEAD_EYE_SIZE[0]:
        raise FrameShapeError(
            f"split produced {left.shape[1]}+{right.shape[1]} wide eyes, expected {HEAD_EYE_SIZE[0]}+{HEAD_EYE_SIZE[0]}"
        )
    return left, right


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


def detect_pair(left, right, pattern=DEFAULT_PATTERN, detect=detect_corners):
    ok_left, corners_left = detect(left, pattern)
    ok_right, corners_right = detect(right, pattern)
    return ok_left, corners_left, ok_right, corners_right


def corner_count(ok, corners):
    if not ok or corners is None:
        return 0
    return int(np.asarray(corners).reshape(-1, 2).shape[0])


def overlay_label(n_left, n_right, saved, stable, expected=54):
    if n_left == expected and n_right == expected:
        tag = "PAIR OK" if stable else "PAIR OK hold"
        return f"{tag}  L={n_left} R={n_right}  saved={saved}"
    return f"L={n_left} R={n_right}  saved={saved}"


def annotate_stereo(left, right, pattern, corners_left, corners_right, ok_left, ok_right,
                    saved, stable, expected=None):
    expected = inner_corner_count(pattern) if expected is None else expected
    vis_left = left.copy()
    vis_right = right.copy()
    if ok_left and corners_left is not None:
        cv2.drawChessboardCorners(vis_left, tuple(pattern), corners_left, True)
    if ok_right and corners_right is not None:
        cv2.drawChessboardCorners(vis_right, tuple(pattern), corners_right, True)
    vis = np.hstack([vis_left, vis_right])
    n_left = corner_count(ok_left, corners_left)
    n_right = corner_count(ok_right, corners_right)
    both = n_left == expected and n_right == expected
    if both and stable:
        color = (0, 255, 0)
    elif both:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)
    text = overlay_label(n_left, n_right, saved, stable, expected=expected)
    cv2.putText(vis, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(vis, "SPACE=save pair   q=quit (NOT e-stop)", (12, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


class PairStability:
    """Accept a save only after both eyes stay PAIR OK for ``stable_s`` seconds."""

    def __init__(self, stable_s=0.3):
        self.stable_s = float(stable_s)
        self.ok_since = None
        self.last_good_left = None
        self.last_good_right = None

    def update(self, left, right, ok_left, ok_right, now):
        if ok_left and ok_right:
            if self.ok_since is None:
                self.ok_since = now
            self.last_good_left = left.copy()
            self.last_good_right = right.copy()
            return (now - self.ok_since) >= self.stable_s
        self.ok_since = None
        return False


def next_pair_index(left_dir, right_dir):
    existing = []
    for folder in (left_dir, right_dir):
        if folder is None or not Path(folder).is_dir():
            continue
        for path in Path(folder).glob("*.png"):
            if path.stem.isdigit():
                existing.append(int(path.stem))
    return (max(existing) + 1) if existing else 0


def save_pair(left_dir, right_dir, index, left_img, right_img):
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    left_path = left_dir / f"{index:03d}.png"
    right_path = right_dir / f"{index:03d}.png"
    if left_path.exists() or right_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing pair {index:03d}: {left_path} / {right_path}"
        )
    left_tmp = left_dir / f".{index:03d}.tmp.png"
    right_tmp = right_dir / f".{index:03d}.tmp.png"
    try:
        if not cv2.imwrite(str(left_tmp), left_img):
            raise IOError(f"cv2.imwrite failed for {left_tmp}")
        if not cv2.imwrite(str(right_tmp), right_img):
            raise IOError(f"cv2.imwrite failed for {right_tmp}")
        os.replace(left_tmp, left_path)
        os.replace(right_tmp, right_path)
    finally:
        left_tmp.unlink(missing_ok=True)
        right_tmp.unlink(missing_ok=True)
    return left_path, right_path


def capture_day_dirs(out_root, day=None):
    root = Path(out_root).expanduser()
    day = day or time.strftime("%Y%m%d")
    captures = root / day / "captures"
    left_dir = captures / "head_left"
    right_dir = captures / "head_right"
    return root / day, left_dir, right_dir


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


def run_capture_loop(frames, left_dir, right_dir, pattern=DEFAULT_PATTERN, stable_s=0.3,
                     split_x=HEAD_SPLIT_X, expected_size=HEAD_FRAME_SIZE, clock=time.monotonic,
                     wait_key=None, imshow=None, detect=detect_corners, log=print,
                     window_name=WINDOW_NAME):
    """Consume BGR frames, overlay detection, save paired PNGs on SPACE.

    ``wait_key`` and ``imshow`` are injectable so tests do not need a GUI.
    Returns the number of pairs written in this session.
    """
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    index = next_pair_index(left_dir, right_dir)
    saved = 0
    gate = PairStability(stable_s=stable_s)
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
            left, right = split_head_stereo(frame, split_x=split_x, expected_size=expected_size)
        except FrameShapeError as error:
            if not warned_shape:
                log(str(error))
                warned_shape = True
            key = wait_key(1) & 0xFF
            if key in (ord("q"), 27):
                log(QUIT_NOTE)
                break
            continue
        ok_left, corners_left, ok_right, corners_right = detect_pair(
            left, right, pattern=pattern, detect=detect)
        now = clock()
        stable = gate.update(left, right, ok_left, ok_right, now)
        vis = annotate_stereo(
            left, right, pattern, corners_left, corners_right, ok_left, ok_right,
            next_pair_index(left_dir, right_dir), stable, expected=expected)
        imshow(window_name, vis)
        key = wait_key(1) & 0xFF
        if key in (ord("q"), 27):
            log(QUIT_NOTE)
            break
        if key != ord(" "):
            continue
        n_left = corner_count(ok_left, corners_left)
        n_right = corner_count(ok_right, corners_right)
        if not stable or gate.last_good_left is None or gate.last_good_right is None:
            if n_left != expected or n_right != expected:
                log(f"not saved: need {expected} corners on BOTH eyes (L={n_left} R={n_right})")
            else:
                log(f"not saved: hold still for {int(round(stable_s * 1000))} ms (motion-blur guard)")
            continue
        pair_index = next_pair_index(left_dir, right_dir)
        try:
            left_path, right_path = save_pair(
                left_dir, right_dir, pair_index, gate.last_good_left, gate.last_good_right)
        except FileExistsError as error:
            log(str(error))
            continue
        saved += 1
        log(f"saved pair {pair_index:03d}  {left_path}  {right_path}  total={saved}")
    return saved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture R1 head-stereo chessboard pairs. Does not start teleop or move the robot.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="TeleImageClient ZMQ host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="head_camera ZMQ port (not 60002/60003)")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--pattern", nargs=2, type=int, default=list(DEFAULT_PATTERN), metavar=("COLS", "ROWS"),
                        help="OpenCV inner corners, not square count")
    parser.add_argument("--square", type=float, default=DEFAULT_SQUARE_M, help="Square size in metres")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--stable-ms", type=int, default=300,
                        help="Require PAIR OK this long before SPACE is accepted")
    parser.add_argument("--day", default=None, help="YYYYMMDD capture folder (default: today)")
    args = parser.parse_args(argv)
    if args.port in (60002, 60003):
        parser.error("do not use WebRTC ports 60002/60003; head stereo is ZMQ 55555")
    if args.pattern[0] <= 0 or args.pattern[1] <= 0:
        parser.error("--pattern must be two positive inner-corner counts")
    if args.square <= 0:
        parser.error("--square must be a positive length in metres")
    if args.stable_ms < 0:
        parser.error("--stable-ms must be >= 0")
    return args


def open_head_client(host, port, topic=DEFAULT_TOPIC):
    from teleimager.client import TeleImageClient
    return TeleImageClient(topic, host, port, request_bgr=True)


def main(argv=None):
    args = parse_args(argv)
    require_display()
    pattern = tuple(args.pattern)
    day_dir, left_dir, right_dir = capture_day_dirs(args.out_root, day=args.day)
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    print("R1 head stereo capture")
    print(f"  host={args.host} port={args.port} topic={args.topic}")
    print(f"  board inner corners {pattern[0]}x{pattern[1]} = {inner_corner_count(pattern)}, "
          f"square={args.square:.3f} m, landscape")
    resume_at = next_pair_index(left_dir, right_dir)
    print(f"  writing {left_dir} and {right_dir}")
    print(f"  resuming at {resume_at:03d}.png; existing PNGs are kept (will not overwrite)")
    print("  SPACE = save pair (only when overlay says PAIR OK and the board is held still)")
    print("  q = quit capture (NOT e-stop; the robot is not commanded)")
    print("  need >=40 PAIR OK (30 train + 10 holdout). Stand 0.4-1.2 m; cover centre, corners, edges.")
    print("  robot stays still. do not start teleop. do not use wrist JSON.")
    client = None
    saved = 0
    try:
        try:
            client = open_head_client(args.host, args.port, topic=args.topic)
        except Exception as error:
            raise SystemExit(
                f"could not subscribe to {args.topic} at {args.host}:{args.port}: {error}\n"
                "preflight: export XDG_RUNTIME_DIR=/run/user/$(id -u); "
                "systemctl --user is-active r1-camera-forward-55555.service\n"
                "do not start teleop; do not restart services unless that unit is clearly dead."
            ) from error
        saved = run_capture_loop(
            iter_client_frames(client), left_dir, right_dir,
            pattern=pattern, stable_s=args.stable_ms / 1000.0)
    except KeyboardInterrupt:
        print("\ninterrupted (Ctrl+C). " + QUIT_NOTE)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
    print(f"saved {saved} pairs under {day_dir}")
    print(QUIT_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
