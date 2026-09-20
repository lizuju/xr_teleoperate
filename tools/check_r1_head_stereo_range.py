#!/usr/bin/env python3
"""Live tape-measure check of R1 head fisheye stereo ranging.

Reads ZMQ ``head_camera`` (default 192.168.124.147:55555) via TeleImageClient,
splits one 1088x448 frame into left [:, :544] and right [:, 544:], detects a
chessboard on both eyes, undistorts the corners with fisheye Knew (never None),
and triangulates with P = Knew[I|0], P' = Knew'[R|t] from
``assets/r1/camera_calibration.json``.

Does not start teleoperation, does not command the robot, and does not touch
WebRTC 60001 or XR undistort.

SPACE snapshots a reading. q quits. That is not an e-stop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from calibrate_r1_head_stereo_capture import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TOPIC,
    DEFAULT_PATTERN,
    DEFAULT_SQUARE_M,
    HEAD_EYE_SIZE,
    HEAD_FRAME_SIZE,
    HEAD_SPLIT_X,
    FrameShapeError,
    corner_count,
    detect_corners,
    detect_pair,
    inner_corner_count,
    iter_client_frames,
    open_head_client,
    require_display,
    split_head_stereo,
)
from teleop.utils.camera_calibration import (  # noqa: E402
    CalibrationError,
    DEFAULT_CALIBRATION_RELPATH,
    load_camera_calibration,
)


WEBRTC_PORTS = (60001, 60002, 60003)
WINDOW_NAME = "head_stereo_range"
QUIT_NOTE = "q is NOT e-stop; range check stopped. The robot was not commanded."
KNEW_REQUIRED = "Knew is required; cv2.fisheye.undistortImage(Knew=None) goes black"
HOLD_TAPE_HINT = "hold tape at camera midpoint to board plane; compare mid_plane"
MIN_VALID_Z_M = 0.05
MAX_VALID_Z_M = 3.00


class RangeError(RuntimeError):
    """Raised when calibration or triangulation cannot be used for ranging."""


@dataclass
class HeadStereoGeometry:
    source: str
    sha256: str
    K_left: np.ndarray
    D_left: np.ndarray
    Knew_left: np.ndarray
    K_right: np.ndarray
    D_right: np.ndarray
    Knew_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    baseline_m: float
    image_size: tuple
    left_rms_px: float | None
    right_rms_px: float | None


@dataclass
class RangeReading:
    z_med_m: float
    z_p25_m: float
    z_p75_m: float
    n_valid: int
    n_corners: int
    left_rms_px: float
    right_rms_px: float
    disparity_med_px: float
    square_med_m: float
    mid_plane_m: float
    xyz: np.ndarray
    tape_m: float | None = None
    err_m: float | None = None
    err_pct: float | None = None


def require_knew(Knew, fallback=None):
    """Return a concrete 3x3 Knew. Never None — undistortImage(Knew=None) goes black."""
    if Knew is None:
        if fallback is None:
            raise RangeError(KNEW_REQUIRED)
        Knew = fallback
    Knew = np.asarray(Knew, dtype=np.float64).reshape(3, 3).copy()
    if Knew.shape != (3, 3) or not np.isfinite(Knew).all():
        raise RangeError("Knew must be a finite 3x3 camera matrix")
    if Knew[0, 0] <= 0.0 or Knew[1, 1] <= 0.0:
        raise RangeError("Knew fx and fy must be positive")
    return Knew


def undistort_fisheye_image(bgr, K, D, Knew, new_size=None):
    """Undistort a fisheye BGR image. ``Knew`` is required (None yields a black image)."""
    if Knew is None:
        raise RangeError(KNEW_REQUIRED)
    Knew = require_knew(Knew)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
    if new_size is None:
        height, width = bgr.shape[:2]
        new_size = (int(width), int(height))
    return cv2.fisheye.undistortImage(bgr, K, D, Knew=Knew, new_size=tuple(int(v) for v in new_size))


def undistort_fisheye_points(points, K, D, Knew):
    """Undistort fisheye pixel corners into the pinhole image of ``Knew``."""
    if Knew is None:
        raise RangeError(KNEW_REQUIRED)
    Knew = require_knew(Knew)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
    return cv2.fisheye.undistortPoints(pts, K, D, P=Knew)


def projection_matrices(K_left, K_right, R, T):
    """P = K[I|0], P' = K'[R|t] in the left-camera frame."""
    K_left = np.asarray(K_left, dtype=np.float64).reshape(3, 3)
    K_right = np.asarray(K_right, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    p_left = K_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p_right = K_right @ np.hstack([R, T])
    return p_left, p_right


def homogeneous_to_xyz(hom):
    hom = np.asarray(hom, dtype=np.float64)
    if hom.shape[0] == 4 and hom.shape[1] != 4:
        hom = hom.T
    xyz = np.full((hom.shape[0], 3), np.nan, dtype=np.float64)
    w = hom[:, 3]
    valid = np.abs(w) > 1e-8
    xyz[valid] = hom[valid, :3] / w[valid, None]
    return xyz


def stereo_midpoint_left(R, T):
    """Stereo camera midpoint expressed in the left-camera frame."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    c_right = (-R.T @ T).ravel()
    return 0.5 * c_right


def plane_distance_m(xyz, origin):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    if xyz.shape[0] < 3:
        return float("nan")
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return float("nan")
    normal = normal / norm
    return float(abs(np.dot(normal, origin - centroid)))


def reconstructed_square_m(xyz, pattern):
    cols, rows = int(pattern[0]), int(pattern[1])
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] != cols * rows:
        return float("nan")
    widths = []
    for row in range(rows):
        for col in range(cols - 1):
            index = row * cols + col
            if not np.isfinite(pts[index]).all() or not np.isfinite(pts[index + 1]).all():
                continue
            widths.append(float(np.linalg.norm(pts[index + 1] - pts[index])))
    if not widths:
        return float("nan")
    return float(np.median(np.asarray(widths)))


def _camera_arrays(entry, location):
    if entry.get("distortion_model") != "fisheye":
        raise RangeError(
            f"{location}: expected distortion_model 'fisheye', got {entry.get('distortion_model')!r}"
        )
    K = np.asarray(entry["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(entry["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    if D.shape[0] < 4:
        raise RangeError(f"{location}: fisheye distortion_coefficients need 4 values")
    D = D[:4].reshape(4, 1)
    size = tuple(int(v) for v in entry["image_size"])
    rms = entry.get("reprojection_error_px")
    return K, D, size, None if rms is None else float(rms)


def load_head_stereo_geometry(path):
    """Load head fisheye stereo from a validated r1_camera_calibration_v1 JSON."""
    path = Path(path).expanduser()
    try:
        calibration = load_camera_calibration(path)
    except CalibrationError as error:
        raise RangeError(str(error)) from error
    cameras = calibration["cameras"]
    stereo = (calibration.get("stereo") or {}).get("head")
    if stereo is None:
        raise RangeError(f"{path}: missing stereo.head")
    for name in (stereo["left"], stereo["right"]):
        if name not in cameras:
            raise RangeError(f"{path}: stereo.head names {name} which is not in cameras")
    left = cameras[stereo["left"]]
    right = cameras[stereo["right"]]
    K_left, D_left, size_left, left_rms = _camera_arrays(left, f"{path}: cameras.{stereo['left']}")
    K_right, D_right, size_right, right_rms = _camera_arrays(
        right, f"{path}: cameras.{stereo['right']}")
    if size_left != size_right:
        raise RangeError(
            f"{path}: head image sizes differ ({size_left} vs {size_right})"
        )
    if tuple(size_left) != tuple(HEAD_EYE_SIZE):
        raise RangeError(
            f"{path}: head image_size is {size_left[0]}x{size_left[1]}, expected "
            f"{HEAD_EYE_SIZE[0]}x{HEAD_EYE_SIZE[1]}"
        )
    R = np.asarray(stereo["rotation"], dtype=np.float64).reshape(3, 3)
    T = np.asarray(stereo["translation_m"], dtype=np.float64).reshape(3, 1)
    baseline = float(stereo.get("baseline_m", np.linalg.norm(T)))
    Knew_left = require_knew(K_left)
    Knew_right = require_knew(K_right)
    return HeadStereoGeometry(
        source=str(calibration.get("source") or path),
        sha256=str(calibration.get("sha256") or ""),
        K_left=K_left,
        D_left=D_left,
        Knew_left=Knew_left,
        K_right=K_right,
        D_right=D_right,
        Knew_right=Knew_right,
        R=R,
        T=T,
        baseline_m=baseline,
        image_size=tuple(size_left),
        left_rms_px=left_rms,
        right_rms_px=right_rms,
    )


def resolve_calib_path(path, root=None):
    """Resolve --calib. Relative paths prefer ``root`` (repo), then cwd.

    A bare relative path is not checked first: ``is_file()`` would follow
    the process cwd and pick production ``assets/`` while tests pass a temp root.
    """
    root = Path(root) if root is not None else REPO
    path = Path(path).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path.resolve()
        raise RangeError(f"calibration file not found: {path}")
    candidates = [root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(item) for item in candidates)
    raise RangeError(f"calibration file not found: {path}. Tried {tried}")


def triangulate_corners(left_points, right_points, geometry, pattern=DEFAULT_PATTERN,
                        tape=None):
    """Triangulate matched fisheye corners in the left-camera frame.

    ``z_med`` is the median left-camera Z of those points. ``mid_plane`` is the
    perpendicular distance from the stereo midpoint to the fitted board plane
    (the tape-measure quantity). Both summaries use the same triangulation,
    not a separate PnP solve.
    """
    left_points = np.asarray(left_points, dtype=np.float64).reshape(-1, 2)
    right_points = np.asarray(right_points, dtype=np.float64).reshape(-1, 2)
    if left_points.shape != right_points.shape or left_points.shape[0] == 0:
        raise RangeError("left and right corner counts must match and be non-empty")
    knew_left = require_knew(geometry.Knew_left, fallback=geometry.K_left)
    knew_right = require_knew(geometry.Knew_right, fallback=geometry.K_right)
    undist_left = undistort_fisheye_points(
        left_points, geometry.K_left, geometry.D_left, knew_left)
    undist_right = undistort_fisheye_points(
        right_points, geometry.K_right, geometry.D_right, knew_right)
    p_left, p_right = projection_matrices(knew_left, knew_right, geometry.R, geometry.T)
    left_px = np.ascontiguousarray(undist_left.reshape(-1, 2).T)
    right_px = np.ascontiguousarray(undist_right.reshape(-1, 2).T)
    hom = cv2.triangulatePoints(p_left, p_right, left_px, right_px)
    xyz = homogeneous_to_xyz(hom)
    z = xyz[:, 2]
    valid = np.isfinite(xyz).all(axis=1) & (z > MIN_VALID_Z_M) & (z < MAX_VALID_Z_M)
    if int(np.count_nonzero(valid)) < max(8, left_points.shape[0] // 3):
        raise RangeError(
            f"triangulation produced too few valid depths "
            f"({int(np.count_nonzero(valid))}/{left_points.shape[0]})"
        )
    z_valid = z[valid]
    xyz_valid = xyz[valid]
    disparity = undist_left.reshape(-1, 2)[:, 0] - undist_right.reshape(-1, 2)[:, 0]
    left_rms, right_rms = reprojection_rms(
        xyz, left_points, right_points, geometry)
    origin = stereo_midpoint_left(geometry.R, geometry.T)
    reading = RangeReading(
        z_med_m=float(np.median(z_valid)),
        z_p25_m=float(np.percentile(z_valid, 25)),
        z_p75_m=float(np.percentile(z_valid, 75)),
        n_valid=int(np.count_nonzero(valid)),
        n_corners=int(left_points.shape[0]),
        left_rms_px=float(left_rms),
        right_rms_px=float(right_rms),
        disparity_med_px=float(np.median(disparity)),
        square_med_m=reconstructed_square_m(xyz, pattern),
        mid_plane_m=plane_distance_m(xyz_valid, origin),
        xyz=xyz,
    )
    return attach_tape(reading, tape)


def reprojection_rms(xyz, left_points, right_points, geometry):
    """Reproject triangulated points through the fisheye model (cheap L/R RMS)."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 1, 3)
    left_points = np.asarray(left_points, dtype=np.float64).reshape(-1, 2)
    right_points = np.asarray(right_points, dtype=np.float64).reshape(-1, 2)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    proj_left, _ = cv2.fisheye.projectPoints(
        xyz, rvec, tvec, geometry.K_left, geometry.D_left)
    xyz_right = (
        geometry.R @ xyz.reshape(-1, 3).T + geometry.T.reshape(3, 1)
    ).T.reshape(-1, 1, 3)
    proj_right, _ = cv2.fisheye.projectPoints(
        xyz_right, rvec, tvec, geometry.K_right, geometry.D_right)
    delta_left = proj_left.reshape(-1, 2) - left_points
    delta_right = proj_right.reshape(-1, 2) - right_points
    finite_left = np.isfinite(delta_left).all(axis=1)
    finite_right = np.isfinite(delta_right).all(axis=1)
    left_rms = float(np.sqrt(np.mean(np.sum(delta_left[finite_left] ** 2, axis=1)))) if np.any(finite_left) else float("nan")
    right_rms = float(np.sqrt(np.mean(np.sum(delta_right[finite_right] ** 2, axis=1)))) if np.any(finite_right) else float("nan")
    return left_rms, right_rms


def attach_tape(reading, tape):
    """Compare tape (midpoint to board plane) against mid_plane, not left-cam z_med."""
    if tape is None:
        reading.tape_m = None
        reading.err_m = None
        reading.err_pct = None
        return reading
    tape = float(tape)
    reading.tape_m = tape
    reading.err_m = float(reading.mid_plane_m - tape)
    reading.err_pct = float(100.0 * reading.err_m / tape) if tape else float("nan")
    return reading


def format_snapshot(reading, tape=None):
    if tape is not None:
        reading = attach_tape(reading, tape)
    parts = [
        "SNAP",
        f"mid_plane={reading.mid_plane_m:.3f} m",
        f"z_med={reading.z_med_m:.3f} m (left-cam)",
        f"disp={reading.disparity_med_px:.2f} px",
        f"n={reading.n_valid}/{reading.n_corners}",
        f"L_rms={reading.left_rms_px:.2f} px",
        f"R_rms={reading.right_rms_px:.2f} px",
        f"square={reading.square_med_m:.4f} m",
    ]
    if reading.tape_m is not None:
        parts.append(f"tape={reading.tape_m:.3f} m")
        parts.append(f"err={reading.err_m * 1000.0:+.0f} mm ({reading.err_pct:+.1f}%)")
    parts.append("| " + HOLD_TAPE_HINT)
    return "  ".join(parts)


def overlay_text(reading, expected, n_left, n_right, tape=None):
    """Return (primary, secondary) overlay lines. Primary is mid_plane vs tape."""
    if reading is None:
        return f"L={n_left} R={n_right}  need {expected} on BOTH", ""
    primary = f"mid_plane={reading.mid_plane_m:.3f} m  L={n_left} R={n_right}"
    if tape is not None:
        reading = attach_tape(reading, tape)
        err_mm = reading.err_m * 1000.0
        primary += f"  tape={float(tape):.3f} err={err_mm:+.0f}mm"
    secondary = (f"z_med={reading.z_med_m:.3f} m (left-cam)  "
                 f"L_rms={reading.left_rms_px:.2f} R_rms={reading.right_rms_px:.2f}")
    return primary, secondary


def annotate_stereo(left, right, pattern, corners_left, corners_right, ok_left, ok_right,
                    reading, expected=None, tape=None):
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
    if reading is not None:
        color = (0, 255, 0)
    elif both:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)
    primary, secondary = overlay_text(
        reading, expected, n_left, n_right, tape=tape)
    cv2.putText(vis, primary, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2, cv2.LINE_AA)
    if secondary:
        cv2.putText(vis, secondary, (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        help_y, hint_y = 92, 120
    else:
        help_y, hint_y = 64, 92
    cv2.putText(vis, "SPACE=snapshot   q=quit (NOT e-stop)", (12, help_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, HOLD_TAPE_HINT, (12, hint_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def try_range_reading(ok_left, corners_left, ok_right, corners_right, geometry,
                      pattern, expected, tape=None):
    n_left = corner_count(ok_left, corners_left)
    n_right = corner_count(ok_right, corners_right)
    if n_left != expected or n_right != expected:
        return None
    try:
        return triangulate_corners(
            corners_left, corners_right, geometry, pattern=pattern, tape=tape)
    except (RangeError, cv2.error):
        return None


def run_range_loop(frames, geometry, pattern=DEFAULT_PATTERN, tape=None,
                   split_x=HEAD_SPLIT_X, expected_size=HEAD_FRAME_SIZE,
                   wait_key=None, imshow=None, detect=detect_corners, log=print,
                   window_name=WINDOW_NAME):
    """Consume BGR stereo frames, overlay Z, snapshot on SPACE. Returns readings."""
    expected = inner_corner_count(pattern)
    warned_shape = False
    readings = []
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
        reading = try_range_reading(
            ok_left, corners_left, ok_right, corners_right, geometry,
            pattern, expected, tape=tape)
        vis = annotate_stereo(
            left, right, pattern, corners_left, corners_right, ok_left, ok_right,
            reading, expected=expected, tape=tape)
        imshow(window_name, vis)
        key = wait_key(1) & 0xFF
        if key in (ord("q"), 27):
            log(QUIT_NOTE)
            break
        if key != ord(" "):
            continue
        n_left = corner_count(ok_left, corners_left)
        n_right = corner_count(ok_right, corners_right)
        if reading is None:
            if n_left != expected or n_right != expected:
                log(f"not snapped: need {expected} corners on BOTH eyes (L={n_left} R={n_right})")
            else:
                log("not snapped: triangulation failed (board may be too close/tilted)")
            continue
        line = format_snapshot(reading, tape=tape)
        log(line)
        readings.append(reading)
    return readings


def k_line(name, K, D, rms=None):
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    rms_txt = "" if rms is None else f"  calib_rms={rms:.3f} px"
    return (f"{name} fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} "
            f"cx={K[0, 2]:.2f} cy={K[1, 2]:.2f} D={D.tolist()}{rms_txt}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tape-measure check of R1 head fisheye stereo ranging. "
                    "Does not start teleop, move the robot, or touch WebRTC 60001.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIBRATION_RELPATH,
                        help="r1_camera_calibration_v1 JSON (default: assets/r1/camera_calibration.json)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="TeleImageClient ZMQ host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="head_camera ZMQ port (not 60001)")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--pattern", nargs=2, type=int, default=list(DEFAULT_PATTERN),
                        metavar=("COLS", "ROWS"),
                        help="OpenCV inner corners, not square count")
    parser.add_argument("--square", type=float, default=DEFAULT_SQUARE_M,
                        help="Square size in metres (used to score reconstructed square size)")
    parser.add_argument("--tape", type=float, default=None,
                        help="Tape-measure distance in metres from camera midpoint to the board plane")
    args = parser.parse_args(argv)
    if args.port in WEBRTC_PORTS:
        parser.error("do not use WebRTC ports 60001/60002/60003; head stereo is ZMQ 55555")
    if args.pattern[0] <= 0 or args.pattern[1] <= 0:
        parser.error("--pattern must be two positive inner-corner counts")
    if args.square <= 0:
        parser.error("--square must be a positive length in metres")
    if args.tape is not None and args.tape <= 0:
        parser.error("--tape must be a positive length in metres")
    return args


def main(argv=None):
    args = parse_args(argv)
    require_display()
    pattern = tuple(args.pattern)
    try:
        calib_path = resolve_calib_path(args.calib)
        geometry = load_head_stereo_geometry(calib_path)
    except RangeError as error:
        raise SystemExit(str(error)) from error
    print("R1 head stereo range check")
    print(f"  calib={geometry.source}")
    print(f"  sha256={geometry.sha256}")
    print(f"  host={args.host} port={args.port} topic={args.topic}")
    print(f"  board inner corners {pattern[0]}x{pattern[1]} = {inner_corner_count(pattern)}, "
          f"square={args.square:.3f} m, landscape")
    print("  " + k_line("left ", geometry.K_left, geometry.D_left, geometry.left_rms_px))
    print("  " + k_line("right", geometry.K_right, geometry.D_right, geometry.right_rms_px))
    print(f"  baseline={geometry.baseline_m:.4f} m")
    print("  Knew = K from JSON (never None; fisheye undistortImage Knew=None goes black)")
    print("  triangulate: undistortPoints(P=Knew) then P=Knew[I|0], P'=Knew'[R|t]")
    if args.tape is not None:
        print(f"  tape={args.tape:.3f} m (compare to mid_plane; z_med is left-cam median Z)")
    print("  SPACE = snapshot mid_plane (tape from midpoint to board) and z_med (left-cam)")
    print("  q = quit (NOT e-stop; the robot is not commanded)")
    print("  measure tape from BETWEEN the two head cameras to the BOARD PLANE")
    print("  that tape number is mid_plane, not left-cam z_med")
    print("  robot stays still. do not start teleop. do not use WebRTC 60001.")
    client = None
    readings = []
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
        readings = run_range_loop(
            iter_client_frames(client), geometry, pattern=pattern, tape=args.tape)
    except KeyboardInterrupt:
        print("\ninterrupted (Ctrl+C). " + QUIT_NOTE)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
    print(f"snapshots={len(readings)}")
    print(QUIT_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
