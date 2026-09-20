#!/usr/bin/env python3
"""Fit fisheye head-stereo intrinsics from captured chessboard pairs.

Reads ``head_left`` / ``head_right`` PNGs with matching indices, holds out the
last 10 pairs (or 20%), runs cv2.fisheye.calibrate + fisheye.stereoCalibrate,
and writes ``camera_calibration.json`` under the capture day directory.

Does not copy into assets/, does not start teleoperation, and does not
calibrate wrists. Exits non-zero if the result is not physically plausible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
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
    DEFAULT_PATTERN,
    DEFAULT_SQUARE_M,
    HEAD_EYE_SIZE,
    detect_corners,
    inner_corner_count,
)
from teleop.utils.camera_calibration import (  # noqa: E402
    CALIBRATION_SCHEMA,
    camera_calibration_metadata,
    load_camera_calibration,
)


MIN_TRAIN_PAIRS = 20
MAX_RMS_PX = 1.0
BASELINE_MIN_M = 0.04
BASELINE_MAX_M = 0.08
EXPECTED_BASELINE_M = 0.060
HOLD_OUT_N = 10
HOLD_OUT_FRAC = 0.20
MIN_BOARD_SPAN_FRAC = 0.05
SIMILAR_VIEW_PX = 3.0
RANSAC_REPROJ_PX = 8.0
RANSAC_MIN_INLIER_FRAC = 0.50
RANSAC_MIN_INLIERS = 20
RANSAC_MIN_DEPTH_M = 0.02
FOCAL_SCALES = (0.9, 0.35, 0.5, 0.7)
MAX_ABS_DISTORTION = 8.0
MIN_FOCAL_PX = 40.0
MAX_FOCAL_PX = 2000.0
DEFAULT_NOTE = (
    "DFOPTIX A3 10x7 squares, OpenCV (9,6) inner, 0.030 m; "
    "head fisheye @ 544x448; head stereo only this pass (wrists not included)"
)


class FitError(RuntimeError):
    """Raised when capture data cannot be turned into a calibration."""


class FitRejected(FitError):
    """Raised when the numeric result is not plausible enough to keep."""


@dataclass
class DetectedPair:
    index: int
    left_path: Path
    right_path: Path
    object_points: np.ndarray
    left_points: np.ndarray
    right_points: np.ndarray
    image_size: tuple


@dataclass
class FitResult:
    n_usable: int
    n_train: int
    n_holdout: int
    image_size: tuple
    pattern: tuple
    square: float
    K_left: np.ndarray
    D_left: np.ndarray
    K_right: np.ndarray
    D_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    baseline_m: float
    left_rms: float
    right_rms: float
    stereo_rms: float
    holdout_left_rms: float
    holdout_right_rms: float
    holdout_stereo_rms: float
    created: str
    note: str
    dropped_train: int = 0
    errors: list = field(default_factory=list)


def object_points(pattern, square):
    cols, rows = int(pattern[0]), int(pattern[1])
    points = np.zeros((cols * rows, 1, 3), dtype=np.float64)
    points[:, 0, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square)
    return points


def _as_object_views(points_list):
    return [np.asarray(item, dtype=np.float64).reshape(-1, 1, 3) for item in points_list]


def _as_image_views(points_list):
    return [np.asarray(item, dtype=np.float64).reshape(-1, 1, 2) for item in points_list]


def project_so3(matrix):
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


def created_date(day_dir):
    for part in reversed(Path(day_dir).parts):
        if len(part) == 8 and part.isdigit():
            return f"{part[:4]}-{part[4:6]}-{part[6:8]}"
    return time.strftime("%Y-%m-%d")


def resolve_capture_root(day_dir):
    day_dir = Path(day_dir).expanduser().resolve()
    if not day_dir.exists():
        raise FitError(f"capture day directory does not exist: {day_dir}")
    for candidate in (day_dir / "captures", day_dir):
        if (candidate / "head_left").is_dir() and (candidate / "head_right").is_dir():
            return candidate
    raise FitError(
        f"no head_left/head_right directories under {day_dir}. "
        "Expected ~/r1-cam-calib/YYYYMMDD or .../YYYYMMDD/captures"
    )


def paired_paths(capture_root):
    left_dir = Path(capture_root) / "head_left"
    right_dir = Path(capture_root) / "head_right"
    left = {path.stem: path for path in sorted(left_dir.glob("*.png")) if path.stem.isdigit()}
    right = {path.stem: path for path in sorted(right_dir.glob("*.png")) if path.stem.isdigit()}
    keys = sorted(set(left) & set(right), key=lambda stem: int(stem))
    unpaired_left = sorted(set(left) - set(right), key=lambda stem: int(stem))
    unpaired_right = sorted(set(right) - set(left), key=lambda stem: int(stem))
    return [(left[key], right[key]) for key in keys], unpaired_left, unpaired_right


def holdout_count(n, holdout_n=HOLD_OUT_N, holdout_frac=HOLD_OUT_FRAC):
    n = int(n)
    if n <= 0:
        return 0
    return min(n, max(int(holdout_n), int(round(n * float(holdout_frac)))))


def holdout_split(pairs, holdout_n=HOLD_OUT_N, holdout_frac=HOLD_OUT_FRAC):
    pairs = list(pairs)
    n_hold = holdout_count(len(pairs), holdout_n=holdout_n, holdout_frac=holdout_frac)
    if n_hold == 0:
        return pairs, []
    return pairs[:-n_hold], pairs[-n_hold:]


def load_detected_pairs(capture_root, pattern=DEFAULT_PATTERN, square=DEFAULT_SQUARE_M,
                        detect=detect_corners, expected_size=HEAD_EYE_SIZE, log=print):
    paths, unpaired_left, unpaired_right = paired_paths(capture_root)
    if unpaired_left:
        log(f"warning: left-only images skipped: {', '.join(unpaired_left)}")
    if unpaired_right:
        log(f"warning: right-only images skipped: {', '.join(unpaired_right)}")
    obj = object_points(pattern, square)
    expected = inner_corner_count(pattern)
    pairs = []
    skipped = []
    for left_path, right_path in paths:
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            skipped.append((left_path.name, "unreadable"))
            continue
        if left.shape[1] != expected_size[0] or left.shape[0] != expected_size[1] \
                or right.shape[1] != expected_size[0] or right.shape[0] != expected_size[1]:
            skipped.append((left_path.name,
                            f"size {left.shape[1]}x{left.shape[0]}/{right.shape[1]}x{right.shape[0]} "
                            f"!= {expected_size[0]}x{expected_size[1]}"))
            continue
        ok_left, corners_left = detect(left, pattern)
        ok_right, corners_right = detect(right, pattern)
        if not ok_left or not ok_right:
            skipped.append((left_path.name,
                            f"detect L={ok_left} R={ok_right} (need {expected} corners each)"))
            continue
        pairs.append(DetectedPair(
            index=int(left_path.stem),
            left_path=left_path,
            right_path=right_path,
            object_points=obj.copy(),
            left_points=canonicalize_corners(corners_left, pattern),
            right_points=canonicalize_corners(corners_right, pattern),
            image_size=expected_size,
        ))
    if skipped:
        log(f"warning: skipped {len(skipped)} paired files that were not dual 54-corner detections")
        for name, reason in skipped[:12]:
            log(f"  {name}: {reason}")
        if len(skipped) > 12:
            log(f"  ... {len(skipped) - 12} more")
    return pairs


def canonicalize_corners(corners, pattern):
    """Force chessboard origin to the image-top-left-ish corner, row-major L→R.

    findChessboardCornersSB can start at any of the four board corners; the
    object-point grid is always row-major from (0,0). Canonicalizing both eyes
    keeps stereo correspondences consistent.
    """
    cols, rows = int(pattern[0]), int(pattern[1])
    pts = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    if pts[0, 0, 0] > pts[0, -1, 0]:
        pts = np.fliplr(pts)
    if pts[0, 0, 1] > pts[-1, 0, 1]:
        pts = np.flipud(pts)
    return np.ascontiguousarray(pts.reshape(-1, 1, 2))


def fisheye_flags(fix_k34=False):
    flags = (
        cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    if fix_k34:
        flags |= cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
    return flags


def _initial_K(image_size, focal_scale=0.9):
    """Non-identity fisheye guess: f ≈ scale * min(w, h), principal at centre."""
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 1 or height <= 1:
        raise FitError(f"invalid image_size {image_size}")
    focal = float(focal_scale) * float(min(width, height))
    if not np.isfinite(focal) or focal < MIN_FOCAL_PX:
        raise FitError(f"refusing degenerate focal length {focal}")
    return np.array(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _initial_D():
    return np.zeros((4, 1), dtype=np.float64)


def candidate_Ks(image_size):
    width, height = int(image_size[0]), int(image_size[1])
    focals = [float(scale) * float(min(width, height)) for scale in FOCAL_SCALES]
    focals.append(float(max(width, height)) / 2.0)
    unique = []
    seen = set()
    for focal in focals:
        key = round(focal, 3)
        if key in seen or focal < MIN_FOCAL_PX:
            continue
        seen.add(key)
        unique.append(_initial_K(image_size, focal_scale=focal / float(min(width, height))))
    return unique


def board_span_px(image_points):
    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    return float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))


def views_too_similar(points_a, points_b, thresh=SIMILAR_VIEW_PX):
    a = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    if a.shape != b.shape:
        return False
    return float(np.mean(np.linalg.norm(a - b, axis=1))) < float(thresh)


def pair_too_small(pair, min_frac=MIN_BOARD_SPAN_FRAC):
    width, height = pair.image_size
    min_w = float(min_frac) * float(width)
    min_h = float(min_frac) * float(height)
    left_x, left_y = board_span_px(pair.left_points)
    right_x, right_y = board_span_px(pair.right_points)
    return (
        left_x < min_w or left_y < min_h
        or right_x < min_w or right_y < min_h
    )


def view_pose_ok(object_points, image_points, K, min_inliers=None,
                 reprojection_error=RANSAC_REPROJ_PX):
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    if min_inliers is None:
        min_inliers = max(RANSAC_MIN_INLIERS, int(RANSAC_MIN_INLIER_FRAC * len(obj)))
    undistorted = cv2.fisheye.undistortPoints(img, K, _initial_D(), P=K)
    try:
        ok, _rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, undistorted.reshape(-1, 1, 2), K, None,
            flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=float(reprojection_error),
            confidence=0.99)
    except cv2.error:
        return False
    if not ok or tvec is None:
        return False
    if float(np.asarray(tvec, dtype=np.float64).reshape(3)[2]) <= RANSAC_MIN_DEPTH_M:
        return False
    n_inliers = 0 if inliers is None else int(len(inliers))
    return n_inliers >= int(min_inliers)


def view_pose_ok_any(object_points, image_points, Ks):
    """True if any seeded K yields a usable pose. Close/edge views often need this."""
    for K in Ks:
        if view_pose_ok(object_points, image_points, K):
            return True
    return False


def view_inits_ok(object_points, image_points, image_size, K):
    """True if OpenCV InitExtrinsics accepts this view with the seeded K."""
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    flags = (
        cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_FIX_K1 | cv2.fisheye.CALIB_FIX_K2
        | cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
        | cv2.fisheye.CALIB_FIX_FOCAL_LENGTH
        | cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
    )
    try:
        cv2.fisheye.calibrate(
            [obj], [img], tuple(image_size),
            np.asarray(K, dtype=np.float64).copy(), _initial_D(),
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1e-3))
        return True
    except cv2.error:
        return False


def select_train_pairs(pairs, K=None, log=print):
    kept = []
    dropped = []
    weak_ransac = []
    if not pairs:
        return kept, dropped
    image_size = pairs[0].image_size
    probe_Ks = candidate_Ks(image_size)
    for pair in pairs:
        if pair_too_small(pair):
            dropped.append((pair.index, "board span too small"))
            continue
        left_inits = any(
            view_inits_ok(pair.object_points, pair.left_points, pair.image_size, probe)
            for probe in probe_Ks)
        right_inits = any(
            view_inits_ok(pair.object_points, pair.right_points, pair.image_size, probe)
            for probe in probe_Ks)
        if not left_inits:
            dropped.append((pair.index, "left InitExtrinsics"))
            continue
        if not right_inits:
            dropped.append((pair.index, "right InitExtrinsics"))
            continue
        if kept and (
            views_too_similar(pair.left_points, kept[-1].left_points)
            and views_too_similar(pair.right_points, kept[-1].right_points)
        ):
            dropped.append((pair.index, "near-duplicate of previous kept view"))
            continue
        # RANSAC with D=0 is advisory: close/edge/tilt views often fail it even
        # when InitExtrinsics is fine. Do not drop those; they are the diverse poses.
        if not view_pose_ok_any(pair.object_points, pair.left_points, probe_Ks) \
                or not view_pose_ok_any(pair.object_points, pair.right_points, probe_Ks):
            weak_ransac.append(pair.index)
        kept.append(pair)
    if dropped:
        log(f"dropped {len(dropped)} train views before fisheye.calibrate")
        for index, reason in dropped[:12]:
            log(f"  pair {index:03d}: {reason}")
        if len(dropped) > 12:
            log(f"  ... {len(dropped) - 12} more")
    if weak_ransac:
        log(
            f"kept {len(weak_ransac)} InitExtrinsics-ok views with weak ransac "
            f"(close/edge diversity): {', '.join(f'{i:03d}' for i in weak_ransac[:12])}"
            + (" ..." if len(weak_ransac) > 12 else "")
        )
    if len(kept) < MIN_TRAIN_PAIRS:
        # InitExtrinsics/size/similar survivors if the hard filters left too few.
        relaxed = [
            pair for pair in pairs
            if not pair_too_small(pair)
            and any(
                view_inits_ok(pair.object_points, pair.left_points, pair.image_size, probe)
                and view_inits_ok(pair.object_points, pair.right_points, pair.image_size, probe)
                for probe in probe_Ks)
        ]
        deduped = []
        for pair in relaxed:
            if deduped and (
                views_too_similar(pair.left_points, deduped[-1].left_points)
                and views_too_similar(pair.right_points, deduped[-1].right_points)
            ):
                continue
            deduped.append(pair)
        if len(deduped) >= MIN_TRAIN_PAIRS:
            log(
                f"hard filters left only {len(kept)} views; "
                f"keeping {len(deduped)} after size/similar/InitExtrinsics only"
            )
            kept_ids = {pair.index for pair in deduped}
            relaxed_dropped = [
                (pair.index, "init-relaxed") for pair in pairs if pair.index not in kept_ids]
            return deduped, relaxed_dropped
        return kept, dropped
    return kept, dropped


def _plausible_intrinsics(rms, K, D):
    if rms is None or not np.isfinite(rms) or rms > 5.0:
        return False
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)):
        return False
    if not (MIN_FOCAL_PX <= float(K[0, 0]) <= MAX_FOCAL_PX):
        return False
    if not (MIN_FOCAL_PX <= float(K[1, 1]) <= MAX_FOCAL_PX):
        return False
    if float(np.max(np.abs(D))) > MAX_ABS_DISTORTION:
        return False
    return True


def _try_fisheye_calibrate(object_list, image_list, image_size, K, flags):
    obj = _as_object_views(object_list)
    img = _as_image_views(image_list)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    if abs(float(K[0, 0]) - 1.0) < 1e-9 and abs(float(K[1, 1]) - 1.0) < 1e-9:
        raise FitError("refusing identity camera matrix for fisheye.calibrate")
    if abs(float(K[0, 0])) < 1e-6 or abs(float(K[1, 1])) < 1e-6:
        raise FitError("refusing zero camera matrix for fisheye.calibrate")
    D = _initial_D()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
    rms, K_out, D_out, _rvecs, _tvecs = cv2.fisheye.calibrate(
        obj, img, tuple(image_size), K, D, flags=flags, criteria=criteria)
    return (
        float(rms),
        np.asarray(K_out, dtype=np.float64).reshape(3, 3),
        np.asarray(D_out, dtype=np.float64).reshape(4, 1),
    )


def pinhole_rational_diagnostic(object_list, image_list, image_size, log=print):
    obj = [np.asarray(item, dtype=np.float32).reshape(-1, 3) for item in object_list]
    img = [np.asarray(item, dtype=np.float32).reshape(-1, 1, 2) for item in image_list]
    K = _initial_K(image_size)
    dist = np.zeros((8, 1), dtype=np.float64)
    try:
        rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
            obj, img, tuple(image_size), K, dist,
            flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_INTRINSIC_GUESS)
    except cv2.error as error:
        log(f"pinhole+rational diagnostic also failed: {error}")
        return None
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    log(
        f"pinhole+rational diagnostic RMS={float(rms):.4f} px "
        f"fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}"
    )
    log(
        "head FOV likely needs fisheye after a better K / more edge+tilt poses; "
        "pinhole is diagnostic only and was not written"
    )
    return float(rms), K, np.asarray(dist, dtype=np.float64)


def _calibrate_with_flags(object_list, image_list, image_size, flags, log=print, label=""):
    last_error = None
    for K in candidate_Ks(image_size):
        try:
            rms, K_out, D_out = _try_fisheye_calibrate(
                object_list, image_list, image_size, K, flags)
        except cv2.error as error:
            last_error = error
            continue
        if _plausible_intrinsics(rms, K_out, D_out):
            log(
                f"fisheye.calibrate ok ({label}) n={len(object_list)} "
                f"RMS={rms:.4f} fx={K_out[0, 0]:.2f} fy={K_out[1, 1]:.2f} "
                f"init_fx={K[0, 0]:.2f}"
            )
            return rms, K_out, D_out
        last_error = FitError(
            f"fisheye.calibrate returned implausible K/D (RMS={rms:.3f} "
            f"fx={float(np.asarray(K_out)[0, 0]):.1f})"
        )
    if last_error is None:
        last_error = FitError("fisheye.calibrate produced no result")
    raise last_error


def _calibrate_subset(object_list, image_list, image_size, log=print):
    last_error = None
    for fix_k34, label in ((False, "full k"), (True, "FIX_K3|FIX_K4")):
        try:
            return _calibrate_with_flags(
                object_list, image_list, image_size,
                fisheye_flags(fix_k34=fix_k34), log=log, label=label)
        except (cv2.error, FitError) as error:
            last_error = error
    if last_error is None:
        raise FitError("fisheye.calibrate produced no result")
    raise last_error


def calibrate_fisheye_mono(object_list, image_list, image_size, log=print):
    obj = _as_object_views(object_list)
    img = _as_image_views(image_list)
    if not obj:
        raise FitError("no views for fisheye.calibrate")
    try:
        return _calibrate_subset(obj, img, image_size, log=log)
    except (cv2.error, FitError) as error:
        order = sorted(
            range(len(img)),
            key=lambda i: board_span_px(img[i])[0] * board_span_px(img[i])[1],
            reverse=True,
        )
        last_good = None
        kept_n = 0
        silent = lambda *_args, **_kwargs: None
        for count in range(len(order), MIN_TRAIN_PAIRS - 1, -1):
            chosen = order[:count]
            try:
                last_good = _calibrate_subset(
                    [obj[i] for i in chosen], [img[i] for i in chosen],
                    image_size, log=silent)
                kept_n = count
                break
            except (cv2.error, FitError):
                log(f"InitExtrinsics on {count} views; dropping the smallest remaining board")
        if last_good is None:
            pinhole_rational_diagnostic(obj, img, image_size, log=log)
            detail = error if isinstance(error, FitError) else f"cv2.fisheye.calibrate failed: {error}"
            raise FitError(
                f"{detail}; could not find a {MIN_TRAIN_PAIRS}+ view subset that inits. "
                "Recapture with the board closer and at the image edges/tilts."
            ) from error
        log(f"fisheye.calibrate recovered by dropping small views; kept {kept_n}/{len(obj)}")
        return last_good


def calibrate_fisheye_stereo(object_list, left_list, right_list, image_size, K_left, D_left, K_right, D_right):
    obj = _as_object_views(object_list)
    left = _as_image_views(left_list)
    right = _as_image_views(right_list)
    K1 = np.asarray(K_left, dtype=np.float64).copy()
    D1 = np.asarray(D_left, dtype=np.float64).reshape(4, 1).copy()
    K2 = np.asarray(K_right, dtype=np.float64).copy()
    D2 = np.asarray(D_right, dtype=np.float64).reshape(4, 1).copy()
    R = np.eye(3, dtype=np.float64)
    T = np.zeros((3, 1), dtype=np.float64)
    flags = cv2.fisheye.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
    try:
        unpacked = cv2.fisheye.stereoCalibrate(
            obj, left, right, K1, D1, K2, D2, tuple(image_size), R, T,
            flags=flags, criteria=criteria)
        rms, K1, D1, K2, D2, R, T = unpacked[:7]
    except cv2.error as error:
        raise FitError(f"cv2.fisheye.stereoCalibrate failed: {error}") from error
    R = project_so3(R)
    T = np.asarray(T, dtype=np.float64).reshape(3)
    baseline = float(np.linalg.norm(T))
    return (float(rms), K1, np.asarray(D1, dtype=np.float64).reshape(4, 1),
            K2, np.asarray(D2, dtype=np.float64).reshape(4, 1), R, T, baseline)


def fisheye_reprojection_rms(object_list, image_list, K, D):
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(4, 1)
    squares = []
    for obj, img in zip(object_list, image_list):
        obj = np.asarray(obj, dtype=np.float64).reshape(-1, 1, 3)
        img = np.asarray(img, dtype=np.float64).reshape(-1, 1, 2)
        undistorted = cv2.fisheye.undistortPoints(img, K, D, P=K)
        ok, rvec, tvec = cv2.solvePnP(
            obj.reshape(-1, 3), undistorted.reshape(-1, 2), K, None,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise FitError("solvePnP failed while scoring reprojection")
        projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
        delta = projected.reshape(-1, 2) - img.reshape(-1, 2)
        squares.append(np.sum(delta * delta, axis=1))
    stacked = np.concatenate(squares)
    return float(np.sqrt(np.mean(stacked)))


def stereo_reprojection_rms(object_list, left_list, right_list, K_left, D_left, K_right, D_right, R, T):
    K_left = np.asarray(K_left, dtype=np.float64).reshape(3, 3)
    D_left = np.asarray(D_left, dtype=np.float64).reshape(4, 1)
    K_right = np.asarray(K_right, dtype=np.float64).reshape(3, 3)
    D_right = np.asarray(D_right, dtype=np.float64).reshape(4, 1)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    squares = []
    for obj, left, right in zip(object_list, left_list, right_list):
        obj = np.asarray(obj, dtype=np.float64).reshape(-1, 1, 3)
        left = np.asarray(left, dtype=np.float64).reshape(-1, 1, 2)
        right = np.asarray(right, dtype=np.float64).reshape(-1, 1, 2)
        undistorted = cv2.fisheye.undistortPoints(left, K_left, D_left, P=K_left)
        ok, rvec, tvec = cv2.solvePnP(
            obj.reshape(-1, 3), undistorted.reshape(-1, 2), K_left, None,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise FitError("solvePnP failed while scoring stereo holdout")
        projected_left, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K_left, D_left)
        R_left, _ = cv2.Rodrigues(rvec)
        R_right = R @ R_left
        t_right = (R @ np.asarray(tvec, dtype=np.float64).reshape(3, 1) + T)
        rvec_right, _ = cv2.Rodrigues(R_right)
        projected_right, _ = cv2.fisheye.projectPoints(obj, rvec_right, t_right, K_right, D_right)
        delta_left = projected_left.reshape(-1, 2) - left.reshape(-1, 2)
        delta_right = projected_right.reshape(-1, 2) - right.reshape(-1, 2)
        squares.append(np.sum(delta_left * delta_left, axis=1))
        squares.append(np.sum(delta_right * delta_right, axis=1))
    stacked = np.concatenate(squares)
    return float(np.sqrt(np.mean(stacked)))


def _safe_holdout_rms(holdout, K_left, D_left, K_right, D_right, R, T, log=print):
    if not holdout:
        return float("nan"), float("nan"), float("nan")
    used = []
    for pair in holdout:
        try:
            fisheye_reprojection_rms([pair.object_points], [pair.left_points], K_left, D_left)
            fisheye_reprojection_rms([pair.object_points], [pair.right_points], K_right, D_right)
            stereo_reprojection_rms(
                [pair.object_points], [pair.left_points], [pair.right_points],
                K_left, D_left, K_right, D_right, R, T)
        except (cv2.error, FitError):
            log(f"holdout pair {pair.index:03d} skipped (could not score)")
            continue
        used.append(pair)
    if not used:
        log("warning: no holdout pair could be scored")
        return float("nan"), float("nan"), float("nan")
    hold_obj = [pair.object_points for pair in used]
    hold_left = [pair.left_points for pair in used]
    hold_right = [pair.right_points for pair in used]
    return (
        fisheye_reprojection_rms(hold_obj, hold_left, K_left, D_left),
        fisheye_reprojection_rms(hold_obj, hold_right, K_right, D_right),
        stereo_reprojection_rms(
            hold_obj, hold_left, hold_right, K_left, D_left, K_right, D_right, R, T),
    )


def fit_detected_pairs(train, holdout, pattern=DEFAULT_PATTERN, square=DEFAULT_SQUARE_M,
                       created=None, note=DEFAULT_NOTE, log=print):
    if not train:
        raise FitError("no training pairs")
    image_size = train[0].image_size
    pattern = (int(pattern[0]), int(pattern[1]))
    square = float(square)
    n_usable = len(train) + len(holdout)
    seed_K = _initial_K(image_size)
    log(
        f"fisheye init K fx={seed_K[0, 0]:.2f} fy={seed_K[1, 1]:.2f} "
        f"cx={seed_K[0, 2]:.2f} cy={seed_K[1, 2]:.2f} (never identity)"
    )
    filtered, dropped = select_train_pairs(train, log=log)
    if len(filtered) < MIN_TRAIN_PAIRS:
        raise FitError(
            f"only {len(filtered)} train views remain after dropping tiny/similar/"
            f"InitExtrinsics failures (need >= {MIN_TRAIN_PAIRS})"
        )
    obj = [pair.object_points for pair in filtered]
    left = [pair.left_points for pair in filtered]
    right = [pair.right_points for pair in filtered]
    last_error = None
    calibrated = None
    for fix_k34, label in ((False, "full k"), (True, "FIX_K3|FIX_K4")):
        flags = fisheye_flags(fix_k34=fix_k34)
        try:
            left_rms, K_left, D_left = _calibrate_with_flags(
                obj, left, image_size, flags, log=log, label=f"left {label}")
            right_rms, K_right, D_right = _calibrate_with_flags(
                obj, right, image_size, flags, log=log, label=f"right {label}")
            calibrated = (left_rms, K_left, D_left, right_rms, K_right, D_right)
            break
        except (cv2.error, FitError) as error:
            last_error = error
            log(f"{label} failed on one eye; trying the next flag set")
    if calibrated is None:
        pinhole_rational_diagnostic(obj, left, image_size, log=log)
        if last_error is None:
            last_error = FitError("fisheye.calibrate failed for both eyes")
        raise last_error if isinstance(last_error, FitError) else FitError(
            f"cv2.fisheye.calibrate failed: {last_error}") from last_error
    left_rms, K_left, D_left, right_rms, K_right, D_right = calibrated
    stereo_rms, K_left, D_left, K_right, D_right, R, T, baseline = calibrate_fisheye_stereo(
        obj, left, right, image_size, K_left, D_left, K_right, D_right)
    holdout_left_rms, holdout_right_rms, holdout_stereo_rms = _safe_holdout_rms(
        holdout, K_left, D_left, K_right, D_right, R, T, log=log)
    return FitResult(
        n_usable=n_usable,
        n_train=len(filtered),
        n_holdout=len(holdout),
        dropped_train=len(dropped),
        image_size=tuple(image_size),
        pattern=pattern,
        square=square,
        K_left=K_left,
        D_left=D_left,
        K_right=K_right,
        D_right=D_right,
        R=R,
        T=T,
        baseline_m=baseline,
        left_rms=left_rms,
        right_rms=right_rms,
        stereo_rms=stereo_rms,
        holdout_left_rms=holdout_left_rms,
        holdout_right_rms=holdout_right_rms,
        holdout_stereo_rms=holdout_stereo_rms,
        created=created or time.strftime("%Y-%m-%d"),
        note=note,
    )


def quality_errors(result, min_train=MIN_TRAIN_PAIRS, max_rms=MAX_RMS_PX,
                   baseline_min=BASELINE_MIN_M, baseline_max=BASELINE_MAX_M,
                   expected_size=HEAD_EYE_SIZE):
    errors = []
    if result.n_train < min_train:
        errors.append(
            f"not enough training pairs: {result.n_train} (need >= {min_train} after holdout; "
            "capture >=40 PAIR OK so 30 remain after holding out 10)"
        )
    if tuple(result.image_size) != tuple(expected_size):
        errors.append(
            f"image_size {result.image_size[0]}x{result.image_size[1]} is not "
            f"{expected_size[0]}x{expected_size[1]}"
        )
    for name, rms in (
        ("left train RMS", result.left_rms),
        ("right train RMS", result.right_rms),
        ("stereo train RMS", result.stereo_rms),
        ("left holdout RMS", result.holdout_left_rms),
        ("right holdout RMS", result.holdout_right_rms),
        ("stereo holdout RMS", result.holdout_stereo_rms),
    ):
        if rms is None or not np.isfinite(rms):
            continue
        if rms > max_rms:
            errors.append(f"{name} {rms:.3f} px exceeds {max_rms:.1f} px")
    baseline = float(result.baseline_m)
    if not (baseline_min <= baseline <= baseline_max):
        errors.append(
            f"baseline {baseline:.4f} m is outside {baseline_min:.2f}–{baseline_max:.2f} m "
            f"(expected ~{EXPECTED_BASELINE_M:.3f} m)"
        )
    return errors


def _matrix3(value):
    return np.asarray(value, dtype=float).reshape(3, 3).tolist()


def _vector(value):
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def camera_entry(image_size, K, D, rms, model="fisheye"):
    return {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "camera_matrix": _matrix3(K),
        "distortion_model": model,
        "distortion_coefficients": _vector(D),
        "reprojection_error_px": float(rms),
    }


def build_document(result):
    return {
        "schema": CALIBRATION_SCHEMA,
        "created": result.created,
        "note": result.note,
        "board": {
            "type": "checkerboard",
            "squares_x": int(result.pattern[0]),
            "squares_y": int(result.pattern[1]),
            "square_size_m": float(result.square),
        },
        "cameras": {
            "head_left": camera_entry(result.image_size, result.K_left, result.D_left, result.left_rms),
            "head_right": camera_entry(result.image_size, result.K_right, result.D_right, result.right_rms),
        },
        "stereo": {
            "head": {
                "left": "head_left",
                "right": "head_right",
                "rotation": _matrix3(result.R),
                "translation_m": _vector(result.T),
                "baseline_m": float(result.baseline_m),
                "epipolar_error_px": float(result.stereo_rms),
            }
        },
    }


def format_report(result, output_path=None, schema_ok=None, errors=None):
    def fmt(rms):
        return "n/a" if rms is None or not np.isfinite(rms) else f"{rms:.4f} px"

    def k_line(name, K, D):
        K = np.asarray(K, dtype=float).reshape(3, 3)
        D = np.asarray(D, dtype=float).reshape(-1)
        return (f"{name} fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} "
                f"cx={K[0, 2]:.2f} cy={K[1, 2]:.2f} D={_vector(D)}")

    lines = [
        "=== R1 head stereo fisheye fit ===",
        f"usable pairs: {result.n_usable}",
        f"train: {result.n_train}",
        f"holdout: {result.n_holdout} (last 10 or 20%)",
        f"dropped train views: {getattr(result, 'dropped_train', 0)}",
        f"image_size: {result.image_size[0]}x{result.image_size[1]}",
        f"board: inner {result.pattern[0]}x{result.pattern[1]}, square={result.square:.3f} m",
        k_line("left ", result.K_left, result.D_left),
        k_line("right", result.K_right, result.D_right),
        f"left  RMS (train): {fmt(result.left_rms)}",
        f"right RMS (train): {fmt(result.right_rms)}",
        f"stereo RMS (train): {fmt(result.stereo_rms)}",
        f"baseline: {result.baseline_m:.4f} m  (expected ~{EXPECTED_BASELINE_M:.3f} m)",
        f"T: {_vector(result.T)}",
        f"left  RMS (holdout): {fmt(result.holdout_left_rms)}",
        f"right RMS (holdout): {fmt(result.holdout_right_rms)}",
        f"stereo RMS (holdout): {fmt(result.holdout_stereo_rms)}",
    ]
    if output_path is not None:
        lines.append(f"wrote: {output_path}")
    if schema_ok is not None:
        lines.append(f"load_camera_calibration: {'OK' if schema_ok else 'FAILED'}")
    lines.append("status if installed now: partial (wrists not in this file)")
    lines.append("NOT copied to assets/r1/camera_calibration.json")
    if errors:
        lines.append("REJECTED:")
        lines.extend(f"  - {item}" for item in errors)
    return "\n".join(lines)


def atomic_write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    payload = json.dumps(document, indent=2) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    return path


def write_calibration(result, output_path):
    document = build_document(result)
    extra = set(document) - {"schema", "created", "note", "board", "cameras", "stereo", "hand_eye"}
    if extra:
        raise FitError(f"refusing to write unknown top-level keys: {sorted(extra)}")
    atomic_write_json(output_path, document)
    loaded = load_camera_calibration(output_path)
    return loaded


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit R1 head-stereo fisheye calibration. Writes JSON under the day dir, not assets/.")
    parser.add_argument("--day-dir", type=Path, required=True,
                        help="Capture day directory, e.g. ~/r1-cam-calib/20260920")
    parser.add_argument("--pattern", nargs=2, type=int, default=list(DEFAULT_PATTERN), metavar=("COLS", "ROWS"))
    parser.add_argument("--square", type=float, default=DEFAULT_SQUARE_M)
    parser.add_argument("--holdout", type=int, default=HOLD_OUT_N)
    parser.add_argument("--holdout-frac", type=float, default=HOLD_OUT_FRAC)
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN_PAIRS)
    parser.add_argument("--max-rms", type=float, default=MAX_RMS_PX)
    parser.add_argument("--baseline-min", type=float, default=BASELINE_MIN_M)
    parser.add_argument("--baseline-max", type=float, default=BASELINE_MAX_M)
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="JSON path (default: <day-dir>/camera_calibration.json)")
    args = parser.parse_args(argv)
    if args.pattern[0] <= 0 or args.pattern[1] <= 0:
        parser.error("--pattern must be two positive inner-corner counts")
    if args.square <= 0:
        parser.error("--square must be a positive length in metres")
    return args


def main(argv=None):
    args = parse_args(argv)
    day_dir = args.day_dir.expanduser().resolve()
    capture_root = resolve_capture_root(day_dir)
    output_path = args.output.expanduser() if args.output else day_dir / "camera_calibration.json"
    if args.output is None and capture_root.name == "captures":
        output_path = capture_root.parent / "camera_calibration.json"
    report_path = output_path.with_name("fit_report.txt")
    pairs = load_detected_pairs(
        capture_root, pattern=tuple(args.pattern), square=args.square)
    if not pairs:
        message = (f"no usable left/right pairs with {inner_corner_count(args.pattern)} corners "
                   f"under {capture_root}")
        print(message, file=sys.stderr)
        return 1
    train, holdout = holdout_split(pairs, holdout_n=args.holdout, holdout_frac=args.holdout_frac)
    print(f"loaded {len(pairs)} usable pairs from {capture_root}")
    print(f"train {len(train)}, holdout {len(holdout)} (last {len(holdout)} by filename index)")
    if len(train) < args.min_train:
        errors = [
            f"not enough training pairs: {len(train)} (need >= {args.min_train} after holdout; "
            "capture >=40 PAIR OK so 30 remain after holding out 10)"
        ]
        print("REJECTED:")
        for item in errors:
            print(f"  - {item}")
        report_path.write_text("REJECTED\n" + "\n".join(errors) + "\n", encoding="utf-8")
        print(f"wrote: {report_path}")
        print("did not write camera_calibration.json")
        return 2
    try:
        result = fit_detected_pairs(
            train, holdout, pattern=tuple(args.pattern), square=args.square,
            created=created_date(day_dir), note=DEFAULT_NOTE, log=print)
    except FitError as error:
        print(f"fit failed: {error}", file=sys.stderr)
        return 1
    errors = quality_errors(
        result, min_train=args.min_train, max_rms=args.max_rms,
        baseline_min=args.baseline_min, baseline_max=args.baseline_max)
    result.errors = errors
    if errors:
        report = format_report(result, output_path=None, schema_ok=None, errors=errors)
        print(report)
        report_path.write_text(report + "\n", encoding="utf-8")
        print(f"wrote: {report_path}")
        print("did not write camera_calibration.json")
        return 2
    loaded = write_calibration(result, output_path)
    metadata = camera_calibration_metadata(
        loaded, ["head_left", "head_right", "left_wrist", "right_wrist"])
    report = format_report(result, output_path=output_path, schema_ok=True, errors=None)
    report += f"\nmetadata status vs four-camera run: {metadata['status']}"
    print(report)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"fit_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
