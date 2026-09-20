#!/usr/bin/env python3
"""Fit plumb_bob wrist intrinsics from captured chessboard images.

Reads ``left_wrist`` / ``right_wrist`` PNGs, holds out the last 10 (or 20%),
runs cv2.calibrateCamera, and merges into ``camera_calibration.json`` under
the capture day directory.

The merge base is the existing day-dir JSON if present, otherwise
assets/r1/camera_calibration.json, so head stereo is preserved. Does not
write assets/. Does not start teleoperation. Exits non-zero if a requested
side is not physically plausible.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
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

from calibrate_r1_wrist_capture import (  # noqa: E402
    DEFAULT_PATTERN,
    DEFAULT_SQUARE_M,
    SIDES,
    WRIST_FRAME_SIZE,
    detect_corners,
    inner_corner_count,
    side_spec,
)
from teleop.utils.camera_calibration import (  # noqa: E402
    CALIBRATION_SCHEMA,
    camera_calibration_metadata,
    load_camera_calibration,
)


MIN_TRAIN_IMAGES = 20
MAX_RMS_PX = 1.0
HOLD_OUT_N = 10
HOLD_OUT_FRAC = 0.20
MIN_BOARD_SPAN_FRAC = 0.05
SIMILAR_VIEW_PX = 3.0
MIN_FOCAL_PX = 80.0
MAX_FOCAL_PX = 2000.0
MAX_ABS_DISTORTION = 8.0
HEAD_CAMERAS = ("head_left", "head_right")
DEFAULT_ASSETS = Path("assets/r1/camera_calibration.json")
DEFAULT_NOTE = (
    "DFOPTIX A3 10x7 squares, OpenCV (9,6) inner, 0.030 m; "
    "wrist plumb_bob @ 640x480; head stereo preserved"
)


class FitError(RuntimeError):
    """Raised when capture data cannot be turned into a calibration."""


class FitRejected(FitError):
    """Raised when the numeric result is not plausible enough to keep."""


@dataclass
class DetectedView:
    index: int
    path: Path
    object_points: np.ndarray
    image_points: np.ndarray
    image_size: tuple


@dataclass
class FitResult:
    side: str
    camera: str
    n_usable: int
    n_train: int
    n_holdout: int
    image_size: tuple
    pattern: tuple
    square: float
    K: np.ndarray
    D: np.ndarray
    rms: float
    holdout_rms: float
    created: str
    note: str
    dropped_train: int = 0
    errors: list = field(default_factory=list)


def object_points(pattern, square):
    cols, rows = int(pattern[0]), int(pattern[1])
    points = np.zeros((cols * rows, 1, 3), dtype=np.float64)
    points[:, 0, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square)
    return points


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
        if (candidate / "left_wrist").is_dir() or (candidate / "right_wrist").is_dir():
            return candidate
    raise FitError(
        f"no left_wrist/right_wrist directories under {day_dir}. "
        "Expected ~/r1-cam-calib/YYYYMMDD or .../YYYYMMDD/captures"
    )


def camera_folder(capture_root, side):
    return Path(capture_root) / side_spec(side)["camera"]


def holdout_count(n, holdout_n=HOLD_OUT_N, holdout_frac=HOLD_OUT_FRAC):
    n = int(n)
    if n <= 0:
        return 0
    return min(n, max(int(holdout_n), int(round(n * float(holdout_frac)))))


def holdout_split(views, holdout_n=HOLD_OUT_N, holdout_frac=HOLD_OUT_FRAC):
    views = list(views)
    n_hold = holdout_count(len(views), holdout_n=holdout_n, holdout_frac=holdout_frac)
    if n_hold == 0:
        return views, []
    return views[:-n_hold], views[-n_hold:]


def canonicalize_corners(corners, pattern):
    """Force chessboard origin to the image-top-left-ish corner, row-major L→R."""
    cols, rows = int(pattern[0]), int(pattern[1])
    pts = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    if pts[0, 0, 0] > pts[0, -1, 0]:
        pts = np.fliplr(pts)
    if pts[0, 0, 1] > pts[-1, 0, 1]:
        pts = np.flipud(pts)
    return np.ascontiguousarray(pts.reshape(-1, 1, 2))


def load_detected_views(folder, pattern=DEFAULT_PATTERN, square=DEFAULT_SQUARE_M,
                        detect=None, expected_size=WRIST_FRAME_SIZE, log=print):
    if detect is None:
        detect = detect_corners
    folder = Path(folder)
    paths = sorted(
        [path for path in folder.glob("*.png") if path.stem.isdigit()],
        key=lambda path: int(path.stem),
    )
    obj = object_points(pattern, square)
    expected = inner_corner_count(pattern)
    views = []
    skipped = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append((path.name, "unreadable"))
            continue
        height, width = image.shape[:2]
        if width != expected_size[0] or height != expected_size[1]:
            skipped.append((path.name, f"size {width}x{height} != {expected_size[0]}x{expected_size[1]}"))
            continue
        ok, corners = detect(image, pattern)
        if not ok:
            skipped.append((path.name, f"detect failed (need {expected} corners)"))
            continue
        views.append(DetectedView(
            index=int(path.stem),
            path=path,
            object_points=obj.copy(),
            image_points=canonicalize_corners(corners, pattern),
            image_size=expected_size,
        ))
    if skipped:
        log(f"warning: skipped {len(skipped)} files that were not 54-corner {expected_size[0]}x{expected_size[1]} detections")
        for name, reason in skipped[:12]:
            log(f"  {name}: {reason}")
        if len(skipped) > 12:
            log(f"  ... {len(skipped) - 12} more")
    return views


def _initial_K(image_size, focal_scale=0.9):
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


def board_span_px(image_points):
    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    return float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))


def views_too_similar(points_a, points_b, thresh=SIMILAR_VIEW_PX):
    a = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    if a.shape != b.shape:
        return False
    return float(np.mean(np.linalg.norm(a - b, axis=1))) < float(thresh)


def view_too_small(view, min_frac=MIN_BOARD_SPAN_FRAC):
    width, height = view.image_size
    span_x, span_y = board_span_px(view.image_points)
    return span_x < float(min_frac) * float(width) or span_y < float(min_frac) * float(height)


def select_train_views(views, log=print):
    kept = []
    dropped = []
    for view in views:
        if view_too_small(view):
            dropped.append((view.index, "board span too small"))
            continue
        if kept and views_too_similar(view.image_points, kept[-1].image_points):
            dropped.append((view.index, "near-duplicate of previous kept view"))
            continue
        kept.append(view)
    if dropped:
        log(f"dropped {len(dropped)} train views before calibrateCamera")
        for index, reason in dropped[:12]:
            log(f"  image {index:03d}: {reason}")
        if len(dropped) > 12:
            log(f"  ... {len(dropped) - 12} more")
    return kept, dropped


def pinhole_flags():
    return cv2.CALIB_USE_INTRINSIC_GUESS


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


def calibrate_pinhole_mono(object_list, image_list, image_size, log=print, label=""):
    obj = [np.asarray(item, dtype=np.float32).reshape(-1, 3) for item in object_list]
    img = [np.asarray(item, dtype=np.float32).reshape(-1, 1, 2) for item in image_list]
    if not obj:
        raise FitError("no views for calibrateCamera")
    K = _initial_K(image_size)
    dist = np.zeros((5, 1), dtype=np.float64)
    flags = pinhole_flags()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
    try:
        rms, K_out, D_out, _rvecs, _tvecs = cv2.calibrateCamera(
            obj, img, tuple(image_size), K, dist, flags=flags, criteria=criteria)
    except cv2.error as error:
        raise FitError(f"cv2.calibrateCamera failed: {error}") from error
    rms = float(rms)
    K_out = np.asarray(K_out, dtype=np.float64).reshape(3, 3)
    D_out = np.asarray(D_out, dtype=np.float64).reshape(-1)
    if D_out.size < 5:
        raise FitError(f"expected plumb_bob (5) distortion coefficients, got {D_out.size}")
    D_out = D_out[:5].reshape(5, 1)
    if not _plausible_intrinsics(rms, K_out, D_out):
        raise FitError(
            f"calibrateCamera returned implausible K/D (RMS={rms:.3f} fx={float(K_out[0, 0]):.1f})"
        )
    log(
        f"calibrateCamera ok ({label}) n={len(obj)} "
        f"RMS={rms:.4f} fx={K_out[0, 0]:.2f} fy={K_out[1, 1]:.2f}"
    )
    return rms, K_out, D_out


def pinhole_reprojection_rms(object_list, image_list, K, D):
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    squares = []
    for obj, img in zip(object_list, image_list):
        obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
        img = np.asarray(img, dtype=np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise FitError("solvePnP failed while scoring reprojection")
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        delta = projected.reshape(-1, 2) - img.reshape(-1, 2)
        squares.append(np.sum(delta * delta, axis=1))
    stacked = np.concatenate(squares)
    return float(np.sqrt(np.mean(stacked)))


def _safe_holdout_rms(holdout, K, D, log=print):
    if not holdout:
        return float("nan")
    used = []
    for view in holdout:
        try:
            pinhole_reprojection_rms([view.object_points], [view.image_points], K, D)
        except (cv2.error, FitError):
            log(f"holdout image {view.index:03d} skipped (could not score)")
            continue
        used.append(view)
    if not used:
        log("warning: no holdout image could be scored")
        return float("nan")
    return pinhole_reprojection_rms(
        [view.object_points for view in used],
        [view.image_points for view in used],
        K, D,
    )


def fit_detected_views(train, holdout, side, pattern=DEFAULT_PATTERN, square=DEFAULT_SQUARE_M,
                       created=None, note=DEFAULT_NOTE, log=print):
    if not train:
        raise FitError("no training images")
    image_size = train[0].image_size
    pattern = (int(pattern[0]), int(pattern[1]))
    square = float(square)
    n_usable = len(train) + len(holdout)
    seed_K = _initial_K(image_size)
    log(
        f"{side} pinhole init K fx={seed_K[0, 0]:.2f} fy={seed_K[1, 1]:.2f} "
        f"cx={seed_K[0, 2]:.2f} cy={seed_K[1, 2]:.2f} (never identity)"
    )
    filtered, dropped = select_train_views(train, log=log)
    if len(filtered) < MIN_TRAIN_IMAGES:
        raise FitError(
            f"only {len(filtered)} train views remain after dropping tiny/similar "
            f"boards (need >= {MIN_TRAIN_IMAGES})"
        )
    obj = [view.object_points for view in filtered]
    img = [view.image_points for view in filtered]
    rms, K, D = calibrate_pinhole_mono(obj, img, image_size, log=log, label=side)
    holdout_rms = _safe_holdout_rms(holdout, K, D, log=log)
    spec = side_spec(side)
    return FitResult(
        side=side,
        camera=spec["camera"],
        n_usable=n_usable,
        n_train=len(filtered),
        n_holdout=len(holdout),
        dropped_train=len(dropped),
        image_size=tuple(image_size),
        pattern=pattern,
        square=square,
        K=K,
        D=D,
        rms=rms,
        holdout_rms=holdout_rms,
        created=created or time.strftime("%Y-%m-%d"),
        note=note,
    )


def quality_errors(result, min_train=MIN_TRAIN_IMAGES, max_rms=MAX_RMS_PX,
                   expected_size=WRIST_FRAME_SIZE):
    errors = []
    if result.n_train < min_train:
        errors.append(
            f"not enough training images: {result.n_train} (need >= {min_train} after holdout; "
            "capture >=40 OK so 30 remain after holding out 10)"
        )
    if tuple(result.image_size) != tuple(expected_size):
        errors.append(
            f"image_size {list(result.image_size)} is not {list(expected_size)}"
        )
    for name, rms in (("train RMS", result.rms), ("holdout RMS", result.holdout_rms)):
        if rms is None or not np.isfinite(rms):
            continue
        if rms > max_rms:
            errors.append(f"{name} {rms:.3f} px exceeds {max_rms:.1f} px")
    K = np.asarray(result.K, dtype=float).reshape(3, 3)
    if not (MIN_FOCAL_PX <= float(K[0, 0]) <= MAX_FOCAL_PX):
        errors.append(f"fx {float(K[0, 0]):.1f} is not plausible for 640x480")
    return errors


def _matrix3(value):
    return np.asarray(value, dtype=float).reshape(3, 3).tolist()


def _vector(value):
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def camera_entry(image_size, K, D, rms, model="plumb_bob"):
    return {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "camera_matrix": _matrix3(K),
        "distortion_model": model,
        "distortion_coefficients": _vector(D),
        "reprojection_error_px": float(rms),
    }


def is_assets_calibration_path(path):
    path = Path(path).expanduser().resolve()
    return path.name == "camera_calibration.json" and path.parent.name == "r1" and path.parent.parent.name == "assets"


def refuse_assets_path(path):
    if is_assets_calibration_path(path):
        raise FitError(
            f"refusing to write {path}; wrist merge updates the day-dir copy only, not assets/"
        )


def validate_head_base(document, source):
    if not isinstance(document, dict):
        raise FitError(f"{source}: expected a JSON object")
    if document.get("schema") != CALIBRATION_SCHEMA:
        raise FitError(f"{source}: schema must be {CALIBRATION_SCHEMA!r}")
    cameras = document.get("cameras")
    if not isinstance(cameras, dict):
        raise FitError(f"{source}: cameras must be an object")
    missing = [name for name in HEAD_CAMERAS if name not in cameras]
    if missing:
        raise FitError(
            f"{source}: missing {', '.join(missing)}; refusing a merge that would drop head stereo"
        )
    stereo = document.get("stereo")
    if not isinstance(stereo, dict) or "head" not in stereo:
        raise FitError(f"{source}: missing stereo.head; refusing a merge that would drop head stereo")
    extra = set(document) - {"schema", "created", "note", "board", "cameras", "stereo", "hand_eye"}
    if extra:
        raise FitError(f"{source}: unknown top-level keys: {', '.join(sorted(extra))}")
    return document


def resolve_base_document(day_dir, assets=None):
    day_json = Path(day_dir).expanduser().resolve() / "camera_calibration.json"
    assets_json = Path(assets).expanduser() if assets else (REPO / DEFAULT_ASSETS)
    if not assets_json.is_absolute():
        assets_json = (REPO / assets_json).resolve()
    else:
        assets_json = assets_json.resolve()
    if day_json.is_file():
        document = json.loads(day_json.read_text(encoding="utf-8"))
        validate_head_base(document, day_json)
        return document, day_json
    if assets_json.is_file():
        document = json.loads(assets_json.read_text(encoding="utf-8"))
        validate_head_base(document, assets_json)
        return document, assets_json
    raise FitError(
        "no base camera_calibration.json with head stereo. "
        f"Looked at {day_json} then {assets_json}. "
        "Will not write a wrist-only file that drops head_left/head_right."
    )


def compose_note(base_note, fitted_cameras):
    names = ", ".join(sorted(fitted_cameras))
    wrist_bit = f"{names} plumb_bob @ 640x480"
    if not base_note:
        return (
            "DFOPTIX A3 10x7 squares, OpenCV (9,6) inner, 0.030 m; "
            f"head stereo preserved; {wrist_bit}"
        )
    note = str(base_note)
    note = note.replace(
        "head stereo only this pass (wrists not included)",
        f"head stereo preserved; {wrist_bit}",
    )
    if wrist_bit not in note:
        note = note.rstrip(".") + f"; {wrist_bit}"
    return note


def merge_wrist_cameras(base_document, results, created=None):
    document = copy.deepcopy(base_document)
    validate_head_base(document, "base")
    head_left = copy.deepcopy(document["cameras"]["head_left"])
    head_right = copy.deepcopy(document["cameras"]["head_right"])
    stereo_head = copy.deepcopy(document["stereo"]["head"])
    fitted = {}
    for result in results:
        if result.camera in HEAD_CAMERAS:
            raise FitError(f"refusing to overwrite {result.camera}")
        fitted[result.camera] = camera_entry(
            result.image_size, result.K, result.D, result.rms, model="plumb_bob")
        document["cameras"][result.camera] = fitted[result.camera]
    if document["cameras"]["head_left"] != head_left or document["cameras"]["head_right"] != head_right:
        raise FitError("merge mutated head cameras")
    if document["stereo"]["head"] != stereo_head:
        raise FitError("merge mutated stereo.head")
    if created:
        document["created"] = created
    document["note"] = compose_note(document.get("note"), fitted)
    extra = set(document) - {"schema", "created", "note", "board", "cameras", "stereo", "hand_eye"}
    if extra:
        raise FitError(f"refusing to write unknown top-level keys: {sorted(extra)}")
    return document


def format_side_line(result):
    K = np.asarray(result.K, dtype=float).reshape(3, 3)
    return (
        f"{result.camera} RMS={result.rms:.4f} px  fx={K[0, 0]:.2f}  "
        f"image_size=[{int(result.image_size[0])}, {int(result.image_size[1])}]"
    )


def format_report(results, output_path=None, schema_ok=None, errors=None,
                  base_path=None, status=None, skipped=None):
    def fmt(rms):
        return "n/a" if rms is None or not np.isfinite(rms) else f"{rms:.4f} px"

    lines = ["=== R1 wrist plumb_bob fit ==="]
    if base_path is not None:
        lines.append(f"base: {base_path}")
    for result in results:
        K = np.asarray(result.K, dtype=float).reshape(3, 3)
        D = np.asarray(result.D, dtype=float).reshape(-1)
        lines.extend([
            format_side_line(result),
            f"  usable: {result.n_usable}  train: {result.n_train}  holdout: {result.n_holdout} "
            f"(last 10 or 20%)  dropped: {result.dropped_train}",
            f"  fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f} D={_vector(D)}",
            f"  holdout RMS: {fmt(result.holdout_rms)}",
        ])
    if skipped:
        lines.append("skipped:")
        lines.extend(f"  - {item}" for item in skipped)
    if output_path is not None:
        lines.append(f"json: {output_path}")
        lines.append(f"wrote: {output_path}")
    if schema_ok is not None:
        lines.append(f"load_camera_calibration: {'OK' if schema_ok else 'FAILED'}")
    lines.append("head stereo preserved: yes")
    if status is not None:
        lines.append(f"status if installed now: {status}")
    lines.append("NOT copied to assets/r1/camera_calibration.json")
    if errors:
        lines.append("REJECTED:")
        lines.extend(f"  - {item}" for item in errors)
    return "\n".join(lines)


def atomic_write_json(path, document):
    path = Path(path)
    refuse_assets_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        sidecar = path.with_name(path.name + ".pre-wrist-fit")
        if not sidecar.exists():
            shutil.copy2(path, sidecar)
    temporary = path.with_name("." + path.name + ".tmp")
    payload = json.dumps(document, indent=2) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    return path


def write_merged_calibration(base_document, results, output_path, created=None):
    document = merge_wrist_cameras(base_document, results, created=created)
    atomic_write_json(output_path, document)
    loaded = load_camera_calibration(output_path)
    for name in HEAD_CAMERAS:
        if name not in loaded["cameras"]:
            raise FitError(f"written JSON lost {name}")
    if "head" not in loaded["stereo"]:
        raise FitError("written JSON lost stereo.head")
    return loaded


def requested_sides(side_arg):
    if side_arg == "both":
        return ["left", "right"]
    return [side_arg]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit R1 wrist plumb_bob calibration. Writes JSON under the day dir, not assets/.")
    parser.add_argument("--day-dir", type=Path, required=True,
                        help="Capture day directory, e.g. ~/r1-cam-calib/20260920")
    parser.add_argument("--side", default="both", choices=["left", "right", "both"],
                        help="both = fit whichever folders have enough images")
    parser.add_argument("--pattern", nargs=2, type=int, default=list(DEFAULT_PATTERN), metavar=("COLS", "ROWS"))
    parser.add_argument("--square", type=float, default=DEFAULT_SQUARE_M)
    parser.add_argument("--holdout", type=int, default=HOLD_OUT_N)
    parser.add_argument("--holdout-frac", type=float, default=HOLD_OUT_FRAC)
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN_IMAGES)
    parser.add_argument("--max-rms", type=float, default=MAX_RMS_PX)
    parser.add_argument("--assets", type=Path, default=None,
                        help="Fallback base JSON if the day dir has none (default: <repo>/assets/r1/camera_calibration.json)")
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
    refuse_assets_path(output_path)
    report_path = output_path.with_name("wrist_fit_report.txt")
    base_document, base_path = resolve_base_document(day_dir, assets=args.assets)
    print(f"base: {base_path} cameras={sorted(base_document.get('cameras', {}))}")
    skipped = []
    results = []
    rejection_errors = []
    for side in requested_sides(args.side):
        folder = camera_folder(capture_root, side)
        camera = side_spec(side)["camera"]
        if not folder.is_dir():
            message = f"{camera}: no folder {folder}"
            if args.side == "both":
                skipped.append(message)
                continue
            print(message, file=sys.stderr)
            return 1
        views = load_detected_views(
            folder, pattern=tuple(args.pattern), square=args.square)
        if not views:
            message = (
                f"{camera}: no usable images with {inner_corner_count(args.pattern)} corners "
                f"under {folder}"
            )
            if args.side == "both":
                skipped.append(message)
                continue
            print(message, file=sys.stderr)
            return 1
        train, holdout = holdout_split(
            views, holdout_n=args.holdout, holdout_frac=args.holdout_frac)
        print(f"{camera}: loaded {len(views)} usable from {folder}")
        print(f"  train {len(train)}, holdout {len(holdout)} (last {len(holdout)} by filename index)")
        if len(train) < args.min_train:
            message = (
                f"{camera}: not enough training images: {len(train)} "
                f"(need >= {args.min_train} after holdout; capture >=40 OK so 30 remain after holding out 10)"
            )
            if args.side == "both":
                skipped.append(message)
                continue
            rejection_errors.append(message)
            print("REJECTED:")
            print(f"  - {message}")
            report_path.write_text("REJECTED\n" + message + "\n", encoding="utf-8")
            print(f"wrote: {report_path}")
            print("did not write camera_calibration.json")
            return 2
        try:
            result = fit_detected_views(
                train, holdout, side, pattern=tuple(args.pattern), square=args.square,
                created=created_date(day_dir), note=DEFAULT_NOTE, log=print)
        except FitError as error:
            message = f"{camera}: fit failed: {error}"
            if args.side == "both":
                rejection_errors.append(message)
                print(message, file=sys.stderr)
                continue
            print(message, file=sys.stderr)
            return 1
        errors = quality_errors(
            result, min_train=args.min_train, max_rms=args.max_rms)
        result.errors = errors
        if errors:
            tagged = [f"{camera}: {item}" for item in errors]
            rejection_errors.extend(tagged)
            continue
        results.append(result)
        print(format_side_line(result))
    if rejection_errors:
        report = format_report(
            results, output_path=None, schema_ok=None, errors=rejection_errors,
            base_path=base_path, skipped=skipped)
        print(report)
        report_path.write_text(report + "\n", encoding="utf-8")
        print(f"wrote: {report_path}")
        print("did not write camera_calibration.json")
        return 2
    if not results:
        message = "no wrist side had enough images to fit"
        if skipped:
            message += "\n" + "\n".join(skipped)
        print(message, file=sys.stderr)
        report_path.write_text("REJECTED\n" + message + "\n", encoding="utf-8")
        print(f"wrote: {report_path}")
        print("did not write camera_calibration.json")
        return 2
    loaded = write_merged_calibration(
        base_document, results, output_path, created=created_date(day_dir))
    metadata = camera_calibration_metadata(
        loaded, ["head_left", "head_right", "left_wrist", "right_wrist"])
    report = format_report(
        results, output_path=output_path, schema_ok=True, errors=None,
        base_path=base_path, status=metadata["status"], skipped=skipped)
    print(report)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"wrist_fit_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
