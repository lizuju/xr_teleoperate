#!/usr/bin/env python3
"""Fit R1 eye-in-hand transforms into hand_eye{} of camera_calibration.json.

Reads captured PNG+JSON pairs under ``~/r1-cam-calib/YYYYMMDD/hand_eye/<camera>/``,
solves PnP with installed intrinsics, then OpenCV calibrateHandEye (eye-in-hand).

Wrist UVC frames are horizontally mirrored. Fitting H-unmirrors them before PnP
so SE(3) hand-eye can converge, then stores the unmirrored-optical extrinsic
together with ``image_mirror: "horizontal"`` (on hand_eye and cameras). Runtime
consumers must use ``prepare_image_for_hand_eye`` /
``project_link_points_to_pixels`` so raw wrist JPEGs need no manual flip at call
sites. Head cameras are never mirrored.

Writes the day-dir JSON only (merge preserves cameras/stereo). Does NOT write
assets/r1/camera_calibration.json. Does not start teleop or move the robot.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (REPO, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrate_r1_wrist_capture import detect_corners, inner_corner_count  # noqa: E402
from calibrate_r1_wrist_fit import (  # noqa: E402
    canonicalize_corners,
    is_assets_calibration_path,
    refuse_assets_path,
)
from r1_hand_eye_common import (  # noqa: E402
    DEFAULT_PATTERN,
    DEFAULT_SQUARE_M,
    DEFAULT_URDF,
    HandEyeError,
    R1A7FK,
    TARGETS,
    invert_se3,
    motor_q_to_pinocchio_q,
    object_points,
    rotation_geodesic_deg,
    rt_from_se3,
    se3_from_rt,
    target_spec,
)
from teleop.utils.camera_calibration import (  # noqa: E402
    CALIBRATION_SCHEMA,
    load_camera_calibration,
)

MIN_TRAIN = 15
HOLD_OUT_N = 5
HOLD_OUT_FRAC = 0.20
MAX_ROT_RMS_DEG = 5.0
MAX_TRANS_RMS_M = 0.03
DEFAULT_ASSETS = Path("assets/r1/camera_calibration.json")
HEAD_CAMERAS = ("head_left", "head_right")
HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


class FitError(HandEyeError):
    pass


class FitRejected(FitError):
    pass


@dataclass
class Sample:
    index: int
    path: Path
    meta_path: Path
    image_points: np.ndarray
    object_points: np.ndarray
    T_gripper2base: np.ndarray
    T_target2cam: np.ndarray
    camera: str
    frame: str


@dataclass
class FitResult:
    key: str
    camera: str
    frame: str
    n_usable: int
    n_train: int
    n_holdout: int
    R: np.ndarray
    t: np.ndarray
    rotation_rms_deg: float
    translation_rms_m: float
    method: str
    holdout_rotation_rms_deg: float = float("nan")
    holdout_translation_rms_m: float = float("nan")
    errors: list = field(default_factory=list)


def created_date(day_dir):
    for part in reversed(Path(day_dir).parts):
        if len(part) == 8 and part.isdigit():
            return f"{part[:4]}-{part[4:6]}-{part[6:8]}"
    return time.strftime("%Y-%m-%d")


def holdout_count(n, holdout_n=HOLD_OUT_N, holdout_frac=HOLD_OUT_FRAC):
    n = int(n)
    if n <= 0:
        return 0
    return min(n, max(int(holdout_n), int(round(n * float(holdout_frac)))))


def holdout_split(samples, holdout_n=HOLD_OUT_N, holdout_frac=HOLD_OUT_FRAC):
    samples = list(samples)
    n_hold = holdout_count(len(samples), holdout_n=holdout_n, holdout_frac=holdout_frac)
    if n_hold == 0:
        return samples, []
    return samples[:-n_hold], samples[-n_hold:]


def load_intrinsics(document, camera):
    entry = document.get("cameras", {}).get(camera)
    if not isinstance(entry, dict):
        raise FitError(f"base JSON has no cameras.{camera}; calibrate intrinsics first")
    K = np.asarray(entry["camera_matrix"], dtype=float).reshape(3, 3)
    D = np.asarray(entry.get("distortion_coefficients", []), dtype=float).reshape(-1)
    model = entry.get("distortion_model", "plumb_bob")
    size = tuple(int(x) for x in entry["image_size"])
    return K, D, model, size


def camera_needs_wrist_unmirror(camera):
    """R1 wrist UVC frames are horizontally mirrored vs a right-handed pinhole frame."""
    return camera in ("left_wrist", "right_wrist")


def unmirror_wrist_frame(image, K, D, model):
    """Undo horizontal mirror on a wrist image and map intrinsics into that frame.

    Mirrored images still give low monocular reprojection error (cx near centre),
    but the implied camera frame is improper relative to the robot, so SE(3)
    hand-eye cannot find a rigid cam→gripper. H-unmirror restores a right-handed
    optical frame. Head cameras must not use this path.
    """
    image = np.asarray(image)
    if image.ndim < 2:
        raise FitError("wrist unmirror expects an image array")
    height, width = image.shape[:2]
    flipped = cv2.flip(image, 1)
    K_out = np.asarray(K, dtype=float).reshape(3, 3).copy()
    K_out[0, 2] = float(width - 1) - float(K_out[0, 2])
    D_out = np.asarray(D, dtype=float).reshape(-1).copy()
    if model in ("plumb_bob", "rational_polynomial") and D_out.size >= 3:
        # Tangential p1 flips sign under a horizontal image reflection.
        D_out[2] = -D_out[2]
    return flipped, K_out, D_out


def undistort_points(image_points, K, D, model):
    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    if model == "fisheye":
        D4 = np.zeros((4, 1), dtype=np.float64)
        D4[: min(4, D.size), 0] = D.reshape(-1)[:4]
        und = cv2.fisheye.undistortPoints(pts, K, D4, P=K)
    elif model in ("plumb_bob", "rational_polynomial", "none"):
        dist = D.reshape(-1) if model != "none" else np.zeros(5, dtype=np.float64)
        und = cv2.undistortPoints(pts, K, dist, P=K)
    else:
        raise FitError(f"unsupported distortion_model {model!r}")
    return np.asarray(und, dtype=np.float64).reshape(-1, 1, 2)


def solve_target2cam(object_pts, image_pts, K, D, model):
    und = undistort_points(image_pts, K, D, model)
    zero_dist = np.zeros((5, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(object_pts, dtype=np.float64).reshape(-1, 3),
        und,
        K,
        zero_dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise FitError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return se3_from_rt(R, tvec.reshape(3))


def resolve_hand_eye_root(day_dir):
    day_dir = Path(day_dir).expanduser().resolve()
    if not day_dir.exists():
        raise FitError(f"day directory does not exist: {day_dir}")
    for candidate in (day_dir / "hand_eye", day_dir):
        if any((candidate / name).is_dir() for name in ("left_wrist", "right_wrist", "head_left")):
            return candidate
    raise FitError(f"no hand_eye camera folders under {day_dir}")


def load_samples(folder, camera, frame, document, pattern=DEFAULT_PATTERN, square=DEFAULT_SQUARE_M,
                 urdf=DEFAULT_URDF, fk=None, detect=None, log=print, wrist_unmirror=None):
    if detect is None:
        detect = detect_corners
    if fk is None:
        fk = R1A7FK(urdf_path=urdf)
    K0, D0, model, expected_size = load_intrinsics(document, camera)
    if wrist_unmirror is None:
        wrist_unmirror = camera_needs_wrist_unmirror(camera)
    if wrist_unmirror and not camera_needs_wrist_unmirror(camera):
        raise FitError(f"wrist_unmirror is only valid for wrist cameras, got {camera!r}")
    if wrist_unmirror:
        log(f"{camera}: applying horizontal unmirror before PnP (UVC mirror vs SE3 hand-eye)")
    obj = object_points(pattern, square)
    expected = inner_corner_count(pattern)
    folder = Path(folder)
    paths = sorted(
        [path for path in folder.glob("*.png") if path.stem.isdigit()],
        key=lambda path: int(path.stem),
    )
    samples = []
    skipped = []
    for path in paths:
        meta_path = path.with_suffix(".json")
        if not meta_path.is_file():
            skipped.append((path.name, "missing sidecar JSON"))
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError as error:
            skipped.append((path.name, f"bad JSON: {error}"))
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append((path.name, "unreadable"))
            continue
        height, width = image.shape[:2]
        if (width, height) != expected_size:
            skipped.append((path.name, f"size {width}x{height} != {expected_size}"))
            continue
        K, D = K0, D0
        if wrist_unmirror:
            image, K, D = unmirror_wrist_frame(image, K0, D0, model)
        ok, corners = detect(image, pattern)
        if not ok:
            skipped.append((path.name, f"detect failed (need {expected})"))
            continue
        image_points = canonicalize_corners(corners, pattern)
        if "T_link_in_root" in meta:
            R = meta["T_link_in_root"]["rotation"]
            t = meta["T_link_in_root"]["translation_m"]
            T_g2b = se3_from_rt(R, t)
        elif "pinocchio_q" in meta:
            T_g2b = fk.link_pose(meta["pinocchio_q"], frame)
        elif "motor_q" in meta:
            T_g2b = fk.link_pose(motor_q_to_pinocchio_q(meta["motor_q"]), frame)
        else:
            skipped.append((path.name, "JSON missing T_link_in_root / pinocchio_q / motor_q"))
            continue
        try:
            T_t2c = solve_target2cam(obj, image_points, K, D, model)
        except FitError as error:
            skipped.append((path.name, str(error)))
            continue
        samples.append(Sample(
            index=int(path.stem),
            path=path,
            meta_path=meta_path,
            image_points=image_points,
            object_points=obj.copy(),
            T_gripper2base=T_g2b,
            T_target2cam=T_t2c,
            camera=camera,
            frame=frame,
        ))
    if skipped:
        log(f"warning: skipped {len(skipped)} samples for {camera}")
        for name, reason in skipped[:12]:
            log(f"  {name}: {reason}")
        if len(skipped) > 12:
            log(f"  ... {len(skipped) - 12} more")
    return samples


def _rt_lists(samples):
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for sample in samples:
        Rg, tg = rt_from_se3(sample.T_gripper2base)
        Rt, tt = rt_from_se3(sample.T_target2cam)
        R_g2b.append(Rg)
        t_g2b.append(tg.reshape(3, 1))
        R_t2c.append(Rt)
        t_t2c.append(tt.reshape(3, 1))
    return R_g2b, t_g2b, R_t2c, t_t2c


def calibrate_hand_eye(samples, method="tsai"):
    if method not in HAND_EYE_METHODS:
        raise FitError(f"unknown method {method!r}")
    if len(samples) < 3:
        raise FitError(f"need >=3 samples for calibrateHandEye, got {len(samples)}")
    R_g2b, t_g2b, R_t2c, t_t2c = _rt_lists(samples)
    try:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_t2c, t_t2c, method=HAND_EYE_METHODS[method])
    except cv2.error as error:
        raise FitError(f"cv2.calibrateHandEye failed: {error}") from error
    R = np.asarray(R_cam2gripper, dtype=float).reshape(3, 3)
    t = np.asarray(t_cam2gripper, dtype=float).reshape(3)
    if abs(float(np.linalg.det(R)) - 1.0) > 1e-2:
        raise FitError(f"hand-eye rotation det={np.linalg.det(R):.4f}, expected ~1")
    return R, t


def residual_stats(samples, R_cam2gripper, t_cam2gripper):
    """Board-in-base consistency across views using estimated cam2gripper."""
    T_c2g = se3_from_rt(R_cam2gripper, t_cam2gripper)
    boards = []
    for sample in samples:
        T_g2b = sample.T_gripper2base
        T_t2c = sample.T_target2cam
        T_t2b = T_g2b @ T_c2g @ T_t2c
        boards.append(T_t2b)
    if len(boards) < 2:
        return 0.0, 0.0
    ref = boards[0]
    rot_errs = []
    trans_errs = []
    for other in boards[1:]:
        rot_errs.append(rotation_geodesic_deg(ref[:3, :3], other[:3, :3]))
        trans_errs.append(float(np.linalg.norm(ref[:3, 3] - other[:3, 3])))
    return float(np.sqrt(np.mean(np.square(rot_errs)))), float(np.sqrt(np.mean(np.square(trans_errs))))


def fit_samples(samples, key, camera, frame, method="tsai", min_train=MIN_TRAIN,
                holdout_n=HOLD_OUT_N, holdout_frac=HOLD_OUT_FRAC, log=print):
    if len(samples) < min_train + 1:
        raise FitError(
            f"{camera}: only {len(samples)} usable samples (need >= {min_train + 1} including holdout)"
        )
    train, holdout = holdout_split(samples, holdout_n=holdout_n, holdout_frac=holdout_frac)
    if len(train) < min_train:
        raise FitError(f"{camera}: only {len(train)} train samples after holdout (need >= {min_train})")
    log(f"{camera}: train={len(train)} holdout={len(holdout)} method={method}")
    R, t = calibrate_hand_eye(train, method=method)
    rot_rms, trans_rms = residual_stats(train, R, t)
    hold_rot, hold_trans = (float("nan"), float("nan"))
    if holdout:
        hold_rot, hold_trans = residual_stats(holdout, R, t)
    return FitResult(
        key=key,
        camera=camera,
        frame=frame,
        n_usable=len(samples),
        n_train=len(train),
        n_holdout=len(holdout),
        R=R,
        t=t,
        rotation_rms_deg=rot_rms,
        translation_rms_m=trans_rms,
        method=method,
        holdout_rotation_rms_deg=hold_rot,
        holdout_translation_rms_m=hold_trans,
    )


def quality_errors(result, max_rot=MAX_ROT_RMS_DEG, max_trans=MAX_TRANS_RMS_M, min_train=MIN_TRAIN):
    errors = []
    if result.n_train < min_train:
        errors.append(f"train samples {result.n_train} < {min_train}")
    if result.rotation_rms_deg > max_rot:
        errors.append(f"train rotation RMS {result.rotation_rms_deg:.2f} deg > {max_rot}")
    if result.translation_rms_m > max_trans:
        errors.append(f"train translation RMS {result.translation_rms_m:.4f} m > {max_trans}")
    if math.isfinite(result.holdout_rotation_rms_deg) and result.holdout_rotation_rms_deg > max_rot * 1.5:
        errors.append(
            f"holdout rotation RMS {result.holdout_rotation_rms_deg:.2f} deg too large"
        )
    if math.isfinite(result.holdout_translation_rms_m) and result.holdout_translation_rms_m > max_trans * 1.5:
        errors.append(
            f"holdout translation RMS {result.holdout_translation_rms_m:.4f} m too large"
        )
    return errors


def hand_eye_entry(result, image_mirror=None):
    entry = {
        "camera": result.camera,
        "frame": result.frame,
        "rotation": np.asarray(result.R, dtype=float).reshape(3, 3).tolist(),
        "translation_m": [float(x) for x in np.asarray(result.t, dtype=float).reshape(3)],
        "rotation_rms_deg": float(result.rotation_rms_deg),
        "translation_rms_m": float(result.translation_rms_m),
    }
    if image_mirror is None and camera_needs_wrist_unmirror(result.camera):
        # Extrinsic is for the unmirrored optical frame; raw stream stays mirrored.
        image_mirror = "horizontal"
    if image_mirror and image_mirror != "none":
        entry["image_mirror"] = image_mirror
    return entry


def annotate_wrist_camera_mirror(document, cameras=("left_wrist", "right_wrist")):
    """Mark wrist camera streams as horizontally mirrored (raw pixels unchanged)."""
    cameras_block = document.setdefault("cameras", {})
    for name in cameras:
        entry = cameras_block.get(name)
        if isinstance(entry, dict):
            entry["image_mirror"] = "horizontal"
    return document


def head_right_from_left_and_stereo(left_entry, stereo_head):
    """Compose head_right cam→head from head_left and stereo (P_left = R P_right + t)."""
    R_left = np.asarray(left_entry["rotation"], dtype=float).reshape(3, 3)
    t_left = np.asarray(left_entry["translation_m"], dtype=float).reshape(3)
    R_stereo = np.asarray(stereo_head["rotation"], dtype=float).reshape(3, 3)
    t_stereo = np.asarray(stereo_head["translation_m"], dtype=float).reshape(3)
    R_right = R_left @ R_stereo
    t_right = R_left @ t_stereo + t_left
    return {
        "camera": "head_right",
        "frame": left_entry["frame"],
        "rotation": R_right.tolist(),
        "translation_m": [float(x) for x in t_right],
        "rotation_rms_deg": float(left_entry.get("rotation_rms_deg", 0.0)),
        "translation_rms_m": float(left_entry.get("translation_rms_m", 0.0)),
    }


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
        raise FitError(f"{source}: missing {', '.join(missing)}; refusing merge that drops head")
    stereo = document.get("stereo")
    if not isinstance(stereo, dict) or "head" not in stereo:
        raise FitError(f"{source}: missing stereo.head")
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
    raise FitError(f"no base camera_calibration.json (looked at {day_json} then {assets_json})")


def merge_hand_eye(base_document, entries, created=None, note_suffix=None,
                   mark_wrist_mirror=True):
    document = copy.deepcopy(base_document)
    validate_head_base(document, "base")
    head_snapshot = {name: copy.deepcopy(document["cameras"][name]) for name in HEAD_CAMERAS}
    stereo_snapshot = copy.deepcopy(document["stereo"]["head"])
    hand_eye = dict(document.get("hand_eye") or {})
    for key, entry in entries.items():
        hand_eye[key] = entry
    document["hand_eye"] = hand_eye
    if mark_wrist_mirror and any(
            camera_needs_wrist_unmirror(entry.get("camera", key))
            for key, entry in entries.items()):
        annotate_wrist_camera_mirror(document)
    if created:
        document["created"] = created
    if note_suffix:
        note = document.get("note") or ""
        if note_suffix not in note:
            document["note"] = (note.rstrip(".") + "; " + note_suffix).lstrip("; ")
    if any(document["cameras"][name] != head_snapshot[name] for name in HEAD_CAMERAS):
        raise FitError("merge mutated head camera intrinsics")
    if document["stereo"]["head"] != stereo_snapshot:
        raise FitError("merge mutated stereo.head")
    return document


def atomic_write_json(path, document):
    path = Path(path)
    refuse_assets_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        sidecar = path.with_name(path.name + ".pre-hand-eye-fit")
        if not sidecar.exists():
            shutil.copy2(path, sidecar)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def format_report(results, output_path=None, errors=None, base_path=None, skipped=None):
    lines = ["=== R1 hand-eye fit ==="]
    if base_path is not None:
        lines.append(f"base: {base_path}")
    for result in results:
        lines.append(
            f"{result.key}: camera={result.camera} frame={result.frame} "
            f"train={result.n_train} holdout={result.n_holdout} method={result.method}"
        )
        lines.append(
            f"  t_cam2frame_m={np.asarray(result.t).reshape(3).tolist()} "
            f"rot_rms={result.rotation_rms_deg:.3f} deg  "
            f"trans_rms={result.translation_rms_m:.4f} m"
        )
        if math.isfinite(result.holdout_rotation_rms_deg):
            lines.append(
                f"  holdout rot_rms={result.holdout_rotation_rms_deg:.3f} deg  "
                f"trans_rms={result.holdout_translation_rms_m:.4f} m"
            )
    if skipped:
        lines.append("skipped:")
        lines.extend(f"  - {item}" for item in skipped)
    if output_path is not None:
        lines.append(f"wrote: {output_path}")
    lines.append("head intrinsics preserved: yes")
    lines.append("NOT copied to assets/r1/camera_calibration.json")
    if errors:
        lines.append("REJECTED:")
        lines.extend(f"  - {item}" for item in errors)
    return "\n".join(lines)


def requested_targets(arg):
    if arg == "all":
        return ["left_wrist", "right_wrist", "head"]
    return [arg]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit R1 hand-eye into day-dir camera_calibration.json (not assets/).")
    parser.add_argument("--day-dir", type=Path, required=True)
    parser.add_argument("--target", default="all",
                        choices=["left_wrist", "right_wrist", "head", "all"])
    parser.add_argument("--pattern", nargs=2, type=int, default=list(DEFAULT_PATTERN))
    parser.add_argument("--square", type=float, default=DEFAULT_SQUARE_M)
    parser.add_argument("--method", default="tsai", choices=sorted(HAND_EYE_METHODS))
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN)
    parser.add_argument("--holdout", type=int, default=HOLD_OUT_N)
    parser.add_argument("--holdout-frac", type=float, default=HOLD_OUT_FRAC)
    parser.add_argument("--max-rot-rms-deg", type=float, default=MAX_ROT_RMS_DEG)
    parser.add_argument("--max-trans-rms-m", type=float, default=MAX_TRANS_RMS_M)
    parser.add_argument("--assets", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--wrist-unmirror",
        choices=["auto", "on", "off"],
        default="auto",
        help="H-unmirror wrist images before PnP (auto=on for left/right_wrist; "
             "needed because R1 wrist UVC frames are mirrored vs SE3 hand-eye)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    day_dir = args.day_dir.expanduser().resolve()
    hand_eye_root = resolve_hand_eye_root(day_dir)
    output_path = args.output.expanduser() if args.output else day_dir / "camera_calibration.json"
    refuse_assets_path(output_path)
    if is_assets_calibration_path(output_path):
        raise FitError("refusing assets path")
    report_path = output_path.with_name("hand_eye_fit_report.txt")
    base_document, base_path = resolve_base_document(day_dir, assets=args.assets)
    print(f"base: {base_path}")
    fk = R1A7FK(urdf_path=args.urdf)
    results = []
    entries = {}
    skipped = []
    rejection = []
    for target in requested_targets(args.target):
        spec = target_spec(target)
        camera = spec["camera"]
        folder = hand_eye_root / camera
        if not folder.is_dir():
            message = f"{camera}: no folder {folder}"
            if args.target == "all":
                skipped.append(message)
                continue
            print(message, file=sys.stderr)
            return 1
        if args.wrist_unmirror == "off":
            do_unmirror = False
        elif args.wrist_unmirror == "on":
            do_unmirror = camera_needs_wrist_unmirror(camera)
        else:
            do_unmirror = None  # auto: wrists on, head off
        samples = load_samples(
            folder, camera, spec["frame"], base_document,
            pattern=tuple(args.pattern), square=args.square, fk=fk, log=print,
            wrist_unmirror=do_unmirror)
        if not samples:
            message = f"{camera}: no usable PNG+JSON samples under {folder}"
            if args.target == "all":
                skipped.append(message)
                continue
            print(message, file=sys.stderr)
            return 1
        try:
            result = fit_samples(
                samples, key=camera, camera=camera, frame=spec["frame"],
                method=args.method, min_train=args.min_train,
                holdout_n=args.holdout, holdout_frac=args.holdout_frac, log=print)
        except FitError as error:
            message = f"{camera}: {error}"
            if args.target == "all":
                rejection.append(message)
                print(message, file=sys.stderr)
                continue
            print(message, file=sys.stderr)
            return 1
        errors = quality_errors(
            result, max_rot=args.max_rot_rms_deg, max_trans=args.max_trans_rms_m,
            min_train=args.min_train)
        if errors:
            rejection.extend(f"{camera}: {item}" for item in errors)
            continue
        results.append(result)
        entries[camera] = hand_eye_entry(result)
        if target == "head":
            entries["head_right"] = head_right_from_left_and_stereo(
                entries["head_left"], base_document["stereo"]["head"])
    if rejection:
        report = format_report(results, errors=rejection, base_path=base_path, skipped=skipped)
        print(report)
        report_path.write_text(report + "\n", encoding="utf-8")
        print("did not write camera_calibration.json")
        return 2
    if not entries:
        message = "no target had enough samples"
        if skipped:
            message += "\n" + "\n".join(skipped)
        print(message, file=sys.stderr)
        report_path.write_text("REJECTED\n" + message + "\n", encoding="utf-8")
        return 2
    document = merge_hand_eye(
        base_document, entries, created=created_date(day_dir),
        note_suffix=(
            "hand_eye eye-in-hand (wrist→wrist_yaw_link solved after H-unmirror; "
            "image_mirror=horizontal so prepare_image_for_hand_eye / "
            "project_link_points_to_pixels consume raw wrist JPEGs; "
            "head→head_yaw_link, no mirror)"
        ))
    atomic_write_json(output_path, document)
    loaded = load_camera_calibration(output_path)
    for name in HEAD_CAMERAS:
        if name not in loaded["cameras"]:
            raise FitError(f"written JSON lost {name}")
    for key in entries:
        if key not in loaded["hand_eye"]:
            raise FitError(f"written JSON lost hand_eye.{key}")
    report = format_report(results, output_path=output_path, base_path=base_path, skipped=skipped)
    print(report)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"hand_eye_fit_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
