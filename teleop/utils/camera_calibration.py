"""Camera calibration schema, validation and per-episode metadata.

Every recorded episode embeds the calibration that was in force when it was
captured. The block is emitted unconditionally: with no calibration file the
status is ``uncalibrated``, so a downstream consumer can tell "nobody has
calibrated this rig yet" apart from "this recorder never wrote the field".

Calibration is a one-way door for a dataset. Once episodes are on disk without
intrinsics they can never be undistorted, deprojected or aligned to another
camera or embodiment, because the rig that produced them may since have been
bumped, rebuilt or scrapped. Embedding the block costs a few kilobytes per
episode and keeps that door open.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


CALIBRATION_SCHEMA = "r1_camera_calibration_v1"
DEFAULT_CALIBRATION_RELPATH = Path("assets/r1/camera_calibration.json")

# colour key -> camera name. color_0/color_1 are the head stereo pair (the
# 1088-wide frame split into two 544 halves); color_2/color_3 are the palm
# cameras. The mapping is fixed, so a key always names the same camera no
# matter which subset of cameras a given run happens to record. This matches
# the long-standing upstream mapping in the non-R1_A7 recording path.
COLOR_KEYS = {
    "color_0": "head_left",
    "color_1": "head_right",
    "color_2": "left_wrist",
    "color_3": "right_wrist",
}
HEAD_COLOR_KEYS = ("color_0", "color_1")
WRIST_COLOR_KEYS = {"left": "color_2", "right": "color_3"}

DISTORTION_MODELS = ("plumb_bob", "rational_polynomial", "fisheye", "none")


class CalibrationError(ValueError):
    """Raised when a calibration file exists but cannot be trusted."""


def _finite_number(value, location):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{location}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationError(f"{location}: expected a finite number")
    return number


def _vector(value, location, length=None):
    if not isinstance(value, list) or not value:
        raise CalibrationError(f"{location}: expected a non-empty list")
    if length is not None and len(value) != length:
        raise CalibrationError(f"{location}: expected {length} values, got {len(value)}")
    return [_finite_number(item, f"{location}[{index}]") for index, item in enumerate(value)]


def _matrix(value, location, rows, columns):
    if not isinstance(value, list) or len(value) != rows:
        raise CalibrationError(f"{location}: expected {rows} rows")
    return [_vector(row, f"{location}[{index}]", columns) for index, row in enumerate(value)]


def _rotation(value, location):
    matrix = np.asarray(_matrix(value, location, 3, 3), dtype=float)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3):
        raise CalibrationError(f"{location}: expected an orthonormal rotation matrix")
    if abs(float(np.linalg.det(matrix)) - 1.0) > 1e-3:
        raise CalibrationError(f"{location}: expected det(R) = +1, got {np.linalg.det(matrix):.6f}")
    return matrix.tolist()


def _camera_entry(name, entry, location):
    if not isinstance(entry, dict):
        raise CalibrationError(f"{location}: expected an object")
    intrinsics = _matrix(entry.get("camera_matrix"), f"{location}.camera_matrix", 3, 3)
    if [intrinsics[2][0], intrinsics[2][1], intrinsics[2][2]] != [0.0, 0.0, 1.0]:
        raise CalibrationError(f"{location}.camera_matrix: last row must be [0, 0, 1]")
    if intrinsics[0][0] <= 0.0 or intrinsics[1][1] <= 0.0:
        raise CalibrationError(f"{location}.camera_matrix: fx and fy must be positive")
    size = _vector(entry.get("image_size"), f"{location}.image_size", 2)
    if size[0] <= 0 or size[1] <= 0 or size[0] != int(size[0]) or size[1] != int(size[1]):
        raise CalibrationError(f"{location}.image_size: expected positive integer width and height")
    model = entry.get("distortion_model", "plumb_bob")
    if model not in DISTORTION_MODELS:
        raise CalibrationError(f"{location}.distortion_model: expected one of {', '.join(DISTORTION_MODELS)}")
    coefficients = entry.get("distortion_coefficients", [])
    if model == "none":
        if coefficients:
            raise CalibrationError(
                f"{location}.distortion_coefficients: must be empty when distortion_model is 'none'")
        coefficients = []
    elif not coefficients:
        raise CalibrationError(f"{location}.distortion_coefficients: required unless distortion_model is 'none'")
    else:
        coefficients = _vector(coefficients, f"{location}.distortion_coefficients")
    result = {
        "camera_matrix": intrinsics,
        "image_size": [int(size[0]), int(size[1])],
        "distortion_model": model,
        "distortion_coefficients": coefficients,
    }
    for optional in ("reprojection_error_px", "calibration_frame"):
        if optional in entry:
            if optional == "calibration_frame":
                if not isinstance(entry[optional], str) or not entry[optional]:
                    raise CalibrationError(f"{location}.{optional}: expected a non-empty string")
                result[optional] = entry[optional]
            else:
                result[optional] = _finite_number(entry[optional], f"{location}.{optional}")
    if "rectification_matrix" in entry:
        result["rectification_matrix"] = _matrix(
            entry["rectification_matrix"], f"{location}.rectification_matrix", 3, 3)
    if "projection_matrix" in entry:
        result["projection_matrix"] = _matrix(
            entry["projection_matrix"], f"{location}.projection_matrix", 3, 4)
    return result


def _stereo_entry(name, entry, location, cameras):
    if not isinstance(entry, dict):
        raise CalibrationError(f"{location}: expected an object")
    pair = []
    for side in ("left", "right"):
        camera = entry.get(side)
        if camera not in cameras:
            raise CalibrationError(f"{location}.{side}: {camera!r} is not a calibrated camera")
        pair.append(camera)
    if pair[0] == pair[1]:
        raise CalibrationError(f"{location}: left and right must name different cameras")
    result = {"left": pair[0], "right": pair[1],
              "rotation": _rotation(entry.get("rotation"), f"{location}.rotation"),
              "translation_m": _vector(entry.get("translation_m"), f"{location}.translation_m", 3)}
    for optional in ("baseline_m", "epipolar_error_px"):
        if optional in entry:
            result[optional] = _finite_number(entry[optional], f"{location}.{optional}")
    if "rectified_image_size" in entry:
        result["rectified_image_size"] = _vector(entry["rectified_image_size"],
                                                 f"{location}.rectified_image_size", 2)
    return result


def _hand_eye_entry(name, entry, location, cameras):
    if not isinstance(entry, dict):
        raise CalibrationError(f"{location}: expected an object")
    camera = entry.get("camera")
    if camera not in cameras:
        raise CalibrationError(f"{location}.camera: {camera!r} is not a calibrated camera")
    frame = entry.get("frame")
    if not isinstance(frame, str) or not frame:
        raise CalibrationError(f"{location}.frame: expected the robot link the camera is rigidly mounted to")
    result = {"camera": camera, "frame": frame,
              "rotation": _rotation(entry.get("rotation"), f"{location}.rotation"),
              "translation_m": _vector(entry.get("translation_m"), f"{location}.translation_m", 3)}
    for optional in ("rotation_rms_deg", "translation_rms_m"):
        if optional in entry:
            result[optional] = _finite_number(entry[optional], f"{location}.{optional}")
    return result


def _board_entry(entry, location):
    if not isinstance(entry, dict):
        raise CalibrationError(f"{location}: expected an object")
    kind = entry.get("type")
    if kind not in ("checkerboard", "charuco"):
        raise CalibrationError(f"{location}.type: expected 'checkerboard' or 'charuco'")
    result = {"type": kind}
    for required in ("squares_x", "squares_y"):
        value = entry.get(required)
        if type(value) is not int or value <= 0:
            raise CalibrationError(f"{location}.{required}: expected a positive integer")
        result[required] = value
    for required in ("square_size_m",):
        result[required] = _finite_number(entry.get(required), f"{location}.{required}")
        if result[required] <= 0.0:
            raise CalibrationError(f"{location}.{required}: expected a positive length in metres")
    if kind == "charuco":
        marker = _finite_number(entry.get("marker_size_m"), f"{location}.marker_size_m")
        if not 0.0 < marker < result["square_size_m"]:
            raise CalibrationError(f"{location}.marker_size_m: expected a positive length below square_size_m")
        result["marker_size_m"] = marker
        dictionary = entry.get("dictionary")
        if not isinstance(dictionary, str) or not dictionary:
            raise CalibrationError(f"{location}.dictionary: expected the ArUco dictionary name")
        result["dictionary"] = dictionary
    return result


def load_camera_calibration(path):
    """Parse and validate a calibration file, returning a normalised mapping."""
    path = Path(path).expanduser()
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CalibrationError(f"{path}: {error}") from error
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise CalibrationError(f"{path}: not valid UTF-8 JSON: {error}") from error
    if not isinstance(document, dict):
        raise CalibrationError(f"{path}: expected a JSON object at the top level")
    if document.get("schema") != CALIBRATION_SCHEMA:
        raise CalibrationError(f"{path}: schema must be {CALIBRATION_SCHEMA!r}, got {document.get('schema')!r}")
    raw_cameras = document.get("cameras")
    if not isinstance(raw_cameras, dict) or not raw_cameras:
        raise CalibrationError(f"{path}: cameras must be a non-empty object")
    cameras = {name: _camera_entry(name, entry, f"{path}: cameras.{name}")
               for name, entry in raw_cameras.items()}
    calibration = {"schema": CALIBRATION_SCHEMA, "source": str(path),
                   "sha256": hashlib.sha256(raw).hexdigest(), "cameras": cameras,
                   "stereo": {}, "hand_eye": {}, "board": None, "created": None, "note": None}
    if "created" in document:
        if not isinstance(document["created"], str) or not document["created"]:
            raise CalibrationError(f"{path}: created must be a non-empty string")
        calibration["created"] = document["created"]
    if "note" in document:
        if not isinstance(document["note"], str):
            raise CalibrationError(f"{path}: note must be a string")
        calibration["note"] = document["note"]
    if "board" in document:
        calibration["board"] = _board_entry(document["board"], f"{path}: board")
    raw_stereo = document.get("stereo", {})
    if not isinstance(raw_stereo, dict):
        raise CalibrationError(f"{path}: stereo must be an object")
    calibration["stereo"] = {name: _stereo_entry(name, entry, f"{path}: stereo.{name}", cameras)
                             for name, entry in raw_stereo.items()}
    raw_hand_eye = document.get("hand_eye", {})
    if not isinstance(raw_hand_eye, dict):
        raise CalibrationError(f"{path}: hand_eye must be an object")
    calibration["hand_eye"] = {name: _hand_eye_entry(name, entry, f"{path}: hand_eye.{name}", cameras)
                               for name, entry in raw_hand_eye.items()}
    extra = set(document) - {"schema", "created", "note", "board", "cameras", "stereo", "hand_eye"}
    if extra:
        raise CalibrationError(f"{path}: unknown top-level keys: {', '.join(sorted(extra))}")
    return calibration


def resolve_camera_calibration(path, root=None):
    """Resolve the calibration file for a run.

    An explicitly requested file must exist and be valid: silently recording an
    episode with no calibration when one was asked for is exactly the failure
    this module exists to prevent. The default location is optional, but a file
    found there is still validated rather than ignored.
    """
    if path:
        return Path(path).expanduser(), load_camera_calibration(path)
    if root is None:
        return None, None
    default = Path(root) / DEFAULT_CALIBRATION_RELPATH
    if not default.is_file():
        return None, None
    return default, load_camera_calibration(default)


def expected_cameras(camera_config):
    """Camera names a run should have calibration for."""
    names = list(COLOR_KEYS[key] for key in HEAD_COLOR_KEYS)
    for side in ("left", "right"):
        if (camera_config.get(f"{side}_wrist_camera") or {}).get("enable_zmq"):
            names.append(COLOR_KEYS[WRIST_COLOR_KEYS[side]])
    return names


def camera_calibration_metadata(calibration, cameras, actual_image_sizes=None):
    """Build the per-episode ``camera_calibration`` block.

    ``cameras`` is the list of names this run should have calibration for and
    ``actual_image_sizes`` maps names to the ``(width, height)`` actually being
    streamed, so a calibration captured at a different resolution is reported
    instead of being applied silently.
    """
    cameras = list(cameras)
    block = {
        "schema": CALIBRATION_SCHEMA,
        "status": "uncalibrated",
        "source": None,
        "sha256": None,
        "created": None,
        "expected_cameras": cameras,
        "missing_cameras": cameras,
        "cameras": {},
        "board": None,
        "stereo": {},
        "hand_eye": {},
        "warnings": [],
    }
    if calibration is None:
        block["note"] = ("No camera calibration was supplied for this run. Colour frames are raw "
                         "camera output and cannot be undistorted, deprojected or aligned after the fact.")
        return block
    block["source"] = calibration.get("source")
    block["sha256"] = calibration.get("sha256")
    block["created"] = calibration.get("created")
    block["board"] = calibration.get("board")
    block["stereo"] = calibration.get("stereo", {})
    block["hand_eye"] = calibration.get("hand_eye", {})
    available = calibration.get("cameras", {})
    block["cameras"] = {name: available[name] for name in cameras if name in available}
    block["missing_cameras"] = [name for name in cameras if name not in available]
    if not block["cameras"]:
        block["status"] = "uncalibrated"
        block["note"] = (f"{block['source']} calibrates no camera this run records; "
                         "expected " + ", ".join(cameras) + ".")
    elif block["missing_cameras"]:
        block["status"] = "partial"
        block["note"] = "Calibration is missing for: " + ", ".join(block["missing_cameras"]) + "."
    else:
        block["status"] = "calibrated"
        block["note"] = None
    for name, entry in block["cameras"].items():
        actual = (actual_image_sizes or {}).get(name)
        expected_size = entry.get("image_size")
        if actual is not None and expected_size is not None and list(actual) != list(expected_size):
            block["warnings"].append(
                f"{name}: calibrated at {expected_size[0]}x{expected_size[1]} but streaming "
                f"{actual[0]}x{actual[1]}; intrinsics do not apply to this resolution.")
    for name, entry in block["stereo"].items():
        for side in ("left", "right"):
            if entry[side] not in block["cameras"]:
                block["warnings"].append(f"stereo.{name}.{side}: {entry[side]} has no intrinsics in this run.")
    for name, entry in block["hand_eye"].items():
        if entry["camera"] not in block["cameras"]:
            block["warnings"].append(f"hand_eye.{name}: {entry['camera']} has no intrinsics in this run.")
    return block
