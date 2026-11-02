"""Build the Vision Pro app's head-optics.json for the raw head cameras.

Reads the factory calibration from the PC1 page, converts it to the app's fisheye
model at the stream size, then removes the remaining left/right pitch and roll using
feature matches on one frame pair. Use a static scene with detail at several depths.

Usage: fit_head_optics.py LEFT.jpg RIGHT.jpg OUTPUT.json
"""
import argparse
import hashlib
import json
import re
import urllib.request

import cv2
import numpy as np

CAM_PARAM_URL = "http://192.168.123.161:9080/cam_param.txt"
RAW_SIZE = (1280, 1088)
EYE_SIZE = (960, 816)
RENDER_SIZE = (1280, 1088)
SCALE = EYE_SIZE[0] / RAW_SIZE[0]
MAX_FIT_RADIUS_PX = 760  # Beyond the lens circle there is no image.
REFINE_STEPS = 3


def read_section(text, name):
    body = text.split(name)[1].split("====")[0]
    return {key: float(value) for key, value in re.findall(r"(\w+):\s*(-?[\d.eE+-]+)", body)}


def camera_matrix(eye):
    return np.array([[eye["fx"], 0, eye["cx"]], [0, eye["fy"], eye["cy"]], [0, 0, 1.0]])


def rational_distortion(eye):
    return np.array([eye["k1"], eye["k2"], eye["p1"], eye["p2"], eye["k3"], eye["k4"], eye["k5"], eye["k6"]])


def fit_fisheye(eye):
    theta = np.linspace(1e-3, np.radians(82), 800)
    t2 = np.tan(theta) ** 2
    radius = np.tan(theta) * (1 + eye["k1"] * t2 + eye["k2"] * t2**2 + eye["k3"] * t2**3) / (
        1 + eye["k4"] * t2 + eye["k5"] * t2**2 + eye["k6"] * t2**3)
    keep = radius * eye["fx"] <= MAX_FIT_RADIUS_PX
    theta, radius = theta[keep], radius[keep]
    powers = np.stack([theta**3, theta**5, theta**7, theta**9], axis=1)
    coefficients = np.linalg.lstsq(powers, radius - theta, rcond=None)[0]
    error = np.abs(theta + powers @ coefficients - radius) * eye["fx"]
    print(f"fisheye fit: max error {error.max():.2f} px, p95 {np.percentile(error, 95):.2f} px")
    return coefficients


def scaled_matrix(matrix):
    matrix = matrix.copy()
    matrix[0, 0] *= SCALE
    matrix[1, 1] *= SCALE
    matrix[0, 2] = (matrix[0, 2] + 0.5) * SCALE - 0.5
    matrix[1, 2] = (matrix[1, 2] + 0.5) * SCALE - 0.5
    return matrix


def sampling_map(eye, fov=120):
    """Same mapping as r1OpticsFragment in RobotVideoOptics.metal."""
    matrix, d, rotation = (np.array(eye[key]) for key in
                           ("camera_matrix", "distortion_coefficients", "rectification_rotation"))
    focal = RENDER_SIZE[0] / (2 * np.tan(np.radians(fov) / 2))
    px, py = np.meshgrid(np.arange(RENDER_SIZE[0], dtype=np.float32), np.arange(RENDER_SIZE[1], dtype=np.float32))
    rays = np.stack(((px - (RENDER_SIZE[0] - 1) / 2) / focal, (py - (RENDER_SIZE[1] - 1) / 2) / focal,
                     np.ones_like(px)), -1) @ rotation
    nx, ny = rays[..., 0] / rays[..., 2], rays[..., 1] / rays[..., 2]
    radius = np.hypot(nx, ny)
    theta = np.arctan(radius)
    t2 = theta * theta
    distorted = theta * (1 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3]))))
    scale = np.where(radius > 1e-8, distorted / np.maximum(radius, 1e-8), 1)
    return ((matrix[0, 0] * nx * scale + matrix[0, 2]).astype(np.float32),
            (matrix[1, 1] * ny * scale + matrix[1, 2]).astype(np.float32))


def load_eye(path):
    data = open(path, "rb").read()
    # A frame saved straight from rtpjpegdepay has a generated header before the real JPEG.
    start = data.find(b"\xff\xd8", 2)
    image = cv2.imdecode(np.frombuffer(data[start:] if start > 0 else data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(image, EYE_SIZE, interpolation=cv2.INTER_AREA)


def vertical_errors(left, right, focal):
    orb = cv2.ORB_create(6000)
    left_points, left_features = orb.detectAndCompute(left, None)
    right_points, right_features = orb.detectAndCompute(right, None)
    rows = []
    for match in cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(left_features, right_features):
        lx, ly = left_points[match.queryIdx].pt
        rx, ry = right_points[match.trainIdx].pt
        if abs(ly - ry) < 25 and -5 < lx - rx < 250:
            rows.append(((lx - (left.shape[1] - 1) / 2) / focal, ly - ry))
    return np.array(rows)


def refine(profile, left, right):
    focal = RENDER_SIZE[0] / (2 * np.tan(np.radians(60)))
    for step in range(REFINE_STEPS):
        views = [cv2.remap(image, *sampling_map(profile[eye]), cv2.INTER_LINEAR)
                 for eye, image in (("left", left), ("right", right))]
        rows = vertical_errors(*views, focal)
        for _ in range(4):
            design = np.stack((np.ones(len(rows)), rows[:, 0]), 1)
            fit = np.linalg.lstsq(design, rows[:, 1], rcond=None)[0]
            error = rows[:, 1] - design @ fit
            rows = rows[np.abs(error) < max(1.0, 2.5 * np.std(error))]
        pitch, roll = fit[0] / focal, fit[1] / focal
        print(f"step {step}: {len(rows)} matches, pitch {np.degrees(pitch):.3f} deg, roll {np.degrees(roll):.3f} deg")
        for eye, sign in (("left", 1), ("right", -1)):
            correction = cv2.Rodrigues(np.array([sign * pitch / 2, 0.0, -sign * roll / 2]))[0]
            profile[eye]["rectification_rotation"] = (
                correction @ np.array(profile[eye]["rectification_rotation"])).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("output")
    args = parser.parse_args()
    raw = urllib.request.urlopen(CAM_PARAM_URL, timeout=8).read()
    text = raw.decode()
    left_eye, right_eye = read_section(text, "m_d_info_l"), read_section(text, "m_d_info_r")
    extrinsics = read_section(text, "r_t_info")
    baseline_m = read_section(text, "intin_info")["ty"] / 1000.0
    rotation = np.array([[extrinsics[f"r{row}{column}"] for column in (1, 2, 3)] for row in (1, 2, 3)])
    translation = np.array([extrinsics["tx"], extrinsics["ty"], extrinsics["tz"]])
    translation = translation / np.linalg.norm(translation) * baseline_m
    left_matrix, right_matrix = camera_matrix(left_eye), camera_matrix(right_eye)
    left_rectify, right_rectify = cv2.stereoRectify(
        left_matrix, rational_distortion(left_eye), right_matrix, rational_distortion(right_eye), RAW_SIZE,
        rotation, translation, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)[:2]
    profile = {
        "schema": "r1_head_optics_v1", "image_size": list(EYE_SIZE), "source_calibration_sha256": hashlib.sha256(raw).hexdigest(),
        "source_calibration_path": CAM_PARAM_URL, "source_document_created": "2026-11-02",
        "head_calibrated_at": "factory EEPROM, read 2026-11-02", "baseline_m": baseline_m,
        "pixel_convention": "OpenCV integer coordinates are pixel centres; normalized texture UV=(pixel+0.5)/size",
        "rectification_rotation_convention": "raw optical ray to rectified optical ray; inverse sampling uses transpose",
        "source_image_geometry": "1920x816 SBS from raw 1280x1088 RTP JPEG 5004/5005 scaled by 0.75; left then right; fisheye coefficients fitted to the factory rational model",
        "left": {"camera_matrix": scaled_matrix(left_matrix).tolist(), "distortion_coefficients": fit_fisheye(left_eye).tolist(),
                 "rectification_rotation": left_rectify.tolist()},
        "right": {"camera_matrix": scaled_matrix(right_matrix).tolist(), "distortion_coefficients": fit_fisheye(right_eye).tolist(),
                  "rectification_rotation": right_rectify.tolist()},
    }
    refine(profile, load_eye(args.left), load_eye(args.right))
    profile["rectification_note"] = "factory R and T, then pitch and roll refined from ORB matches on one static scene"
    with open(args.output, "w") as output:
        json.dump(profile, output, indent=2)


if __name__ == "__main__":
    main()
