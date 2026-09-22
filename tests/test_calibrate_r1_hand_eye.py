#!/usr/bin/env python3
"""Unit tests for R1 hand-eye capture/fit (no robot motion, no DDS)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from calibrate_r1_hand_eye_capture import (  # noqa: E402
    StaticQSource,
    annotate_frame,
    build_sample_payload,
    run_capture_loop,
    save_sample,
)
from calibrate_r1_hand_eye_fit import (  # noqa: E402
    FitResult,
    Sample,
    calibrate_hand_eye,
    camera_needs_wrist_unmirror,
    head_right_from_left_and_stereo,
    hand_eye_entry,
    merge_hand_eye,
    quality_errors,
    refuse_assets_path,
    residual_stats,
    unmirror_wrist_frame,
)
from calibrate_r1_wrist_capture import detect_corners  # noqa: E402
from r1_hand_eye_common import (  # noqa: E402
    DEFAULT_PATTERN,
    DEFAULT_URDF,
    motor_q_to_pinocchio_q,
    object_points,
    pinocchio_q_to_named,
    rotation_geodesic_deg,
    rt_from_se3,
    se3_from_rt,
)
from teleop.utils.camera_calibration import load_camera_calibration  # noqa: E402

PATTERN = DEFAULT_PATTERN
SQUARE = 0.030
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def head_document():
    return {
        "schema": "r1_camera_calibration_v1",
        "created": "2026-09-20",
        "note": "head+wrists intrinsics",
        "board": {"type": "checkerboard", "squares_x": 9, "squares_y": 6, "square_size_m": 0.030},
        "cameras": {
            "head_left": {
                "image_size": [544, 448],
                "camera_matrix": [[131.59, 0.0, 287.48], [0.0, 130.56, 214.52], [0.0, 0.0, 1.0]],
                "distortion_model": "fisheye",
                "distortion_coefficients": [0.2485, 0.3494, 0.0, 0.0],
                "reprojection_error_px": 0.25,
            },
            "head_right": {
                "image_size": [544, 448],
                "camera_matrix": [[131.91, 0.0, 286.69], [0.0, 131.07, 215.06], [0.0, 0.0, 1.0]],
                "distortion_model": "fisheye",
                "distortion_coefficients": [0.2479, 0.3462, 0.0, 0.0],
                "reprojection_error_px": 0.24,
            },
            "left_wrist": {
                "image_size": [640, 480],
                "camera_matrix": [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
                "distortion_model": "plumb_bob",
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                "reprojection_error_px": 0.16,
            },
            "right_wrist": {
                "image_size": [640, 480],
                "camera_matrix": [[401.0, 0.0, 321.0], [0.0, 401.0, 241.0], [0.0, 0.0, 1.0]],
                "distortion_model": "plumb_bob",
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                "reprojection_error_px": 0.16,
            },
        },
        "stereo": {
            "head": {
                "left": "head_left",
                "right": "head_right",
                "rotation": IDENTITY,
                "translation_m": [-0.059, 0.0, 0.0],
                "baseline_m": 0.059,
            }
        },
        "hand_eye": {},
    }


def random_rotation(rng):
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    angle = float(rng.uniform(0.15, 0.9))
    return cv2.Rodrigues(axis * angle)[0]


def make_full_q(values=None):
    q = np.zeros(35, dtype=float)
    if values:
        for index, value in values.items():
            q[int(index)] = float(value)
    return q


class MotorMapTest(unittest.TestCase):
    def test_motor_q_maps_to_17_pinocchio_slots(self):
        full = make_full_q({13: 0.1, 29: -0.2, 30: 0.3, 15: 0.4, 28: -0.5})
        pin_q = motor_q_to_pinocchio_q(full)
        self.assertEqual(pin_q.shape, (17,))
        self.assertAlmostEqual(pin_q[0], 0.1)
        self.assertAlmostEqual(pin_q[1], -0.2)
        self.assertAlmostEqual(pin_q[2], 0.3)
        self.assertAlmostEqual(pin_q[3], 0.4)
        self.assertAlmostEqual(pin_q[16], -0.5)
        named = pinocchio_q_to_named(pin_q)
        self.assertEqual(named["waist_yaw"], 0.1)


class SyntheticHandEyeTest(unittest.TestCase):
    def test_calibrate_hand_eye_recovers_known_cam2gripper(self):
        rng = np.random.default_rng(7)
        R_true = random_rotation(rng)
        t_true = np.array([0.03, -0.01, 0.05], dtype=float)
        T_c2g = se3_from_rt(R_true, t_true)
        T_board = se3_from_rt(np.eye(3), np.array([0.4, 0.0, 0.3]))
        samples = []
        for index in range(24):
            R_g = random_rotation(rng)
            t_g = np.array([0.2, rng.uniform(-0.15, 0.15), 0.35 + 0.02 * index], dtype=float)
            T_g2b = se3_from_rt(R_g, t_g)
            T_t2c = np.linalg.inv(T_c2g) @ np.linalg.inv(T_g2b) @ T_board
            samples.append(Sample(
                index=index, path=Path(f"{index:03d}.png"), meta_path=Path(f"{index:03d}.json"),
                image_points=np.zeros((54, 1, 2)), object_points=object_points(),
                T_gripper2base=T_g2b, T_target2cam=T_t2c,
                camera="left_wrist", frame="left_wrist_yaw_link",
            ))
        R_est, t_est = calibrate_hand_eye(samples[:20], method="tsai")
        rot_err = rotation_geodesic_deg(R_true, R_est)
        trans_err = float(np.linalg.norm(t_true - t_est))
        self.assertLess(rot_err, 1.0)
        self.assertLess(trans_err, 0.005)
        rot_rms, trans_rms = residual_stats(samples[20:], R_est, t_est)
        self.assertLess(rot_rms, 1.0)
        self.assertLess(trans_rms, 0.01)


class MergeSchemaTest(unittest.TestCase):
    def test_merge_does_not_drop_head_intrinsics(self):
        base = head_document()
        head_left_before = json.loads(json.dumps(base["cameras"]["head_left"]))
        stereo_before = json.loads(json.dumps(base["stereo"]["head"]))
        result = FitResult(
            key="left_wrist", camera="left_wrist", frame="left_wrist_yaw_link",
            n_usable=20, n_train=16, n_holdout=4,
            R=np.eye(3), t=np.array([0.02, 0.0, 0.04]),
            rotation_rms_deg=0.5, translation_rms_m=0.002, method="tsai",
        )
        entry = hand_eye_entry(result)
        merged = merge_hand_eye(base, {"left_wrist": entry}, created="2026-09-22")
        self.assertEqual(merged["cameras"]["head_left"], head_left_before)
        self.assertEqual(merged["cameras"]["head_right"], base["cameras"]["head_right"])
        self.assertEqual(merged["stereo"]["head"], stereo_before)
        self.assertEqual(merged["hand_eye"]["left_wrist"]["frame"], "left_wrist_yaw_link")
        self.assertEqual(merged["cameras"]["left_wrist"]["camera_matrix"][0][0], 400.0)

    def test_head_right_composed_from_stereo(self):
        left = {
            "camera": "head_left",
            "frame": "head_yaw_link",
            "rotation": IDENTITY,
            "translation_m": [0.01, 0.02, 0.03],
            "rotation_rms_deg": 0.4,
            "translation_rms_m": 0.001,
        }
        stereo = {"rotation": IDENTITY, "translation_m": [-0.059, 0.0, 0.0]}
        right = head_right_from_left_and_stereo(left, stereo)
        self.assertEqual(right["camera"], "head_right")
        self.assertEqual(right["frame"], "head_yaw_link")
        np.testing.assert_allclose(right["translation_m"], [-0.049, 0.02, 0.03], atol=1e-9)

    def test_merged_json_loads_with_schema(self):
        base = head_document()
        result = FitResult(
            key="left_wrist", camera="left_wrist", frame="left_wrist_yaw_link",
            n_usable=20, n_train=16, n_holdout=4,
            R=np.eye(3), t=np.array([0.02, 0.0, 0.04]),
            rotation_rms_deg=0.5, translation_rms_m=0.002, method="tsai",
        )
        merged = merge_hand_eye(base, {"left_wrist": hand_eye_entry(result)})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera_calibration.json"
            path.write_text(json.dumps(merged), encoding="utf-8")
            loaded = load_camera_calibration(path)
            self.assertIn("left_wrist", loaded["hand_eye"])
            self.assertIn("head_left", loaded["cameras"])
            self.assertEqual(loaded["cameras"]["head_left"]["distortion_model"], "fisheye")

    def test_refuse_assets_path(self):
        with tempfile.TemporaryDirectory() as folder:
            assets = Path(folder) / "assets" / "r1" / "camera_calibration.json"
            assets.parent.mkdir(parents=True)
            assets.write_text("{}", encoding="utf-8")
            with self.assertRaises(Exception):
                refuse_assets_path(assets)


class CaptureLoopTest(unittest.TestCase):
    def test_space_saves_png_and_json_when_ok(self):
        image = np.full((480, 640, 3), 200, dtype=np.uint8)
        # Draw a crude board that detect may fail; inject detect instead.
        frames = [image, image, image, image]
        keys = [255, 255, ord(" "), ord("q")]
        key_iter = iter(keys)

        def wait_key(_delay):
            try:
                return next(key_iter)
            except StopIteration:
                return ord("q")

        class Clock:
            def __init__(self):
                self.t = 0.0

            def __call__(self):
                self.t += 0.2
                return self.t

        def fake_detect(_image, pattern=PATTERN):
            cols, rows = pattern
            grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
            return True, (grid * 20.0 + 40.0).reshape(-1, 1, 2)

        class FakeFK:
            def link_pose(self, pinocchio_q, frame_name):
                return se3_from_rt(np.eye(3), np.array([0.1, 0.2, 0.3]))

        with tempfile.TemporaryDirectory() as folder:
            q_source = StaticQSource(make_full_q({13: 0.0, 15: 0.1}))
            saved = run_capture_loop(
                frames, folder, "left_wrist", q_source, FakeFK(),
                pattern=PATTERN, square=SQUARE, stable_s=0.0,
                wait_key=wait_key, imshow=lambda *_: None, detect=fake_detect,
                clock=Clock(), log=lambda *_: None, window_name="test")
            self.assertEqual(saved, 1)
            self.assertTrue((Path(folder) / "000.png").is_file())
            meta = json.loads((Path(folder) / "000.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["camera"], "left_wrist")
            self.assertEqual(meta["frame"], "left_wrist_yaw_link")
            self.assertEqual(len(meta["pinocchio_q"]), 17)
            self.assertEqual(meta["T_link_in_root"]["translation_m"], [0.1, 0.2, 0.3])


@unittest.skipUnless(DEFAULT_URDF.is_file(), "r1_a7.urdf not in this tree")
class LiveUrdfFkTest(unittest.TestCase):
    def test_fk_returns_finite_wrist_pose(self):
        from r1_hand_eye_common import R1A7FK
        fk = R1A7FK(urdf_path=DEFAULT_URDF)
        q = np.zeros(17)
        q[3] = 0.2
        pose = fk.link_pose(q, "left_wrist_yaw_link")
        self.assertEqual(pose.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(pose)))
        head = fk.link_pose(q, "head_yaw_link")
        self.assertTrue(np.all(np.isfinite(head)))


class QualityGateTest(unittest.TestCase):
    def test_quality_errors_flag_large_residual(self):
        result = FitResult(
            key="left_wrist", camera="left_wrist", frame="left_wrist_yaw_link",
            n_usable=20, n_train=16, n_holdout=4,
            R=np.eye(3), t=np.zeros(3),
            rotation_rms_deg=9.0, translation_rms_m=0.001, method="tsai",
        )
        errors = quality_errors(result)
        self.assertTrue(any("rotation RMS" in item for item in errors))



class WristUnmirrorTest(unittest.TestCase):
    def test_camera_needs_wrist_unmirror(self):
        self.assertTrue(camera_needs_wrist_unmirror("left_wrist"))
        self.assertTrue(camera_needs_wrist_unmirror("right_wrist"))
        self.assertFalse(camera_needs_wrist_unmirror("head_left"))

    def test_unmirror_flips_image_and_cx(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :100] = 255  # bright strip on the left
        K = np.array([[400.0, 0.0, 300.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
        D = np.array([0.01, -0.02, 0.03, 0.04, 0.05])
        flipped, K2, D2 = unmirror_wrist_frame(image, K, D, "plumb_bob")
        self.assertEqual(flipped.shape, image.shape)
        self.assertGreater(int(flipped[:, -100:].mean()), int(flipped[:, :100].mean()))
        self.assertAlmostEqual(float(K2[0, 2]), 640 - 1 - 300.0)
        self.assertAlmostEqual(float(K2[1, 2]), 240.0)
        self.assertAlmostEqual(float(D2[2]), -0.03)
        self.assertAlmostEqual(float(D2[3]), 0.04)

    def test_hand_eye_entry_marks_wrist_mirror(self):
        result = FitResult(
            key="left_wrist", camera="left_wrist", frame="left_wrist_yaw_link",
            n_usable=20, n_train=16, n_holdout=4,
            R=np.eye(3), t=np.array([0.02, 0.0, 0.04]),
            rotation_rms_deg=0.5, translation_rms_m=0.002, method="tsai",
        )
        entry = hand_eye_entry(result)
        self.assertEqual(entry["image_mirror"], "horizontal")
        head = FitResult(
            key="head_left", camera="head_left", frame="head_yaw_link",
            n_usable=20, n_train=16, n_holdout=4,
            R=np.eye(3), t=np.array([0.06, 0.04, -0.02]),
            rotation_rms_deg=0.5, translation_rms_m=0.002, method="tsai",
        )
        self.assertNotIn("image_mirror", hand_eye_entry(head))

    def test_merge_marks_wrist_camera_mirror(self):
        base = head_document()
        result = FitResult(
            key="left_wrist", camera="left_wrist", frame="left_wrist_yaw_link",
            n_usable=20, n_train=16, n_holdout=4,
            R=np.eye(3), t=np.array([0.02, 0.0, 0.04]),
            rotation_rms_deg=0.5, translation_rms_m=0.002, method="tsai",
        )
        merged = merge_hand_eye(base, {"left_wrist": hand_eye_entry(result)})
        self.assertEqual(merged["cameras"]["left_wrist"]["image_mirror"], "horizontal")
        self.assertEqual(merged["cameras"]["right_wrist"]["image_mirror"], "horizontal")
        self.assertNotIn("image_mirror", merged["cameras"]["head_left"])
        self.assertEqual(merged["hand_eye"]["left_wrist"]["image_mirror"], "horizontal")


class RawWristProjectionHelperTest(unittest.TestCase):
    def test_helper_flips_exactly_once_and_projects_to_raw_pixels(self):
        from teleop.utils.camera_calibration import (
            prepare_image_for_hand_eye,
            project_link_points_to_pixels,
        )

        width, height = 640, 480
        cx_raw, cy = 300.0, 240.0
        fx = fy = 400.0
        camera_entry = {
            "image_size": [width, height],
            "camera_matrix": [[fx, 0.0, cx_raw], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "distortion_model": "plumb_bob",
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_mirror": "horizontal",
        }
        hand_eye = {
            "camera": "left_wrist",
            "frame": "left_wrist_yaw_link",
            "rotation": IDENTITY,
            "translation_m": [0.0, 0.0, 0.0],
            "image_mirror": "horizontal",
        }
        # Point 0.1 m to the +X of the (unmirrored) camera, 0.5 m ahead.
        point_cam = np.array([[0.10, 0.0, 0.50]])
        # With identity hand_eye, link == cam.
        cx_u = float(width - 1) - cx_raw
        u_unmirrored = fx * (0.10 / 0.50) + cx_u
        v_unmirrored = fy * (0.0 / 0.50) + cy
        u_raw_expected = float(width - 1) - u_unmirrored
        pixels = project_link_points_to_pixels(point_cam, hand_eye, camera_entry)
        self.assertEqual(pixels.shape, (1, 2))
        self.assertAlmostEqual(float(pixels[0, 0]), u_raw_expected, places=5)
        self.assertAlmostEqual(float(pixels[0, 1]), v_unmirrored, places=5)

        raw = np.zeros((height, width, 3), dtype=np.uint8)
        raw[:, :80] = 255
        once, K1, D1, model = prepare_image_for_hand_eye(raw, camera_entry, hand_eye)
        self.assertEqual(model, "plumb_bob")
        self.assertGreater(int(once[:, -80:].mean()), int(once[:, :80].mean()))
        self.assertAlmostEqual(float(K1[0, 2]), cx_u)
        # Applying the helper to an already-unmirrored frame would flip again —
        # callers must always pass the raw mirrored stream. Document that by
        # checking a second prepare on `once` without the mirror flag is identity.
        camera_no_mirror = dict(camera_entry)
        camera_no_mirror["image_mirror"] = "none"
        hand_no = dict(hand_eye)
        hand_no["image_mirror"] = "none"
        again, K2, _, _ = prepare_image_for_hand_eye(once, camera_no_mirror, hand_no)
        np.testing.assert_array_equal(again, once)
        # No-mirror path leaves the camera_matrix untouched (still the raw cx).
        self.assertAlmostEqual(float(K2[0, 2]), cx_raw)


if __name__ == "__main__":
    unittest.main()
