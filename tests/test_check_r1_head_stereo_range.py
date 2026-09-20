import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from check_r1_head_stereo_range import (
    DEFAULT_PATTERN,
    HEAD_EYE_SIZE,
    HOLD_TAPE_HINT,
    KNEW_REQUIRED,
    RangeError,
    RangeReading,
    attach_tape,
    format_snapshot,
    homogeneous_to_xyz,
    load_head_stereo_geometry,
    overlay_text,
    parse_args,
    projection_matrices,
    reconstructed_square_m,
    require_knew,
    resolve_calib_path,
    run_range_loop,
    stereo_midpoint_left,
    triangulate_corners,
    undistort_fisheye_image,
    undistort_fisheye_points,
)
from calibrate_r1_head_stereo_capture import inner_corner_count, split_head_stereo
from teleop.utils.camera_calibration import load_camera_calibration


PATTERN = DEFAULT_PATTERN
SQUARE = 0.030
EXPECTED = inner_corner_count(PATTERN)
HEAD_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def object_points(pattern=PATTERN, square=SQUARE):
    cols, rows = int(pattern[0]), int(pattern[1])
    points = np.zeros((cols * rows, 1, 3), dtype=np.float64)
    points[:, 0, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square)
    return points


def head_document(fx=132.0, baseline=0.0591, distortion=None, size=HEAD_EYE_SIZE):
    if distortion is None:
        distortion = [0.25, 0.35, 0.0, 0.0]
    width, height = size
    K = [[float(fx), 0.0, width / 2.0], [0.0, float(fx), height / 2.0], [0.0, 0.0, 1.0]]
    return {
        "schema": "r1_camera_calibration_v1",
        "created": "2026-09-20",
        "note": "unit test head fisheye stereo",
        "board": {"type": "checkerboard", "squares_x": 9, "squares_y": 6, "square_size_m": 0.030},
        "cameras": {
            "head_left": {
                "image_size": [int(width), int(height)],
                "camera_matrix": K,
                "distortion_model": "fisheye",
                "distortion_coefficients": list(distortion),
                "reprojection_error_px": 0.25,
            },
            "head_right": {
                "image_size": [int(width), int(height)],
                "camera_matrix": K,
                "distortion_model": "fisheye",
                "distortion_coefficients": list(distortion),
                "reprojection_error_px": 0.24,
            },
        },
        "stereo": {
            "head": {
                "left": "head_left",
                "right": "head_right",
                "rotation": HEAD_IDENTITY,
                "translation_m": [-float(baseline), 0.0, 0.0],
                "baseline_m": float(baseline),
                "epipolar_error_px": 0.28,
            }
        },
    }


def write_calib(folder, document=None):
    path = Path(folder) / "camera_calibration.json"
    path.write_text(json.dumps(document or head_document(), indent=2) + "\n", encoding="utf-8")
    return path


def in_image(points, width, height, margin=4):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.all(
        (points[:, 0] >= margin) & (points[:, 0] < width - margin)
        & (points[:, 1] >= margin) & (points[:, 1] < height - margin)
    )


def project_stereo_board(geometry, z_m=0.50, rvec=None, center_x=None, noise=0.0, seed=0,
                         pattern=PATTERN, square=SQUARE):
    obj = object_points(pattern, square)
    cols, rows = pattern
    board_center = np.array([(cols - 1) * square / 2.0, (rows - 1) * square / 2.0, 0.0])
    if center_x is None:
        center_x = float(geometry.baseline_m) / 2.0
    if rvec is None:
        rvec = np.zeros(3, dtype=np.float64)
    R_left, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t_left = np.array([center_x, 0.0, float(z_m)], dtype=np.float64) - R_left @ board_center
    t_left = t_left.reshape(3, 1)
    left, _ = cv2.fisheye.projectPoints(obj, rvec.reshape(3, 1), t_left, geometry.K_left, geometry.D_left)
    R_right = geometry.R @ R_left
    t_right = geometry.R @ t_left + geometry.T
    rvec_right, _ = cv2.Rodrigues(R_right)
    right, _ = cv2.fisheye.projectPoints(obj, rvec_right, t_right, geometry.K_right, geometry.D_right)
    left = left.reshape(-1, 1, 2).astype(np.float64)
    right = right.reshape(-1, 1, 2).astype(np.float64)
    if noise:
        rng = np.random.default_rng(seed)
        left = left + rng.normal(0.0, noise, left.shape)
        right = right + rng.normal(0.0, noise, right.shape)
    return left, right


def stereo_frame(left, right):
    return np.hstack([left, right])


def fake_corners(points):
    return True, np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


class ScriptedDetect:
    def __init__(self, left_points, right_points, ok_left=True, ok_right=True):
        self.left_points = left_points
        self.right_points = right_points
        self.ok_left = ok_left
        self.ok_right = ok_right
        self.calls = 0

    def __call__(self, image, pattern=PATTERN):
        side = self.calls % 2
        self.calls += 1
        if side == 0:
            if not self.ok_left:
                return False, None
            return fake_corners(self.left_points)
        if not self.ok_right:
            return False, None
        return fake_corners(self.right_points)


def scripted_wait_key(keys):
    iterator = iter(keys)

    def wait_key(delay):
        try:
            return next(iterator)
        except StopIteration:
            return ord("q")

    return wait_key


class KnewAndCliTests(unittest.TestCase):
    def test_require_knew_never_returns_none_and_copies_k(self):
        K = np.array([[132.0, 0.0, 272.0], [0.0, 132.0, 224.0], [0.0, 0.0, 1.0]])
        knew = require_knew(None, fallback=K)
        self.assertIsNotNone(knew)
        self.assertEqual(knew.shape, (3, 3))
        self.assertEqual(knew[0, 0], 132.0)
        K[0, 0] = 1.0
        self.assertEqual(knew[0, 0], 132.0)

    def test_undistort_image_rejects_none_knew(self):
        image = np.full((448, 544, 3), 180, dtype=np.uint8)
        K = np.array([[132.0, 0.0, 272.0], [0.0, 132.0, 224.0], [0.0, 0.0, 1.0]])
        D = np.array([[0.25], [0.35], [0.0], [0.0]])
        with self.assertRaises(RangeError) as ctx:
            undistort_fisheye_image(image, K, D, Knew=None)
        self.assertIn("black", str(ctx.exception))
        self.assertIn("Knew", str(ctx.exception))

    def test_opencv_undistort_image_knew_none_goes_black(self):
        image = np.zeros((448, 544, 3), dtype=np.uint8)
        image[:] = (40, 90, 180)
        image[174:274, 222:322] = 255
        K = np.array([[132.0, 0.0, 272.0], [0.0, 132.0, 224.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.array([[0.25], [0.35], [0.0], [0.0]], dtype=np.float64)
        raw_none = cv2.fisheye.undistortImage(image, K, D, Knew=None)
        self.assertLess(float(raw_none.mean()), 8.0)
        recovered = undistort_fisheye_image(image, K, D, Knew=K)
        self.assertGreater(float(recovered.mean()), 20.0)
        self.assertGreater(int(recovered[224, 272].max()), 80)

    def test_undistort_points_passes_concrete_knew(self):
        K = np.array([[132.0, 0.0, 272.0], [0.0, 132.0, 224.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
        pts = np.array([[[272.0, 224.0]]], dtype=np.float64)
        with mock.patch("check_r1_head_stereo_range.cv2.fisheye.undistortPoints",
                        wraps=cv2.fisheye.undistortPoints) as wrapped:
            out = undistort_fisheye_points(pts, K, D, Knew=K)
        self.assertEqual(wrapped.call_count, 1)
        kwargs = wrapped.call_args.kwargs
        positional = wrapped.call_args.args
        P = kwargs.get("P")
        if P is None and len(positional) >= 5:
            P = positional[4]
        self.assertIsNotNone(P)
        self.assertTrue(np.allclose(P, K))
        self.assertEqual(out.shape[-1], 2)

    def test_cli_defaults_match_the_field_card(self):
        args = parse_args([])
        self.assertEqual(str(args.calib), "assets/r1/camera_calibration.json")
        self.assertEqual(args.host, "192.168.124.147")
        self.assertEqual(args.port, 55555)
        self.assertEqual(args.topic, "head_camera")
        self.assertEqual(tuple(args.pattern), (9, 6))
        self.assertAlmostEqual(args.square, 0.030)
        self.assertIsNone(args.tape)

    def test_cli_tape_and_rejects_webrtc_ports(self):
        args = parse_args(["--tape", "0.50", "--calib", "/tmp/x.json"])
        self.assertAlmostEqual(args.tape, 0.50)
        with self.assertRaises(SystemExit):
            parse_args(["--port", "60001"])
        with self.assertRaises(SystemExit):
            parse_args(["--port", "60002"])
        with self.assertRaises(SystemExit):
            parse_args(["--tape", "0"])


class GeometryLoadTests(unittest.TestCase):
    def test_load_uses_k_as_knew_and_preserves_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_calib(folder, head_document(fx=131.59, baseline=0.0591))
            loaded = load_camera_calibration(path)
            self.assertEqual(loaded["stereo"]["head"]["left"], "head_left")
            geometry = load_head_stereo_geometry(path)
            self.assertIsNotNone(geometry.Knew_left)
            self.assertTrue(np.allclose(geometry.Knew_left, geometry.K_left))
            self.assertAlmostEqual(geometry.baseline_m, 0.0591, places=4)
            self.assertEqual(geometry.image_size, (544, 448))
            self.assertAlmostEqual(geometry.K_left[0, 0], 131.59, places=2)

    def test_load_rejects_missing_stereo_and_wrong_model(self):
        with tempfile.TemporaryDirectory() as folder:
            document = head_document()
            del document["stereo"]
            path = write_calib(folder, document)
            with self.assertRaises(RangeError):
                load_head_stereo_geometry(path)
            document = head_document()
            document["cameras"]["head_left"]["distortion_model"] = "plumb_bob"
            document["cameras"]["head_left"]["distortion_coefficients"] = [0.0, 0.0, 0.0, 0.0, 0.0]
            path = write_calib(folder, document)
            with self.assertRaises(RangeError) as ctx:
                load_head_stereo_geometry(path)
            self.assertIn("fisheye", str(ctx.exception))

    def test_resolve_calib_path_finds_relative_under_repo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dest = root / "assets" / "r1"
            dest.mkdir(parents=True)
            path = dest / "camera_calibration.json"
            path.write_text("{}\n", encoding="utf-8")
            found = resolve_calib_path("assets/r1/camera_calibration.json", root=root)
            self.assertEqual(found, path.resolve())
            with self.assertRaises(RangeError):
                resolve_calib_path("missing.json", root=root)


class TriangulationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = write_calib(self.temporary.name, head_document(fx=132.0, baseline=0.0591))
        self.geometry = load_head_stereo_geometry(self.path)

    def test_synthetic_board_at_half_metre_recovers_median_z(self):
        left, right = project_stereo_board(self.geometry, z_m=0.50)
        self.assertTrue(in_image(left, 544, 448))
        self.assertTrue(in_image(right, 544, 448))
        reading = triangulate_corners(left, right, self.geometry, pattern=PATTERN, tape=0.50)
        self.assertAlmostEqual(reading.z_med_m, 0.50, delta=0.005)
        self.assertAlmostEqual(reading.mid_plane_m, 0.50, delta=0.008)
        self.assertAlmostEqual(reading.err_m, reading.mid_plane_m - 0.50, places=9)
        self.assertLess(abs(reading.err_m), 0.005)
        self.assertLess(abs(reading.err_pct), 1.0)
        self.assertLess(reading.left_rms_px, 0.15)
        self.assertLess(reading.right_rms_px, 0.15)
        self.assertAlmostEqual(reading.square_med_m, SQUARE, delta=0.001)
        self.assertGreater(reading.disparity_med_px, 8.0)
        self.assertEqual(reading.n_valid, EXPECTED)

    def test_synthetic_board_at_one_metre(self):
        left, right = project_stereo_board(self.geometry, z_m=1.00)
        reading = triangulate_corners(left, right, self.geometry, pattern=PATTERN, tape=1.00)
        self.assertAlmostEqual(reading.z_med_m, 1.00, delta=0.012)
        self.assertLess(abs(reading.err_pct), 1.5)

    def test_noisy_corners_stay_within_a_few_percent(self):
        left, right = project_stereo_board(self.geometry, z_m=0.50, noise=0.3, seed=3)
        reading = triangulate_corners(left, right, self.geometry, pattern=PATTERN, tape=0.50)
        self.assertAlmostEqual(reading.z_med_m, 0.50, delta=0.025)
        self.assertLess(abs(reading.err_pct), 5.0)

    def test_triangulate_uses_knew_not_none(self):
        left, right = project_stereo_board(self.geometry, z_m=0.60)
        with mock.patch("check_r1_head_stereo_range.undistort_fisheye_points",
                        wraps=undistort_fisheye_points) as wrapped:
            triangulate_corners(left, right, self.geometry, pattern=PATTERN)
        self.assertGreaterEqual(wrapped.call_count, 2)
        for call in wrapped.call_args_list:
            knew = call.args[3] if len(call.args) >= 4 else call.kwargs.get("Knew")
            self.assertIsNotNone(knew)
            self.assertEqual(np.asarray(knew).shape, (3, 3))

    def test_projection_matrices_are_k_times_extrinsics(self):
        P1, P2 = projection_matrices(
            self.geometry.Knew_left, self.geometry.Knew_right,
            self.geometry.R, self.geometry.T)
        self.assertEqual(P1.shape, (3, 4))
        self.assertTrue(np.allclose(P1[:, :3], self.geometry.Knew_left))
        self.assertTrue(np.allclose(P2[:, :3], self.geometry.Knew_right @ self.geometry.R))

    def test_homogeneous_divide_and_square_spacing(self):
        hom = np.array([[1.0, 2.0, 4.0, 2.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64).T
        xyz = homogeneous_to_xyz(hom)
        self.assertAlmostEqual(xyz[0, 2], 2.0)
        self.assertTrue(np.isnan(xyz[1, 0]))
        pts = object_points().reshape(-1, 3)
        pts[:, 2] = 0.5
        self.assertAlmostEqual(reconstructed_square_m(pts, PATTERN), SQUARE, places=6)

    def test_snapshot_line_includes_tape_hint(self):
        reading = RangeReading(
            z_med_m=0.512, z_p25_m=0.508, z_p75_m=0.516, n_valid=54, n_corners=54,
            left_rms_px=0.28, right_rms_px=0.31, disparity_med_px=15.4,
            square_med_m=0.0302, mid_plane_m=0.510, xyz=np.zeros((54, 3)),
        )
        line = format_snapshot(reading, tape=0.50)
        self.assertIn("SNAP", line)
        self.assertIn("mid_plane=0.510 m", line)
        self.assertIn("z_med=0.512 m (left-cam)", line)
        self.assertIn("tape=0.500 m", line)
        self.assertIn("err=+10 mm", line)
        self.assertIn(HOLD_TAPE_HINT, line)
        self.assertIn("+2.0%", line)
        self.assertIn("compare mid_plane", HOLD_TAPE_HINT)

    def test_attach_tape_uses_mid_plane_not_left_z(self):
        reading = RangeReading(
            z_med_m=0.443, z_p25_m=0.44, z_p75_m=0.45, n_valid=54, n_corners=54,
            left_rms_px=0.2, right_rms_px=0.2, disparity_med_px=15.0,
            square_med_m=0.03, mid_plane_m=0.499, xyz=np.zeros((1, 3)),
        )
        attach_tape(reading, 0.50)
        self.assertAlmostEqual(reading.err_m, -0.001)
        self.assertAlmostEqual(reading.err_pct, -0.2)
        primary, secondary = overlay_text(reading, EXPECTED, 54, 54, tape=0.50)
        self.assertIn("mid_plane=0.499 m", primary)
        self.assertIn("err=-1mm", primary)
        self.assertNotIn("err=-57mm", primary)
        self.assertIn("z_med=0.443 m (left-cam)", secondary)
        self.assertNotIn("z_med", primary)

    def test_midpoint_is_half_baseline_for_parallel_cameras(self):
        mid = stereo_midpoint_left(self.geometry.R, self.geometry.T)
        self.assertAlmostEqual(mid[0], 0.0591 / 2.0, places=4)
        self.assertAlmostEqual(mid[2], 0.0, places=4)


class RangeLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.geometry = load_head_stereo_geometry(
            write_calib(self.temporary.name, head_document(fx=132.0, baseline=0.0591)))
        self.left_pts, self.right_pts = project_stereo_board(self.geometry, z_m=0.50)
        self.blank = np.full((448, 544, 3), 40, dtype=np.uint8)

    def test_space_snapshots_median_z_and_q_quits(self):
        detect = ScriptedDetect(self.left_pts, self.right_pts)
        messages = []
        readings = run_range_loop(
            [stereo_frame(self.blank, self.blank)] * 3,
            self.geometry, pattern=PATTERN, tape=0.50,
            wait_key=scripted_wait_key([ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            detect=detect, log=messages.append,
        )
        self.assertEqual(len(readings), 1)
        self.assertAlmostEqual(readings[0].z_med_m, 0.50, delta=0.01)
        self.assertTrue(any("SNAP" in str(item) and "mid_plane=" in str(item) for item in messages))
        self.assertTrue(any("left-cam" in str(item) for item in messages))
        self.assertTrue(any("NOT e-stop" in str(item) for item in messages))

    def test_space_without_both_eyes_does_not_snapshot(self):
        detect = ScriptedDetect(self.left_pts, self.right_pts, ok_right=False)
        messages = []
        readings = run_range_loop(
            [stereo_frame(self.blank, self.blank)] * 2,
            self.geometry, pattern=PATTERN,
            wait_key=scripted_wait_key([ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            detect=detect, log=messages.append,
        )
        self.assertEqual(readings, [])
        self.assertTrue(any("not snapped" in str(item) for item in messages))

    def test_split_layout_matches_live_jpeg(self):
        left = np.full((448, 544, 3), 10, dtype=np.uint8)
        right = np.full((448, 544, 3), 200, dtype=np.uint8)
        split_left, split_right = split_head_stereo(stereo_frame(left, right))
        self.assertEqual(split_left.shape, (448, 544, 3))
        self.assertEqual(int(split_left[0, 0, 0]), 10)
        self.assertEqual(int(split_right[0, 0, 0]), 200)


if __name__ == "__main__":
    unittest.main()
