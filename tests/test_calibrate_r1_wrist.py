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

from calibrate_r1_wrist_capture import (
    DEFAULT_PATTERN,
    WRIST_FRAME_SIZE,
    FrameShapeError,
    FrameStability,
    capture_day_dirs,
    detect_corners,
    inner_corner_count,
    next_image_index,
    overlay_label,
    parse_args as parse_capture_args,
    require_wrist_frame,
    run_capture_loop,
    save_image,
)
from calibrate_r1_wrist_fit import (
    DetectedView,
    FitResult,
    _initial_K,
    atomic_write_json,
    camera_entry,
    fit_detected_views,
    holdout_count,
    holdout_split,
    is_assets_calibration_path,
    load_detected_views,
    main as fit_main,
    merge_wrist_cameras,
    object_points,
    parse_args as parse_fit_args,
    pinhole_flags,
    quality_errors,
    refuse_assets_path,
    resolve_base_document,
    write_merged_calibration,
)
from teleop.utils.camera_calibration import camera_calibration_metadata, load_camera_calibration


PATTERN = DEFAULT_PATTERN
SQUARE = 0.030
EXPECTED = inner_corner_count(PATTERN)
HEAD_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def checkerboard_bgr(width=640, height=480, pattern=PATTERN, square_px=48, margin=40):
    image = np.full((height, width, 3), 240, dtype=np.uint8)
    cols, rows = pattern
    for row in range(rows + 1):
        for col in range(cols + 1):
            if (col + row) % 2 == 0:
                x0 = margin + col * square_px
                y0 = margin + row * square_px
                image[y0:y0 + square_px, x0:x0 + square_px] = 15
    return image


def fake_corners(pattern=PATTERN, ok=True):
    if not ok:
        return False, None
    cols, rows = pattern
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    points = (grid * 20.0 + 40.0).reshape(-1, 1, 2)
    return True, points


class ScriptedDetect:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = 0

    def __call__(self, image, pattern=PATTERN):
        ok = self.sequence[min(self.calls, len(self.sequence) - 1)]
        self.calls += 1
        return fake_corners(pattern, ok=bool(ok))


class AutoClock:
    def __init__(self, dt=0.05):
        self.t = 0.0
        self.dt = dt

    def __call__(self):
        self.t += self.dt
        return self.t


def scripted_wait_key(keys):
    iterator = iter(keys)

    def wait_key(delay):
        try:
            return next(iterator)
        except StopIteration:
            return ord("q")

    return wait_key


def in_image(points, width, height, margin=4):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.all(
        (points[:, 0] >= margin) & (points[:, 0] < width - margin)
        & (points[:, 1] >= margin) & (points[:, 1] < height - margin)
    )


def head_document():
    return {
        "schema": "r1_camera_calibration_v1",
        "created": "2026-09-20",
        "note": "DFOPTIX A3 10x7 squares, OpenCV (9,6) inner, 0.030 m; head fisheye @ 544x448; head stereo only this pass (wrists not included)",
        "board": {"type": "checkerboard", "squares_x": 9, "squares_y": 6, "square_size_m": 0.030},
        "cameras": {
            "head_left": {
                "image_size": [544, 448],
                "camera_matrix": [[131.59, 0.0, 287.48], [0.0, 130.56, 214.52], [0.0, 0.0, 1.0]],
                "distortion_model": "fisheye",
                "distortion_coefficients": [0.2485, 0.3494, 0.0, 0.0],
                "reprojection_error_px": 0.2545,
            },
            "head_right": {
                "image_size": [544, 448],
                "camera_matrix": [[131.91, 0.0, 286.69], [0.0, 131.07, 215.06], [0.0, 0.0, 1.0]],
                "distortion_model": "fisheye",
                "distortion_coefficients": [0.2479, 0.3462, 0.0, 0.0],
                "reprojection_error_px": 0.2396,
            },
        },
        "stereo": {
            "head": {
                "left": "head_left",
                "right": "head_right",
                "rotation": HEAD_IDENTITY,
                "translation_m": [-0.05909, -0.000096, 0.000869],
                "baseline_m": 0.0591,
                "epipolar_error_px": 0.2805,
            }
        },
    }


def synthetic_detected_views(count, image_size=WRIST_FRAME_SIZE, pattern=PATTERN,
                             square=SQUARE, fx=450.0, seed=0, noise=0.0):
    width, height = image_size
    K = np.array([[fx, 0.0, width / 2.0],
                  [0.0, fx, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([0.04, -0.015, 0.001, 0.0, 0.0], dtype=np.float64)
    obj = object_points(pattern, square)
    rng = np.random.default_rng(seed)
    views = []
    attempts = 0
    while len(views) < count and attempts < count * 40:
        attempts += 1
        rvec = np.array([
            rng.uniform(-0.30, 0.30),
            rng.uniform(-0.35, 0.35),
            rng.uniform(-0.18, 0.18),
        ], dtype=np.float64)
        tvec = np.array([
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.04, 0.04),
            rng.uniform(0.28, 0.65),
        ], dtype=np.float64)
        projected, _ = cv2.projectPoints(obj.reshape(-1, 3), rvec, tvec, K, D)
        if not in_image(projected, width, height):
            continue
        if noise:
            projected = projected + rng.normal(0.0, noise, size=projected.shape)
        index = len(views)
        views.append(DetectedView(
            index=index,
            path=Path(f"left_wrist/{index:03d}.png"),
            object_points=obj.copy(),
            image_points=np.asarray(projected, dtype=np.float64).reshape(-1, 1, 2),
            image_size=image_size,
        ))
    if len(views) < count:
        raise RuntimeError(f"could only synthesise {len(views)} of {count} in-image poses")
    return views


def write_head_json(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(head_document(), indent=2) + "\n", encoding="utf-8")
    return path


class DetectAndShapeTests(unittest.TestCase):
    def test_require_wrist_frame_accepts_640x480(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = require_wrist_frame(frame)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_wrong_frame_size_is_rejected(self):
        with self.assertRaisesRegex(FrameShapeError, "1088x448"):
            require_wrist_frame(np.zeros((448, 1088, 3), dtype=np.uint8))
        with self.assertRaisesRegex(FrameShapeError, "544x448"):
            require_wrist_frame(np.zeros((448, 544, 3), dtype=np.uint8))

    def test_detect_corners_finds_54_on_a_synthetic_board(self):
        image = checkerboard_bgr()
        ok, corners = detect_corners(image, PATTERN)
        self.assertTrue(ok)
        self.assertEqual(len(corners), EXPECTED)

    def test_detect_corners_rejects_blank(self):
        ok, corners = detect_corners(np.full((480, 640, 3), 180, dtype=np.uint8), PATTERN)
        self.assertFalse(ok)
        self.assertIsNone(corners)


class CaptureLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / "left_wrist"
        self.folder.mkdir()
        self.board = checkerboard_bgr()
        self.blank = np.full((480, 640, 3), 40, dtype=np.uint8)

    def test_saves_only_when_54_corners(self):
        frames = [self.blank, self.board, self.board]
        saved = run_capture_loop(
            frames, self.folder, pattern=PATTERN, stable_s=0.0,
            clock=AutoClock(0.05),
            wait_key=scripted_wait_key([ord(" "), ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            log=lambda *args, **kwargs: None,
        )
        self.assertEqual(saved, 1)
        self.assertTrue((self.folder / "000.png").is_file())
        image = cv2.imread(str(self.folder / "000.png"))
        self.assertEqual(image.shape, (480, 640, 3))

    def test_space_without_corners_does_not_write(self):
        saved = run_capture_loop(
            [self.blank] * 3, self.folder, pattern=PATTERN, stable_s=0.0,
            clock=AutoClock(0.05),
            wait_key=scripted_wait_key([ord(" "), ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            log=lambda *args, **kwargs: None,
        )
        self.assertEqual(saved, 0)
        self.assertEqual(list(self.folder.glob("*.png")), [])

    def test_stability_gate_blocks_immediate_save(self):
        detect = ScriptedDetect([True] * 20)
        messages = []
        saved = run_capture_loop(
            [self.blank] * 4, self.folder, pattern=PATTERN, stable_s=0.3,
            clock=AutoClock(0.05),
            wait_key=scripted_wait_key([ord(" "), ord(" "), ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            detect=detect, log=messages.append,
        )
        self.assertEqual(saved, 0)
        self.assertTrue(any("hold still" in str(item) for item in messages))

    def test_dir_with_000_png_resumes_at_001_without_clobber(self):
        save_image(self.folder, 0, self.board)
        original = (self.folder / "000.png").read_bytes()
        self.assertEqual(next_image_index(self.folder), 1)
        messages = []
        detect = ScriptedDetect([True] * 8)
        saved = run_capture_loop(
            [self.blank] * 8, self.folder, pattern=PATTERN, stable_s=0.3,
            clock=AutoClock(0.05),
            wait_key=scripted_wait_key([-1] * 7 + [ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            detect=detect, log=messages.append,
        )
        self.assertEqual(saved, 1)
        self.assertTrue(any("resuming at 001.png" in str(item) for item in messages))
        self.assertEqual((self.folder / "000.png").read_bytes(), original)
        self.assertTrue((self.folder / "001.png").is_file())
        self.assertNotEqual((self.folder / "001.png").read_bytes(), original)

    def test_save_image_refuses_to_overwrite_existing_index(self):
        save_image(self.folder, 0, self.board)
        original = (self.folder / "000.png").read_bytes()
        with self.assertRaises(FileExistsError):
            save_image(self.folder, 0, self.blank)
        self.assertEqual((self.folder / "000.png").read_bytes(), original)

    def test_overlay_mentions_ok_count_and_next_index(self):
        text = overlay_label(54, 3, True)
        self.assertIn("OK", text)
        self.assertIn("corners=54", text)
        self.assertIn("next=003", text)
        self.assertNotIn("OK", overlay_label(40, 0, False).split("corners")[0])

    def test_capture_cli_requires_side_and_defaults_match_field_card(self):
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parse_capture_args([])
        left = parse_capture_args(["--side", "left"])
        self.assertEqual(left.host, "192.168.124.147")
        self.assertEqual(left.port, 55556)
        self.assertEqual(left.topic, "left_wrist_camera")
        self.assertEqual(tuple(left.pattern), (9, 6))
        self.assertAlmostEqual(left.square, 0.030)
        self.assertEqual(left.out_root, Path("~/r1-cam-calib"))
        right = parse_capture_args(["--side", "right"])
        self.assertEqual(right.port, 55557)
        self.assertEqual(right.topic, "right_wrist_camera")

    def test_capture_cli_refuses_webrtc_ports(self):
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parse_capture_args(["--side", "left", "--port", "60002"])
            with self.assertRaises(SystemExit):
                parse_capture_args(["--side", "right", "--port", "60003"])

    def test_day_dirs_layout(self):
        day_dir, image_dir = capture_day_dirs(self.root, "left", day="20260920")
        self.assertEqual(image_dir, day_dir / "captures" / "left_wrist")
        _, right_dir = capture_day_dirs(self.root, "right", day="20260920")
        self.assertEqual(right_dir, day_dir / "captures" / "right_wrist")


class StabilityTests(unittest.TestCase):
    def test_frame_stability_requires_hold(self):
        gate = FrameStability(stable_s=0.3)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        self.assertFalse(gate.update(image, True, 0.0))
        self.assertFalse(gate.update(image, True, 0.2))
        self.assertTrue(gate.update(image, True, 0.3))
        self.assertFalse(gate.update(image, False, 0.4))
        self.assertFalse(gate.update(image, True, 0.5))


class HoldoutAndQualityTests(unittest.TestCase):
    def test_holdout_is_last_10_or_20_percent(self):
        self.assertEqual(holdout_count(40), 10)
        self.assertEqual(holdout_count(100), 20)
        train, hold = holdout_split(list(range(40)))
        self.assertEqual(train, list(range(30)))
        self.assertEqual(hold, list(range(30, 40)))

    def _result(self, n_train=30, rms=0.2, holdout_rms=0.3, image_size=(640, 480), fx=450.0):
        K = np.array([[fx, 0.0, 320.0], [0.0, fx, 240.0], [0.0, 0.0, 1.0]])
        return FitResult(
            side="left", camera="left_wrist", n_usable=n_train + 10, n_train=n_train,
            n_holdout=10, image_size=image_size, pattern=(9, 6), square=0.030,
            K=K, D=np.zeros((5, 1)), rms=rms, holdout_rms=holdout_rms,
            created="2026-09-20", note="test",
        )

    def test_quality_rejects_too_few_images(self):
        errors = quality_errors(self._result(n_train=8))
        self.assertTrue(any("training images" in item for item in errors))

    def test_quality_rejects_high_rms(self):
        errors = quality_errors(self._result(rms=1.4))
        self.assertTrue(any("train RMS" in item for item in errors))

    def test_quality_rejects_wrong_image_size(self):
        errors = quality_errors(self._result(image_size=(544, 448)))
        self.assertTrue(any("image_size" in item for item in errors))


class FitAndMergeTests(unittest.TestCase):
    def test_pinhole_fit_recovers_fx_on_synthetic_points(self):
        views = synthetic_detected_views(40, fx=450.0, seed=7)
        train, hold = holdout_split(views)
        self.assertEqual(len(hold), 10)
        self.assertEqual([view.index for view in hold], list(range(30, 40)))
        result = fit_detected_views(
            train, hold, "left", pattern=PATTERN, square=SQUARE,
            log=lambda *_args, **_kwargs: None)
        self.assertGreaterEqual(result.n_train, 20)
        self.assertLess(result.rms, 0.5)
        self.assertLess(result.holdout_rms, 0.7)
        self.assertAlmostEqual(float(result.K[0, 0]), 450.0, delta=15.0)
        self.assertEqual(tuple(result.image_size), (640, 480))
        self.assertEqual(np.asarray(result.D).reshape(-1).size, 5)
        self.assertEqual(quality_errors(result), [])

    def test_fit_cli_rejects_too_few_images_without_writing_json(self):
        with tempfile.TemporaryDirectory() as folder:
            day = Path(folder) / "20260920"
            left = day / "captures" / "left_wrist"
            left.mkdir(parents=True)
            write_head_json(day / "camera_calibration.json")
            board = checkerboard_bgr()
            for index in range(12):
                cv2.imwrite(str(left / f"{index:03d}.png"), board)
            original = (day / "camera_calibration.json").read_bytes()
            code = fit_main(["--day-dir", str(day), "--side", "left"])
            self.assertEqual(code, 2)
            self.assertEqual((day / "camera_calibration.json").read_bytes(), original)
            self.assertNotIn(b"left_wrist", original)

    def test_load_detected_views_picks_up_appended_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            left = root / "left_wrist"
            left.mkdir()
            board = checkerboard_bgr()
            cv2.imwrite(str(left / "000.png"), board)
            cv2.imwrite(str(left / "046.png"), board)
            views = load_detected_views(left, pattern=PATTERN, square=SQUARE, log=lambda *_a, **_k: None)
            self.assertEqual([view.index for view in views], [0, 46])

    def test_merge_preserves_head_stereo_and_loader_passes(self):
        views = synthetic_detected_views(40, fx=440.0, seed=3)
        train, hold = holdout_split(views)
        silent = lambda *_args, **_kwargs: None
        left = fit_detected_views(train, hold, "left", pattern=PATTERN, square=SQUARE, log=silent)
        right_views = synthetic_detected_views(40, fx=455.0, seed=11)
        right_train, right_hold = holdout_split(right_views)
        right = fit_detected_views(
            right_train, right_hold, "right", pattern=PATTERN, square=SQUARE, log=silent)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera_calibration.json"
            loaded = write_merged_calibration(head_document(), [left, right], path)
            self.assertEqual(
                sorted(loaded["cameras"]),
                ["head_left", "head_right", "left_wrist", "right_wrist"],
            )
            self.assertEqual(loaded["cameras"]["left_wrist"]["distortion_model"], "plumb_bob")
            self.assertEqual(loaded["cameras"]["left_wrist"]["image_size"], [640, 480])
            self.assertEqual(len(loaded["cameras"]["left_wrist"]["distortion_coefficients"]), 5)
            self.assertEqual(loaded["cameras"]["head_left"]["distortion_model"], "fisheye")
            self.assertEqual(loaded["stereo"]["head"]["left"], "head_left")
            self.assertEqual(loaded["stereo"]["head"]["translation_m"], [-0.05909, -0.000096, 0.000869])
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(set(raw) <= {"schema", "created", "note", "board", "cameras", "stereo", "hand_eye"})
            meta = camera_calibration_metadata(
                loaded, ["head_left", "head_right", "left_wrist", "right_wrist"])
            self.assertEqual(meta["status"], "calibrated")

    def test_one_wrist_keeps_status_partial(self):
        views = synthetic_detected_views(40, fx=440.0, seed=5)
        train, hold = holdout_split(views)
        silent = lambda *_args, **_kwargs: None
        left = fit_detected_views(train, hold, "left", pattern=PATTERN, square=SQUARE, log=silent)
        document = merge_wrist_cameras(head_document(), [left])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera_calibration.json"
            path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            loaded = load_camera_calibration(path)
            meta = camera_calibration_metadata(
                loaded, ["head_left", "head_right", "left_wrist", "right_wrist"])
            self.assertEqual(meta["status"], "partial")
            self.assertEqual(meta["missing_cameras"], ["right_wrist"])

    def test_fit_both_skips_empty_side_and_merges_the_other(self):
        with tempfile.TemporaryDirectory() as folder:
            day = Path(folder) / "20260920"
            left = day / "captures" / "left_wrist"
            left.mkdir(parents=True)
            write_head_json(day / "camera_calibration.json")
            assets = Path(folder) / "assets" / "r1" / "camera_calibration.json"
            write_head_json(assets)
            views = synthetic_detected_views(40, fx=442.0, seed=9)
            with mock.patch("calibrate_r1_wrist_fit.load_detected_views", return_value=views):
                code = fit_main(["--day-dir", str(day), "--side", "both", "--assets", str(assets)])
            self.assertEqual(code, 0)
            loaded = load_camera_calibration(day / "camera_calibration.json")
            self.assertIn("left_wrist", loaded["cameras"])
            self.assertNotIn("right_wrist", loaded["cameras"])
            self.assertEqual(loaded["cameras"]["left_wrist"]["image_size"], [640, 480])
            self.assertEqual(loaded["stereo"]["head"]["baseline_m"], 0.0591)
            self.assertEqual(
                camera_calibration_metadata(
                    loaded, ["head_left", "head_right", "left_wrist", "right_wrist"])["status"],
                "partial",
            )
            self.assertTrue((day / "camera_calibration.json.pre-wrist-fit").is_file())
            self.assertNotIn("left_wrist", (day / "camera_calibration.json.pre-wrist-fit").read_text(encoding="utf-8"))

    def test_prefers_day_dir_json_over_assets_and_never_writes_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            day = Path(folder) / "20260920"
            day.mkdir()
            (day / "captures" / "left_wrist").mkdir(parents=True)
            day_json = write_head_json(day / "camera_calibration.json")
            assets = Path(folder) / "xr" / "assets" / "r1" / "camera_calibration.json"
            write_head_json(assets)
            original_assets = assets.read_bytes()
            document, source = resolve_base_document(day, assets=assets)
            self.assertEqual(source, day_json)
            self.assertEqual(sorted(document["cameras"]), ["head_left", "head_right"])
            self.assertTrue(is_assets_calibration_path(assets))
            with self.assertRaisesRegex(Exception, "not assets"):
                refuse_assets_path(assets)
            with self.assertRaisesRegex(Exception, "not assets"):
                atomic_write_json(assets, document)
            self.assertEqual(assets.read_bytes(), original_assets)

    def test_fit_cli_defaults(self):
        args = parse_fit_args(["--day-dir", "/tmp/r1-cam-calib/20260920"])
        self.assertEqual(args.side, "both")
        self.assertEqual(tuple(args.pattern), (9, 6))
        self.assertAlmostEqual(args.square, 0.030)
        self.assertEqual(args.holdout, 10)

    def test_initial_K_is_never_identity_and_flags_are_plumb_bob(self):
        K = _initial_K((640, 480))
        self.assertGreater(K[0, 0], 80.0)
        self.assertAlmostEqual(K[0, 0], 0.9 * 480.0)
        self.assertAlmostEqual(K[0, 2], 320.0)
        self.assertAlmostEqual(K[1, 2], 240.0)
        self.assertNotAlmostEqual(K[0, 0], 1.0)
        flags = pinhole_flags()
        self.assertTrue(flags & cv2.CALIB_USE_INTRINSIC_GUESS)
        self.assertFalse(flags & cv2.CALIB_RATIONAL_MODEL)
        self.assertEqual(camera_entry((640, 480), K, np.zeros(5), 0.2)["distortion_model"], "plumb_bob")

    def test_missing_base_refuses_wrist_only_json(self):
        with tempfile.TemporaryDirectory() as folder:
            day = Path(folder) / "20260920"
            day.mkdir()
            (day / "captures" / "left_wrist").mkdir(parents=True)
            missing_assets = Path(folder) / "no-assets.json"
            with self.assertRaisesRegex(Exception, "head stereo"):
                resolve_base_document(day, assets=missing_assets)


if __name__ == "__main__":
    unittest.main()
