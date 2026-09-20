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

from calibrate_r1_head_stereo_capture import (
    DEFAULT_PATTERN,
    HEAD_EYE_SIZE,
    HEAD_FRAME_SIZE,
    FrameShapeError,
    PairStability,
    capture_day_dirs,
    detect_corners,
    inner_corner_count,
    next_pair_index,
    overlay_label,
    parse_args as parse_capture_args,
    run_capture_loop,
    save_pair,
    split_head_stereo,
)
from calibrate_r1_head_stereo_fit import (
    DetectedPair,
    FitResult,
    _as_image_views,
    _as_object_views,
    _initial_K,
    build_document,
    calibrate_fisheye_mono,
    canonicalize_corners,
    fit_detected_pairs,
    fisheye_flags,
    holdout_count,
    holdout_split,
    load_detected_pairs,
    main as fit_main,
    object_points,
    parse_args as parse_fit_args,
    quality_errors,
    select_train_pairs,
    write_calibration,
)
from teleop.utils.camera_calibration import camera_calibration_metadata, load_camera_calibration


PATTERN = DEFAULT_PATTERN
SQUARE = 0.030
EXPECTED = inner_corner_count(PATTERN)


def checkerboard_bgr(width=544, height=448, pattern=PATTERN, square_px=40, margin=32):
    image = np.full((height, width, 3), 240, dtype=np.uint8)
    cols, rows = pattern
    for row in range(rows + 1):
        for col in range(cols + 1):
            if (col + row) % 2 == 0:
                x0 = margin + col * square_px
                y0 = margin + row * square_px
                image[y0:y0 + square_px, x0:x0 + square_px] = 15
    return image


def stereo_frame(left, right):
    return np.hstack([left, right])


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
        pair = self.sequence[min(self.calls // 2, len(self.sequence) - 1)]
        side = self.calls % 2
        self.calls += 1
        return fake_corners(pattern, ok=bool(pair[side]))


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


def synthetic_detected_pairs(count, baseline=0.060, seed=0, image_size=HEAD_EYE_SIZE,
                             pattern=PATTERN, square=SQUARE, noise=0.0):
    width, height = image_size
    K = np.array([[220.0, 0.0, width / 2.0],
                  [0.0, 220.0, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([[0.02], [-0.01], [0.0], [0.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    T = np.array([[-float(baseline)], [0.0], [0.0]], dtype=np.float64)
    obj = object_points(pattern, square)
    rng = np.random.default_rng(seed)
    pairs = []
    attempts = 0
    while len(pairs) < count and attempts < count * 40:
        attempts += 1
        rvec = np.array([
            rng.uniform(-0.35, 0.35),
            rng.uniform(-0.40, 0.40),
            rng.uniform(-0.20, 0.20),
        ], dtype=np.float64)
        tvec = np.array([
            rng.uniform(-0.06, 0.06),
            rng.uniform(-0.05, 0.05),
            rng.uniform(0.50, 0.95),
        ], dtype=np.float64)
        left, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
        R_left, _ = cv2.Rodrigues(rvec)
        R_right = R @ R_left
        t_right = (R @ tvec.reshape(3, 1) + T).reshape(3)
        rvec_right, _ = cv2.Rodrigues(R_right)
        right, _ = cv2.fisheye.projectPoints(obj, rvec_right, t_right, K, D)
        if not in_image(left, width, height) or not in_image(right, width, height):
            continue
        if noise:
            left = left + rng.normal(0.0, noise, size=left.shape)
            right = right + rng.normal(0.0, noise, size=right.shape)
        index = len(pairs)
        pairs.append(DetectedPair(
            index=index,
            left_path=Path(f"head_left/{index:03d}.png"),
            right_path=Path(f"head_right/{index:03d}.png"),
            object_points=obj.copy(),
            left_points=np.asarray(left, dtype=np.float64).reshape(-1, 1, 2),
            right_points=np.asarray(right, dtype=np.float64).reshape(-1, 1, 2),
            image_size=image_size,
        ))
    if len(pairs) < count:
        raise RuntimeError(f"could only synthesise {len(pairs)} of {count} in-image poses")
    return pairs


class SplitAndDetectTests(unittest.TestCase):
    def test_split_is_hard_544_not_half_of_something_else(self):
        frame = np.zeros((448, 1088, 3), dtype=np.uint8)
        frame[:, :544] = (0, 0, 255)
        frame[:, 544:] = (0, 255, 0)
        left, right = split_head_stereo(frame)
        self.assertEqual(left.shape, (448, 544, 3))
        self.assertEqual(right.shape, (448, 544, 3))
        self.assertTrue(np.all(left[:, :, 2] == 255))
        self.assertTrue(np.all(right[:, :, 1] == 255))

    def test_wrong_frame_size_is_rejected(self):
        with self.assertRaisesRegex(FrameShapeError, "640x480"):
            split_head_stereo(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_detect_corners_finds_54_on_a_synthetic_board(self):
        image = checkerboard_bgr()
        ok, corners = detect_corners(image, PATTERN)
        self.assertTrue(ok)
        self.assertEqual(len(corners), EXPECTED)

    def test_detect_corners_rejects_blank(self):
        ok, corners = detect_corners(np.full((448, 544, 3), 180, dtype=np.uint8), PATTERN)
        self.assertFalse(ok)
        self.assertIsNone(corners)


class CaptureLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.left = self.root / "head_left"
        self.right = self.root / "head_right"
        self.left.mkdir()
        self.right.mkdir()
        self.board = checkerboard_bgr()
        self.blank = np.full((448, 544, 3), 40, dtype=np.uint8)

    def test_saves_only_when_both_eyes_have_54_corners(self):
        frames = [
            stereo_frame(self.board, self.blank),
            stereo_frame(self.board, self.board),
            stereo_frame(self.board, self.board),
        ]
        saved = run_capture_loop(
            frames, self.left, self.right, pattern=PATTERN, stable_s=0.0,
            clock=AutoClock(0.05),
            wait_key=scripted_wait_key([ord(" "), ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            log=lambda *args, **kwargs: None,
        )
        self.assertEqual(saved, 1)
        self.assertTrue((self.left / "000.png").is_file())
        self.assertTrue((self.right / "000.png").is_file())
        left = cv2.imread(str(self.left / "000.png"))
        right = cv2.imread(str(self.right / "000.png"))
        self.assertEqual(left.shape, (448, 544, 3))
        self.assertEqual(right.shape, (448, 544, 3))

    def test_space_on_single_eye_does_not_write(self):
        saved = run_capture_loop(
            [stereo_frame(self.board, self.blank)] * 3, self.left, self.right,
            pattern=PATTERN, stable_s=0.0, clock=AutoClock(0.05),
            wait_key=scripted_wait_key([ord(" "), ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            log=lambda *args, **kwargs: None,
        )
        self.assertEqual(saved, 0)
        self.assertEqual(list(self.left.glob("*.png")), [])
        self.assertEqual(list(self.right.glob("*.png")), [])

    def test_stability_gate_blocks_immediate_save(self):
        detect = ScriptedDetect([(True, True)] * 20)
        messages = []
        saved = run_capture_loop(
            [stereo_frame(self.blank, self.blank)] * 4, self.left, self.right,
            pattern=PATTERN, stable_s=0.3, clock=AutoClock(0.05),
            wait_key=scripted_wait_key([ord(" "), ord(" "), ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            detect=detect, log=messages.append,
        )
        self.assertEqual(saved, 0)
        self.assertTrue(any("hold still" in str(item) for item in messages))

    def test_same_index_is_used_for_both_eyes_and_continues(self):
        save_pair(self.left, self.right, 0, self.board, self.board)
        self.assertEqual(next_pair_index(self.left, self.right), 1)
        detect = ScriptedDetect([(True, True)] * 8)
        saved = run_capture_loop(
            [stereo_frame(self.blank, self.blank)] * 8, self.left, self.right,
            pattern=PATTERN, stable_s=0.3, clock=AutoClock(0.05),
            wait_key=scripted_wait_key([-1] * 7 + [ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            detect=detect, log=lambda *args, **kwargs: None,
        )
        self.assertEqual(saved, 1)
        self.assertTrue((self.left / "001.png").is_file())
        self.assertTrue((self.right / "001.png").is_file())
        self.assertFalse((self.left / "002.png").exists())

    def test_dir_with_000_png_resumes_at_001_without_clobber(self):
        save_pair(self.left, self.right, 0, self.board, self.board)
        original_left = (self.left / "000.png").read_bytes()
        original_right = (self.right / "000.png").read_bytes()
        self.assertEqual(next_pair_index(self.left, self.right), 1)
        messages = []
        detect = ScriptedDetect([(True, True)] * 8)
        saved = run_capture_loop(
            [stereo_frame(self.blank, self.blank)] * 8, self.left, self.right,
            pattern=PATTERN, stable_s=0.3, clock=AutoClock(0.05),
            wait_key=scripted_wait_key([-1] * 7 + [ord(" "), ord("q")]),
            imshow=lambda *args, **kwargs: None,
            detect=detect, log=messages.append,
        )
        self.assertEqual(saved, 1)
        self.assertTrue(any("resuming at 001.png" in str(item) for item in messages))
        self.assertEqual((self.left / "000.png").read_bytes(), original_left)
        self.assertEqual((self.right / "000.png").read_bytes(), original_right)
        self.assertTrue((self.left / "001.png").is_file())
        self.assertTrue((self.right / "001.png").is_file())
        self.assertNotEqual((self.left / "001.png").read_bytes(), original_left)

    def test_save_pair_refuses_to_overwrite_existing_index(self):
        save_pair(self.left, self.right, 0, self.board, self.board)
        original = (self.left / "000.png").read_bytes()
        with self.assertRaises(FileExistsError):
            save_pair(self.left, self.right, 0, self.blank, self.blank)
        self.assertEqual((self.left / "000.png").read_bytes(), original)
        self.assertEqual((self.right / "000.png").read_bytes(), original)

    def test_overlay_mentions_pair_ok_and_counts(self):
        text = overlay_label(54, 54, 3, True)
        self.assertIn("PAIR OK", text)
        self.assertIn("L=54", text)
        self.assertIn("R=54", text)
        self.assertIn("saved=3", text)
        self.assertNotIn("PAIR OK", overlay_label(54, 0, 0, False))

    def test_capture_cli_defaults_match_the_field_card(self):
        args = parse_capture_args([])
        self.assertEqual(args.host, "192.168.124.147")
        self.assertEqual(args.port, 55555)
        self.assertEqual(tuple(args.pattern), (9, 6))
        self.assertAlmostEqual(args.square, 0.030)
        self.assertEqual(args.out_root, Path("~/r1-cam-calib"))
        self.assertEqual(args.stable_ms, 300)

    def test_capture_cli_refuses_webrtc_ports(self):
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parse_capture_args(["--port", "60002"])

    def test_day_dirs_layout(self):
        day_dir, left_dir, right_dir = capture_day_dirs(self.root, day="20260920")
        self.assertEqual(left_dir, day_dir / "captures" / "head_left")
        self.assertEqual(right_dir, day_dir / "captures" / "head_right")


class StabilityTests(unittest.TestCase):
    def test_pair_stability_requires_hold(self):
        gate = PairStability(stable_s=0.3)
        left = np.zeros((4, 4, 3), dtype=np.uint8)
        right = left.copy()
        self.assertFalse(gate.update(left, right, True, True, 0.0))
        self.assertFalse(gate.update(left, right, True, True, 0.2))
        self.assertTrue(gate.update(left, right, True, True, 0.3))
        self.assertFalse(gate.update(left, right, True, False, 0.4))
        self.assertFalse(gate.update(left, right, True, True, 0.5))


class HoldoutAndQualityTests(unittest.TestCase):
    def test_holdout_is_last_10_or_20_percent(self):
        self.assertEqual(holdout_count(40), 10)
        self.assertEqual(holdout_count(100), 20)
        train, hold = holdout_split(list(range(40)))
        self.assertEqual(train, list(range(30)))
        self.assertEqual(hold, list(range(30, 40)))

    def test_quality_rejects_too_few_pairs_and_insane_baseline(self):
        result = FitResult(
            n_usable=12, n_train=8, n_holdout=4, image_size=(544, 448),
            pattern=(9, 6), square=0.030,
            K_left=np.eye(3), D_left=np.zeros((4, 1)),
            K_right=np.eye(3), D_right=np.zeros((4, 1)),
            R=np.eye(3), T=np.array([-0.12, 0.0, 0.0]), baseline_m=0.12,
            left_rms=0.2, right_rms=0.2, stereo_rms=0.3,
            holdout_left_rms=0.4, holdout_right_rms=0.4, holdout_stereo_rms=0.5,
            created="2026-09-20", note="test",
        )
        errors = quality_errors(result)
        self.assertTrue(any("training pairs" in item for item in errors))
        self.assertTrue(any("baseline" in item for item in errors))

    def test_quality_rejects_high_rms(self):
        result = FitResult(
            n_usable=40, n_train=30, n_holdout=10, image_size=(544, 448),
            pattern=(9, 6), square=0.030,
            K_left=np.eye(3), D_left=np.zeros((4, 1)),
            K_right=np.eye(3), D_right=np.zeros((4, 1)),
            R=np.eye(3), T=np.array([-0.06, 0.0, 0.0]), baseline_m=0.06,
            left_rms=0.2, right_rms=0.2, stereo_rms=1.4,
            holdout_left_rms=0.3, holdout_right_rms=0.3, holdout_stereo_rms=0.4,
            created="2026-09-20", note="test",
        )
        errors = quality_errors(result)
        self.assertTrue(any("stereo train RMS" in item for item in errors))


class FitAndSchemaTests(unittest.TestCase):
    def test_head_only_json_passes_the_production_loader(self):
        result = FitResult(
            n_usable=40, n_train=30, n_holdout=10, image_size=(544, 448),
            pattern=(9, 6), square=0.030,
            K_left=np.array([[180.0, 0.0, 272.0], [0.0, 180.0, 224.0], [0.0, 0.0, 1.0]]),
            D_left=np.array([0.01, -0.02, 0.0, 0.0]),
            K_right=np.array([[181.0, 0.0, 271.0], [0.0, 181.0, 225.0], [0.0, 0.0, 1.0]]),
            D_right=np.array([0.012, -0.018, 0.0, 0.0]),
            R=np.eye(3), T=np.array([-0.060, 0.001, 0.0]), baseline_m=0.060,
            left_rms=0.21, right_rms=0.22, stereo_rms=0.31,
            holdout_left_rms=0.40, holdout_right_rms=0.41, holdout_stereo_rms=0.44,
            created="2026-09-20", note="unit test",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera_calibration.json"
            loaded = write_calibration(result, path)
            self.assertEqual(sorted(loaded["cameras"]), ["head_left", "head_right"])
            self.assertEqual(loaded["board"]["squares_x"], 9)
            self.assertEqual(loaded["board"]["squares_y"], 6)
            self.assertEqual(loaded["cameras"]["head_left"]["distortion_model"], "fisheye")
            self.assertEqual(loaded["stereo"]["head"]["left"], "head_left")
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(raw), {"schema", "created", "note", "board", "cameras", "stereo"})
            meta = camera_calibration_metadata(
                loaded, ["head_left", "head_right", "left_wrist", "right_wrist"])
            self.assertEqual(meta["status"], "partial")

    def test_fisheye_fit_recovers_baseline_on_synthetic_points(self):
        pairs = synthetic_detected_pairs(40, baseline=0.060, seed=7)
        train, hold = holdout_split(pairs)
        self.assertEqual(len(hold), 10)
        self.assertEqual([pair.index for pair in hold], list(range(30, 40)))
        result = fit_detected_pairs(
            train, hold, pattern=PATTERN, square=SQUARE, log=lambda *_args, **_kwargs: None)
        self.assertGreaterEqual(result.n_train, 20)
        self.assertLess(result.left_rms, 0.5)
        self.assertLess(result.right_rms, 0.5)
        self.assertLess(result.stereo_rms, 0.5)
        self.assertTrue(0.04 <= result.baseline_m <= 0.08)
        self.assertEqual(quality_errors(result), [])

    def test_fit_cli_rejects_too_few_pairs_without_writing_json(self):
        with tempfile.TemporaryDirectory() as folder:
            day = Path(folder) / "20260920"
            left = day / "captures" / "head_left"
            right = day / "captures" / "head_right"
            left.mkdir(parents=True)
            right.mkdir(parents=True)
            board = checkerboard_bgr()
            for index in range(12):
                cv2.imwrite(str(left / f"{index:03d}.png"), board)
                cv2.imwrite(str(right / f"{index:03d}.png"), board)
            code = fit_main(["--day-dir", str(day)])
            self.assertEqual(code, 2)
            self.assertFalse((day / "camera_calibration.json").exists())

    def test_load_detected_pairs_requires_both_sides_and_matching_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            left = root / "head_left"
            right = root / "head_right"
            left.mkdir()
            right.mkdir()
            board = checkerboard_bgr()
            blank = np.full((448, 544, 3), 80, dtype=np.uint8)
            cv2.imwrite(str(left / "000.png"), board)
            cv2.imwrite(str(right / "000.png"), board)
            cv2.imwrite(str(left / "001.png"), board)
            cv2.imwrite(str(right / "001.png"), blank)
            cv2.imwrite(str(left / "002.png"), board)
            messages = []
            pairs = load_detected_pairs(root, pattern=PATTERN, square=SQUARE, log=messages.append)
            self.assertEqual([pair.index for pair in pairs], [0])
            self.assertTrue(any("001" in item for item in messages))

    def test_load_detected_pairs_picks_up_appended_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            left = root / "head_left"
            right = root / "head_right"
            left.mkdir()
            right.mkdir()
            board = checkerboard_bgr()
            cv2.imwrite(str(left / "000.png"), board)
            cv2.imwrite(str(right / "000.png"), board)
            cv2.imwrite(str(left / "046.png"), board)
            cv2.imwrite(str(right / "046.png"), board)
            pairs = load_detected_pairs(root, pattern=PATTERN, square=SQUARE, log=lambda *_a, **_k: None)
            self.assertEqual([pair.index for pair in pairs], [0, 46])

    def test_select_keeps_initextrinsics_ok_views_when_ransac_fails(self):
        pairs = synthetic_detected_pairs(40, baseline=0.060, seed=7)
        train, _hold = holdout_split(pairs)
        silent = lambda *_args, **_kwargs: None
        with mock.patch("calibrate_r1_head_stereo_fit.view_pose_ok", return_value=False):
            kept, dropped = select_train_pairs(train, log=silent)
        self.assertGreaterEqual(len(kept), 20)
        self.assertFalse(any("ransac" in reason for _index, reason in dropped))
        collapsed = np.full_like(train[0].left_points, [272.0, 224.0])
        bad = DetectedPair(
            index=999,
            left_path=Path("head_left/999.png"),
            right_path=Path("head_right/999.png"),
            object_points=train[0].object_points.copy(),
            left_points=collapsed.copy(),
            right_points=collapsed.copy(),
            image_size=train[0].image_size,
        )
        kept_with_bad, dropped_with_bad = select_train_pairs([bad] + list(train), log=silent)
        self.assertTrue(any(index == 999 for index, _reason in dropped_with_bad))
        self.assertFalse(any(pair.index == 999 for pair in kept_with_bad))

    def test_fit_cli_defaults(self):
        args = parse_fit_args(["--day-dir", "/tmp/r1-cam-calib/20260920"])
        self.assertEqual(tuple(args.pattern), (9, 6))
        self.assertAlmostEqual(args.square, 0.030)
        self.assertEqual(args.holdout, 10)

    def test_initial_K_is_never_identity_or_zero(self):
        K = _initial_K((544, 448))
        self.assertGreater(K[0, 0], 50.0)
        self.assertGreater(K[1, 1], 50.0)
        self.assertAlmostEqual(K[0, 0], 0.9 * 448.0)
        self.assertAlmostEqual(K[0, 2], 272.0)
        self.assertAlmostEqual(K[1, 2], 224.0)
        self.assertNotAlmostEqual(K[0, 0], 1.0)
        self.assertNotAlmostEqual(K[1, 1], 1.0)
        self.assertGreater(abs(K[0, 0]), 1.0)
        flags = fisheye_flags(fix_k34=False)
        self.assertTrue(flags & cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
        self.assertTrue(flags & cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC)
        self.assertTrue(flags & cv2.fisheye.CALIB_FIX_SKEW)
        self.assertFalse(flags & cv2.fisheye.CALIB_CHECK_COND)

    def test_canonicalize_puts_origin_at_top_left(self):
        cols, rows = PATTERN
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
        reversed_pts = grid[::-1].reshape(-1, 1, 2)
        ordered = canonicalize_corners(reversed_pts, PATTERN)
        pts = ordered.reshape(-1, 2)
        self.assertLess(pts[0, 0], pts[cols - 1, 0])
        self.assertLess(pts[0, 1], pts[-1, 1])

    def test_zero_or_identity_K_cannot_init_but_seeded_K_succeeds(self):
        pairs = synthetic_detected_pairs(30, baseline=0.060, seed=11)
        train, _hold = holdout_split(pairs)
        obj = _as_object_views([pair.object_points for pair in train])
        img = _as_image_views([pair.left_points for pair in train])
        image_size = train[0].image_size
        flags = fisheye_flags(fix_k34=False)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-6)
        with self.assertRaises(cv2.error) as zero_ctx:
            cv2.fisheye.calibrate(
                obj, img, image_size,
                np.zeros((3, 3), dtype=np.float64), np.zeros((4, 1), dtype=np.float64),
                flags=flags, criteria=criteria)
        self.assertIn("InitExtrinsics", str(zero_ctx.exception))
        identity_K = np.eye(3, dtype=np.float64)
        identity_failed = False
        identity_exploded = False
        try:
            rms_id, K_id, D_id, _r, _t = cv2.fisheye.calibrate(
                obj, img, image_size,
                identity_K.copy(), np.zeros((4, 1), dtype=np.float64),
                flags=flags, criteria=criteria)
            identity_exploded = (
                not np.isfinite(rms_id)
                or abs(float(K_id[0, 0])) < 5.0
                or abs(float(K_id[0, 0])) > 2000.0
                or float(np.max(np.abs(D_id))) > 8.0
            )
        except cv2.error:
            identity_failed = True
        self.assertTrue(identity_failed or identity_exploded)
        silent = lambda *_args, **_kwargs: None
        rms, K, D = calibrate_fisheye_mono(obj, img, image_size, log=silent)
        self.assertLess(rms, 0.5)
        self.assertGreater(K[0, 0], 50.0)
        self.assertLess(K[0, 0], 800.0)
        self.assertLess(float(np.max(np.abs(D))), 8.0)

    def test_fit_skips_collapsed_view_that_would_trip_initextrinsics(self):
        pairs = synthetic_detected_pairs(40, baseline=0.060, seed=7)
        train, hold = holdout_split(pairs)
        collapsed = np.full_like(train[0].left_points, [272.0, 224.0])
        bad = DetectedPair(
            index=999,
            left_path=Path("head_left/999.png"),
            right_path=Path("head_right/999.png"),
            object_points=train[0].object_points.copy(),
            left_points=collapsed.copy(),
            right_points=collapsed.copy(),
            image_size=train[0].image_size,
        )
        silent = lambda *_args, **_kwargs: None
        result = fit_detected_pairs([bad] + list(train), hold, pattern=PATTERN, square=SQUARE, log=silent)
        self.assertGreaterEqual(result.n_train, 20)
        self.assertGreaterEqual(result.dropped_train, 1)
        self.assertLess(result.left_rms, 0.5)
        self.assertLess(result.right_rms, 0.5)
        self.assertLess(result.stereo_rms, 0.5)
        self.assertTrue(0.04 <= result.baseline_m <= 0.08)
        self.assertEqual(quality_errors(result), [])


if __name__ == "__main__":
    unittest.main()
