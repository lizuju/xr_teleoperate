import copy
import json
from pathlib import Path
import tempfile
import unittest

from teleop.utils.camera_calibration import (CALIBRATION_SCHEMA, CalibrationError,
                                             DEFAULT_CALIBRATION_RELPATH, camera_calibration_metadata,
                                             default_camera_calibration_root, expected_cameras,
                                             load_camera_calibration, resolve_camera_calibration)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = REPO_ROOT / "teleop/teleop_hand_and_arm.py"


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def camera(width, height, fx=None):
    fx = fx if fx is not None else width * 0.9
    return {
        "image_size": [width, height],
        "camera_matrix": [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
        "distortion_model": "plumb_bob",
        "distortion_coefficients": [0.01, -0.02, 0.0, 0.0, 0.003],
        "reprojection_error_px": 0.21,
    }


def document():
    return {
        "schema": CALIBRATION_SCHEMA,
        "created": "2026-09-15",
        "board": {"type": "checkerboard", "squares_x": 9, "squares_y": 6, "square_size_m": 0.030},
        "cameras": {
            "head_left": camera(544, 448),
            "head_right": camera(544, 448),
            "left_wrist": camera(640, 480, fx=400.0),
            "right_wrist": camera(640, 480, fx=400.0),
        },
        "stereo": {"head": {"left": "head_left", "right": "head_right",
                            "rotation": IDENTITY, "translation_m": [-0.06, 0.0, 0.0],
                            "baseline_m": 0.06, "epipolar_error_px": 0.31}},
        "hand_eye": {"left_wrist": {"camera": "left_wrist", "frame": "left_wrist_yaw_link",
                                    "rotation": IDENTITY, "translation_m": [0.02, 0.0, 0.05],
                                    "rotation_rms_deg": 1.2, "translation_rms_m": 0.004}},
    }


class CalibrationFileTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, value, name="calibration.json"):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def assertRejected(self, value, fragment):
        path = self.write(value)
        with self.assertRaises(CalibrationError) as caught:
            load_camera_calibration(path)
        self.assertIn(fragment, str(caught.exception))

    def test_valid_file_is_normalised_and_hashed(self):
        path = self.write(document())
        calibration = load_camera_calibration(path)
        self.assertEqual(calibration["schema"], CALIBRATION_SCHEMA)
        self.assertEqual(calibration["source"], str(path))
        self.assertEqual(len(calibration["sha256"]), 64)
        self.assertEqual(sorted(calibration["cameras"]), ["head_left", "head_right",
                                                          "left_wrist", "right_wrist"])
        self.assertEqual(calibration["cameras"]["head_left"]["image_size"], [544, 448])
        self.assertEqual(calibration["board"]["square_size_m"], 0.030)
        self.assertEqual(calibration["stereo"]["head"]["baseline_m"], 0.06)
        self.assertEqual(calibration["hand_eye"]["left_wrist"]["frame"], "left_wrist_yaw_link")

    def test_non_json_and_non_object_are_rejected(self):
        path = self.root / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(CalibrationError, "not valid UTF-8 JSON"):
            load_camera_calibration(path)
        self.assertRejected([1, 2, 3], "expected a JSON object")

    def test_schema_is_enforced(self):
        broken = document()
        broken["schema"] = "something_else"
        self.assertRejected(broken, "schema must be")

    def test_camera_matrix_is_checked(self):
        broken = document()
        broken["cameras"]["head_left"]["camera_matrix"][2] = [0.0, 0.0, 2.0]
        self.assertRejected(broken, "last row must be [0, 0, 1]")

        broken = document()
        broken["cameras"]["head_left"]["camera_matrix"][0][0] = -100.0
        self.assertRejected(broken, "fx and fy must be positive")

        broken = document()
        del broken["cameras"]["head_left"]["camera_matrix"]
        self.assertRejected(broken, "expected 3 rows")

    def test_image_size_is_checked(self):
        broken = document()
        broken["cameras"]["head_left"]["image_size"] = [544.5, 448]
        self.assertRejected(broken, "positive integer width and height")

    def test_distortion_model_is_checked(self):
        broken = document()
        broken["cameras"]["head_left"]["distortion_model"] = "brown_conrady"
        self.assertRejected(broken, "distortion_model")

        broken = document()
        broken["cameras"]["head_left"]["distortion_coefficients"] = []
        self.assertRejected(broken, "distortion_coefficients: required")

        none_model = document()
        none_model["cameras"]["head_left"]["distortion_model"] = "none"
        none_model["cameras"]["head_left"]["distortion_coefficients"] = []
        calibration = load_camera_calibration(self.write(none_model))
        self.assertEqual(calibration["cameras"]["head_left"]["distortion_coefficients"], [])

    def test_rotation_must_be_a_proper_rotation(self):
        broken = document()
        broken["stereo"]["head"]["rotation"] = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertRejected(broken, "det(R) = +1")

        broken = document()
        broken["stereo"]["head"]["rotation"] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]
        self.assertRejected(broken, "orthonormal")

    def test_stereo_and_hand_eye_must_name_calibrated_cameras(self):
        broken = document()
        broken["stereo"]["head"]["right"] = "head_center"
        self.assertRejected(broken, "is not a calibrated camera")

        broken = document()
        broken["hand_eye"]["left_wrist"]["camera"] = "left_palm"
        self.assertRejected(broken, "is not a calibrated camera")

        broken = document()
        broken["hand_eye"]["left_wrist"]["frame"] = ""
        self.assertRejected(broken, "expected the robot link")

    def test_board_geometry_is_checked(self):
        broken = document()
        broken["board"] = {"type": "charuco", "squares_x": 10, "squares_y": 7, "square_size_m": 0.030,
                           "marker_size_m": 0.030, "dictionary": "DICT_5X5_100"}
        self.assertRejected(broken, "below square_size_m")

        broken = document()
        broken["board"] = {"type": "charuco", "squares_x": 10, "squares_y": 7, "square_size_m": 0.030,
                           "marker_size_m": 0.022}
        self.assertRejected(broken, "dictionary")

        broken = document()
        broken["board"] = {"type": "grid", "squares_x": 9, "squares_y": 6, "square_size_m": 0.030}
        self.assertRejected(broken, "checkerboard")

    def test_unknown_top_level_keys_are_rejected(self):
        broken = document()
        broken["extrinsics"] = {}
        self.assertRejected(broken, "unknown top-level keys: extrinsics")

    def test_empty_camera_map_is_rejected(self):
        broken = document()
        broken["cameras"] = {}
        self.assertRejected(broken, "cameras must be a non-empty object")


class ResolveCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def install_default(self, value):
        path = self.root / DEFAULT_CALIBRATION_RELPATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_no_default_file_is_not_an_error(self):
        self.assertEqual(resolve_camera_calibration("", root=self.root), (None, None))
        self.assertEqual(resolve_camera_calibration(None, root=self.root), (None, None))

    def test_default_file_is_picked_up_and_validated(self):
        path = self.install_default(document())
        for value in ("", None):
            resolved, calibration = resolve_camera_calibration(value, root=self.root)
            self.assertEqual(resolved, path)
            self.assertEqual(calibration["schema"], CALIBRATION_SCHEMA)
            self.assertEqual(sorted(calibration["cameras"]),
                             ["head_left", "head_right", "left_wrist", "right_wrist"])

    def test_broken_default_file_is_not_silently_ignored(self):
        self.install_default({"schema": CALIBRATION_SCHEMA, "cameras": {}})
        with self.assertRaises(CalibrationError):
            resolve_camera_calibration("", root=self.root)

    def test_explicit_path_must_exist(self):
        with self.assertRaisesRegex(CalibrationError, "No such file"):
            resolve_camera_calibration(str(self.root / "absent.json"), root=self.root)

    def test_explicit_path_wins_over_the_default(self):
        self.install_default(document())
        explicit = self.root / "elsewhere.json"
        explicit.write_text(json.dumps(document()), encoding="utf-8")
        resolved, _ = resolve_camera_calibration(str(explicit), root=self.root)
        self.assertEqual(resolved, explicit)


class DefaultCalibrationRootTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        # Production layout: unitree_r1_dev/xr_teleoperate/teleop/teleop_hand_and_arm.py
        # parents[2] from the main script is unitree_r1_dev, which has no assets/.
        self.dev = Path(self.temporary.name) / "unitree_r1_dev"
        self.repo = self.dev / "xr_teleoperate"
        self.main = self.repo / "teleop/teleop_hand_and_arm.py"
        self.helper = self.repo / "teleop/utils/camera_calibration.py"
        self.main.parent.mkdir(parents=True)
        self.helper.parent.mkdir(parents=True, exist_ok=True)
        self.main.write_text("# stub\n", encoding="utf-8")
        self.helper.write_text("# stub\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def install_repo_default(self):
        path = self.repo / DEFAULT_CALIBRATION_RELPATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document()), encoding="utf-8")
        return path.resolve()

    def test_root_from_main_script_is_repo_not_dev(self):
        self.assertEqual(default_camera_calibration_root(self.main), self.repo.resolve())
        self.assertEqual(self.main.resolve().parents[1], self.repo.resolve())
        self.assertEqual(self.main.resolve().parents[2], self.dev.resolve())

    def test_root_from_helper_is_also_repo(self):
        self.assertEqual(default_camera_calibration_root(self.helper), self.repo.resolve())
        self.assertEqual(self.helper.resolve().parents[2], self.repo.resolve())

    def test_empty_arg_finds_repo_assets_not_dev_assets(self):
        default = self.install_repo_default()
        self.assertFalse((self.dev / DEFAULT_CALIBRATION_RELPATH).exists())
        resolved, calibration = resolve_camera_calibration(
            "", root=default_camera_calibration_root(self.main))
        self.assertEqual(resolved, default)
        self.assertEqual(sorted(calibration["cameras"]),
                         ["head_left", "head_right", "left_wrist", "right_wrist"])
        self.assertEqual(resolve_camera_calibration("", root=self.main.resolve().parents[2]),
                         (None, None))

    def test_explicit_path_still_wins_on_the_fake_tree(self):
        self.install_repo_default()
        explicit = self.dev / "other.json"
        explicit.write_text(json.dumps(document()), encoding="utf-8")
        resolved, _ = resolve_camera_calibration(
            str(explicit), root=default_camera_calibration_root(self.main))
        self.assertEqual(resolved, explicit)

    def test_main_script_no_longer_passes_parents_2(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn("default_camera_calibration_root(__file__)", source)
        self.assertIn("resolve_run_camera_calibration(args)", source)
        self.assertNotIn("Path(__file__).resolve().parents[2]", source)
        self.assertIn("raise SystemExit(2)", source)

    def test_wrappers_still_pass_empty_camera_calibration(self):
        for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
            script = (REPO_ROOT / "teleop" / name).read_text(encoding="utf-8")
            self.assertIn('--camera-calibration "${CAMERA_CALIBRATION:-}"', script)


class CalibrationMetadataTest(unittest.TestCase):
    def test_uncalibrated_block_is_explicit_and_not_missing(self):
        block = camera_calibration_metadata(None, ["head_left", "head_right", "left_wrist"])
        self.assertEqual(block["status"], "uncalibrated")
        self.assertIsNone(block["source"])
        self.assertEqual(block["expected_cameras"], ["head_left", "head_right", "left_wrist"])
        self.assertEqual(block["missing_cameras"], ["head_left", "head_right", "left_wrist"])
        self.assertEqual(block["cameras"], {})
        self.assertIn("cannot be undistorted", block["note"])

    def calibration(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "calibration.json"
        path.write_text(json.dumps(document()), encoding="utf-8")
        return load_camera_calibration(path)

    def test_full_coverage_reports_calibrated(self):
        block = camera_calibration_metadata(
            self.calibration(), ["head_left", "head_right", "left_wrist", "right_wrist"],
            {"head_left": (544, 448), "head_right": (544, 448),
             "left_wrist": (640, 480), "right_wrist": (640, 480)})
        self.assertEqual(block["status"], "calibrated")
        self.assertEqual(block["missing_cameras"], [])
        self.assertEqual(block["warnings"], [])
        self.assertIsNone(block["note"])
        self.assertEqual(sorted(block["cameras"]), ["head_left", "head_right",
                                                    "left_wrist", "right_wrist"])
        self.assertEqual(block["stereo"]["head"]["left"], "head_left")
        self.assertEqual(block["hand_eye"]["left_wrist"]["camera"], "left_wrist")

    def test_partial_coverage_is_reported_not_hidden(self):
        calibration = self.calibration()
        del calibration["cameras"]["right_wrist"]
        block = camera_calibration_metadata(calibration,
                                            ["head_left", "head_right", "left_wrist", "right_wrist"])
        self.assertEqual(block["status"], "partial")
        self.assertEqual(block["missing_cameras"], ["right_wrist"])
        self.assertIn("right_wrist", block["note"])

    def test_calibrating_nothing_this_run_records_is_uncalibrated(self):
        block = camera_calibration_metadata(self.calibration(), ["front_stereo"])
        self.assertEqual(block["status"], "uncalibrated")
        self.assertEqual(block["cameras"], {})
        self.assertIn("calibrates no camera this run records", block["note"])

    def test_resolution_mismatch_is_warned_about(self):
        block = camera_calibration_metadata(
            self.calibration(), ["head_left", "head_right", "left_wrist", "right_wrist"],
            {"left_wrist": (1280, 720)})
        self.assertEqual(block["status"], "calibrated")
        self.assertEqual(len(block["warnings"]), 1)
        self.assertIn("calibrated at 640x480 but streaming 1280x720", block["warnings"][0])

    def test_expected_cameras_follow_the_streaming_config(self):
        config = {"head_camera": {}, "left_wrist_camera": {"enable_zmq": True},
                  "right_wrist_camera": {"enable_zmq": False}}
        self.assertEqual(expected_cameras(config), ["head_left", "head_right", "left_wrist"])
        self.assertEqual(expected_cameras({}), ["head_left", "head_right"])

    def test_metadata_never_mutates_the_loaded_calibration(self):
        calibration = self.calibration()
        before = copy.deepcopy(calibration)
        camera_calibration_metadata(calibration, ["head_left"])
        self.assertEqual(calibration, before)


if __name__ == "__main__":
    unittest.main()
