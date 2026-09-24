import builtins
from copy import deepcopy
from dataclasses import fields
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / "teleop/televuer/src/televuer/tv_wrapper.py"


class RawMotionSource:
    def __init__(self):
        self.head_pose = np.eye(4)
        self.head_pose[:3, 3] = [0.1, 1.6, -0.2]
        self.snapshot = {
            "motion_data_ready": True,
            "motion_data_timestamp": 100.0,
            "motion_sample_seq": 3,
            "left_hand_timestamp": 100.0,
            "right_hand_timestamp": 100.0,
        }
        for side, x in (("left", -0.3), ("right", 0.3)):
            pose = np.eye(4)
            pose[:3, 3] = [x, 1.25, -0.6]
            points = np.tile(pose[:3, 3], (25, 1))
            points[:, 2] -= np.arange(25) * 0.005
            points[:, 0] += np.sin(np.arange(25)) * 0.03
            self.snapshot.update({
                f"{side}_arm_pose": pose,
                f"{side}_hand_positions": points,
                f"{side}_hand_orientations": np.tile(np.eye(3), (25, 1, 1)),
                f"{side}_hand_pinch": False,
                f"{side}_hand_pinchValue": 0.025,
                f"{side}_hand_squeeze": False,
                f"{side}_hand_squeezeValue": 0.2,
            })
        self.render_to_xr = mock.Mock()
        self.render_wrist_to_xr = mock.Mock()
        self.render_torque_hud_to_xr = mock.Mock()
        self.close = mock.Mock()

    def get_hand_motion_snapshot(self, include_orientations=False):
        return deepcopy(self.snapshot)

    def pop_hand_motion_sample(self):
        return self.get_hand_motion_snapshot(include_orientations=True)


class VisionProLegacyContractTest(unittest.TestCase):
    def setUp(self):
        self.legacy_source = RawMotionSource()
        self.legacy_factory = mock.Mock(return_value=self.legacy_source)
        package_name = "visionpro_legacy_contract_under_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(WRAPPER_PATH.parent)]
        backend = types.ModuleType(f"{package_name}.televuer")
        backend.TeleVuer = self.legacy_factory
        module_name = f"{package_name}.tv_wrapper"
        spec = importlib.util.spec_from_file_location(module_name, WRAPPER_PATH)
        self.module = importlib.util.module_from_spec(spec)
        self.modules_patch = mock.patch.dict(sys.modules, {
            package_name: package,
            f"{package_name}.televuer": backend,
            module_name: self.module,
        })
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        spec.loader.exec_module(self.module)

    def test_default_constructs_only_the_existing_backend(self):
        original_import = builtins.__import__

        def import_without_native_dependencies(name, *args, **kwargs):
            if name.split(".")[0] in ("avp_stream", "grpc", "visionpro_source"):
                raise AssertionError(f"Legacy mode imported optional dependency {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_native_dependencies):
            wrapper = self.module.TeleVuerWrapper(
                use_hand_tracking=True,
                display_mode="immersive",
                webrtc=True,
                webrtc_url="https://robot.example/offer",
            )
            sample = wrapper.get_tele_data()
        self.legacy_factory.assert_called_once()
        self.assertIs(wrapper.tvuer, self.legacy_source)
        self.assertEqual(self.legacy_factory.call_args.kwargs["display_mode"], "immersive")
        self.assertTrue(self.legacy_factory.call_args.kwargs["webrtc"])
        self.assertTrue(sample.motion_data_ready)

    def test_injected_source_does_not_start_a_webxr_backend(self):
        source = RawMotionSource()
        wrapper = self.module.TeleVuerWrapper(
            use_hand_tracking=True, motion_source=source, display_mode="pass-through",
        )
        self.legacy_factory.assert_not_called()
        self.assertIs(wrapper.tvuer, source)
        self.assertEqual(wrapper.get_tele_data().left_hand_timestamp, 100.0)
        wrapper.close()
        source.close.assert_called_once_with()

    def test_injected_and_legacy_sources_use_identical_transforms(self):
        for reference_mode in ("head_yaw", "head_position"):
            with self.subTest(reference_mode=reference_mode):
                legacy = self.module.TeleVuerWrapper(
                    use_hand_tracking=True, return_hand_rot_data=True,
                    arm_reference_mode=reference_mode,
                )
                native = self.module.TeleVuerWrapper(
                    use_hand_tracking=True, return_hand_rot_data=True,
                    arm_reference_mode=reference_mode, motion_source=RawMotionSource(),
                )
                expected, actual = legacy.get_tele_data(), native.get_tele_data()
                for field in fields(expected):
                    left, right = getattr(expected, field.name), getattr(actual, field.name)
                    if isinstance(left, np.ndarray):
                        np.testing.assert_allclose(left, right, err_msg=field.name)
                    else:
                        self.assertEqual(left, right, field.name)
                self.assertEqual(actual.left_hand_pos.shape, (25, 3))
                self.assertEqual(actual.right_hand_rot.shape, (25, 3, 3))
                self.assertAlmostEqual(actual.left_hand_pinchValue, 2.5)

    def test_lost_side_keeps_the_other_skeleton_usable(self):
        source = RawMotionSource()
        wrapper = self.module.TeleVuerWrapper(use_hand_tracking=True, motion_source=source)
        before = wrapper.get_tele_data()
        source.snapshot.update(right_hand_timestamp=0.0, motion_data_timestamp=0.0)
        after = wrapper.get_tele_data()
        self.assertTrue(after.motion_data_ready)
        self.assertEqual(after.left_hand_timestamp, 100.0)
        self.assertEqual(after.right_hand_timestamp, 0.0)
        self.assertEqual(after.motion_data_timestamp, 0.0)
        np.testing.assert_array_equal(after.left_hand_pos, before.left_hand_pos)
        self.assertGreater(np.max(np.linalg.norm(after.left_hand_pos, axis=1)), 0.01)

    def test_known_world_fixture_reaches_existing_robot_and_hand_frames(self):
        wrapper = self.module.TeleVuerWrapper(use_hand_tracking=True, motion_source=RawMotionSource())
        sample = wrapper.get_tele_data()
        np.testing.assert_allclose(sample.head_pose[:3, 3], [0.2, -0.1, 1.6])
        np.testing.assert_allclose(sample.left_wrist_pose[:3, 3], [0.55, 0.4, 0.1])
        np.testing.assert_allclose(sample.right_wrist_pose[:3, 3], [0.55, -0.2, 0.1])
        np.testing.assert_allclose(sample.left_wrist_pose[:3, :3], [
            [1, 0, 0], [0, 0, -1], [0, 1, 0],
        ])
        np.testing.assert_allclose(sample.right_wrist_pose[:3, :3], [
            [1, 0, 0], [0, 0, 1], [0, -1, 0],
        ])
        for hand in (sample.left_hand_pos, sample.right_hand_pos):
            np.testing.assert_allclose(hand[0], [0.0, 0.0, 0.0], atol=1e-12)
            np.testing.assert_allclose(hand[24], [0.0, -0.12, np.sin(24) * 0.03], atol=1e-12)

    def test_explicit_loss_replaces_both_consumer_caches(self):
        source = RawMotionSource()
        wrapper = self.module.TeleVuerWrapper(use_hand_tracking=True, motion_source=source)
        wrapper.get_tele_data()
        wrapper.get_arm_tele_data()
        source.snapshot.update(
            motion_data_ready=False,
            left_hand_timestamp=0.0,
            right_hand_timestamp=0.0,
            motion_data_timestamp=0.0,
        )
        for sample in (wrapper.get_tele_data(), wrapper.get_arm_tele_data()):
            self.assertFalse(sample.motion_data_ready)
            self.assertEqual(sample.left_hand_timestamp, 0.0)
            self.assertEqual(sample.right_hand_timestamp, 0.0)
            self.assertEqual(sample.motion_data_timestamp, 0.0)

    def test_repeated_reads_do_not_refresh_receive_timestamps(self):
        source = RawMotionSource()
        wrapper = self.module.TeleVuerWrapper(use_hand_tracking=True, motion_source=source)
        for _ in range(5):
            sample = wrapper.get_tele_data()
            self.assertEqual(sample.left_hand_timestamp, 100.0)
            self.assertEqual(sample.right_hand_timestamp, 100.0)
            self.assertEqual(sample.motion_data_timestamp, 100.0)
        np.testing.assert_array_equal(
            source.snapshot["left_hand_positions"], RawMotionSource().snapshot["left_hand_positions"],
        )


if __name__ == "__main__":
    unittest.main()
