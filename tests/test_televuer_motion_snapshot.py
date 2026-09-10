import ast
import asyncio
import importlib.util
from multiprocessing import Array, Value
from msgpack import ExtType, unpackb
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[1]
TELEVUER_PATH = STAGE_ROOT / "teleop" / "televuer" / "src" / "televuer" / "televuer.py"


def load_televuer_class():
    vuer_module = types.ModuleType("vuer")
    vuer_module.Vuer = object
    schemas_module = types.ModuleType("vuer.schemas")
    for name in (
        "ImageBackground",
        "Hands",
        "MotionControllers",
        "WebRTCVideoPlane",
        "WebRTCStereoVideoPlane",
    ):
        setattr(schemas_module, name, object)
    cv2_module = types.ModuleType("cv2")
    with mock.patch.dict(
        sys.modules,
        {"vuer": vuer_module, "vuer.schemas": schemas_module, "cv2": cv2_module},
    ):
        spec = importlib.util.spec_from_file_location("televuer_snapshot_under_test", TELEVUER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.TeleVuer


def bare_televuer():
    tele_vuer_class = load_televuer_class()
    tele_vuer = tele_vuer_class.__new__(tele_vuer_class)
    tele_vuer.use_hand_tracking = True
    tele_vuer.left_arm_pose_shared = Array("d", 16, lock=True)
    tele_vuer.right_arm_pose_shared = Array("d", 16, lock=True)
    tele_vuer.left_hand_position_shared = Array("d", 75, lock=True)
    tele_vuer.right_hand_position_shared = Array("d", 75, lock=True)
    tele_vuer.left_hand_orientation_shared = Array("d", 225, lock=True)
    tele_vuer.right_hand_orientation_shared = Array("d", 225, lock=True)
    tele_vuer.left_hand_pinch_shared = Value("b", False, lock=True)
    tele_vuer.left_hand_pinchValue_shared = Value("d", 0.0, lock=True)
    tele_vuer.left_hand_squeeze_shared = Value("b", False, lock=True)
    tele_vuer.left_hand_squeezeValue_shared = Value("d", 0.0, lock=True)
    tele_vuer.right_hand_pinch_shared = Value("b", False, lock=True)
    tele_vuer.right_hand_pinchValue_shared = Value("d", 0.0, lock=True)
    tele_vuer.right_hand_squeeze_shared = Value("b", False, lock=True)
    tele_vuer.right_hand_squeezeValue_shared = Value("d", 0.0, lock=True)
    tele_vuer.motion_data_ready_shared = Value("b", False, lock=True)
    tele_vuer.motion_data_timestamp_shared = Value("d", 0.0, lock=True)
    tele_vuer.left_hand_timestamp_shared = Value("d", 0.0, lock=True)
    tele_vuer.right_hand_timestamp_shared = Value("d", 0.0, lock=True)
    tele_vuer.motion_sample_seq_shared = Value("L", 0, lock=True)
    tele_vuer.tracking_event_counts_shared = Array("L", 9, lock=True)
    tele_vuer.tracking_hand_status_shared = Array("i", 2, lock=True)
    guard_type = tele_vuer.on_hand_move.__func__.__globals__["HandPoseGuard"]
    tele_vuer.hand_pose_guards = (guard_type(), guard_type())
    return tele_vuer


class TeleVuerMotionSnapshotTest(unittest.TestCase):
    def test_next_writer_recovers_from_abandoned_odd_sequence(self):
        tele_vuer = bare_televuer()

        abandoned_seq = tele_vuer._begin_motion_sample()
        self.assertEqual(abandoned_seq, 1)
        self.assertIsNone(tele_vuer.get_hand_motion_snapshot(max_attempts=1))

        recovered_seq = tele_vuer._begin_motion_sample()
        self.assertEqual(recovered_seq, 3)
        with tele_vuer.motion_data_timestamp_shared.get_lock():
            tele_vuer.motion_data_timestamp_shared.value = 2.0
        with tele_vuer.motion_data_ready_shared.get_lock():
            tele_vuer.motion_data_ready_shared.value = True
        tele_vuer._commit_motion_sample(recovered_seq)

        snapshot = tele_vuer.get_hand_motion_snapshot(max_attempts=1)
        self.assertEqual(snapshot["motion_sample_seq"], 4)
        self.assertEqual(snapshot["motion_data_timestamp"], 2.0)

    def test_reader_rejects_in_progress_mixed_hand_sample(self):
        tele_vuer = bare_televuer()
        left_written = threading.Event()
        allow_commit = threading.Event()

        def writer():
            sample_seq = tele_vuer._begin_motion_sample()
            with tele_vuer.left_hand_position_shared.get_lock():
                tele_vuer.left_hand_position_shared[:] = [1.0] * 75
            left_written.set()
            allow_commit.wait(1.0)
            with tele_vuer.right_hand_position_shared.get_lock():
                tele_vuer.right_hand_position_shared[:] = [1.0] * 75
            with tele_vuer.motion_data_timestamp_shared.get_lock():
                tele_vuer.motion_data_timestamp_shared.value = 1.0
            with tele_vuer.motion_data_ready_shared.get_lock():
                tele_vuer.motion_data_ready_shared.value = True
            tele_vuer._commit_motion_sample(sample_seq)

        writer_thread = threading.Thread(target=writer)
        writer_thread.start()
        self.assertTrue(left_written.wait(1.0))
        self.assertIsNone(tele_vuer.get_hand_motion_snapshot(max_attempts=1))

        allow_commit.set()
        writer_thread.join(timeout=1.0)
        self.assertFalse(writer_thread.is_alive())
        snapshot = tele_vuer.get_hand_motion_snapshot(max_attempts=1)

        self.assertEqual(snapshot["motion_sample_seq"], 2)
        self.assertEqual(snapshot["motion_data_timestamp"], 1.0)
        self.assertTrue(snapshot["motion_data_ready"])
        np.testing.assert_allclose(snapshot["left_hand_positions"], 1.0)
        np.testing.assert_allclose(snapshot["right_hand_positions"], 1.0)


class TeleVuerHandFreshnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (STAGE_ROOT / "teleop" / "teleop_hand_and_arm.py").read_text()
        guard = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == "is_fresh_motion_data")
        namespace = {}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "<tracking-guard>", "exec"), namespace)
        cls.is_fresh = staticmethod(namespace["is_fresh_motion_data"])

    def setUp(self):
        self.tele_vuer = bare_televuer()
        self.pose = np.tile(np.eye(4).reshape(-1), 25).tolist()

    def publish(self, timestamp, *hands, **payload):
        event = types.SimpleNamespace(value={hand: self.pose for hand in hands} | payload)
        with mock.patch.dict(self.tele_vuer.on_hand_move.__func__.__globals__,
                             {"time": types.SimpleNamespace(monotonic=lambda: timestamp)}):
            asyncio.run(self.tele_vuer.on_hand_move(event, None))
        snapshot = self.tele_vuer.get_hand_motion_snapshot()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["motion_sample_seq"] % 2, 0)
        return types.SimpleNamespace(**snapshot)

    def settle(self, timestamp, **payload):
        for step in range(13):
            sample = self.publish(timestamp - 0.4 + step / 30, "left", "right", **payload)
        return sample

    def test_valid_both_hands_are_available_without_input_stability_delay(self):
        sample = self.publish(99.6, "left", "right")
        self.assertTrue(sample.motion_data_ready)
        self.assertTrue(self.is_fresh(sample, 0.25, now=99.6))
        sample = self.settle(100.0)
        self.assertTrue(self.is_fresh(sample, 0.25, now=100.0))
        self.assertEqual(sample.left_hand_timestamp, 100.0)
        self.assertEqual(sample.right_hand_timestamp, 100.0)

    def test_missing_either_hand_invalidates_it_immediately(self):
        for moving in ("left", "right"):
            self.tele_vuer = bare_televuer()
            self.settle(100.0)
            sample = self.publish(100.01, moving)
            missing = "right" if moving == "left" else "left"
            self.assertEqual(getattr(sample, missing + "_hand_timestamp"), 0.0)
            self.assertEqual(getattr(sample, moving + "_hand_timestamp"), 100.01)
            self.assertFalse(self.is_fresh(sample, 0.25, now=100.01))
            self.assertTrue(sample.motion_data_ready)

    def test_msgpackr_undefined_is_missing_not_malformed(self):
        self.settle(100.0)
        undefined = unpackb(bytes.fromhex("d40000"), raw=False)
        self.assertIsInstance(undefined, ExtType)
        sample = self.publish(100.01, left=undefined, right=undefined)
        self.assertEqual((sample.left_hand_timestamp, sample.right_hand_timestamp), (0.0, 0.0))
        counts = self.tele_vuer.tracking_event_counts_shared[:]
        self.assertEqual(counts[4], 0)
        self.assertEqual(counts[5:7], [1, 1])

    def test_unknown_extension_is_reported_and_does_not_block_other_hand(self):
        self.settle(100.0)
        with mock.patch("builtins.print") as output:
            sample = self.publish(100.01, "right", left=ExtType(116, b"bad"))
        self.assertEqual(sample.left_hand_timestamp, 0.0)
        self.assertEqual(sample.right_hand_timestamp, 100.01)
        self.assertIn("code=116", output.call_args.args[0])
        self.assertEqual(self.tele_vuer.tracking_event_counts_shared[4], 1)

    def test_bad_joint_transforms_and_gestures_hold_only_bad_side(self):
        self.settle(100.0)
        nan = self.pose.copy(); nan[12] = float("nan")
        reflection = self.pose.copy(); reflection[0] = -1
        skew = self.pose.copy(); skew[16 + 1] = 0.2
        distant = self.pose.copy(); distant[16 + 12] = 1.0
        for invalid in ([0.] * 400, nan, reflection, skew, distant, "invalid"):
            sample = self.publish(100.01, "right", left=invalid)
            self.assertEqual(sample.left_hand_timestamp, 0.0)
            self.assertEqual(sample.right_hand_timestamp, 100.01)
        sample = self.settle(100.5)
        sample = self.publish(100.51, "left", "right", leftState={"pinchValue": float("nan")})
        self.assertEqual(sample.left_hand_timestamp, 0.0)
        self.assertEqual(sample.right_hand_timestamp, 100.51)

    def test_large_valid_movement_updates_both_wrist_and_fingers(self):
        self.settle(100.0)
        before = self.tele_vuer.get_hand_motion_snapshot(include_orientations=True)
        jumped = np.asarray(self.pose).reshape(25, 16).copy()
        jumped[:, 12] += 0.3
        sample = self.publish(100.03, "right", left=jumped.flatten().tolist())
        self.assertEqual(sample.left_hand_timestamp, 100.03)
        self.assertEqual(sample.right_hand_timestamp, 100.03)
        expected_pose = before["left_arm_pose"].copy()
        expected_pose[0, 3] += 0.3
        np.testing.assert_allclose(sample.left_arm_pose, expected_pose)
        np.testing.assert_allclose(sample.left_hand_positions, before["left_hand_positions"] + [0.3, 0, 0])
        diagnostics = self.tele_vuer.get_tracking_diagnostics()
        self.assertEqual(diagnostics["left_state"], "tracking")
        self.assertEqual(diagnostics["left_suspect"], 0)

    def test_column_major_joint_pose_mapping_is_preserved(self):
        matrices = np.tile(np.eye(4), (25, 1, 1))
        matrices[:, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        matrices[:, :3, 3] = np.column_stack((np.linspace(0.1, 0.25, 25),
                                              np.full(25, 0.2), np.full(25, 0.3)))
        encoded = matrices.transpose(0, 2, 1).flatten().tolist()
        self.settle(100.0, left=encoded, right=encoded)
        sample = self.tele_vuer.get_hand_motion_snapshot(include_orientations=True)
        np.testing.assert_array_equal(sample["left_arm_pose"], matrices[0])
        np.testing.assert_array_equal(sample["left_hand_positions"], matrices[:, :3, 3])
        np.testing.assert_array_equal(sample["right_hand_orientations"], matrices[:, :3, :3])

    def test_moving_reappearance_resumes_wrist_and_fingers_without_activation(self):
        self.settle(100.0)
        self.publish(100.01, "right")
        for step in range(120):
            timestamp = 100.04 + step / 30
            moving = np.asarray(self.pose).reshape(25, 16).copy()
            x = 0.15 + 0.2 * step / 30
            moving[:, 12] = x
            sample = self.publish(timestamp, "right", left=moving.flatten().tolist())
            self.assertEqual(sample.left_hand_timestamp, timestamp)
            self.assertEqual(sample.right_hand_timestamp, timestamp)
            self.assertAlmostEqual(sample.left_arm_pose[0, 3], x)
            np.testing.assert_allclose(sample.left_hand_positions[:, 0], x)
            self.assertTrue(self.is_fresh(sample, 0.25, now=timestamp))

    def test_isolated_new_frames_do_not_disable_the_no_events_timeout(self):
        for timestamp in (100.0, 100.4, 101.0, 102.0):
            sample = self.publish(timestamp, "left", "right")
            self.assertTrue(self.is_fresh(sample, 0.25, now=timestamp))
            self.assertFalse(self.is_fresh(sample, 0.25, now=timestamp + 0.251))

    def test_invalid_then_valid_frame_recovers_only_valid_side_immediately(self):
        self.settle(100.0)
        invalid = self.pose.copy()
        invalid[12] = float("nan")
        with mock.patch("builtins.print"):
            sample = self.publish(100.01, left=invalid, right=invalid)
        self.assertEqual((sample.left_hand_timestamp, sample.right_hand_timestamp), (0.0, 0.0))
        sample = self.publish(100.02, "left")
        self.assertEqual((sample.left_hand_timestamp, sample.right_hand_timestamp), (100.02, 0.0))
        self.assertEqual(self.tele_vuer.get_tracking_diagnostics()["left_state"], "tracking")

    def test_no_new_events_expires_last_sample(self):
        sample = self.settle(100.0)
        self.assertTrue(self.is_fresh(sample, 0.25, now=100.25))
        self.assertFalse(self.is_fresh(sample, 0.25, now=100.251))

    def test_both_missing_still_commits_readable_snapshot_and_clear_diagnostics(self):
        self.settle(100.0)
        self.publish(100.01)
        result = self.tele_vuer.get_tracking_diagnostics()
        self.assertEqual(result["left_state"], "missing")
        self.assertEqual(result["right_state"], "missing")
        self.assertFalse(result["left_fresh"])
        self.assertFalse(result["right_fresh"])
        self.assertIsNone(result["pair_age_ms"])
        self.assertTrue(result["ready"])

    def test_wrapper_exports_both_timestamps_from_the_same_snapshot(self):
        import sys
        source_dir = TELEVUER_PATH.parent
        package = types.ModuleType("televuer_freshness_test")
        package.__path__ = [str(source_dir)]
        module = types.ModuleType("televuer_freshness_test.televuer")
        module.TeleVuer = object
        name = "televuer_freshness_test.tv_wrapper"
        with mock.patch.dict(sys.modules, {
            "televuer_freshness_test": package,
            "televuer_freshness_test.televuer": module,
        }):
            spec = importlib.util.spec_from_file_location(name, source_dir / "tv_wrapper.py")
            wrapper_module = importlib.util.module_from_spec(spec)
            with mock.patch.dict(sys.modules, {name: wrapper_module}):
                spec.loader.exec_module(wrapper_module)
            wrapper = wrapper_module.TeleVuerWrapper.__new__(wrapper_module.TeleVuerWrapper)
            self.settle(100.0)
            self.publish(100.03, "left")
            wrapper.use_hand_tracking = True
            wrapper.return_hand_rot_data = False
            wrapper.arm_reference_mode = "head_yaw"
            wrapper.tvuer = types.SimpleNamespace(
                head_pose=np.eye(4),
                get_hand_motion_snapshot=self.tele_vuer.get_hand_motion_snapshot,
            )
            wrapper._last_hand_motion_snapshot = {}
            wrapper._tele_data_lock = wrapper_module.threading.Lock()
            sample = wrapper.get_tele_data()
            self.assertEqual(sample.left_hand_timestamp, 100.03)
            self.assertEqual(sample.right_hand_timestamp, 0.0)
            self.assertEqual(sample.motion_data_timestamp, 0.0)

    def test_runtime_diagnostics_precede_tracking_hold_early_continue(self):
        source = (STAGE_ROOT / "teleop" / "teleop_hand_and_arm.py").read_text()
        tree = ast.parse(source)
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While)
                    and any(isinstance(child, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "loop_monotonic"
                        for target in child.targets) for child in node.body))
        diagnostic = next(index for index, node in enumerate(loop.body)
                          if "get_tracking_diagnostics" in ast.unparse(node))
        first_continue = next(index for index, node in enumerate(loop.body)
                              if any(isinstance(child, ast.Continue) for child in ast.walk(node)))
        self.assertLess(diagnostic, first_continue)
        self.assertIn("+ 2.0", ast.unparse(loop.body[diagnostic]))


if __name__ == "__main__":
    unittest.main()
