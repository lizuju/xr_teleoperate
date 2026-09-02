import importlib.util
from multiprocessing import Array, Value
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
    tele_vuer.motion_sample_seq_shared = Value("L", 0, lock=True)
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


if __name__ == "__main__":
    unittest.main()
