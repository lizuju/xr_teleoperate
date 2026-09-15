import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from test_televuer_motion_snapshot import load_televuer_class


Guard = load_televuer_class().on_hand_move.__globals__["HandPoseGuard"]


def pose(x=0.0, angle=0.0):
    result = np.eye(4)
    result[0, 3] = x
    result[:3, :3] = Rotation.from_rotvec([0, angle, 0]).as_matrix()
    return result


class HandPoseGuardTest(unittest.TestCase):
    def setUp(self):
        self.guard = Guard()
        for step in range(13):
            self.guard.accept_wrist(pose(), 100 + step / 30)
        self.assertTrue(self.guard.tracking)

    def test_normal_slow_movement_and_jitter_pass(self):
        for step in range(1, 121):
            target = pose(0.003 * step + (-1)**step * 0.002, 0.015 * step)
            self.assertTrue(self.guard.accept_wrist(target, 100.4 + step / 30))

    def test_large_rotation_without_translation_is_not_speed_limited(self):
        target = pose(angle=np.deg2rad(90))
        self.assertTrue(self.guard.accept_wrist(target, 100.43))
        np.testing.assert_array_equal(self.guard.pose, target)

    def test_valid_fast_motion_is_not_classified_as_tracking_loss(self):
        for step in range(1, 61):
            target = pose(0.30 + (-1)**step * 0.08, (-1)**step * 0.7)
            self.assertTrue(self.guard.accept_wrist(target, 100.4 + step / 30))
            np.testing.assert_array_equal(self.guard.pose, target)

    def test_new_valid_pose_after_long_gap_does_not_require_stationarity(self):
        target = pose(0.3, 1.0)
        self.assertTrue(self.guard.accept_wrist(target, 110.0))
        np.testing.assert_array_equal(self.guard.pose, target)

    def test_reappearance_at_different_pose_resumes_on_first_valid_frame(self):
        self.guard.lose("missing")
        raw = pose(0.15, 0.4)
        self.assertTrue(self.guard.accept_wrist(raw, 100.5))
        np.testing.assert_allclose(self.guard.pose, raw)

    def test_continuous_movement_after_missing_does_not_reset_recovery(self):
        self.guard.lose("missing")
        for step in range(1, 121):
            raw = pose(0.2 * step / 30, 0.02 * step)
            self.assertTrue(self.guard.accept_wrist(raw, 100.4 + step / 30))
            np.testing.assert_array_equal(self.guard.pose, raw)
        self.assertEqual(self.guard.status, "tracking")

    def test_accepted_pose_is_copied(self):
        raw = pose(0.1)
        self.guard.accept_wrist(raw, 100.5)
        raw[0, 3] = 9.0
        self.assertEqual(self.guard.pose[0, 3], 0.1)


if __name__ == "__main__":
    unittest.main()
