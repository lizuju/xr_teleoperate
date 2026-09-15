import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teleop" / "robot_control"))
from r1_hand_tracking import R1WristHold, hand_tracking_freshness


def pose(x=0.0, yaw=0.0):
    result = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[0, 3] = x
    return result


class R1IndependentHandTrackingTest(unittest.TestCase):
    def test_freshness_is_separate_and_rejects_invalid_times(self):
        sample = SimpleNamespace(motion_data_ready=True, left_hand_timestamp=100.4, right_hand_timestamp=100.0)
        self.assertEqual(hand_tracking_freshness(sample, 0.25, 100.4), (True, False))
        sample.left_hand_timestamp, sample.right_hand_timestamp = 100.0, 100.4
        self.assertEqual(hand_tracking_freshness(sample, 0.25, 100.4), (False, True))
        for timestamp in (0.0, -1.0, 100.5, float("nan"), float("inf")):
            sample.left_hand_timestamp = timestamp
            self.assertFalse(hand_tracking_freshness(sample, 0.25, 100.4)[0])
        sample.motion_data_ready = False
        self.assertEqual(hand_tracking_freshness(sample, 0.25, 100.4), (False, False))

    def test_each_arm_can_move_while_the_other_holds(self):
        for moving_side in (0, 1):
            with self.subTest(moving_side=moving_side):
                holds = [R1WristHold(pose()), R1WristHold(pose())]
                for step in range(1, 30):
                    for side, hold in enumerate(holds):
                        previous = hold.target.copy()
                        target = hold.prepare(pose(step * 0.01, step * 0.01), side == moving_side)
                        hold.commit(target)
                        if side == moving_side:
                            self.assertGreater(target[0, 3], previous[0, 3])
                            np.testing.assert_allclose(target, pose(step * 0.01, step * 0.01))
                        else:
                            np.testing.assert_allclose(target, pose())

    def test_reacquisition_holds_once_then_passes_the_complete_latest_target(self):
        hold = R1WristHold(pose(0.2, 0.3))
        hold.hold()
        raw = pose(0.3, -0.2)
        resumed = hold.prepare(raw, True)
        np.testing.assert_allclose(resumed, pose(0.2, 0.3), atol=1e-12)
        hold.commit(resumed)
        latest = pose(0.4, 1.2)
        target = hold.prepare(latest, True)
        np.testing.assert_array_equal(target, latest)
        np.testing.assert_array_equal(hold.target, resumed)
        hold.commit(target)
        np.testing.assert_array_equal(hold.target, latest)

    def test_unpublished_candidate_is_not_used_as_the_held_pose(self):
        hold = R1WristHold(pose(0.2))
        hold.prepare(pose(0.3), True)
        hold.hold()
        np.testing.assert_allclose(hold.prepare(pose(0.8), True), pose(0.2))

    def test_90_and_180_degree_recovery_has_no_multi_second_rotation_ramp(self):
        held = pose(0.2, 0.3)
        for angle in (np.pi / 2, np.pi - 1e-7, np.pi):
            with self.subTest(angle=angle):
                raw = pose(0.22)
                raw[:3, :3] = Rotation.from_rotvec([angle, 0, 0]).as_matrix() @ held[:3, :3]
                hold = R1WristHold(held)
                hold.hold()
                first = hold.prepare(raw, True)
                np.testing.assert_array_equal(first, held)
                hold.commit(first)
                target = hold.prepare(raw, True)
                np.testing.assert_array_equal(target, raw)
                rotation = target[:3, :3]
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
                hold.commit(target)
                np.testing.assert_array_equal(hold.target, raw)

    def test_repeated_occlusions_do_not_accumulate_pose_offset(self):
        hold = R1WristHold(pose())
        for cycle in range(10):
            hold.hold()
            hold.commit(hold.prepare(pose(0.01, math.radians(5)), True))
            for step in range(60):
                raw = pose(0.01, math.radians(5)) if step < 15 else pose()
                hold.commit(hold.prepare(raw, True))
            np.testing.assert_allclose(hold.target, pose(), atol=1e-8)

    def test_recovery_does_not_cap_any_translation_axis_or_rotation(self):
        hold = R1WristHold(pose())
        hold.hold()
        raw = pose(0.5, 1.0)
        raw[1:3, 3] = [-0.4, 0.3]
        hold.commit(hold.prepare(raw, True))
        np.testing.assert_array_equal(hold.prepare(raw, True), raw)

    def test_normal_tracking_has_no_extra_wrist_filter(self):
        hold = R1WristHold(pose())
        for step in range(120):
            sign = (-1) ** step
            raw = pose(sign * 0.002, sign * math.radians(1))
            target = hold.prepare(raw, True)
            np.testing.assert_array_equal(target, raw)
            hold.commit(target)

    def test_continuous_translation_and_rotation_follow_each_latest_sample_after_recovery(self):
        hold = R1WristHold(pose())
        hold.hold()
        hold.commit(hold.prepare(pose(0.1), True))
        for step in range(1, 30):
            raw = pose(0.1 + step * 0.02, step * math.radians(10))
            target = hold.prepare(raw, True)
            hold.commit(target)
            np.testing.assert_array_equal(target, raw)

    def test_repeated_occlusion_preserves_the_last_committed_target(self):
        hold = R1WristHold(pose(0.2))
        for x in (0.8, -0.1, 0.6):
            hold.hold()
            target = hold.prepare(pose(x), True)
            np.testing.assert_allclose(target, hold.target)
            hold.commit(target)

    def test_held_target_is_not_modified_by_callers(self):
        source = pose(0.2)
        hold = R1WristHold(source)
        source[0, 3] = 9.0
        result = hold.prepare(pose(1.0), False)
        result[0, 3] = 8.0
        np.testing.assert_allclose(hold.target, pose(0.2))

    def test_commit_and_following_targets_are_copies(self):
        hold = R1WristHold(pose())
        raw = pose(0.3, 1.2)
        target = hold.prepare(raw, True)
        raw[0, 3] = 9.0
        np.testing.assert_array_equal(target, pose(0.3, 1.2))
        hold.commit(target)
        target[0, 3] = 8.0
        np.testing.assert_array_equal(hold.target, pose(0.3, 1.2))

    def test_repeated_losses_hold_last_commit_and_never_unpublished_candidates(self):
        hold = R1WristHold(pose(0.2, 0.3))
        for cycle in range(5):
            committed = hold.target.copy()
            for _ in range(3):
                np.testing.assert_array_equal(hold.prepare(pose(2.0, 2.0), False), committed)
                self.assertFalse(hold.tracking)
            first = hold.prepare(pose(0.8, 1.0), True)
            np.testing.assert_array_equal(first, committed)
            self.assertTrue(hold.tracking)
            unpublished = hold.prepare(pose(1.2, 2.0), True)
            np.testing.assert_array_equal(unpublished, pose(1.2, 2.0))
            np.testing.assert_array_equal(hold.prepare(pose(3.0), False), committed)
            np.testing.assert_array_equal(hold.prepare(pose(4.0), True), committed)
            latest = pose((cycle + 1) * 0.1, (cycle + 1) * 0.2)
            hold.commit(hold.prepare(latest, True))
            np.testing.assert_array_equal(hold.target, latest)


if __name__ == "__main__":
    unittest.main()
