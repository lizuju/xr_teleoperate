import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

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
                        target = hold.prepare(pose(step * 0.01, step * 0.01), side == moving_side)
                        hold.commit(target)
                        np.testing.assert_allclose(target, pose(step * 0.01, step * 0.01) if side == moving_side else pose())

    def test_reacquisition_reclutches_without_position_or_rotation_jump(self):
        hold = R1WristHold(pose(0.2, 0.3))
        hold.hold()
        resumed = hold.prepare(pose(0.8, -0.7), True)
        np.testing.assert_allclose(resumed, pose(0.2, 0.3), atol=1e-12)
        hold.commit(resumed)
        moved = hold.prepare(pose(0.82, -0.6), True)
        np.testing.assert_allclose(moved, pose(0.22, 0.4), atol=1e-12)

    def test_unpublished_candidate_is_not_used_as_the_held_pose(self):
        hold = R1WristHold(pose(0.2))
        hold.prepare(pose(0.3), True)
        hold.hold()
        np.testing.assert_allclose(hold.prepare(pose(0.8), True), pose(0.2))

    def test_reclutch_preserves_world_axes_for_noncommuting_rotations(self):
        held = pose(0.2, 0.3)
        raw = pose(0.8)
        raw[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        hold = R1WristHold(held)
        hold.hold()
        resumed = hold.prepare(raw, True)
        np.testing.assert_allclose(resumed, held, atol=1e-12)
        hold.commit(resumed)
        delta = pose(yaw=0.1)[:3, :3]
        raw[:3, :3] = delta @ raw[:3, :3]
        raw[1, 3] += 0.02
        target = hold.prepare(raw, True)
        np.testing.assert_allclose(target[:3, :3], delta @ held[:3, :3], atol=1e-12)
        np.testing.assert_allclose(target[:3, 3], held[:3, 3] + [0, 0.02, 0], atol=1e-12)

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


if __name__ == "__main__":
    unittest.main()
