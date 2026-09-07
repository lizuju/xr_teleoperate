import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "teleop" / "robot_control"))
from r1_head_waist import R1HeadWaistFollower, compensate_wrist_for_waist


def rotation_yaw(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def rotation_pitch(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def head_pose(yaw):
    pose = np.eye(4)
    pose[:3, :3] = rotation_yaw(yaw)
    return pose


class R1HeadWaistTest(unittest.TestCase):
    def test_short_glance_and_small_turn_do_not_move_waist(self):
        follower = R1HeadWaistFollower(0.1, 0.0)
        for index in range(1, 201):
            now = index * 0.01
            yaw = math.radians(45.0 if now < 0.3 else 15.0)
            head, waist = follower.update(head_pose(yaw), np.eye(4), 0.1, now)
            self.assertAlmostEqual(waist, 0.1)
            self.assertAlmostEqual(head[1], yaw)
            self.assertFalse(follower.following)

    def test_sustained_turn_follows_then_stops_near_center(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        actual = 0.0
        observed_following = False
        for index in range(1, 1001):
            head, actual = follower.update(
                head_pose(math.radians(25.0)), np.eye(4), actual, index * 0.01,
            )
            observed_following |= follower.following
        self.assertTrue(observed_following)
        self.assertFalse(follower.following)
        self.assertGreater(actual, math.radians(20.0))
        self.assertLessEqual(actual, math.radians(25.0))
        self.assertLess(abs(head[1]), math.radians(5.0))
        settled = actual
        for index in range(1001, 1101):
            _, actual = follower.update(
                head_pose(math.radians(25.0)), np.eye(4), actual, index * 0.01,
            )
        self.assertAlmostEqual(actual, settled)

    def test_alternating_short_glances_do_not_accumulate_dwell(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        for index in range(1, 301):
            yaw = 0.8 if (index // 20) % 2 == 0 else -0.8
            _, waist = follower.update(head_pose(yaw), np.eye(4), 0.0, index * 0.01)
            self.assertEqual(waist, 0.0)
            self.assertFalse(follower.following)

    def test_speed_acceleration_angle_bounds_and_smooth_reversal(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        targets = [0.0]
        dt = 0.01
        for index in range(1, 1801):
            yaw = math.radians(65.0 if index < 160 else -65.0)
            _, target = follower.update(head_pose(yaw), np.eye(4), targets[-1], index * dt)
            targets.append(target)
        velocities = np.diff(targets) / dt
        accelerations = np.diff(np.concatenate(([0.0], velocities))) / dt
        self.assertLessEqual(np.max(np.abs(targets)), 2.618 + 1e-12)
        self.assertLessEqual(np.max(np.abs(velocities)), 0.35 + 1e-12)
        self.assertLessEqual(np.max(np.abs(accelerations)), 0.5 + 1e-10)
        self.assertGreater(targets[150], 0.1)
        self.assertLess(targets[-1], -math.radians(60.0))
        self.assertGreaterEqual(targets[-1], -math.radians(65.0))

    def test_waist_can_pass_90_degrees_and_stops_at_urdf_limit(self):
        for direction in (-1.0, 1.0):
            follower = R1HeadWaistFollower(0.0, 0.0)
            actual = 0.0
            for index in range(1, 2001):
                _, actual = follower.update(
                    head_pose(direction * math.radians(175.0)), np.eye(4), actual, index * 0.01,
                )
                self.assertLessEqual(abs(actual), 2.618)
            self.assertAlmostEqual(actual, direction * 2.618, places=5)

    def test_changed_goal_inside_stopping_distance_does_not_jump_velocity(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        targets = [0.0]
        for index in range(1, 151):
            _, target = follower.update(head_pose(1.0), np.eye(4), 0.0, index * 0.01)
            targets.append(target)
        changed_goal = targets[-1] + 0.001
        for index in range(151, 251):
            _, target = follower.update(head_pose(changed_goal), np.eye(4), 0.0, index * 0.01)
            targets.append(target)
        velocity = np.diff(targets) / 0.01
        acceleration = np.diff(np.concatenate(([0.0], velocity))) / 0.01
        self.assertLessEqual(np.max(np.abs(acceleration)), 0.5 + 1e-10)

    def test_heading_unwrap_crosses_pi_without_direction_flip(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        totals = []
        for index, yaw in enumerate((170.0, 179.0, -179.0, -170.0), 1):
            follower.update(head_pose(math.radians(yaw)), np.eye(4), 0.0, index * 0.01)
            totals.append(math.degrees(follower.total_yaw))
        np.testing.assert_allclose(totals, [170.0, 179.0, 181.0, 190.0], atol=1e-12)

    def test_nearly_vertical_heading_noise_cannot_start_waist_motion(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        for index in range(1, 301):
            current = np.eye(4)
            noise_heading = math.sin(index * 1.7) * math.pi
            current[:3, :3] = rotation_yaw(noise_heading) @ rotation_pitch(math.pi / 2.0 - 0.01)
            _, waist = follower.update(current, np.eye(4), 0.0, index * 0.01)
            self.assertEqual(waist, 0.0)
            self.assertEqual(follower.total_yaw, 0.0)
            self.assertFalse(follower.following)

    def test_nearly_vertical_heading_brakes_then_recovery_requires_new_dwell(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        targets = [0.0]
        for index in range(1, 131):
            _, target = follower.update(head_pose(0.8), np.eye(4), targets[-1], index * 0.01)
            targets.append(target)
        self.assertTrue(follower.following)
        retained_heading = follower.total_yaw
        for index in range(131, 251):
            current = np.eye(4)
            current[:3, :3] = rotation_yaw(math.sin(index) * math.pi) @ rotation_pitch(-math.pi / 2.0 + 0.01)
            _, target = follower.update(current, np.eye(4), targets[-1], index * 0.01)
            targets.append(target)
            self.assertEqual(follower.total_yaw, retained_heading)
            self.assertFalse(follower.following)
        velocities = np.diff(targets) / 0.01
        acceleration = np.diff(np.concatenate(([0.0], velocities))) / 0.01
        self.assertLessEqual(np.max(np.abs(acceleration)), 0.5 + 1e-10)
        self.assertAlmostEqual(targets[-1], targets[-2])
        stopped_target = targets[-1]
        for index in range(251, 290):
            _, target = follower.update(head_pose(1.0), np.eye(4), stopped_target, index * 0.01)
            self.assertEqual(target, stopped_target)
            self.assertFalse(follower.following)
        follower.update(head_pose(1.0), np.eye(4), stopped_target, 2.92)
        self.assertTrue(follower.following)

    def test_heading_recovery_selects_nearest_unwrapped_branch(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        follower.update(head_pose(math.radians(170.0)), np.eye(4), 0.0, 0.01)
        vertical = np.eye(4)
        vertical[:3, :3] = rotation_yaw(-1.0) @ rotation_pitch(math.pi / 2.0)
        follower.update(vertical, np.eye(4), 0.0, 0.02)
        follower.update(head_pose(math.radians(-179.0)), np.eye(4), 0.0, 0.03)
        self.assertAlmostEqual(math.degrees(follower.total_yaw), 181.0)
        self.assertFalse(follower.following)
        follower.update(head_pose(math.radians(-170.0)), np.eye(4), 0.0, 0.04)
        self.assertAlmostEqual(math.degrees(follower.total_yaw), 190.0)

    def test_large_heading_preserves_pitch_branch_and_saturates_head(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        current = np.eye(4)
        current[:3, :3] = rotation_pitch(0.2) @ rotation_yaw(math.radians(100.0))
        head, _ = follower.update(current, np.eye(4), 0.0, 0.01)
        np.testing.assert_allclose(head, [0.2, math.radians(100.0)], atol=1e-12)
        for index, angle in enumerate((170.0, 179.0, -179.0, -170.0), 2):
            head, _ = follower.update(head_pose(math.radians(angle)), np.eye(4), 0.0, index * 0.01)
            self.assertAlmostEqual(head[0], 0.0)
            self.assertAlmostEqual(head[1], 2.0071)

    def test_timeout_reanchors_to_actual_and_requires_new_dwell(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        for index in range(1, 81):
            follower.update(head_pose(0.8), np.eye(4), 0.0, index * 0.01)
        self.assertTrue(follower.following)
        self.assertGreater(follower.waist_target, 0.0)
        _, target = follower.update(head_pose(0.8), np.eye(4), 0.05, 1.2)
        self.assertEqual(target, 0.05)
        self.assertFalse(follower.following)
        for index in range(1, 40):
            _, target = follower.update(head_pose(0.8), np.eye(4), 0.05, 1.2 + index * 0.01)
            self.assertEqual(target, 0.05)
            self.assertFalse(follower.following)

    def test_head_uses_actual_waist_and_pitch_before_yaw(self):
        reference = head_pose(0.2)
        current = reference.copy()
        desired = rotation_pitch(0.4) @ rotation_yaw(0.7)
        current[:3, :3] = reference[:3, :3] @ desired
        follower = R1HeadWaistFollower(0.17, 0.0)
        head, _ = follower.update(current, reference, 0.43, 0.01)
        reconstructed = rotation_yaw(0.43 - 0.17) @ rotation_pitch(head[0]) @ rotation_yaw(head[1])
        np.testing.assert_allclose(reconstructed[:, 0], desired[:, 0], atol=1e-12)
        head_at_reference, _ = follower.update(current, reference, 0.17, 0.02)
        np.testing.assert_allclose(head_at_reference, [0.4, 0.7], atol=1e-12)

    def test_explicit_reset_after_short_stale_clears_motion_and_dwell(self):
        follower = R1HeadWaistFollower(0.17, 0.0, tracking_timeout=0.25)
        for index in range(1, 81):
            follower.update(head_pose(0.8), np.eye(4), 0.17, index * 0.01)
        self.assertTrue(follower.following)
        self.assertGreater(follower.waist_target, 0.17)
        follower.reset(0.22, 0.81)
        self.assertEqual(follower.waist_reference, 0.17)
        self.assertEqual(follower.tracking_timeout, 0.25)
        self.assertFalse(follower.following)
        self.assertEqual(follower.waist_target, 0.22)
        for index in range(1, 41):
            _, target = follower.update(head_pose(0.8), np.eye(4), 0.22, 0.81 + index * 0.01)
            self.assertEqual(target, 0.22)
            self.assertFalse(follower.following)
        _, target = follower.update(head_pose(0.8), np.eye(4), 0.22, 1.23)
        self.assertTrue(follower.following)
        self.assertGreater(target, 0.22)
        self.assertLessEqual((target - 0.22) / 0.02, 0.5 * 0.02 + 1e-12)

    def test_wrist_compensation_rotates_complete_pose_and_roundtrips(self):
        target = np.eye(4)
        target[:3, :3] = rotation_pitch(0.2) @ rotation_yaw(-0.4)
        target[:3, 3] = [0.5, 0.3, 0.4]
        original = target.copy()
        compensated = compensate_wrist_for_waist(target, 0.43, 0.17)
        np.testing.assert_allclose(compensated[:3, :3], rotation_yaw(-0.26) @ target[:3, :3], atol=1e-12)
        np.testing.assert_allclose(compensated[:3, 3], rotation_yaw(-0.26) @ target[:3, 3], atol=1e-12)
        np.testing.assert_allclose(compensate_wrist_for_waist(compensated, 0.17, 0.43), target, atol=1e-12)
        np.testing.assert_array_equal(target, original)

    def test_compensation_pivot_matches_current_r1_urdf(self):
        joint = ET.parse(REPO_ROOT / "assets" / "r1" / "r1_a7.urdf").getroot().find("joint[@name='waist_yaw_joint']")
        self.assertEqual(joint.find("origin").attrib, {"xyz": "0 0 0", "rpy": "0 0 0"})
        self.assertEqual(joint.find("axis").get("xyz"), "0 0 1")
        self.assertEqual(float(joint.find("limit").get("lower")), -2.618)
        self.assertEqual(float(joint.find("limit").get("upper")), 2.618)

    def test_nonfinite_or_malformed_inputs_are_rejected_before_state_changes(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                R1HeadWaistFollower(invalid, 0.0)
            with self.assertRaises(ValueError):
                R1HeadWaistFollower(0.0, invalid)
            with self.assertRaises(ValueError):
                R1HeadWaistFollower(0.0, 0.0, invalid)
            follower = R1HeadWaistFollower(0.0, 0.0)
            with self.assertRaises(ValueError):
                follower.update(np.eye(4), np.eye(4), invalid, 0.01)
            with self.assertRaises(ValueError):
                follower.update(np.eye(4), np.eye(4), 0.0, invalid)
            with self.assertRaises(ValueError):
                follower.reset(invalid, 0.01)
            with self.assertRaises(ValueError):
                follower.reset(0.0, invalid)
            pose = np.eye(4)
            pose[1, 2] = invalid
            with self.assertRaises(ValueError):
                follower.update(pose, np.eye(4), 0.0, 0.01)
            with self.assertRaises(ValueError):
                follower.update(np.eye(4), pose, 0.0, 0.01)
            with self.assertRaises(ValueError):
                compensate_wrist_for_waist(pose, 0.0, 0.0)
            with self.assertRaises(ValueError):
                compensate_wrist_for_waist(np.eye(4), invalid, 0.0)
            self.assertEqual(follower.waist_target, 0.0)
            self.assertFalse(follower.following)
        with self.assertRaises(ValueError):
            R1HeadWaistFollower(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            R1HeadWaistFollower(0.0, 0.0).update(np.eye(3), np.eye(4), 0.0, 0.01)
        with self.assertRaises(ValueError):
            R1HeadWaistFollower(0.0, 1.0).update(np.eye(4), np.eye(4), 0.0, 0.5)


if __name__ == "__main__":
    unittest.main()
