import unittest

import numpy as np

from teleop.robot_control.r1_wrist_workspace import R1WristWorkspace
import test_r1_head_waist_integration as integration


def pose(x, y=0.0, z=0.0):
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


class WristWorkspaceTest(unittest.TestCase):
    def test_overreach_is_consumed_and_reverse_motion_retracts(self):
        workspace = R1WristWorkspace(pose(0.4))
        for x in np.linspace(0.4, 0.7, 61):
            workspace.observe(pose(x), pose(0.4))
        self.assertAlmostEqual(workspace.target(pose(0.7))[0, 3], 0.42)
        self.assertAlmostEqual(workspace.target(pose(0.6))[0, 3], 0.305)

    def test_stationary_hand_does_not_drift_and_inward_motion_is_not_erased(self):
        workspace = R1WristWorkspace(pose(0.4))
        workspace.observe(pose(0.7), pose(0.4))
        offset = workspace.offset.copy()
        for _ in range(100):
            workspace.observe(pose(0.7), pose(0.35))
        np.testing.assert_array_equal(workspace.offset, offset)
        target = workspace.target(pose(0.68))
        workspace.observe(pose(0.68), pose(0.35))
        self.assertLess(workspace.offset[0], offset[0])
        self.assertLess(target[0, 3], 0.4)
        np.testing.assert_allclose(workspace.target(pose(0.68)), target)

    def test_reversal_correction_is_bounded_by_motion_and_committed_only_once(self):
        workspace = R1WristWorkspace(pose(0.4))
        workspace.observe(pose(0.7), pose(0.4))
        offset = workspace.offset.copy()
        raw = pose(0.698, 0.01)
        expected = pose(0.416, 0.01)
        for _ in range(10):
            np.testing.assert_allclose(workspace.target(raw), expected)
            np.testing.assert_array_equal(workspace.offset, offset)
        workspace.observe(raw, pose(0.4, 0.01))
        np.testing.assert_allclose(workspace.target(raw), expected)
        for _ in range(100):
            workspace.observe(raw, pose(0.4, 0.01))
            np.testing.assert_allclose(workspace.target(raw), expected)

    def test_return_assist_ends_once_solver_catches_up(self):
        workspace = R1WristWorkspace(pose(0.4))
        workspace.observe(pose(0.7), pose(0.4))
        raw = pose(0.6975)
        target = workspace.target(raw)
        workspace.observe(raw, target)
        np.testing.assert_array_equal(workspace.return_residual, np.zeros(3))
        offset = workspace.offset.copy()
        for x in np.r_[np.linspace(0.69, 0.6, 20), np.linspace(0.6, 0.69, 20)]:
            raw = pose(x)
            target = workspace.target(raw)
            workspace.observe(raw, target)
            np.testing.assert_array_equal(workspace.offset, offset)
            np.testing.assert_allclose(target[:3, 3], raw[:3, 3] + offset)

    def test_hold_discards_pending_return_assist_without_changing_offset(self):
        workspace = R1WristWorkspace(pose(0.4))
        workspace.observe(pose(0.7), pose(0.4))
        offset = workspace.offset.copy()
        workspace.hold()
        np.testing.assert_array_equal(workspace.return_residual, np.zeros(3))
        np.testing.assert_allclose(workspace.target(pose(0.6)), pose(0.6 + offset[0]))

    def test_large_residual_after_occlusion_cannot_grant_unbounded_return_assist(self):
        workspace = R1WristWorkspace(pose(0.4))
        workspace.hold()
        workspace.observe(pose(0.9), pose(0.4))
        workspace.observe(pose(0.905), pose(0.4))
        offset = workspace.offset.copy()
        for x in np.linspace(0.905, 0.7, 42):
            raw = pose(x)
            workspace.target(raw)
            workspace.observe(raw, pose(0.4))
        self.assertLessEqual(np.linalg.norm(workspace.offset - offset), 0.015 + 1e-12)

    def test_reachable_motion_with_small_solver_residual_preserves_mapping(self):
        workspace = R1WristWorkspace(pose(0.2))
        for x in np.r_[np.linspace(0.2, 0.4, 41), np.linspace(0.4, 0.2, 41)]:
            raw = pose(x)
            workspace.observe(raw, pose(x - 0.01))
            np.testing.assert_array_equal(workspace.target(raw), raw)

    def test_occlusion_keeps_offset_without_consuming_reacquisition_jump(self):
        workspace = R1WristWorkspace(pose(0.4))
        workspace.observe(pose(0.7), pose(0.4))
        offset = workspace.offset.copy()
        workspace.hold()
        workspace.observe(pose(0.9), pose(0.4))
        np.testing.assert_array_equal(workspace.offset, offset)
        workspace.observe(pose(0.91), pose(0.4))
        self.assertAlmostEqual(workspace.offset[0] - offset[0], -0.01)

    def test_orientation_and_tangential_motion_are_preserved(self):
        workspace = R1WristWorkspace(pose(0.4))
        raw = pose(0.7)
        raw[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        workspace.observe(raw, pose(0.4))
        raw[1, 3] = 0.1
        workspace.observe(raw, pose(0.4, 0.1))
        target = workspace.target(raw)
        np.testing.assert_array_equal(target[:3, :3], raw[:3, :3])
        self.assertAlmostEqual(target[1, 3], 0.1)
        self.assertAlmostEqual(target[0, 3], 0.42)


class WristWorkspaceControlTest(unittest.TestCase):
    def setUp(self):
        self.fixture = integration.R1HeadWaistIntegrationTest()
        self.fixture.setUp()

    def tick(self, ns):
        integration.execute(self.fixture.control_nodes, ns, loop=True)

    def test_world_and_torso_frames_retract_and_keep_other_hand_independent(self):
        for mode in ('world', 'torso'):
            with self.subTest(mode=mode):
                ns, sample = self.fixture.independent_context()
                ns['args'].waist_follow_compensation = mode
                ik = ns['arm_ik']
                ik.forward_wrist_poses.side_effect = None
                poses = (self.fixture.left, self.fixture.right)
                ik.forward_wrist_poses.return_value = tuple(
                    integration.compensate_wrist_for_waist(p, 0.43, 0.17)
                    if mode == 'world' else p.copy() for p in poses
                )
                for x in np.linspace(0.4, 0.7, 61):
                    sample.left_wrist_pose = self.fixture.left.copy()
                    sample.left_wrist_pose[0, 3] = x
                    self.tick(ns)
                sample.left_wrist_pose[0, 3] = 0.6
                self.tick(ns)
                self.assertAlmostEqual(ns['left_wrist_target'][0, 3], 0.305)
                np.testing.assert_allclose(ns['right_wrist_target'], self.fixture.right)
                np.testing.assert_allclose(ns['wrist_workspaces'][1].offset, np.zeros(3), atol=1e-12)
                np.testing.assert_array_equal(ik.forward_wrist_poses.call_args.args[0], ik.last_raw_q)

    def test_absent_or_aged_hand_cannot_accumulate_workspace_offset(self):
        for timestamp in (0.0, 0.1):
            ns, sample = self.fixture.independent_context(timestamp, 0.99)
            sample.left_wrist_pose = pose(1.0, 0.2, 0.8)
            ns['arm_ik'].forward_wrist_poses.side_effect = None
            self.tick(ns)
            np.testing.assert_array_equal(ns['wrist_workspaces'][0].offset, np.zeros(3))
            self.assertIsNone(ns['wrist_workspaces'][0].previous_position)

    def test_lost_hand_during_ik_cannot_commit_workspace_correction(self):
        ns, sample = self.fixture.independent_context()
        sample.left_wrist_pose = pose(1.0, 0.2, 0.8)
        ns['arm_ik'].forward_wrist_poses.side_effect = None
        def lose_hand(*args, **kwargs):
            sample.left_hand_timestamp = 0.0
            return np.zeros(14), np.zeros(14)
        ns['arm_ik'].solve_ik.side_effect = lose_hand
        self.tick(ns)
        np.testing.assert_array_equal(ns['wrist_workspaces'][0].offset, np.zeros(3))
        ns['arm_ctrl'].ctrl_dual_arm_and_head.assert_not_called()

    def test_stale_sample_cannot_apply_an_uncommitted_return_correction(self):
        ns, sample = self.fixture.independent_context(0.1, 0.99)
        ns['args'].waist_follow = False
        workspace = ns['wrist_workspaces'][0]
        extended = self.fixture.left.copy()
        extended[0, 3] += 0.2
        workspace.observe(extended, self.fixture.left)
        offset = workspace.offset.copy()
        sample.left_wrist_pose = extended.copy()
        sample.left_wrist_pose[0, 3] -= 0.005
        self.tick(ns)
        expected = sample.left_wrist_pose.copy()
        expected[:3, 3] += offset
        np.testing.assert_allclose(ns['left_wrist_target'], expected)
        np.testing.assert_array_equal(workspace.offset, offset)
        np.testing.assert_array_equal(workspace.return_residual, np.zeros(3))


if __name__ == '__main__':
    unittest.main()
