"""Replay of left O6 + right Dex1-1 episodes, and the policy's eased start pose."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from teleop.utils.r1_episode_replay import (O6_DEX1, TWO_O6, ReplayError, ReplayPose, load_replay_plan,
                                            recorded_start_pose)

LIMITS = {"right": {"min_q_rad": 4.28, "max_q_rad": 5.4}}
LAUNCH = {"right_ee": "dex1", "dex1_transport": "lowcmd", "dex1_slot": 33, "dex1_kp": 5.0, "dex1_kd": 0.05,
          "dex1_grip_delta": 0.5}


def write_episode(directory, frames, info):
    manifest = {"schema": "xr_teleop_episode_v2", "status": "complete", "outcome": "success",
                "episode_id": 0, "frame_count": len(frames), "frames": "frames.jsonl", "text": {}, "info": info}
    (directory / "episode.json").write_text(json.dumps(manifest))
    (directory / "frames.jsonl").write_text("".join(json.dumps(frame) + "\n" for frame in frames))


def dex1_info(**changes):
    info = {"robot": "R1_A7", "end_effector": "linker_o6_left+dex1_1_right", "end_effectors": dict(O6_DEX1),
            "gripper_position_limits": LIMITS, "launch_arguments": dict(LAUNCH), "frequency": 40}
    info.update(changes)
    return info


def frame(idx, seconds, arm=0.1, grip=5.0, left_state=0.3, grip_state=5.2):
    mono = 10_000_000_000 + round(seconds * 1e9)
    return {
        "idx": idx, "colors": {}, "actions": {},
        "states": {
            "left_arm": {"qpos": [arm + 0.01 * i for i in range(7)]},
            "right_arm": {"qpos": [-arm - 0.01 * i for i in range(7)]},
            "left_ee": {"qpos": [left_state] * 6},
            "right_ee": {"qpos": [grip_state]},
            "body": {"qpos": [0.02, 0.08, -0.05]},
        },
        "sample": {
            "timestamp_ns": 1_000_000_000_000 + round(seconds * 1e9), "monotonic_ns": mono, "mode": "following",
            "commands": {
                "arm": {"published": {"arm_q": [arm] * 14, "arm_tau": [0.0] * 14, "head_q": [0.01, -0.02],
                                      "waist_q": 0.0, "monotonic_ns": mono, "sequence": idx + 1}},
                "hands": {"published": {
                    "left": {"q": [0.2] * 6, "mode": 1, "monotonic_ns": mono, "sequence": idx + 1},
                    "right": {"q": [grip], "mode": 1, "kp": 5.0, "kd": 0.05, "monotonic_ns": mono, "sequence": idx + 1},
                }},
            },
        },
    }


class Dex1ReplayPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.episode = Path(self.temporary.name) / "episode_0000"
        self.episode.mkdir()

    def test_a_dex1_episode_loads_its_gripper_command_in_rad(self):
        write_episode(self.episode, [frame(0, 0.0, grip=5.0), frame(1, 0.1, grip=4.5)], dex1_info())
        plan = load_replay_plan(self.episode)
        self.assertEqual(plan.end_effectors, O6_DEX1)
        self.assertEqual(plan.gripper_limits, {"min_q": 4.28, "max_q": 5.4})
        self.assertEqual(plan.dex1_settings, {"slot": 33, "kp": 5.0, "kd": 0.05, "grip_delta": 0.5})
        np.testing.assert_allclose(plan.waypoints[0].right_q, [5.0])
        np.testing.assert_allclose(plan.waypoints[1].right_q, [4.5])
        np.testing.assert_allclose(plan.waypoints[0].left_q, [0.2] * 6)

    def test_a_gripper_command_outside_the_recorded_range_is_refused(self):
        write_episode(self.episode, [frame(0, 0.0, grip=6.0)], dex1_info())
        with self.assertRaisesRegex(ReplayError, "outside the recorded gripper range"):
            load_replay_plan(self.episode)
        # A hair outside (within the tolerance) is clipped to the range.
        write_episode(self.episode, [frame(0, 0.0, grip=5.43)], dex1_info())
        np.testing.assert_allclose(load_replay_plan(self.episode).waypoints[0].right_q, [5.4])

    def test_other_hand_layouts_and_transports_are_refused(self):
        write_episode(self.episode, [frame(0, 0.0)], dex1_info(end_effectors={"left": "linker_o6", "right": "dex3"}))
        with self.assertRaisesRegex(ReplayError, "R1_A7"):
            load_replay_plan(self.episode)
        write_episode(self.episode, [frame(0, 0.0)],
                      dex1_info(launch_arguments=dict(LAUNCH, dex1_transport="service")))
        with self.assertRaisesRegex(ReplayError, "lowcmd"):
            load_replay_plan(self.episode)

    def test_the_recorded_start_pose_comes_from_measured_states(self):
        write_episode(self.episode, [frame(0, 0.0, arm=0.2), frame(1, 0.1, arm=0.4, grip_state=4.9)], dex1_info())
        pose, sides = recorded_start_pose(self.episode)
        self.assertEqual(sides, O6_DEX1)
        np.testing.assert_allclose(pose.arm_q[:7], [0.2 + 0.01 * i for i in range(7)])
        np.testing.assert_allclose(pose.arm_q[7:], [-0.2 - 0.01 * i for i in range(7)])
        np.testing.assert_allclose(pose.head_q, [0.08, -0.05])
        self.assertAlmostEqual(pose.waist_q, 0.02)
        np.testing.assert_allclose(pose.left_q, [0.3] * 6)
        np.testing.assert_allclose(pose.right_q, [5.2])
        self.assertEqual((pose.left_mode, pose.right_mode), (1, 1))
        np.testing.assert_allclose(recorded_start_pose(self.episode, 1)[0].right_q, [4.9])
        with self.assertRaisesRegex(ReplayError, "no frame 5"):
            recorded_start_pose(self.episode, 5)


class ReplayToolHandTests(unittest.TestCase):
    def test_the_hand_controller_matches_the_recorded_hands(self):
        import replay_r1_episode_on_robot as tool
        arm = Mock()
        dex1_plan = SimpleNamespace(end_effectors=dict(O6_DEX1), gripper_limits={"min_q": 4.28, "max_q": 5.4},
                                    dex1_settings={"slot": 33, "kp": 5.0, "kd": 0.05, "grip_delta": 0.5})
        with patch("teleop.robot_control.robot_hand_linker_o6.LinkerO6Controller") as o6, \
                patch("teleop.robot_control.robot_hand_dex1_r1.Dex1LowCmdGripper") as gripper, \
                patch("teleop.robot_control.robot_hand_dex1_r1.O6Dex1Hands") as pair:
            hands = tool.make_hand_controller(arm, dex1_plan)
            o6.assert_called_once_with(sides=("left",))
            gripper.assert_called_once_with(arm, slot=33, kp=5.0, kd=0.05, grip_delta=0.5, closed_q=4.28, open_q=5.4)
            pair.assert_called_once_with(o6.return_value, gripper.return_value)
            self.assertIs(hands, pair.return_value)
        with patch("teleop.robot_control.robot_hand_linker_o6.LinkerO6Controller") as o6:
            self.assertIs(tool.make_hand_controller(arm, SimpleNamespace(end_effectors=dict(TWO_O6))), o6.return_value)
            o6.assert_called_once_with()

    def test_the_tool_builds_hands_through_the_layout(self):
        source = (ROOT / "tools/replay_r1_episode_on_robot.py").read_text(encoding="utf-8")
        self.assertIn("hand_ctrl = make_hand_controller(arm_ctrl, plan)", source)
        self.assertNotIn("hand_ctrl = LinkerO6Controller()", source)


class PolicyStartPoseTests(unittest.TestCase):
    def load(self):
        import run_r1_act_policy
        return run_r1_act_policy

    def robot(self):
        arm = Mock()
        arm.get_recording_snapshot.return_value = {"state": {"q": [0.0] * 14, "head_q": [0.0, 0.0], "waist_q": 0.0}}
        hand = Mock()
        hand.get_state.return_value = ([0.2] * 6, [4.8])
        return arm, hand

    @staticmethod
    def target(arm=0.3):
        return ReplayPose(arm_q=np.full(14, arm), arm_tau=np.zeros(14), head_q=np.array([0.08, -0.05]), waist_q=0.02,
                          left_q=np.full(6, 0.3), right_q=np.array([5.2]), left_mode=1, right_mode=1)

    def test_start_pose_arguments(self):
        runner = self.load()
        args = runner.parse_args(["policy.pt"])
        self.assertIsNone(args.start_pose)
        self.assertEqual((args.start_frame, args.start_speed, args.max_start_approach_seconds), (0, 0.3, 15.0))
        args = runner.parse_args(["policy.pt", "--start-pose", "/tmp/episode_0096", "--start-frame", "3"])
        self.assertEqual((args.start_pose, args.start_frame), (Path("/tmp/episode_0096"), 3))
        for bad in (["--start-speed", "0"], ["--start-speed", "1.5"], ["--start-frame", "-1"],
                    ["--max-start-approach-seconds", "0"]):
            with self.assertRaises(SystemExit):
                runner.parse_args(["policy.pt", *bad])

    def test_the_approach_is_slow_and_ends_on_the_start_pose(self):
        runner = self.load()
        arm, hand = self.robot()
        target = self.target(0.3)
        args = SimpleNamespace(start_speed=0.3, max_start_approach_seconds=15.0)
        period = 0.025
        with patch.object(runner.time, "sleep"):
            held = runner.ease_to_start(arm, hand, target, args, period, [False])
        self.assertIs(held, target)
        sent = [np.asarray(call.args[0]) for call in arm.ctrl_dual_arm_and_head.call_args_list]
        self.assertEqual(len(sent), 80)  # 0.3 rad at 0.3 rad/s is 1 s, raised to the 2 s minimum, at 40 Hz
        steps = np.abs(np.diff(np.vstack([np.zeros(14)] + sent), axis=0)).max()
        self.assertLessEqual(steps, 0.3 * period + 1e-9)
        np.testing.assert_allclose(sent[-1], target.arm_q)
        np.testing.assert_allclose(arm.ctrl_dual_arm_and_head.call_args.args[2], target.head_q)
        self.assertAlmostEqual(arm.ctrl_dual_arm_and_head.call_args.kwargs["waist_yaw_target"], 0.02)
        np.testing.assert_allclose(hand.update.call_args.args[1], [5.2])

    def test_q_stops_the_approach_and_a_far_start_is_refused(self):
        runner = self.load()
        arm, hand = self.robot()
        args = SimpleNamespace(start_speed=0.3, max_start_approach_seconds=15.0)
        with patch.object(runner.time, "sleep"):
            self.assertIsNone(runner.ease_to_start(arm, hand, self.target(), args, 0.025, [True]))
        arm.ctrl_dual_arm_and_head.assert_not_called()
        with self.assertRaisesRegex(ReplayError, "too far"):
            runner.ease_to_start(arm, hand, self.target(10.0), args, 0.025, [False])
        arm.ctrl_dual_arm_and_head.assert_not_called()

    def test_main_checks_the_start_pose_before_moving_and_the_held_pose_before_the_policy(self):
        source = (ROOT / "tools/run_r1_act_policy.py").read_text(encoding="utf-8")
        early = source.index('refusing --start-pose: it is outside the training range')
        activate = source.index("MotionSwitcher().Enter_Debug_Mode()")
        approach = source.index("last = ease_to_start(arm, hand, start_target, args, period, stop)")
        held_check = source.index("the held pose is outside the training range")
        policy = source.index("last = execute_loop(run, arm, hand, guard, gravity, period, stop, args, last, blend_next)")
        self.assertLess(early, activate)
        self.assertLess(activate, approach)
        self.assertLess(approach, held_check)
        self.assertLess(held_check, policy)
        self.assertIn("if outside and start_target is None and not args.allow_out_of_range_start:", source)


if __name__ == "__main__":
    unittest.main()
