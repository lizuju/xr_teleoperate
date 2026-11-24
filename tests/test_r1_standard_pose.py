import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from teleop.robot_control.arm_target_shaper import ArmTargetShaper
from teleop.robot_control.r1_standard_pose import (
    DEFAULT_POSE_PATH, MAX_EXTRA_SECONDS, MIN_SECONDS, SETTLE_SECONDS, STANDARD_POSE_KEYS, R1StandardPose,
    StandardPoseMove, load_standard_pose, standard_start_pose,
)
import test_r1_pause as pause_fixture

ROOT = Path(__file__).resolve().parents[1]
HAS_SDK = importlib.util.find_spec("unitree_sdk2py") is not None and importlib.util.find_spec("sshkeyboard") is not None

GRAVITY = np.linspace(-1.0, 1.0, 14)


def fake_arm(arm_q, head_q, waist=0.0):
    """A controller mock whose requested targets follow what it is sent."""
    arm = Mock()
    state = {"arm_q": list(arm_q), "arm_tau": [0.0] * 14, "head_q": list(head_q)}
    arm.get_recording_snapshot.side_effect = lambda: {"requested": dict(state), "published": dict(state)}
    arm.get_current_waist_yaw.return_value = waist

    def send(q, tau, head, waist_yaw_target=None):
        state.update(arm_q=list(q), arm_tau=list(tau), head_q=list(head))

    arm.ctrl_dual_arm_and_head.side_effect = send
    return arm


class PoseFileTest(unittest.TestCase):
    def test_the_shipped_pose_is_the_round_2_mean_start(self):
        pose = load_standard_pose(DEFAULT_POSE_PATH)
        self.assertEqual(pose["arm_q"].shape, (14,))
        self.assertEqual(pose["head_q"].shape, (2,))
        data = json.loads(DEFAULT_POSE_PATH.read_text())
        self.assertIn("split-run2.json", data["source"]["split"])
        self.assertEqual(len(data["arm_joint_names"]), 14)

    def test_bad_files_are_refused(self):
        good = json.loads(DEFAULT_POSE_PATH.read_text())
        bad = [dict(good, arm_q=good["arm_q"][:13]), dict(good, arm_q=good["arm_q"][:13] + [float("nan")]),
               dict(good, head_q=[0.7, 0.0]), dict(good, head_q=None)]
        with tempfile.TemporaryDirectory() as directory:
            for index, data in enumerate(bad):
                path = Path(directory) / f"bad{index}.json"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError, msg=index):
                    load_standard_pose(path)


class PolicyStartTest(unittest.TestCase):
    def test_the_policy_start_is_the_teleop_pose_with_the_waist_at_0_and_the_mean_hands(self):
        pose, sides = standard_start_pose(DEFAULT_POSE_PATH)
        teleop = load_standard_pose(DEFAULT_POSE_PATH)
        np.testing.assert_array_equal(pose.arm_q, teleop["arm_q"])
        np.testing.assert_array_equal(pose.head_q, teleop["head_q"])
        self.assertEqual(pose.waist_q, 0.0)
        self.assertEqual(pose.left_q.shape, (6,))
        self.assertEqual(pose.right_q.shape, (1,))
        self.assertTrue(4.28 < pose.right_q[0] <= 5.4)
        np.testing.assert_array_equal(pose.arm_tau, np.zeros(14))
        self.assertEqual((pose.left_mode, pose.right_mode), (1, 1))
        self.assertEqual(sides, {"left": "linker_o6", "right": "dex1_1"})

    def test_a_file_without_usable_hands_is_refused_for_policy_starts(self):
        good = json.loads(DEFAULT_POSE_PATH.read_text())
        bad = [{k: v for k, v in good.items() if k != "left_hand_q"}, dict(good, right_hand_q=[5.3, 5.3]),
               dict(good, left_hand_q=[1.2] * 6), dict(good, end_effectors={"left": "linker_o6", "right": "linker_o6"})]
        with tempfile.TemporaryDirectory() as directory:
            for index, data in enumerate(bad):
                path = Path(directory) / f"bad{index}.json"
                path.write_text(json.dumps(data))
                load_standard_pose(path)  # teleop o needs only arms and head
                with self.assertRaises(ValueError, msg=index):
                    standard_start_pose(path)

    @unittest.skipUnless(HAS_SDK, "needs the robot SDK environment")
    def test_runner_takes_standard_and_refuses_a_frame_with_it(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import run_r1_act_policy as runner
        self.assertEqual(runner.parse_args(["policy.pt", "--start-pose", "standard"]).start_pose, DEFAULT_POSE_PATH)
        self.assertEqual(runner.parse_args(["policy.pt", "--start-pose", str(DEFAULT_POSE_PATH)]).start_pose,
                         DEFAULT_POSE_PATH)
        self.assertEqual(runner.parse_args(["policy.pt", "--start-pose", "/tmp/episode_0096", "--start-frame", "3"])
                         .start_frame, 3)
        with self.assertRaises(SystemExit):
            runner.parse_args(["policy.pt", "--start-pose", "standard", "--start-frame", "2"])
        source = (ROOT / "tools/run_r1_act_policy.py").read_text(encoding="utf-8")
        self.assertLess(source.index("standard_start_pose(args.start_pose)"),
                        source.index("refusing --start-pose: it is outside the training range"))


class MoveTest(unittest.TestCase):
    def test_duration_keeps_the_peak_speed_at_the_limit(self):
        move = StandardPoseMove(np.zeros(14), np.zeros(2), 0.0, np.full(14, 0.9), np.zeros(2), 0.0, 10.0)
        self.assertAlmostEqual(move.duration, 1.5 * 0.9 / 0.3)
        times = np.linspace(10.0, 10.0 + move.duration, 2001)
        arm = np.array([move.at(t)[0] for t in times])
        peak = np.abs(np.diff(arm, axis=0)).max() / (times[1] - times[0])
        self.assertLessEqual(peak, 0.3 * 1.001)
        self.assertGreater(peak, 0.3 * 0.99)

    def test_ends_exactly_on_the_target_and_finishes_after_settling(self):
        start, target = np.full(14, 0.2), np.linspace(-0.5, 0.5, 14)
        move = StandardPoseMove(start, [0.0, 0.1], 0.3, target, [0.1, -0.03], 0.0, 0.0)
        arm, head, waist, finished = move.at(0.0)
        np.testing.assert_array_equal(arm, start)
        self.assertEqual(waist, 0.3)
        self.assertFalse(finished)
        arm, head, waist, finished = move.at(move.duration)
        np.testing.assert_allclose(arm, target)
        np.testing.assert_allclose(head, [0.1, -0.03])
        self.assertEqual(waist, 0.0)
        self.assertFalse(finished)
        self.assertTrue(move.at(move.duration + SETTLE_SECONDS)[3])

    def test_short_moves_take_the_minimum_and_the_waist_can_set_the_pace(self):
        self.assertEqual(StandardPoseMove(np.zeros(14), np.zeros(2), None, np.zeros(14), np.zeros(2), None, 0).duration,
                         MIN_SECONDS)
        move = StandardPoseMove(np.zeros(14), np.zeros(2), -math.pi / 2, np.zeros(14), np.zeros(2), 0.0, 0)
        self.assertAlmostEqual(move.duration, 1.5 * 90 / 20)
        self.assertIsNone(StandardPoseMove(np.zeros(14), np.zeros(2), None, np.ones(14), np.zeros(2), None, 0).at(1)[2])

    def test_bad_moves_are_refused(self):
        with self.assertRaises(ValueError):
            StandardPoseMove(np.zeros(14), np.zeros(2), 0.0, np.full(14, 5.0), np.zeros(2), 0.0, 0)  # 25 s
        with self.assertRaises(ValueError):
            StandardPoseMove(np.zeros(14), np.zeros(2), 0.0, np.zeros(14), np.zeros(2), None, 0)
        with self.assertRaises(ValueError):
            StandardPoseMove(np.zeros(13), np.zeros(2), None, np.zeros(14), np.zeros(2), None, 0)
        with self.assertRaises(ValueError):
            StandardPoseMove(np.zeros(14), np.zeros(2), None, np.zeros(14), np.zeros(2), None, 0, arm_speed=0.0)


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.pose = {"arm_q": np.full(14, 0.5), "head_q": np.array([0.1, -0.03]), "source": "test"}
        self.logger = Mock()
        self.control = R1StandardPose(self.pose, lambda q: GRAVITY.copy(), self.logger)

    def run_move(self, arm, generation=1, waist_reference=None, head_locked=False, start=1.0):
        commanded, now = [], start
        while True:
            if not self.control.step(arm, generation, waist_reference, now, head_locked=head_locked):
                return commanded
            commanded.append(arm.ctrl_dual_arm_and_head.call_args)
            now += 0.025
            self.assertLess(now, 30.0)

    def test_o_toggles_between_a_move_and_a_stop(self):
        self.assertEqual(self.control.press(), "go")
        self.assertTrue(self.control.moving())
        self.assertEqual(self.control.press(), "stop")
        self.assertFalse(self.control.moving())
        self.assertEqual(self.control.press(), "go")

    def test_a_move_starts_at_the_held_targets_and_ends_held_at_the_standard_pose(self):
        arm = fake_arm(np.full(14, 0.2), [0.3, 0.4], waist=math.radians(-8.0))
        self.control.press()
        calls = self.run_move(arm, waist_reference=0.0)
        first, last = calls[0], calls[-1]
        np.testing.assert_allclose(first.args[0], np.full(14, 0.2))
        np.testing.assert_allclose(first.args[2], [0.3, 0.4])
        self.assertAlmostEqual(first.kwargs["waist_yaw_target"], math.radians(-8.0))
        np.testing.assert_allclose(last.args[0], self.pose["arm_q"])
        np.testing.assert_allclose(last.args[2], self.pose["head_q"])
        self.assertEqual(last.kwargs["waist_yaw_target"], 0.0)
        for call in calls:
            np.testing.assert_array_equal(call.args[1], GRAVITY)
        self.assertFalse(self.control.moving())
        self.assertIn("at the standard start pose", self.logger.info.call_args.args[0])
        self.assertFalse(self.control.step(arm, 1, 0.0, 99.0))

    def test_without_waist_follow_the_waist_is_left_alone(self):
        arm = fake_arm(np.full(14, 0.2), [0.0, 0.0])
        self.control.press()
        calls = self.run_move(arm, waist_reference=None)
        self.assertTrue(calls)
        self.assertTrue(all(call.kwargs["waist_yaw_target"] is None for call in calls))
        arm.get_current_waist_yaw.assert_not_called()

    def test_a_locked_head_stays_where_it_is(self):
        arm = fake_arm(np.full(14, 0.2), [0.3, 0.4])
        self.control.press()
        calls = self.run_move(arm, head_locked=True)
        for call in calls:
            np.testing.assert_allclose(call.args[2], [0.3, 0.4])

    def test_o_again_or_a_new_pause_stops_the_move_where_it_is(self):
        for stop in ("key", "pause"):
            arm = fake_arm(np.full(14, 0.2), [0.0, 0.0])
            self.control.press()
            self.assertTrue(self.control.step(arm, 1, None, 1.0))
            self.assertTrue(self.control.step(arm, 1, None, 2.0))
            sent = arm.ctrl_dual_arm_and_head.call_count
            if stop == "key":
                self.assertEqual(self.control.press(), "stop")
            self.assertFalse(self.control.step(arm, 1 if stop == "key" else 2, None, 2.1))
            self.assertFalse(self.control.step(arm, 2, None, 2.2))
            self.assertEqual(arm.ctrl_dual_arm_and_head.call_count, sent)
            self.assertFalse(self.control.moving())
            self.assertIn("stopped", self.logger.warning.call_args.args[0])

    def test_it_reports_arrival_only_when_the_controller_target_is_there(self):
        arm = fake_arm(np.full(14, 0.2), [0.0, 0.0])
        arm.ctrl_dual_arm_and_head.side_effect = None  # the controller target never moves
        self.control.press()
        calls = self.run_move(arm)
        self.assertFalse(self.control.moving())
        message = self.logger.warning.call_args.args[0]
        self.assertIn("stayed 0.300 rad", message)
        move_time = 0.025 * (len(calls) - 1)
        self.assertGreaterEqual(move_time, MIN_SECONDS + SETTLE_SECONDS + MAX_EXTRA_SECONDS - 0.05)
        self.assertNotIn("at the standard start pose", str(self.logger.info.call_args_list[-1]))

    def test_a_move_that_would_take_too_long_is_refused_and_o_works_again(self):
        control = R1StandardPose(self.pose, lambda q: GRAVITY.copy(), self.logger, arm_speed=0.01)
        arm = fake_arm(np.full(14, 0.2), [0.0, 0.0])
        control.press()
        self.assertFalse(control.step(arm, 1, None, 1.0))
        arm.ctrl_dual_arm_and_head.assert_not_called()
        self.assertIn("not moving", self.logger.warning.call_args.args[0])
        self.assertEqual(control.press(), "go")


class LoopAndKeyTest(unittest.TestCase):
    """The real control-loop slice and on_press, through the pause test fixture."""

    def setUp(self):
        fixture = pause_fixture.R1PauseControlIntegrationTest()
        fixture.setUp()
        self.fixture, self.ns, self.state = fixture, fixture.ns, fixture.state
        self.pose = {"arm_q": np.full(14, 0.5), "head_q": np.array([0.1, -0.03]), "source": "test"}
        self.control = R1StandardPose(self.pose, lambda q: GRAVITY.copy(), Mock())
        self.ns.update(STANDARD_POSE=self.control, STANDARD_POSE_KEYS=STANDARD_POSE_KEYS, START=True,
                       RECORD_RUNNING=False, RECORD_TOGGLE=False)
        # on_press shares the loop's namespace, as the module globals do in production.
        tree = ast.parse(pause_fixture.MAIN_PATH.read_text())
        on_press = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "on_press")
        self.ns.setdefault("logger_mp", Mock())
        self.ns.setdefault("HAND_LOCK_KEYS", pause_fixture.HAND_LOCK_KEYS)
        self.ns.setdefault("HAND_LOCKS", pause_fixture.R1HandLocks())
        pause_fixture.execute([on_press], self.ns)

    def test_o_while_following_pauses_moves_and_s_resumes_at_the_standard_pose(self):
        self.ns["RECORD_TOGGLE"] = True  # a pending start is cleared, as p does
        self.ns["on_press"]("o")
        self.assertTrue(self.state.paused)
        self.assertTrue(self.control.moving())
        self.assertFalse(self.ns["RECORD_TOGGLE"])
        now = 1.0
        self.fixture.tick(now)
        self.state.request_resume(now)  # the operator presses s/r during the move
        while self.control.moving():
            now += 0.1
            self.fixture.tick(now)
            self.assertLess(now, 20.0)
        self.assertTrue(self.state.paused, "resume must wait for the move")
        np.testing.assert_allclose(self.fixture.requested["arm_q"], self.pose["arm_q"])
        np.testing.assert_allclose(self.fixture.requested["head_q"], self.pose["head_q"])
        self.ns["arm_ik"].reset_smoothing.assert_not_called()
        for _ in range(10):  # until the stable-hands check realigns
            now += 0.1
            self.fixture.tick(now)
            if not self.state.paused:
                break
        self.assertFalse(self.state.paused)
        np.testing.assert_allclose(self.ns["arm_ik"].reset_smoothing.call_args.kwargs["reference_q"], self.pose["arm_q"])
        np.testing.assert_allclose(self.ns["arm_ik"].forward_wrist_poses.call_args.args[0], self.pose["arm_q"])

    def test_the_loop_does_not_hold_targets_while_the_move_commands(self):
        hold = self.ns["arm_ctrl"].hold_targets
        self.ns["on_press"]("o")
        now = 1.0
        self.fixture.tick(now)
        before = hold.call_count
        while self.control.moving():
            now += 0.1
            self.fixture.tick(now)
        self.assertEqual(hold.call_count, before, "a hold during the move resets the arm target shaper")
        self.fixture.tick(now + 0.1)
        self.assertGreater(hold.call_count, before)

    def test_arms_reach_the_pose_through_the_real_arm_target_shaper(self):
        """The 2026-11-23 robot test: the head arrived, the arms never moved."""
        shaper = ArmTargetShaper(velocity_limit=6.0, accel_limit=40.0)
        requested = self.fixture.requested
        shaper.reset(np.array(requested["arm_q"]))
        start = list(requested["arm_q"])

        def send(q, tau, head, **kwargs):  # as R1_A7_ArmController: the arm target is shaped, the head is not
            requested.update(arm_q=shaper.shape(q, now=self.fixture.now).tolist(), arm_tau=list(tau),
                             head_q=list(head))

        self.ns["arm_ctrl"].ctrl_dual_arm_and_head.side_effect = send
        self.ns["arm_ctrl"].hold_targets.side_effect = lambda: shaper.reset(np.array(requested["arm_q"]))
        self.ns["on_press"]("o")
        now = 1.0
        while True:
            self.fixture.tick(now)
            if not self.control.moving():
                break
            now += 0.025
            self.assertLess(now, 20.0)
        np.testing.assert_allclose(requested["arm_q"], self.pose["arm_q"], atol=0.01)
        np.testing.assert_allclose(requested["head_q"], self.pose["head_q"])
        self.assertGreater(np.abs(np.array(start) - self.pose["arm_q"]).max(), 0.3)
        self.control.logger.warning.assert_not_called()

    def test_o_is_refused_before_start_and_during_an_episode(self):
        for name, value in (("START", False), ("RECORD_RUNNING", True)):
            self.ns.update(START=True, RECORD_RUNNING=False)
            self.ns[name] = value
            self.ns["on_press"]("o")
            self.assertFalse(self.state.paused, name)
            self.assertFalse(self.control.moving(), name)

    def test_o_again_stops_the_move_and_the_pause_holds(self):
        self.ns["on_press"]("o")
        self.fixture.tick(1.0)
        self.fixture.tick(1.5)
        halfway = list(self.fixture.requested["arm_q"])
        self.ns["on_press"]("O")
        self.assertFalse(self.control.moving())
        for index in range(5):
            self.fixture.tick(1.6 + 0.1 * index)
        self.assertEqual(self.fixture.requested["arm_q"], halfway)
        self.assertTrue(self.state.paused)

    def test_without_a_standard_pose_the_pause_path_is_unchanged(self):
        self.ns["STANDARD_POSE"] = None
        self.ns["on_press"]("o")
        self.assertFalse(self.state.paused)
        self.state.pause()
        self.fixture.tick(1.0)
        self.assertGreaterEqual(self.ns["arm_ctrl"].hold_targets.call_count, 1)


if __name__ == "__main__":
    unittest.main()
