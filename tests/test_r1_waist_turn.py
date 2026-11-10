"""Arrow-key waist turns: stop placement, motion limits, and no turning back."""
import ast
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "teleop" / "teleop_hand_and_arm.py"
sys.path.insert(0, str(ROOT / "teleop" / "robot_control"))
from r1_head_waist import R1HeadWaistFollower

DT = 0.025
QUARTER = math.pi / 2.0


def head_pose(yaw):
    pose = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    pose[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return pose


class Robot:
    """Drives a follower at 40 Hz with a waist that tracks its target with a short lag."""

    def __init__(self, follower, now=0.0, waist=None, lag=0.0):
        self.follower = follower
        self.now = now
        self.waist = follower.waist_target if waist is None else waist
        self.lag = lag
        self.targets, self.heads, self.events = [], [], []

    def run(self, seconds, head_yaw=0.0):
        for _ in range(int(round(seconds / DT))):
            self.now += DT
            head, target = self.follower.update(head_pose(head_yaw), np.eye(4), self.waist, self.now)
            self.waist += (target - self.waist) * (1.0 - self.lag)
            self.targets.append(target)
            self.heads.append(head[1])
            self.events.extend(self.follower.pop_turn_events())
        return self

    def run_until_idle(self, limit=10.0, head_yaw=0.0):
        steps = 0
        while self.follower.turning:
            self.run(DT, head_yaw)
            steps += 1
            if steps * DT > limit:
                raise AssertionError("turn did not finish")
        return steps * DT


class WaistTurnFollowerTest(unittest.TestCase):
    def test_right_turn_lands_exactly_on_minus_90_and_stays(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        robot = Robot(follower)
        self.assertAlmostEqual(follower.command_turn(-1), -QUARTER)
        seconds = robot.run_until_idle()
        self.assertEqual(follower.waist_target, -QUARTER)
        self.assertAlmostEqual(follower.heading_offset, -QUARTER)
        # 30 deg/s with 60 deg/s^2 ramps: 0.5 s up, 2.5 s cruise, 0.5 s down.
        self.assertGreater(seconds, 3.3)
        self.assertLess(seconds, 3.9)
        self.assertEqual([event["event"] for event in robot.events], ["reached"])
        # The head only covers the one-tick lag of the simulated waist.
        self.assertLessEqual(max(abs(yaw) for yaw in robot.heads), follower.turn_velocity * DT + 1e-9)
        # The operator keeps facing forward: nothing turns back, nothing follows.
        robot.heads.clear()
        robot.run(5.0)
        self.assertEqual(follower.waist_target, -QUARTER)
        self.assertFalse(follower.following)
        self.assertLess(max(abs(yaw) for yaw in robot.heads), 1e-9)

    def test_turn_respects_speed_and_acceleration_limits(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        robot = Robot(follower, lag=0.3)
        follower.command_turn(1)
        robot.run_until_idle()
        velocities = np.diff([0.0] + robot.targets) / DT
        self.assertLessEqual(np.max(np.abs(velocities)), follower.turn_velocity + 1e-9)
        velocities = np.append(velocities, 0.0)  # it stays at the stop afterwards
        self.assertLessEqual(
            np.max(np.abs(np.diff(velocities))) / DT, follower.turn_acceleration * 1.0001,
        )
        self.assertLessEqual(max(robot.targets), QUARTER + 1e-12)

    def test_stops_are_start_heading_and_plus_minus_90(self):
        follower = R1HeadWaistFollower(0.1, 0.0)
        robot = Robot(follower)
        self.assertAlmostEqual(follower.command_turn(1), 0.1 + QUARTER)
        robot.run_until_idle()
        self.assertIsNone(follower.command_turn(1))  # +180 is outside the waist range
        self.assertAlmostEqual(follower.command_turn(-1), 0.1)
        robot.run_until_idle()
        self.assertAlmostEqual(follower.command_turn(-1), 0.1 - QUARTER)
        robot.run_until_idle()
        self.assertIsNone(follower.command_turn(-1))
        self.assertAlmostEqual(follower.waist_target, 0.1 - QUARTER)

    def test_waist_a_few_degrees_off_a_stop_needs_one_press(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        follower.waist_target = math.radians(5.0)
        self.assertAlmostEqual(follower.next_turn_stop(-1), -QUARTER)
        follower.waist_target = math.radians(-85.0)
        self.assertIsNone(follower.next_turn_stop(-1))
        self.assertAlmostEqual(follower.next_turn_stop(1), 0.0)
        follower.waist_target = math.radians(-40.0)
        self.assertAlmostEqual(follower.next_turn_stop(-1), -QUARTER)
        self.assertAlmostEqual(follower.next_turn_stop(1), 0.0)

    def test_opposite_key_mid_turn_goes_back_without_overshoot(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        robot = Robot(follower)
        follower.command_turn(-1)
        robot.run(1.5)
        middle = follower.waist_target
        self.assertLess(middle, math.radians(-20.0))
        self.assertAlmostEqual(follower.command_turn(1), 0.0)
        robot.run_until_idle()
        self.assertEqual(follower.waist_target, 0.0)
        self.assertAlmostEqual(follower.heading_offset, 0.0)
        self.assertGreaterEqual(min(robot.targets), -QUARTER)
        self.assertLessEqual(max(robot.targets), 1e-12)
        robot.run(3.0)
        self.assertEqual(follower.waist_target, 0.0)

    def test_tracking_gap_stops_turn_where_it_is_and_does_not_undo_it(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        robot = Robot(follower)
        follower.command_turn(-1)
        robot.run(1.5)
        stopped = robot.waist
        robot.now += 1.0  # longer than the tracking timeout: the next update resets
        robot.run(4.0)
        self.assertFalse(follower.turning)
        self.assertEqual([event["event"] for event in robot.events], ["cancelled"])
        self.assertAlmostEqual(robot.events[0]["waist"], stopped)
        self.assertAlmostEqual(follower.waist_target, stopped)
        self.assertFalse(follower.following)
        # Pressing again finishes the turn to the same stop.
        self.assertAlmostEqual(follower.command_turn(-1), -QUARTER)
        robot.run_until_idle()
        self.assertEqual(follower.waist_target, -QUARTER)

    def test_external_reset_mid_turn_keeps_the_partial_turn(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        robot = Robot(follower, lag=0.2)
        follower.command_turn(1)
        robot.run(1.0)
        follower.reset(robot.waist, robot.now)
        robot.run(3.0)
        self.assertAlmostEqual(follower.waist_target, follower.heading_offset)
        self.assertGreater(follower.waist_target, math.radians(10.0))
        self.assertFalse(follower.following)

    def test_head_turn_after_key_turn_follows_from_the_new_heading(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        robot = Robot(follower)
        follower.command_turn(-1)
        robot.run_until_idle()
        robot.run(6.0, head_yaw=math.radians(30.0))
        self.assertTrue(math.radians(-70.0) <= follower.waist_target <= math.radians(-55.0))

    def test_robot_head_stays_forward_through_the_turn(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        robot = Robot(follower, lag=0.5)
        follower.command_turn(-1)
        robot.run_until_idle()
        # The head only covers the waist's lag behind its own target.
        self.assertLess(max(abs(yaw) for yaw in robot.heads), math.radians(3.0))

    def test_turn_origin_keeps_stops_after_a_resume(self):
        follower = R1HeadWaistFollower(-QUARTER, 0.0, turn_origin=0.0)
        self.assertIsNone(follower.next_turn_stop(-1))
        self.assertAlmostEqual(follower.next_turn_stop(1), 0.0)

    def test_head_follow_off_never_turns_the_waist_from_the_head(self):
        follower = R1HeadWaistFollower(0.0, 0.0, head_follow=False)
        robot = Robot(follower)
        robot.run(5.0, head_yaw=math.radians(60.0))
        self.assertEqual(follower.waist_target, 0.0)
        self.assertFalse(follower.following)
        # The robot head alone follows the operator's head.
        self.assertAlmostEqual(robot.heads[-1], math.radians(60.0))

    def test_head_follow_off_keeps_arrow_key_turns(self):
        follower = R1HeadWaistFollower(0.0, 0.0, head_follow=False)
        robot = Robot(follower)
        self.assertAlmostEqual(follower.command_turn(-1), -QUARTER)
        robot.run_until_idle()
        robot.run(5.0, head_yaw=math.radians(45.0))
        self.assertEqual(follower.waist_target, -QUARTER)
        self.assertAlmostEqual(robot.heads[-1], math.radians(45.0))

    def test_head_target_follows_without_moving_the_waist(self):
        follower = R1HeadWaistFollower(0.0, 0.0, head_follow=False)
        robot = Robot(follower)
        follower.command_turn(-1)
        robot.run_until_idle()
        state = (follower.waist_target, follower.total_yaw, follower.heading_offset, follower._last_time)
        head = follower.head_target(head_pose(math.radians(30.0)), np.eye(4), robot.waist)
        self.assertAlmostEqual(head[1], math.radians(30.0), places=6)
        self.assertEqual(state, (follower.waist_target, follower.total_yaw, follower.heading_offset, follower._last_time))
        # Same answer update() gives for the same pose once it is steady.
        robot.run(0.5, head_yaw=math.radians(30.0))
        np.testing.assert_allclose(follower.head_target(head_pose(math.radians(30.0)), np.eye(4), robot.waist),
                                   [0.0, robot.heads[-1]], atol=1e-9)

    def test_invalid_inputs(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        with self.assertRaises(ValueError):
            follower.command_turn(0)
        with self.assertRaises(ValueError):
            R1HeadWaistFollower(0.0, 0.0, turn_velocity=0.0)
        with self.assertRaises(ValueError):
            R1HeadWaistFollower(0.0, 0.0, turn_origin=float("nan"))


class WaistTurnTeleopTest(unittest.TestCase):
    """The keyboard and loop helpers, executed from the production source."""

    NAMES = ("on_press", "take_waist_turn_request", "waist_follower_from_args",
             "start_waist_turn", "log_waist_turn_events", "waist_turn_label")

    def setUp(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in self.NAMES]
        self.assertEqual(sorted(node.name for node in nodes), sorted(self.NAMES))
        self.logger = Mock()
        self.ns = {
            "math": math, "np": np, "R1HeadWaistFollower": R1HeadWaistFollower,
            "logger_mp": self.logger, "WAIST_TURN_LOCK": threading.Lock(),
            "WAIST_TURN_REQUEST": None, "R1_PAUSE": None, "READY": True, "START": True,
            "STOP": False, "RECORD_TOGGLE": False, "RECORD_OUTCOME": "unspecified",
            "RECORD_RUNNING": False, "RECORD_ENABLED": False, "DRY_RUN_MODE": False,
            "ARM_REQUEST_GENERATION": 0, "R1_A7_DEFERRED_REAL_MODE": False,
        }
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(module, str(MAIN_PATH), "exec"), self.ns)

    def test_arrow_keys_queue_one_turn_each(self):
        self.ns["on_press"]("right")
        self.assertEqual(self.ns["take_waist_turn_request"](), "right")
        self.assertIsNone(self.ns["take_waist_turn_request"]())
        self.ns["on_press"]("left")
        self.ns["on_press"]("right")
        self.assertEqual(self.ns["take_waist_turn_request"](), "right")

    def test_follower_from_args_uses_turn_settings_and_origin(self):
        args = SimpleNamespace(tracking_timeout=0.25, waist_turn_speed_deg=20.0, waist_turn_accel_deg=40.0)
        follower = self.ns["waist_follower_from_args"](args, -QUARTER, 0.0, turn_origin=0.0)
        self.assertAlmostEqual(follower.turn_velocity, math.radians(20.0))
        self.assertAlmostEqual(follower.turn_acceleration, math.radians(40.0))
        self.assertEqual(follower.turn_origin, 0.0)
        self.assertTrue(follower.head_follow)
        old = self.ns["waist_follower_from_args"](SimpleNamespace(tracking_timeout=0.25), 0.2, 0.0)
        self.assertAlmostEqual(old.turn_velocity, math.radians(30.0))
        self.assertEqual(old.turn_origin, 0.2)
        keys_only = self.ns["waist_follower_from_args"](
            SimpleNamespace(tracking_timeout=0.25, waist_head_follow="off"), 0.0, 0.0)
        self.assertFalse(keys_only.head_follow)

    def test_launchers_default_to_arrow_key_turns_only(self):
        for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
            text = (ROOT / "teleop" / name).read_text(encoding="utf-8")
            self.assertIn('--waist-head-follow "${WAIST_HEAD_FOLLOW:-off}"', text)
            self.assertIn("--waist-follow \\\n", text)

    def test_start_turn_logs_goal_and_refuses_world_compensation(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        torso = SimpleNamespace(waist_follow_compensation="torso")
        self.assertAlmostEqual(self.ns["start_waist_turn"](follower, "right", torso), -QUARTER)
        self.assertEqual(self.ns["waist_turn_label"](follower), "to -90 deg")
        world = SimpleNamespace(waist_follow_compensation="world")
        other = R1HeadWaistFollower(0.0, 0.0)
        self.assertIsNone(self.ns["start_waist_turn"](other, "left", world))
        self.assertFalse(other.turning)
        self.assertEqual(self.ns["waist_turn_label"](other), "idle")

    def test_turn_events_are_logged_once(self):
        follower = R1HeadWaistFollower(0.0, 0.0)
        follower.command_turn(1)
        now = 0.0
        while follower.turning:
            now += DT
            follower.update(head_pose(0.0), np.eye(4), follower.waist_target, now)
        self.ns["log_waist_turn_events"](follower)
        self.ns["log_waist_turn_events"](follower)
        self.assertEqual(self.logger.info.call_count, 1)


class WaistTurnLoopTest(unittest.TestCase):
    """The production control-loop slice with a real follower and a pressed arrow key."""

    def setUp(self):
        import test_r1_head_waist_integration as integration
        self.case = integration.R1HeadWaistIntegrationTest("test_tracking_resumption_resets_follower_from_actual_waist")
        self.case.setUp()
        self.execute = integration.execute

    def context(self, key):
        ns = self.case.context()
        ns["args"].waist_follow_compensation = "torso"
        # The stub clock reads 1.0; start just before it so the first update is not a gap.
        ns["waist_follower"] = R1HeadWaistFollower(0.43, 0.98)
        ns["take_waist_turn_request"] = Mock(return_value=key)
        return ns

    def turn_warnings(self, ns):
        return [call for call in ns["logger_mp"].warning.call_args_list
                if "[R1 WAIST TURN]" in str(call.args[0])]

    def test_arrow_key_starts_the_turn_in_the_control_loop(self):
        ns = self.context("right")
        self.execute(self.case.control_nodes, ns, loop=True)
        follower = ns["waist_follower"]
        self.assertTrue(follower.turning)
        self.assertAlmostEqual(follower.turn_goal, 0.43 - QUARTER)
        self.assertLess(follower.waist_target, 0.43)
        self.assertEqual(self.turn_warnings(ns), [])

    def test_arrow_key_is_dropped_with_a_warning_while_tracking_is_held(self):
        ns = self.context("left")
        ns["is_present_motion_data"] = Mock(return_value=False)
        self.execute(self.case.control_nodes, ns, loop=True)
        self.assertFalse(ns["waist_follower"].turning)
        self.assertEqual(len(self.turn_warnings(ns)), 1)
        ns["take_waist_turn_request"].assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
