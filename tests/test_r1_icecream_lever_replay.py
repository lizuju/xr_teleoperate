#!/usr/bin/env python3
"""Unit tests for ice-cream lever playback planner (no robot motion)."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from teleop.utils import r1_icecream_lever_replay as m


TOOLS = REPO / "tools"
RUNNER_PATH = TOOLS / "replay_r1_icecream_lever.py"


def _pinocchio(waist, right7, left7=None, head=None):
    left7 = left7 if left7 is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    head = head if head is not None else [0.01, -0.02]
    return [waist, head[0], head[1], *left7, *right7]


def _sample_waypoints(*, waist_spread="ok", pull_cm=2.43):
    """Build a v2 document. waist_spread: ok | withdraw_far."""
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    grasp = [-0.0382, -0.3965, 0.3335]
    # ~2.43 cm travel matching production Δxyz scale.
    pulled = [-0.0613, -0.4015, 0.3391]
    withdraw = [-0.0211, -0.3785, 0.4461]
    grasp_w = -1.4577
    pulled_w = grasp_w - math.radians(1.5)
    withdraw_w = grasp_w - math.radians(1.0) if waist_spread == "ok" else grasp_w + math.radians(12.0)
    grasp_right = [-1.54, -0.30, 0.16, 0.76, 0.25, -0.04, 0.09]
    pulled_right = [-1.62, -0.31, 0.05, 0.90, 0.56, 0.13, 0.91]
    withdraw_right = [-1.98, -0.42, -0.09, 1.15, 0.19, 0.09, 0.32]
    left_recorded = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]  # must never appear in 14-vector
    lever = {
        "side": "right",
        "frame": "right_wrist_yaw_link",
        "position_m": grasp,
        "rotation": identity,
        "pinocchio_q": _pinocchio(grasp_w, grasp_right, left7=left_recorded),
        "pulled": {
            "position_m": pulled,
            "rotation": identity,
            "pinocchio_q": _pinocchio(pulled_w, pulled_right, left7=left_recorded),
        },
        "wait_s": 3.0,
        "withdraw": {
            "position_m": withdraw,
            "rotation": identity,
            "pinocchio_q": _pinocchio(withdraw_w, withdraw_right, left7=left_recorded),
        },
    }
    return {
        "schema": "icecream_waypoints_v2",
        "turn_waist_yaw_rad": -0.0468,
        "slots": {
            "left_cup": {
                "side": "left",
                "frame": "left_wrist_yaw_link",
                "position_m": [0.34, 0.02, 0.09],
                "rotation": identity,
                "pinocchio_q": _pinocchio(0.0, [0.0] * 7),
            },
            "right_lever_3": lever,
        },
    }, {
        "grasp": grasp,
        "pulled": pulled,
        "withdraw": withdraw,
        "grasp_w": grasp_w,
        "expected_travel": math.sqrt(sum((a - b) ** 2 for a, b in zip(grasp, pulled))),
    }


def _sample_grip():
    return {
        "schema": "o6_grip_cap_v2",
        "sides": {
            "left": {"max_close_q": [0.19, 0.48, 0.31, 0.39, 0.41, 0.39]},
            "right": {"max_close_q": [0.25, 0.05, 0.71, 0.73, 0.74, 0.74]},
        },
    }


class IcecreamLeverReplayTests(unittest.TestCase):
    def test_sequence_order_and_travel_cm(self):
        doc, meta = _sample_waypoints()
        plan = m.extract_lever_slot(doc, _sample_grip()["sides"]["right"]["max_close_q"])
        self.assertAlmostEqual(plan.pull_travel_m, meta["expected_travel"], places=6)
        self.assertAlmostEqual(plan.pull_travel_m * 100.0, meta["expected_travel"] * 100.0, places=4)
        # Production-scale: ~2.43 cm, not scaled up.
        self.assertGreater(plan.pull_travel_m * 100.0, 2.0)
        self.assertLess(plan.pull_travel_m * 100.0, 3.0)
        self.assertAlmostEqual(plan.wait_s, 3.0)
        text = m.format_check_only(plan)
        self.assertIn("sends_robot_commands: false", text)
        self.assertIn("等待 3.0 秒", text)
        self.assertIn("max_close_q=[0.25, 0.05, 0.71, 0.73, 0.74, 0.74]", text)
        for needle in ("1.", "2.", "3.", "4.", "5.", "6."):
            self.assertIn(needle, text)
        self.assertIn("grasp 腰", text)
        self.assertIn("turn_waist_yaw_rad", text)

    def test_left_arm_unchanged_in_built_14_vector(self):
        doc, _ = _sample_waypoints()
        plan = m.extract_lever_slot(doc, [0.25, 0.05, 0.71, 0.73, 0.74, 0.74])
        live_left = [0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77]
        live_right = [0.0] * 7
        live_head = [0.05, -0.05]
        plan, _, _ = m.build_motion_segments(
            plan,
            live_left7=live_left,
            live_right7=live_right,
            live_head2=live_head,
            live_waist=-0.2,
            live_left_hand=[0.1] * 6,
            live_right_hand=[0.05] * 6,
            max_approach_s=15.0,
            hz=20.0,
        )
        names = [s.name for s in plan.segments]
        self.assertEqual(
            names,
            [
                "approach_grasp_close",
                "pull",
                "wait",
                "return_grasp",
                "withdraw_release",
                "turn_waist",
            ],
        )
        live_left_arr = np.asarray(live_left, dtype=np.float64)
        live_head_arr = np.asarray(live_head, dtype=np.float64)
        for segment in plan.segments:
            for pose in segment.poses:
                np.testing.assert_allclose(pose.arm_q[:7], live_left_arr)
                np.testing.assert_allclose(pose.head_q, live_head_arr)
                self.assertTrue(pose.left_fresh)
                # Recorded left (all 9s) must never leak into the command.
                self.assertFalse(np.any(np.isclose(pose.arm_q[:7], 9.0)))

    def test_waist_stays_at_grasp_until_final_segment(self):
        doc, meta = _sample_waypoints()
        plan = m.extract_lever_slot(doc, [0.25, 0.05, 0.71, 0.73, 0.74, 0.74])
        plan, _, _ = m.build_motion_segments(
            plan,
            live_left7=[0.1] * 7,
            live_right7=[0.0] * 7,
            live_head2=[0.0, 0.0],
            live_waist=0.0,
            live_left_hand=[0.2] * 6,
            live_right_hand=[0.0] * 6,
            max_approach_s=15.0,
            hz=20.0,
        )
        grasp_w = meta["grasp_w"]
        for segment in plan.segments:
            if segment.name == "approach_grasp_close":
                # Ramps live waist → grasp waist; must finish on grasp.
                self.assertAlmostEqual(segment.poses[-1].waist_q, grasp_w, places=5)
                continue
            if segment.name == "turn_waist":
                self.assertAlmostEqual(segment.poses[-1].waist_q, -0.0468, places=5)
                self.assertGreater(abs(segment.poses[-1].waist_q - grasp_w), 0.5)
                continue
            for pose in segment.poses:
                self.assertAlmostEqual(pose.waist_q, grasp_w, places=5)

    def test_waist_mismatch_passes_and_withdraw_keeps_grasp_waist(self):
        """Stored withdraw waist may differ; playback keeps grasp waist (no refuse)."""
        doc, meta = _sample_waypoints(waist_spread="withdraw_far")
        plan = m.extract_lever_slot(doc, [0.25, 0.05, 0.71, 0.73, 0.74, 0.74])
        self.assertTrue(any("differ" in w for w in plan.warnings))
        self.assertAlmostEqual(plan.grasp_waist_rad, meta["grasp_w"], places=5)
        # Stored withdraw waist is ~12° away; must not be copied into withdraw segment.
        self.assertGreater(
            abs(math.degrees(plan.withdraw_waist_rad - plan.grasp_waist_rad)), 5.0
        )
        plan, _, _ = m.build_motion_segments(
            plan,
            live_left7=[0.1] * 7,
            live_right7=[0.0] * 7,
            live_head2=[0.0, 0.0],
            live_waist=0.0,
            live_left_hand=[0.2] * 6,
            live_right_hand=[0.0] * 6,
            max_approach_s=15.0,
            hz=20.0,
        )
        withdraw = next(s for s in plan.segments if s.name == "withdraw_release")
        for pose in withdraw.poses:
            self.assertAlmostEqual(pose.waist_q, meta["grasp_w"], places=5)
        # Right arm still goes to the taught withdraw joints.
        np.testing.assert_allclose(
            withdraw.poses[-1].arm_q[7:14], plan.withdraw_right_arm, atol=1e-9
        )

    def test_runner_software_estop_path_releases_right_hand(self):
        """q-stop must freeze LIVE / release right, not keep chasing lever targets."""
        runner_src = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("def software_estop", runner_src)
        self.assertIn("def freeze_live_release_right", runner_src)
        self.assertIn("right_fresh=False", runner_src)
        self.assertIn("Exit_Debug_Mode", runner_src)
        self.assertNotIn("hold_until_quit", runner_src)
        # Source-level: stop handler present and does not advance waypoint index.
        self.assertIn("software e-stop", runner_src.lower())
        tree = ast.parse(runner_src)
        fn_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("software_estop", fn_names)
        self.assertIn("freeze_live_release_right", fn_names)
        self.assertNotIn("hold_until_quit", fn_names)

    def test_approach_refuse_if_too_far(self):
        with self.assertRaisesRegex(m.IcecreamLeverError, "too far"):
            m.approach_duration_right_waist(
                live_right7=[0.0] * 7,
                live_waist=0.0,
                target_right7=[2.0] * 7,  # 2 rad @ 0.6 rad/s = 3.33s arm alone
                target_waist=math.radians(-90),  # 90° @ 10 deg/s = 9s
                max_seconds=5.0,
            )

    def test_check_only_path_does_not_construct_publisher(self):
        """Planner module has no DDS imports; runner defers publisher imports."""
        planner_src = Path(m.__file__).read_text(encoding="utf-8")
        planner_tree = ast.parse(planner_src)
        planner_hits = []
        forbidden = ("ChannelPublisher", "ChannelFactoryInitialize", "unitree_sdk2py", "R1_A7_ArmController")
        for node in ast.walk(planner_tree):
            if isinstance(node, ast.Name) and node.id in forbidden:
                planner_hits.append(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in forbidden:
                planner_hits.append(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    names = [node.module or ""] + [alias.name for alias in node.names]
                for name in names:
                    if any(token in (name or "") for token in forbidden):
                        planner_hits.append(name)
        self.assertEqual(planner_hits, [])

        runner_src = RUNNER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(runner_src)
        # Top-level imports must not pull robot publishers.
        top_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_imports.append(node.module or "")
        joined = " ".join(top_imports)
        self.assertNotIn("unitree_sdk2py", joined)
        self.assertNotIn("robot_arm", joined)
        self.assertNotIn("robot_hand_linker_o6", joined)
        # Motion path imports happen inside run_robot.
        self.assertIn("def run_robot", runner_src)
        self.assertIn("ChannelFactoryInitialize", runner_src)

        doc, _ = _sample_waypoints()
        with tempfile.TemporaryDirectory() as tmp:
            wp = Path(tmp) / "icecream_waypoints.json"
            gc = Path(tmp) / "o6_grip_cap.json"
            wp.write_text(json.dumps(doc), encoding="utf-8")
            gc.write_text(json.dumps(_sample_grip()), encoding="utf-8")
            # Call planner the same way --check-only does (no publisher).
            waypoints = m.load_waypoints(wp)
            right_q = m.load_right_max_close_q(gc)
            plan = m.extract_lever_slot(waypoints, right_q)
            text = m.format_check_only(plan)
            self.assertIn("sends_robot_commands: false", text)
            self.assertFalse(plan.sends_robot_commands)

    def test_runner_check_only_exits_zero(self):
        doc, _ = _sample_waypoints()
        with tempfile.TemporaryDirectory() as tmp:
            wp = Path(tmp) / "icecream_waypoints.json"
            gc = Path(tmp) / "o6_grip_cap.json"
            wp.write_text(json.dumps(doc), encoding="utf-8")
            gc.write_text(json.dumps(_sample_grip()), encoding="utf-8")
            spec = importlib.util.spec_from_file_location(
                "replay_r1_icecream_lever", RUNNER_PATH
            )
            runner = importlib.util.module_from_spec(spec)
            # Avoid executing robot path; load module then call main(--check-only).
            spec.loader.exec_module(runner)
            code = runner.main(
                ["--check-only", "--waypoints", str(wp), "--grip-cap", str(gc)]
            )
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
