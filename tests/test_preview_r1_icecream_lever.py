#!/usr/bin/env python3
"""Unit tests for ice-cream lever DRY-RUN preview (no robot)."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
MODULE_PATH = TOOLS / "preview_r1_icecream_lever.py"


def load_module():
    spec = importlib.util.spec_from_file_location("preview_r1_icecream_lever", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_waypoints(*, include_pulled=True, include_wait=True, include_withdraw=True, include_turn=True):
    grasp = [-0.0382, -0.3965, 0.3335]
    pulled = [-0.0613, -0.4015, 0.3391]
    withdraw = [-0.0211, -0.3785, 0.4461]
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    lever = {
        "side": "right",
        "frame": "right_wrist_yaw_link",
        "position_m": grasp,
        "rotation": identity,
    }
    if include_pulled:
        lever["pulled"] = {"position_m": pulled, "rotation": identity}
    if include_wait:
        lever["wait_s"] = 3.0
    if include_withdraw:
        lever["withdraw"] = {"position_m": withdraw, "rotation": identity}
    doc = {
        "schema": "icecream_waypoints_v2",
        "slots": {
            "left_cup": {
                "side": "left",
                "frame": "left_wrist_yaw_link",
                "position_m": [0.34, 0.02, 0.09],
                "rotation": identity,
            },
            "right_lever_3": lever,
        },
    }
    if include_turn:
        doc["turn_waist_yaw_rad"] = -0.0468
    return doc


def _sample_grip_cap():
    return {
        "schema": "o6_grip_cap_v2",
        "sides": {
            "left": {"max_close_q": [0.19, 0.48, 0.31, 0.39, 0.41, 0.39]},
            "right": {"max_close_q": [0.25, 0.05, 0.71, 0.73, 0.74, 0.74]},
        },
    }


class PreviewLeverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()

    def test_distances_and_sequence_text(self):
        m = self.m
        with tempfile.TemporaryDirectory() as tmp:
            wp = Path(tmp) / "icecream_waypoints.json"
            gc = Path(tmp) / "o6_grip_cap.json"
            wp.write_text(json.dumps(_sample_waypoints()), encoding="utf-8")
            gc.write_text(json.dumps(_sample_grip_cap()), encoding="utf-8")
            waypoints = m.load_waypoints(wp)
            right_q = m.load_right_max_close_q(gc)
            preview = m.build_preview(waypoints, right_q, slot_name="right_lever_3")
            expected_travel = math.sqrt(
                (-0.0613 - -0.0382) ** 2
                + (-0.4015 - -0.3965) ** 2
                + (0.3391 - 0.3335) ** 2
            )
            self.assertAlmostEqual(preview["pull_travel_m"], expected_travel, places=6)
            self.assertAlmostEqual(preview["pull_travel_m"] * 100.0, expected_travel * 100.0, places=4)
            self.assertAlmostEqual(preview["wait_s"], 3.0)
            self.assertFalse(preview["sends_robot_commands"])
            self.assertEqual(preview["right_max_close_q"], [0.25, 0.05, 0.71, 0.73, 0.74, 0.74])
            self.assertTrue(preview["left_cup_present"])
            text = m.format_preview(preview)
            self.assertIn("sends_robot_commands: false", text)
            self.assertIn("等待 3.0 秒", text)
            self.assertIn("max_close_q=[0.25, 0.05, 0.71, 0.73, 0.74, 0.74]", text)
            self.assertIn("本预览不命令左手", text)
            self.assertIn(f"{expected_travel * 100.0:.2f} cm", text)
            # Withdraw distances present.
            self.assertIn("距 grasp", text)
            self.assertIn("距 pulled", text)

    def test_missing_pulled_refuses(self):
        m = self.m
        waypoints = _sample_waypoints(include_pulled=False)
        right_q = [0.25, 0.05, 0.71, 0.73, 0.74, 0.74]
        with self.assertRaises(m.PreviewError) as ctx:
            m.build_preview(waypoints, right_q)
        self.assertIn("pulled", str(ctx.exception).lower())

    def test_missing_wait_withdraw_turn_refuse(self):
        m = self.m
        right_q = [0.25, 0.05, 0.71, 0.73, 0.74, 0.74]
        with self.assertRaises(m.PreviewError):
            m.build_preview(_sample_waypoints(include_wait=False), right_q)
        with self.assertRaises(m.PreviewError):
            m.build_preview(_sample_waypoints(include_withdraw=False), right_q)
        with self.assertRaises(m.PreviewError):
            m.build_preview(_sample_waypoints(include_turn=False), right_q)

    def test_no_robot_imports_or_lowcmd_symbols(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        bad = []
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.append(alias.name)
                if node.module:
                    imported.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append(alias.name)
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in ("ChannelPublisher", "Enter_Debug_Mode"):
                    bad.append(name)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value == "rt/lowcmd":
                    bad.append("rt/lowcmd")
            if isinstance(node, ast.Name) and node.id in ("LowCmd_", "LowCmd", "ChannelPublisher"):
                bad.append(node.id)
        self.assertEqual(bad, [], f"forbidden motion/publish symbols: {bad}")
        self.assertNotIn("ChannelPublisher", imported)
        self.assertNotIn("LowCmd_", imported)
        self.assertNotIn("LowCmd", imported)
        self.assertNotIn("unitree_sdk2py", imported)
        self.assertNotIn('"rt/lowcmd"', source)
        self.assertNotIn("'rt/lowcmd'", source)
        hits = self.m.module_publishes_lowcmd(source)
        self.assertEqual(hits, [])

    def test_cli_prints_and_exits_zero(self):
        m = self.m
        with tempfile.TemporaryDirectory() as tmp:
            wp = Path(tmp) / "icecream_waypoints.json"
            gc = Path(tmp) / "o6_grip_cap.json"
            wp.write_text(json.dumps(_sample_waypoints()), encoding="utf-8")
            gc.write_text(json.dumps(_sample_grip_cap()), encoding="utf-8")
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                code = m.main(["--waypoints", str(wp), "--grip-cap", str(gc)])
            self.assertEqual(code, 0)
            out = buf.getvalue()
            self.assertIn("sends_robot_commands: false", out)
            self.assertIn("等待 3.0 秒", out)


if __name__ == "__main__":
    unittest.main()
