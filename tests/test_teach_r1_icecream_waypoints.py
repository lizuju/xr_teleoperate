#!/usr/bin/env python3
"""Unit tests for ice-cream waypoint teach-in (fake FK; no robot)."""

from __future__ import annotations

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
MODULE_PATH = TOOLS / "teach_r1_icecream_waypoints.py"


def load_module():
    # Stub r1_hand_eye_common so tests run without pinocchio/urdf.
    import sys
    import types

    if "r1_hand_eye_common" not in sys.modules:
        stub = types.ModuleType("r1_hand_eye_common")
        stub.DEFAULT_URDF = Path("/tmp/unused.urdf")
        stub.HandEyeError = RuntimeError

        def motor_q_to_pinocchio_q(full_q):
            full_q = np.asarray(full_q, dtype=float).reshape(-1)
            q = np.zeros(17, dtype=float)
            # Mirror production mapping for waist_yaw (motor 13 → pin 0).
            if full_q.size > 13:
                q[0] = float(full_q[13])
            return q

        def pinocchio_q_to_named(q):
            q = np.asarray(q, dtype=float).reshape(17)
            return {
                "waist_yaw": float(q[0]),
                "pinocchio_q": [float(x) for x in q],
            }

        def rt_from_se3(matrix):
            matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
            return matrix[:3, :3].copy(), matrix[:3, 3].copy()

        class R1A7FK:
            def link_pose(self, pinocchio_q, frame_name):
                raise AssertionError("tests should inject FakeFK")

        stub.motor_q_to_pinocchio_q = motor_q_to_pinocchio_q
        stub.pinocchio_q_to_named = pinocchio_q_to_named
        stub.rt_from_se3 = rt_from_se3
        stub.R1A7FK = R1A7FK
        sys.modules["r1_hand_eye_common"] = stub

    # Prefer tools/ on path for the real import inside the module.
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    spec = importlib.util.spec_from_file_location("teach_r1_icecream_waypoints", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def se3(xyz, R=None):
    T = np.eye(4, dtype=float)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return T


class FakeFK:
    def __init__(self, left_xyz=(0.1, 0.2, 0.3), right_xyz=(0.4, -0.1, 0.5)):
        self.left_xyz = np.asarray(left_xyz, dtype=float)
        self.right_xyz = np.asarray(right_xyz, dtype=float)

    def link_pose(self, pinocchio_q, frame_name):
        if frame_name == "left_wrist_yaw_link":
            return se3(self.left_xyz)
        if frame_name == "right_wrist_yaw_link":
            return se3(self.right_xyz)
        raise KeyError(frame_name)


class FakeQSource:
    def __init__(self, full_q=None):
        self._full_q = np.zeros(35, dtype=float) if full_q is None else np.asarray(full_q, float)

    def get_motor_q(self):
        return self._full_q.copy()

    def set_waist_yaw(self, yaw_rad):
        self._full_q[13] = float(yaw_rad)

    def close(self):
        return None


def _v1_sample_document():
    """Minimal production-like v1 payload (left_cup + lever_3 grasp + unit pull)."""
    return {
        "schema": "icecream_waypoints_v1",
        "notes": "test v1",
        "updated_at": "2026-09-22T16:21:49+0800",
        "slots": {
            "left_cup": {
                "side": "left",
                "frame": "left_wrist_yaw_link",
                "position_m": [0.341, 0.024, 0.089],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "recorded_at": "2026-09-22T16:10:00+0800",
            },
            "right_lever_3": {
                "side": "right",
                "frame": "right_wrist_yaw_link",
                "position_m": [-0.038, -0.397, 0.334],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "pull_direction_base": [-0.134, 0.776, -0.616],
                "pull_direction_recorded_at": "2026-09-22T16:20:00+0800",
                "recorded_at": "2026-09-22T16:15:00+0800",
            },
        },
    }


class TeachIcecreamWaypointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()

    def test_v1_load_keeps_grasp_and_left_cup(self):
        m = self.m
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "icecream_waypoints.json"
            path.write_text(
                __import__("json").dumps(_v1_sample_document(), indent=2) + "\n",
                encoding="utf-8",
            )
            loaded = m.load_document(path)
            self.assertEqual(loaded["schema"], "icecream_waypoints_v2")
            self.assertAlmostEqual(loaded["slots"]["left_cup"]["position_m"][0], 0.341)
            lever = loaded["slots"]["right_lever_3"]
            self.assertAlmostEqual(lever["position_m"][1], -0.397)
            self.assertIn("pull_direction_base", lever)
            # Unit vector preserved (renormalized).
            self.assertAlmostEqual(float(np.linalg.norm(lever["pull_direction_base"])), 1.0)
            # Disk file must still be v1 until caller presses w.
            on_disk = __import__("json").loads(path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["schema"], "icecream_waypoints_v1")

    def test_save_reload_preserves_slots(self):
        m = self.m
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "icecream_waypoints.json"
            doc = m.empty_document()
            left = m.matrix_to_pose_entry(
                se3([0.11, 0.22, 0.33]),
                side="left",
                frame="left_wrist_yaw_link",
                pinocchio_q=np.zeros(17),
            )
            right = m.matrix_to_pose_entry(
                se3([0.41, -0.12, 0.55]),
                side="right",
                frame="right_wrist_yaw_link",
                pinocchio_q=np.zeros(17),
            )
            doc = m.upsert_pose(doc, "left_cup", left)
            doc = m.upsert_pose(doc, "right_lever_1", right)
            m.save_document(doc, path)
            loaded = m.load_document(path)
            self.assertEqual(loaded["schema"], "icecream_waypoints_v2")
            self.assertAlmostEqual(loaded["slots"]["left_cup"]["position_m"][0], 0.11)
            self.assertAlmostEqual(loaded["slots"]["right_lever_1"]["position_m"][2], 0.55)

    def test_left_slot_does_not_overwrite_right(self):
        m = self.m
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.json"
            doc = m.empty_document()
            doc = m.upsert_pose(
                doc,
                "right_lever_2",
                m.matrix_to_pose_entry(
                    se3([1.0, 2.0, 3.0]), side="right", frame="right_wrist_yaw_link"
                ),
            )
            m.save_document(doc, path)
            doc2 = m.load_document(path)
            doc2 = m.upsert_pose(
                doc2,
                "left_cup",
                m.matrix_to_pose_entry(
                    se3([9.0, 8.0, 7.0]), side="left", frame="left_wrist_yaw_link"
                ),
            )
            m.save_document(doc2, path)
            final = m.load_document(path)
            self.assertIn("right_lever_2", final["slots"])
            self.assertAlmostEqual(final["slots"]["right_lever_2"]["position_m"][0], 1.0)
            self.assertAlmostEqual(final["slots"]["left_cup"]["position_m"][0], 9.0)

    def test_pulled_pose_stores_meters_and_travel_length(self):
        m = self.m
        grasp = [0.0, 0.0, 0.0]
        pulled = [0.03, 0.04, 0.0]
        distance = m.travel_length_m(grasp, pulled)
        self.assertAlmostEqual(distance, 0.05)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            fk = FakeFK(right_xyz=(0.0, 0.0, 0.0))
            teacher = m.IcecreamWaypointTeacher(FakeQSource(), fk, save_path=path)
            teacher.select_slot("right_lever_3")
            teacher.save_pose()
            fk.right_xyz = np.array([0.03, 0.04, 0.0])
            pulled_entry, travel = teacher.save_pulled_pose()
            self.assertAlmostEqual(travel, 0.05)
            self.assertAlmostEqual(pulled_entry["position_m"][0], 0.03)
            teacher.write_file()
            loaded = m.load_document(path)
            lever = loaded["slots"]["right_lever_3"]
            self.assertAlmostEqual(lever["position_m"][0], 0.0)
            self.assertAlmostEqual(lever["pulled"]["position_m"][1], 0.04)
            self.assertAlmostEqual(m.travel_length_m(lever["position_m"], lever["pulled"]["position_m"]), 0.05)
            # Legacy unit vector kept in sync.
            self.assertAlmostEqual(float(np.linalg.norm(lever["pull_direction_base"])), 1.0)

    def test_wait_s_must_be_positive(self):
        m = self.m
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wait.json"
            teacher = m.IcecreamWaypointTeacher(FakeQSource(), FakeFK(), save_path=path)
            teacher.select_slot("right_lever_1")
            teacher.save_pose()
            teacher.set_wait_default()
            self.assertAlmostEqual(teacher.document["slots"]["right_lever_1"]["wait_s"], 3.0)
            teacher.nudge_wait(+1.0)
            self.assertAlmostEqual(teacher.document["slots"]["right_lever_1"]["wait_s"], 4.0)
            with self.assertRaises(m.TeachError):
                teacher.nudge_wait(-10.0)
            with self.assertRaises(m.TeachError):
                m.upsert_wait_s(teacher.document, "right_lever_1", 0.0)
            with self.assertRaises(m.TeachError):
                m.upsert_wait_s(teacher.document, "right_lever_1", -1.0)

    def test_withdraw_and_yaw_do_not_overwrite_grasp(self):
        m = self.m
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wy.json"
            q = FakeQSource()
            q.set_waist_yaw(0.42)
            fk = FakeFK(right_xyz=(-0.038, -0.397, 0.334))
            teacher = m.IcecreamWaypointTeacher(q, fk, save_path=path)
            teacher.select_slot("right_lever_3")
            teacher.save_pose()
            grasp_xyz = list(teacher.document["slots"]["right_lever_3"]["position_m"])
            fk.right_xyz = np.array([-0.1, -0.5, 0.4])
            teacher.save_withdraw_pose()
            yaw = teacher.save_turn_yaw()
            self.assertAlmostEqual(yaw, 0.42)
            lever = teacher.document["slots"]["right_lever_3"]
            self.assertAlmostEqual(lever["position_m"][0], grasp_xyz[0])
            self.assertAlmostEqual(lever["position_m"][1], grasp_xyz[1])
            self.assertAlmostEqual(lever["withdraw"]["position_m"][0], -0.1)
            self.assertAlmostEqual(teacher.document["turn_waist_yaw_rad"], 0.42)
            teacher.write_file()
            loaded = m.load_document(path)
            self.assertAlmostEqual(loaded["slots"]["right_lever_3"]["position_m"][2], grasp_xyz[2])
            self.assertAlmostEqual(loaded["turn_waist_yaw_rad"], 0.42)

    def test_writing_one_slot_does_not_clear_another(self):
        m = self.m
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keep.json"
            path.write_text(
                __import__("json").dumps(_v1_sample_document(), indent=2) + "\n",
                encoding="utf-8",
            )
            fk = FakeFK(left_xyz=(0.341, 0.024, 0.089), right_xyz=(-0.038, -0.397, 0.334))
            teacher = m.IcecreamWaypointTeacher(FakeQSource(), fk, save_path=path)
            # Memory already has left_cup + lever_3 from v1 migration.
            self.assertIn("left_cup", teacher.document["slots"])
            self.assertIn("right_lever_3", teacher.document["slots"])
            teacher.select_slot("right_lever_3")
            fk.right_xyz = np.array([-0.05, -0.35, 0.28])
            teacher.save_pulled_pose()
            teacher.set_wait_default()
            teacher.write_file()
            loaded = m.load_document(path)
            self.assertEqual(loaded["schema"], "icecream_waypoints_v2")
            self.assertAlmostEqual(loaded["slots"]["left_cup"]["position_m"][0], 0.341)
            self.assertAlmostEqual(loaded["slots"]["right_lever_3"]["position_m"][1], -0.397)
            self.assertIn("pulled", loaded["slots"]["right_lever_3"])
            self.assertAlmostEqual(loaded["slots"]["right_lever_3"]["wait_s"], 3.0)

    def test_pull_direction_unit_length(self):
        m = self.m
        direction, distance = m.compute_pull_direction([0.0, 0.0, 0.0], [0.03, 0.0, 0.0])
        self.assertAlmostEqual(distance, 0.03)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)
        self.assertAlmostEqual(direction[0], 1.0)

    def test_refuse_tiny_pull_motion(self):
        m = self.m
        with self.assertRaises(m.TeachError) as ctx:
            m.compute_pull_direction([0.0, 0.0, 0.0], [0.005, 0.0, 0.0])
        self.assertIn("too close", str(ctx.exception))

    def test_teacher_save_pose_and_direction_with_fake_fk(self):
        m = self.m
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "icecream_waypoints.json"
            fk = FakeFK(left_xyz=(0.2, 0.0, 0.8), right_xyz=(0.5, -0.2, 0.7))
            teacher = m.IcecreamWaypointTeacher(FakeQSource(), fk, save_path=path)
            teacher.select_slot("right_lever_1")
            teacher.save_pose()
            # Move along +Y by 4 cm, then record direction.
            fk.right_xyz = np.array([0.5, -0.16, 0.7])
            direction, distance = teacher.save_direction()
            self.assertGreater(distance, 0.01)
            self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)
            teacher.write_file()
            loaded = m.load_document(path)
            lever = loaded["slots"]["right_lever_1"]
            self.assertAlmostEqual(lever["position_m"][0], 0.5)
            self.assertAlmostEqual(lever["pull_direction_base"][1], 1.0, places=5)

    def test_direction_before_pose_refused(self):
        m = self.m
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.json"
            teacher = m.IcecreamWaypointTeacher(FakeQSource(), FakeFK(), save_path=path)
            teacher.select_slot("right_lever_3")
            with self.assertRaises(m.TeachError):
                teacher.save_direction()
            with self.assertRaises(m.TeachError):
                teacher.save_pulled_pose()

    def test_no_lowcmd_publish_path_in_source(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        bad = []
        for node in ast.walk(tree):
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
            if isinstance(node, ast.Name) and node.id in ("LowCmd_", "LowCmd"):
                bad.append(node.id)
        self.assertEqual(bad, [], f"forbidden motion/publish symbols: {bad}")
        # Also ensure the helper agrees.
        hits = self.m.module_publishes_lowcmd(source)
        # Documentary strings are ignored; Attribute/Name hits must be empty.
        self.assertEqual(hits, [])

    def test_module_does_not_import_publisher_symbols(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.append(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append(alias.name)
        self.assertNotIn("ChannelPublisher", imported)
        self.assertNotIn("LowCmd_", imported)
        self.assertNotIn("LowCmd", imported)
        # Topic string must never target lowcmd.
        self.assertNotIn('"rt/lowcmd"', source)
        self.assertNotIn("'rt/lowcmd'", source)


if __name__ == "__main__":
    unittest.main()
