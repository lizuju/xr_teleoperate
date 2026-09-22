#!/usr/bin/env python3
"""Unit tests for O6 cup-grip max-close persistence (per-axis, no robot required)."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class O6GripCapModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        cls.module = importlib.import_module("teleop.utils.o6_grip_cap")

    def test_missing_file_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            self.assertIsNone(self.module.load_grip_cap(path))

    def test_round_trip_save_load_vector(self):
        limits = [0.2, 0.3, 0.45, 0.4, 0.35, 0.25]
        document = self.module.build_document(
            max_close_q=limits,
            apply_to="both",
            cup_mouth_diameter_cm=9.0,
            command_torque=1.0,
            left_q=limits,
            right_q=limits,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o6_grip_cap.json"
            saved = self.module.save_grip_cap(document, path)
            self.assertTrue(saved.is_file())
            loaded = self.module.load_grip_cap(path)
            self.assertEqual(loaded["schema"], "o6_grip_cap_v2")
            self.assertEqual(loaded["quantity"], "normalized_close_q")
            self.assertEqual(loaded["cup_mouth_diameter_cm"], 9.0)
            self.assertEqual(loaded["axis_names"], list(self.module.AXIS_NAMES))
            self.assertEqual(loaded["sides"]["left"]["max_close_q"], limits)
            self.assertEqual(loaded["sides"]["right"]["max_close_q"], limits)
            text = path.read_text(encoding="utf-8")
            self.assertIn("normalized_close_q", text)
            self.assertIn("thumb_pitch", text)
            self.assertNotIn('"force_N"', text)

    def test_scalar_v1_migrates_to_equal_axes(self):
        legacy = {
            "schema": "o6_grip_cap_v1",
            "quantity": "normalized_close_q",
            "units": "normalized_q_0_to_1",
            "cup_mouth_diameter_cm": 9.0,
            "apply_to": "both",
            "command_torque": 1.0,
            "sides": {
                "left": {"max_close_q": 0.42},
                "right": {"max_close_q": 0.42},
            },
        }
        loaded = self.module.validate_document(legacy)
        self.assertEqual(loaded["schema"], "o6_grip_cap_v2")
        self.assertEqual(loaded["migrated_from"], "o6_grip_cap_v1")
        self.assertEqual(loaded["sides"]["left"]["max_close_q"], [0.42] * 6)
        self.assertIn("Migrated from o6_grip_cap_v1", loaded["notes"])
        # Clamp after migration is still a vector — no scalar broadcast path.
        capped = self.module.clamp_close_q(
            np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1]),
            loaded["sides"]["left"]["max_close_q"],
        )
        np.testing.assert_allclose(capped, [0.42, 0.1, 0.42, 0.1, 0.42, 0.1])

    def test_reject_fake_schema_and_out_of_range(self):
        with self.assertRaises(self.module.GripCapError):
            self.module.validate_document(
                {
                    "schema": "nope",
                    "quantity": "normalized_close_q",
                    "sides": {"left": {"max_close_q": [0.5] * 6}},
                }
            )
        with self.assertRaises(self.module.GripCapError):
            self.module.build_document(max_close_q=[1.5] * 6)

    def test_per_axis_clamp_does_not_pull_other_axes(self):
        values = np.array([0.1, 0.9, 0.5, 0.0, 1.0, 0.7])
        limits = np.array([0.55, 0.55, 0.55, 0.55, 0.2, 0.55])
        capped = self.module.clamp_close_q(values, limits)
        np.testing.assert_allclose(capped, [0.1, 0.55, 0.5, 0.0, 0.2, 0.55])
        # Axis 0 stayed low; axis 2 stayed at 0.5 — only axes above their own limit move.
        self.assertEqual(capped[0], values[0])
        self.assertEqual(capped[2], values[2])
        self.assertEqual(capped[3], values[3])
        untouched = self.module.clamp_close_q(values, None)
        np.testing.assert_allclose(untouched, values)

    def test_max_close_for_side_both_and_single(self):
        both = self.module.build_document(max_close_q=[0.3] * 6, apply_to="both")
        np.testing.assert_allclose(self.module.max_close_for_side(both, "left"), [0.3] * 6)
        np.testing.assert_allclose(self.module.max_close_for_side(both, "right"), [0.3] * 6)
        left_only = self.module.build_document(
            max_close_q=[0.25, 0.2, 0.3, 0.3, 0.3, 0.3], apply_to="left"
        )
        np.testing.assert_allclose(
            self.module.max_close_for_side(left_only, "left"),
            [0.25, 0.2, 0.3, 0.3, 0.3, 0.3],
        )
        self.assertIsNone(self.module.max_close_for_side(left_only, "right"))
        # Right key must not appear when calibrating left-only.
        self.assertNotIn("right", left_only["sides"])
        limits = self.module.limits_from_document(left_only)
        self.assertIsNone(limits["right"])

    def test_left_only_saved_shape_honored(self):
        """Operator save with apply_to=left and sides.left only — no right copy."""
        limits = [0.19, 0.48, 0.31, 0.39, 0.41, 0.39]
        document = {
            "schema": "o6_grip_cap_v2",
            "quantity": "normalized_close_q",
            "units": "normalized_q_0_to_1",
            "apply_to": "left",
            "command_torque": 1.0,
            "sides": {"left": {"max_close_q": limits, "command_torque": 1.0}},
        }
        loaded = self.module.validate_document(document)
        np.testing.assert_allclose(
            self.module.max_close_for_side(loaded, "left"), limits
        )
        self.assertIsNone(self.module.max_close_for_side(loaded, "right"))

    def test_absent_or_null_right_is_uncapped(self):
        """Missing or null right must not inherit the left vector."""
        limits = [0.22, 0.33, 0.44, 0.55, 0.66, 0.11]
        both_missing_right = {
            "schema": "o6_grip_cap_v2",
            "quantity": "normalized_close_q",
            "apply_to": "both",
            "sides": {"left": {"max_close_q": limits}},
        }
        loaded = self.module.validate_document(both_missing_right)
        np.testing.assert_allclose(
            self.module.max_close_for_side(loaded, "left"), limits
        )
        self.assertIsNone(self.module.max_close_for_side(loaded, "right"))

        null_right = {
            "schema": "o6_grip_cap_v2",
            "quantity": "normalized_close_q",
            "apply_to": "both",
            "sides": {"left": {"max_close_q": limits}, "right": None},
        }
        loaded_null = self.module.validate_document(null_right)
        self.assertNotIn("right", loaded_null["sides"])
        self.assertIsNone(self.module.max_close_for_side(loaded_null, "right"))


class CupGripCalibratorLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        tools_dir = str(ROOT / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        cls.tool = importlib.import_module("calibrate_o6_cup_grip")

    def test_can_record_gates(self):
        close = np.full(6, 0.4)
        ok, _ = self.tool.can_record(armed=False, ever_moved=True, close_q=close)
        self.assertFalse(ok)
        ok, _ = self.tool.can_record(armed=True, ever_moved=False, close_q=close)
        self.assertFalse(ok)
        ok, _ = self.tool.can_record(
            armed=True, ever_moved=True, close_q=np.full(6, 0.05)
        )
        self.assertFalse(ok)
        ok, reason = self.tool.can_record(armed=True, ever_moved=True, close_q=close)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_bump_only_changes_selected_axis(self):
        class Dummy:
            def get_state(self):
                return np.zeros(6), np.zeros(6)

            def get_action(self):
                return np.zeros(6), np.zeros(6)

            def get_recording_snapshot(self):
                return {
                    "state": {
                        "left": {"torque": [0.0] * 6},
                        "right": {"torque": [0.0] * 6},
                    }
                }

            def update(self, *args, **kwargs):
                return None

        calibrator = self.tool.CupGripCalibrator(Dummy())
        calibrator.arm()
        calibrator.select_axis(2)  # index
        before = calibrator.close_q.copy()
        calibrator.bump(+0.01)
        after = calibrator.close_q
        self.assertAlmostEqual(after[2], before[2] + 0.01)
        for index in range(6):
            if index == 2:
                continue
            self.assertEqual(after[index], before[index])
        # Save requires movement away from the start pose.
        path = None
        with tempfile.TemporaryDirectory() as tmp:
            calibrator.save_path = Path(tmp) / "cap.json"
            # Still near start on most axes; one axis at 0.06 may be insufficient
            # if MOVE_EPSILON is 0.02 — ramp further.
            for _ in range(5):
                calibrator.bump(+0.01)
            path = calibrator.record()
            self.assertIsNotNone(path)
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            saved = loaded["sides"]["left"]["max_close_q"]
            self.assertEqual(len(saved), 6)
            self.assertGreater(saved[2], 0.05 + 0.02)
            for index in (0, 1, 3, 4, 5):
                self.assertAlmostEqual(saved[index], 0.05)


class LinkerO6GripCapIntegrationTest(unittest.TestCase):
    """Exercise controller clamp with the existing fake DDS stack."""

    @classmethod
    def setUpClass(cls):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        class FakeMotorCmd:
            def __init__(self):
                self.mode = 0
                self.q = 0.0
                self.dq = 0.0
                self.tau = 0.0

        class FakeMotorCmds:
            def __init__(self):
                self.cmds = []

        class FakePublisher:
            instances = []

            def __init__(self, topic, message_type):
                self.topic = topic
                self.message_type = message_type
                self.writes = []
                self.write_result = True
                self.closed = False
                type(self).instances.append(self)

            def Init(self):
                pass

            def Write(self, message, timeout=None):
                self.writes.append(message)
                return self.write_result

            def Close(self):
                self.closed = True

        class FakeSubscriber:
            instances = []
            queued_messages = {}

            def __init__(self, topic, message_type):
                self.topic = topic
                self.message_type = message_type
                self.handler = None
                self.closed = False
                type(self).instances.append(self)

            def Init(self, handler=None, queueLen=0):
                self.handler = handler
                if handler is not None:
                    messages = type(self).queued_messages.setdefault(self.topic, [])
                    while messages:
                        handler(messages.pop(0))

            def Close(self):
                self.closed = True

            @classmethod
            def deliver(cls, topic, message):
                for subscriber in cls.instances:
                    if (
                        subscriber.topic == topic
                        and subscriber.handler is not None
                        and not subscriber.closed
                    ):
                        subscriber.handler(message)
                        return
                cls.queued_messages.setdefault(topic, []).append(message)

        channel_module = types.ModuleType("unitree_sdk2py.core.channel")
        channel_module.ChannelPublisher = FakePublisher
        channel_module.ChannelSubscriber = FakeSubscriber
        default_module = types.ModuleType("unitree_sdk2py.idl.default")
        default_module.unitree_go_msg_dds__MotorCmd_ = FakeMotorCmd
        dds_module = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
        dds_module.MotorCmds_ = FakeMotorCmds
        dds_module.MotorStates_ = object
        modules = {
            "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
            "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
            "unitree_sdk2py.core.channel": channel_module,
            "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl"),
            "unitree_sdk2py.idl.default": default_module,
            "unitree_sdk2py.idl.unitree_go": types.ModuleType("unitree_sdk2py.idl.unitree_go"),
            "unitree_sdk2py.idl.unitree_go.msg": types.ModuleType(
                "unitree_sdk2py.idl.unitree_go.msg"
            ),
            "unitree_sdk2py.idl.unitree_go.msg.dds_": dds_module,
        }
        cls.FakePublisher = FakePublisher
        cls.FakeSubscriber = FakeSubscriber
        cls.module_patch = mock.patch.dict(sys.modules, modules)
        cls.module_patch.start()
        for name in list(sys.modules):
            if name.startswith("teleop.robot_control.robot_hand_linker_o6") or name == "teleop.utils.o6_grip_cap":
                del sys.modules[name]
        cls.grip = importlib.import_module("teleop.utils.o6_grip_cap")
        cls.hand = importlib.import_module("teleop.robot_control.robot_hand_linker_o6")

    @classmethod
    def tearDownClass(cls):
        cls.module_patch.stop()

    def setUp(self):
        self.FakePublisher.instances.clear()
        self.FakeSubscriber.instances.clear()
        self.FakeSubscriber.queued_messages.clear()

    def _state(self, values, mode=1):
        return types.SimpleNamespace(
            states=[
                types.SimpleNamespace(
                    q=value, mode=mode, dq=0.0, tau_est=0.0, temperature=0.0, reserve=[0]
                )
                for value in values
            ]
        )

    def _queue_pair(self, left, right, modes=(1, 1)):
        self.FakeSubscriber.deliver("rt/linker/left/state", self._state(left, modes[0]))
        self.FakeSubscriber.deliver("rt/linker/right/state", self._state(right, modes[1]))

    def _activate(self, controller, pose=None):
        import threading
        import time

        pose = [0.0] * 6 if pose is None else list(pose)
        errors = []

        def run():
            try:
                controller.activate()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(0.01)
        self.assertTrue(thread.is_alive())
        self._queue_pair(pose, pose)
        deadline = time.monotonic() + 0.5
        while (
            any(len(publisher.writes) == 0 for publisher in self.FakePublisher.instances)
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        self.assertTrue(all(len(publisher.writes) >= 1 for publisher in self.FakePublisher.instances))
        self._queue_pair(pose, pose)
        thread.join(timeout=0.5)
        if errors:
            raise errors[0]
        self.assertFalse(thread.is_alive())
        self.assertTrue(controller.active)

    def test_no_file_leaves_targets_uncapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent.json"
            self._queue_pair([0.0] * 6, [0.0] * 6)
            controller = self.hand.LinkerO6Controller(grip_cap_path=missing, apply_grip_cap=True)
            controller.wait_until_ready(timeout=0.2)
            self.assertIsNone(controller._grip_max_close["left"])
            target = np.array([0.8] * 6)
            np.testing.assert_allclose(controller._capped_target(target, "left"), target)

    def test_saved_vector_cap_clamps_elementwise_before_smooth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cap.json"
            limits = [0.35, 0.5, 0.2, 0.9, 0.4, 0.4]
            self.grip.save_grip_cap(
                self.grip.build_document(max_close_q=limits, apply_to="both"),
                path,
            )
            self._queue_pair([0.0] * 6, [0.0] * 6)
            controller = self.hand.LinkerO6Controller(grip_cap_path=path, apply_grip_cap=True)
            controller.wait_until_ready(timeout=0.2)
            np.testing.assert_allclose(controller._grip_max_close["left"], limits)
            self._activate(controller)
            target = np.array([0.9, 0.9, 0.9, 0.1, 0.9, 0.05])
            expected = np.minimum(target, limits)
            with mock.patch.object(self.hand.time, "monotonic", return_value=100.0):
                controller.action_time = 99.0
                self._queue_pair([0.0] * 6, [0.0] * 6)
                controller.update(target, target, tracking_fresh=(True, True))
            np.testing.assert_allclose(controller.requested_targets[0], expected)
            np.testing.assert_allclose(controller.requested_targets[1], expected)
            # Axes under their own limit were not dragged down by a sibling limit.
            self.assertAlmostEqual(controller.requested_targets[0][3], 0.1)
            self.assertAlmostEqual(controller.requested_targets[0][5], 0.05)

    def test_apply_grip_cap_false_skips_autoload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cap.json"
            self.grip.save_grip_cap(
                self.grip.build_document(max_close_q=[0.2] * 6, apply_to="both"),
                path,
            )
            self._queue_pair([0.0] * 6, [0.0] * 6)
            controller = self.hand.LinkerO6Controller(grip_cap_path=path, apply_grip_cap=False)
            self.assertIsNone(controller._grip_max_close["left"])
            target = np.array([0.7] * 6)
            np.testing.assert_allclose(controller._capped_target(target, "left"), target)

    def test_left_only_cap_clamps_left_leaves_right_uncapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cap.json"
            limits = [0.19, 0.48, 0.31, 0.39, 0.41, 0.39]
            self.grip.save_grip_cap(
                self.grip.build_document(max_close_q=limits, apply_to="left"),
                path,
            )
            self._queue_pair([0.0] * 6, [0.0] * 6)
            controller = self.hand.LinkerO6Controller(grip_cap_path=path, apply_grip_cap=True)
            controller.wait_until_ready(timeout=0.2)
            np.testing.assert_allclose(controller._grip_max_close["left"], limits)
            self.assertIsNone(controller._grip_max_close["right"])
            self._activate(controller)
            left_in = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
            right_in = np.array([0.85, 0.7, 0.6, 0.5, 0.4, 0.95])
            expected_left = np.minimum(left_in, limits)
            with mock.patch.object(self.hand.time, "monotonic", return_value=100.0):
                controller.action_time = 99.0
                self._queue_pair([0.0] * 6, [0.0] * 6)
                controller.update(left_in, right_in, tracking_fresh=(True, True))
            np.testing.assert_allclose(controller.requested_targets[0], expected_left)
            # Right hand must remain exactly the retargeted target (no cap).
            np.testing.assert_allclose(controller.requested_targets[1], right_in)


if __name__ == "__main__":
    unittest.main()
