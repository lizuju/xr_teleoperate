from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


STAGE_ROOT = Path(__file__).resolve().parents[1]
SIM_TOOLS = STAGE_ROOT / "sim_tools"
if not SIM_TOOLS.is_dir():
    SIM_TOOLS = STAGE_ROOT.parent / "unitree_sim_isaaclab" / "tools"
sys.path.insert(0, str(SIM_TOOLS))
SYNTHETIC_PATH = SIM_TOOLS / "o6_synthetic_live_source.py"
KEYBOARD_PATH = SIM_TOOLS / "o6_keyboard_gesture_source.py"
TELEOP_PATH = STAGE_ROOT / "teleop" / "teleop_hand_and_arm.py"


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class O6WriterLockTest(unittest.TestCase):
    def test_keyboard_gestures_use_contact_scanned_limits(self):
        keyboard = load_module("o6_keyboard_contact_limits_test", KEYBOARD_PATH)
        self.assertEqual(keyboard.GESTURES["fist"], [0.7] * 6)
        self.assertEqual(keyboard.GESTURES["thumb_adduction"], [0.15, 0.9, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(
            keyboard.GESTURES["thumb_index_touch"],
            [0.765, 0.765, 0.675, 0.0, 0.0, 0.0],
        )

    def test_arm_cartesian_payload_uses_a_separate_safe_contract(self):
        keyboard = load_module("o6_keyboard_arm_payload_test", KEYBOARD_PATH)
        payload = keyboard.make_arm_payload(
            [0.02, 0.0, -0.02, -0.02, 0.0, 0.02],
            sequence=7,
            mapping=keyboard.ARM_MAPPING,
            timestamp_ns=1_000_000_000,
        )
        self.assertEqual(payload["schema"], "r1_a7_arm_ik_target_v1")
        self.assertEqual(payload["reference"], "armed_actual_wrist_yaw_pose")
        self.assertEqual(payload["frame"], "r1_waist_yaw")
        self.assertEqual(payload["target_units"], "m")
        self.assertEqual(payload["target_hand_order"], ["left", "right"])
        self.assertEqual(payload["position_offset_m"], [0.02, 0.0, -0.02, -0.02, 0.0, 0.02])
        self.assertEqual(payload["published_monotonic_ns"], 1_000_000_000)

    def test_arm_keyboard_selection_and_workspace_constraint(self):
        keyboard = load_module("o6_keyboard_arm_movement_test", KEYBOARD_PATH)
        target = [0.0] * 6
        for _ in range(20):
            keyboard.move_selected_arm(target, "left", 0, -0.005)
        self.assertAlmostEqual(target[0], -0.06)
        self.assertTrue(keyboard.constrain_dual_offset(target)[1] is False)
        keyboard.center_selected_arm(target, "left")
        self.assertEqual(target, [0.0] * 6)

    def test_keyboard_drains_rapid_consecutive_keys(self):
        keyboard = load_module("o6_keyboard_input_test", KEYBOARD_PATH)
        with (
            mock.patch.object(keyboard.sys, "stdin") as stdin,
            mock.patch.object(
                keyboard.select,
                "select",
                side_effect=[([9], [], []), ([9], [], []), ([], [], [])],
            ),
            mock.patch.object(keyboard.os, "read", side_effect=[b"1", b"o"]),
        ):
            stdin.fileno.return_value = 9
            self.assertEqual(keyboard.read_available_keys(), ["1", "o"])

    def test_synthetic_and_keyboard_contend_on_the_same_lock(self):
        synthetic = load_module("o6_synthetic_live_source_test", SYNTHETIC_PATH)
        keyboard = load_module("o6_keyboard_gesture_source_test", KEYBOARD_PATH)
        with tempfile.TemporaryDirectory(prefix="o6-writer-lock-") as directory:
            target = Path(directory) / "target.json"
            first = synthetic.acquire_writer_lock(target, "test")
            try:
                with self.assertRaisesRegex(RuntimeError, "Another O6 live-state writer"):
                    keyboard.acquire_writer_lock(target)
            finally:
                first.close()

    def test_vision_pro_live_writer_uses_the_shared_lock_name(self):
        source = TELEOP_PATH.read_text(encoding="utf-8")
        self.assertIn('lock_path = live_state_path.parent / "writer.lock"', source)
        self.assertIn("live_writer_lock_file = acquire_live_writer_lock(live_state_path)", source)
        self.assertIn("fcntl.LOCK_EX | fcntl.LOCK_NB", source)


if __name__ == "__main__":
    unittest.main()
