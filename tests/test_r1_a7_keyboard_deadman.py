import importlib.util
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEADMAN_PATH = REPO_ROOT / "teleop" / "r1_a7_keyboard_deadman.py"
PUBLISHER_PATH = REPO_ROOT / "teleop" / "r1_a7_arm_publisher_shadow.py"

deadman_spec = importlib.util.spec_from_file_location(
    "r1_a7_keyboard_deadman", DEADMAN_PATH
)
deadman = importlib.util.module_from_spec(deadman_spec)
deadman_spec.loader.exec_module(deadman)
publisher_spec = importlib.util.spec_from_file_location(
    "r1_a7_arm_publisher_shadow_for_deadman_test", PUBLISHER_PATH
)
publisher = importlib.util.module_from_spec(publisher_spec)
publisher_spec.loader.exec_module(publisher)


class KeyboardDeadmanTest(unittest.TestCase):
    def make_state(self):
        return deadman.KeyboardDeadmanState(0.08, 0.75)

    def test_startup_requires_release_then_a_new_press(self):
        state = self.make_state()
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.0)
        self.assertFalse(state.snapshot(1.0)[0])
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.1)
        self.assertFalse(state.snapshot(1.1)[0])
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.2)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.3)
        self.assertFalse(state.snapshot(1.3)[0])
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.35)
        self.assertTrue(state.snapshot(1.35)[0])

    def test_release_stops_immediately(self):
        state = self.make_state()
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.0)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.1)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.15)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.2)
        held, status, _age = state.snapshot(1.2)
        self.assertFalse(held)
        self.assertEqual(status, "released")

    def test_repeat_refreshes_but_stuck_held_times_out(self):
        state = self.make_state()
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.0)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.1)
        self.assertFalse(state.snapshot(1.1)[0])
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.2)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.25)
        self.assertTrue(state.snapshot(1.329)[0])
        held, status, _age = state.snapshot(1.33)
        self.assertFalse(held)
        self.assertEqual(status, "key_event_timeout_requires_release")
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 2.5)
        self.assertFalse(state.snapshot(2.5)[0])

    def test_late_repeat_cannot_refresh_an_expired_lease(self):
        state = self.make_state()
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.0)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.1)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.2)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.3)
        held, status, _age = state.snapshot(1.3)
        self.assertFalse(held)
        self.assertEqual(status, "key_event_timeout_requires_release")

    def test_long_hold_has_no_fixed_duration_limit(self):
        state = self.make_state()
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.0)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.1)
        for index in range(1, 301):
            now = 1.2 + index / 30.0
            state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, now)
            self.assertTrue(state.snapshot(now)[0])
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, now + 0.01)
        self.assertFalse(state.snapshot(now + 0.01)[0])

    def test_late_first_repeat_cannot_revive_an_old_press(self):
        state = self.make_state()
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.0)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.1)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 100.0)
        held, status, _age = state.snapshot(100.0)
        self.assertFalse(held)
        self.assertEqual(status, "first_repeat_timeout_requires_release")

    def test_queued_old_release_press_repeat_is_stale_at_snapshot(self):
        state = self.make_state()
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.0)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.1)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.15)
        held, status, _age = state.snapshot(100.0)
        self.assertFalse(held)
        self.assertEqual(status, "key_event_timeout_requires_release")

    def test_syn_dropped_requires_release_and_repress(self):
        state = self.make_state()
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.0)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.1)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.15)
        state.consume(deadman.EV_SYN, deadman.SYN_DROPPED, 0, 1.2)
        self.assertFalse(state.snapshot(1.2)[0])
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.3)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.4)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.45)
        self.assertFalse(state.snapshot(1.45)[0])
        state.consume(deadman.EV_SYN, deadman.SYN_REPORT, 0, 1.5)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_RELEASE, 1.6)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_PRESS, 1.7)
        state.consume(deadman.EV_KEY, deadman.KEY_SPACE, deadman.KEY_REPEAT, 1.75)
        self.assertTrue(state.snapshot(1.75)[0])

    def test_unrelated_keys_cannot_enable_deadman(self):
        state = self.make_state()
        state.consume(deadman.EV_KEY, 28, deadman.KEY_RELEASE, 1.0)
        state.consume(deadman.EV_KEY, 28, deadman.KEY_PRESS, 1.1)
        self.assertFalse(state.snapshot(1.1)[0])

    def test_heartbeat_contract_is_accepted_by_shadow_consumer(self):
        row = deadman.payload(
            1,
            True,
            "held",
            0.01,
            "/dev/input/event5",
            12_345_678_900,
        )
        now = row["updated_monotonic_ns"] / 1_000_000_000.0
        self.assertEqual(row["updated_monotonic_ns"], 12_345_678_900)
        normalized, status = publisher.validate_deadman(row, now, 0.10)
        self.assertEqual(status, "valid")
        self.assertTrue(normalized["held"])
        self.assertEqual(
            publisher.validate_deadman(row, now + 0.101, 0.10)[1],
            "stale",
        )
        invalid = dict(row)
        invalid["held"] = 1
        self.assertEqual(
            publisher.validate_deadman(invalid, now, 0.10)[1],
            "invalid_held",
        )

    def test_writer_lock_is_single_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "deadman.json"
            first = deadman.acquire_writer_lock(output)
            try:
                with self.assertRaises(RuntimeError):
                    deadman.acquire_writer_lock(output)
            finally:
                first.close()

    def test_source_and_launcher_have_no_robot_output_path(self):
        source = DEADMAN_PATH.read_text(encoding="utf-8")
        launcher = (
            REPO_ROOT / "teleop" / "run_r1_a7_keyboard_deadman.sh"
        ).read_text(encoding="utf-8")
        for text in (source, launcher):
            for token in (
                "unitree_sdk2py",
                "ChannelPublisher",
                "ChannelFactoryInitialize",
                "rt/lowcmd",
                "LowCmd",
                "MotionSwitcher",
                "Enter_Debug_Mode",
                "import socket",
                "import serial",
                "import subprocess",
                "os.system",
                "Popen",
            ):
                with self.subTest(token=token):
                    self.assertNotIn(token, text)
        self.assertIn("EVIOCSCLOCKID", source)
        self.assertIn("event_time = seconds + microseconds / 1_000_000.0", source)
        self.assertIn("signal.SIGHUP", source)


if __name__ == "__main__":
    unittest.main()
