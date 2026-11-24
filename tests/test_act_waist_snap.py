import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from teleop.utils.act_runtime import ActionEnsemble, PolicyGuard, WAIST_INDEX, WaistHold

HAS_SDK = importlib.util.find_spec("unitree_sdk2py") is not None and importlib.util.find_spec("sshkeyboard") is not None
RELEASE, CAPTURE = math.radians(5), math.radians(2)


def action(waist_deg):
    values = np.linspace(-0.6, 0.6, 24)
    values[WAIST_INDEX] = math.radians(waist_deg)
    return values


def commanded(hold, targets_deg):
    return [round(math.degrees(hold.apply(action(waist))[WAIST_INDEX]), 6) for waist in targets_deg]


class HoldTests(unittest.TestCase):
    def test_a_held_stop_absorbs_drift_up_to_release(self):
        hold = WaistHold(RELEASE, CAPTURE, -math.pi / 2)
        self.assertEqual(commanded(hold, [-86.0, -94.9, -85.1, -88.0]), [-90.0] * 4)

    def test_a_turn_passes_once_it_leaves_release_and_stays_free_until_capture(self):
        hold = WaistHold(RELEASE, CAPTURE, 0.0)
        self.assertEqual(commanded(hold, [-1.0, -4.0, -4.9, -5.1, -4.0, -3.0, -9.0, -45.0]),
                         [0.0, 0.0, 0.0, -5.1, -4.0, -3.0, -9.0, -45.0])

    def test_a_target_within_capture_starts_a_new_hold(self):
        hold = WaistHold(RELEASE, CAPTURE, math.radians(-45.0))
        self.assertEqual(commanded(hold, [-60.0, -86.0, -88.5, -86.0, -84.9, -86.0]),
                         [-60.0, -86.0, -90.0, -90.0, -84.9, -86.0])
        hold = WaistHold(RELEASE, CAPTURE, math.radians(45.0))
        self.assertEqual(commanded(hold, [88.5, 85.5, 1.5, -3.0]), [90.0, 90.0, 0.0, 0.0])

    def test_the_starting_waist_decides_the_first_hold(self):
        self.assertEqual(commanded(WaistHold(RELEASE, CAPTURE, math.radians(1.5)), [-4.0]), [0.0])
        self.assertEqual(commanded(WaistHold(RELEASE, CAPTURE, math.radians(-3.0)), [-4.0]), [-4.0])

    def test_zero_release_disables_the_hold(self):
        hold = WaistHold(0.0, CAPTURE, 0.0)
        self.assertEqual(commanded(hold, [-1.0, -3.0, -86.0, -90.5]), [-1.0, -3.0, -86.0, -90.5])

    def test_only_the_waist_changes_and_the_input_is_not_modified(self):
        original = action(-86.0)
        held = WaistHold(RELEASE, CAPTURE, -math.pi / 2).apply(original)
        others = [index for index in range(24) if index != WAIST_INDEX]
        np.testing.assert_array_equal(held[others], original[others])
        self.assertEqual(held[WAIST_INDEX], -math.pi / 2)
        self.assertEqual(original[WAIST_INDEX], math.radians(-86.0))

    def test_bad_windows_or_length_are_refused(self):
        for release in (-0.01, math.radians(45), float("nan")):
            with self.assertRaises(ValueError):
                WaistHold(release, 0.0, 0.0)
        for capture in (-0.01, float("nan"), math.radians(6)):
            with self.assertRaises(ValueError):
                WaistHold(RELEASE, capture, 0.0)
        with self.assertRaises(ValueError):
            WaistHold(RELEASE, CAPTURE, 0.0).apply(np.zeros(23))


@unittest.skipUnless(HAS_SDK, "needs the robot SDK environment")
class RunnerTests(unittest.TestCase):
    NORMALIZATION = {"action_min": [-1.0] * 14 + [0.2] * 6 + [4.3] + [-1.7, -0.5, -0.5],
                     "action_max": [1.0] * 14 + [0.6] * 6 + [5.4] + [0.2, 0.5, 0.5],
                     "qpos_min": [-1.0] * 24, "qpos_max": [1.0] * 24}

    def test_flags_default_to_5_and_2_deg_and_refuse_bad_values(self):
        import run_r1_act_policy as runner
        args = runner.parse_args(["policy.pt"])
        self.assertEqual((args.waist_snap_deg, args.waist_capture_deg), (5.0, 2.0))
        self.assertEqual(runner.parse_args(["policy.pt", "--waist-snap-deg", "0"]).waist_snap_deg, 0.0)
        for bad in (["--waist-snap-deg", "-1"], ["--waist-snap-deg", "45"], ["--waist-snap-deg", "nan"],
                    ["--waist-capture-deg", "-1"], ["--waist-capture-deg", "nan"], ["--waist-capture-deg", "6"]):
            with self.assertRaises(SystemExit):
                runner.parse_args(["policy.pt", *bad])

    def run_loop(self, live_waist, chunk_waist_deg, snap_deg, capture_deg=2.0):
        """Commanded and logged waist values (deg) of an execute_loop that plays one chunk."""
        import run_r1_act_policy as runner
        with tempfile.TemporaryDirectory() as directory:
            arm, hand = Mock(), Mock()
            arm.get_recording_snapshot.return_value = {
                "state": {"q": [0.0] * 14, "head_q": [0.0, 0.0], "waist_q": live_waist},
                "requested": {"arm_q": [0.0] * 14}}
            hand.get_state.return_value = ([0.2] * 6, [4.8])
            client = Mock(idle=False, failed=None)
            chunk = np.zeros((400, 24))
            chunk[:, 14:20], chunk[:, 20], chunk[:, WAIST_INDEX] = 0.2, 4.8, np.radians(chunk_waist_deg)
            client.poll.side_effect = [((0, time.monotonic()), 1.0, chunk, None)] + [None] * 1000
            guard = PolicyGuard(self.NORMALIZATION, max_chunk_age_s=0.1)
            run = runner.PolicyRun(arm, hand, lambda: (None, {}), client, Mock(), ActionEnsemble(400), guard,
                                   Path(directory) / "ticks.jsonl")
            args = SimpleNamespace(max_start_delta=0.35, blend_seconds=0.0, max_seconds=5.0,
                                   waist_snap_deg=snap_deg, waist_capture_deg=capture_deg)
            runner.execute_loop(run, arm, hand, guard, lambda q: np.zeros(14), 0.002, [False], args)
            run.log.close()
            with open(Path(directory) / "ticks.jsonl") as log:
                ticks = [json.loads(line) for line in log]
        self.assertIn("no fresh policy output", guard.tripped)
        ticks = [tick for tick in ticks if tick["mode"] == "execute"]
        self.assertTrue(ticks)
        sent = [call.kwargs["waist_yaw_target"] for call in arm.ctrl_dual_arm_and_head.call_args_list]
        return (np.degrees(sent), np.degrees([tick["command"][WAIST_INDEX] for tick in ticks]),
                np.degrees([tick["target"][WAIST_INDEX] for tick in ticks]))

    def test_a_drifted_target_is_commanded_at_the_held_stop_and_logged_as_the_policy_gave_it(self):
        sent, command, target = self.run_loop(math.radians(-89.4), np.full(400, -86.0), 5.0)
        self.assertAlmostEqual(sent[-1], -90.0)
        np.testing.assert_allclose(command, -90.0)
        np.testing.assert_allclose(target, -86.0)

    def test_a_waist_starting_outside_capture_is_not_held(self):
        sent, command, _ = self.run_loop(math.radians(-85.9), np.full(400, -86.0), 5.0)
        self.assertAlmostEqual(sent[-1], -86.0)
        np.testing.assert_allclose(command, -86.0)

    def test_zero_release_commands_the_policy_waist_target(self):
        sent, command, _ = self.run_loop(math.radians(-89.4), np.full(400, -86.0), 0.0)
        self.assertAlmostEqual(sent[-1], -86.0)
        np.testing.assert_allclose(command, -86.0)

    def test_a_turn_creeping_off_a_stop_is_released_past_5_deg(self):
        _, command, target = self.run_loop(0.0, np.maximum(-0.5 * np.arange(400), -30.0), 5.0)
        released = np.abs(target) > 5.0 + 1e-9
        self.assertTrue(released.any() and not released[0], target[:5])
        first = int(np.argmax(released))
        np.testing.assert_allclose(command[:first], 0.0)
        np.testing.assert_allclose(command[first:], target[first:])


if __name__ == "__main__":
    unittest.main()
