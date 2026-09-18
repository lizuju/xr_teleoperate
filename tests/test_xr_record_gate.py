import ast
from pathlib import Path
import unittest

from teleop.utils.xr_record_gate import (
    DEFAULT_RECORD_MAX_TRACKING_AGE_MS,
    recording_blocked_by_tracking,
)


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "teleop" / "teleop_hand_and_arm.py"


class RecordingTrackingGateTest(unittest.TestCase):
    def test_fresh_hands_are_allowed(self):
        self.assertIsNone(recording_blocked_by_tracking(
            {"left_age_ms": 24.0, "right_age_ms": 31.0},
        ))

    def test_one_fresh_hand_is_enough(self):
        self.assertIsNone(recording_blocked_by_tracking(
            {"left_age_ms": 24.0, "right_age_ms": None},
        ))

    def test_stale_uplink_blocks_start(self):
        reason = recording_blocked_by_tracking(
            {"left_age_ms": 106.0, "right_age_ms": 98.0},
        )
        self.assertIn("left_age_ms=106.0", reason)
        self.assertIn("exceeds 100", reason)

    def test_one_stale_hand_blocks_even_if_the_other_is_fresh(self):
        reason = recording_blocked_by_tracking(
            {"left_age_ms": 24.0, "right_age_ms": 180.0},
        )
        self.assertIn("right_age_ms=180.0", reason)

    def test_no_hand_at_all_blocks(self):
        reason = recording_blocked_by_tracking({"left_age_ms": None, "right_age_ms": None})
        self.assertIn("no XR hand sample", reason)

    def test_zero_or_negative_limit_disables_the_gate(self):
        stale = {"left_age_ms": 500.0, "right_age_ms": 500.0}
        self.assertIsNone(recording_blocked_by_tracking(stale, 0))
        self.assertIsNone(recording_blocked_by_tracking(stale, -1))
        self.assertIsNone(recording_blocked_by_tracking(stale, None))

    def test_missing_diagnostics_do_not_block(self):
        from teleop.utils.xr_record_gate import tracking_diagnostics
        self.assertIsNone(recording_blocked_by_tracking(None))
        self.assertIsNone(tracking_diagnostics(object()))
        self.assertEqual(DEFAULT_RECORD_MAX_TRACKING_AGE_MS, 100.0)


class RecordingTrackingGateWiringTest(unittest.TestCase):
    def test_main_program_refuses_to_start_an_episode_on_a_stale_uplink(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn("from teleop.utils.xr_record_gate import recording_blocked_by_tracking, tracking_diagnostics", source)
        self.assertIn("'--record-max-tracking-age-ms'", source)
        self.assertIn("recording_blocked_by_tracking(", source)
        self.assertIn("[RECORD] not started:", source)

    def test_wrappers_pass_the_gate_and_keep_wrist_recording_on_zmq(self):
        for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
            script = (ROOT / "teleop" / name).read_text(encoding="utf-8")
            self.assertIn(
                '--record-max-tracking-age-ms "${RECORD_MAX_TRACKING_AGE_MS:-100}"',
                script,
                name,
            )

    def test_paused_loop_polls_resume_once_per_frame(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        loop = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.While)
            and any(
                isinstance(child, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "time_ik_start" for t in child.targets)
                for child in ast.walk(node)
            )
        )
        calls = [
            node for node in ast.walk(loop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "poll_resume"
        ]
        self.assertEqual(len(calls), 1, "poll_resume must run once per paused frame")
