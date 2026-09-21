import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop.utils.r1_episode_replay import (
    ReplayError,
    ReplayPose,
    approach_duration,
    build_approach_poses,
    load_replay_plan,
    playback_clock_ns,
    sample_at_time,
)


def pose(arm=0.0, head=0.0, waist=0.0, hand=0.2, left_mode=1, right_mode=1):
    return ReplayPose(
        arm_q=np.full(14, arm, dtype=np.float64),
        arm_tau=np.zeros(14, dtype=np.float64),
        head_q=np.full(2, head, dtype=np.float64),
        waist_q=float(waist),
        left_q=np.full(6, hand, dtype=np.float64),
        right_q=np.full(6, hand, dtype=np.float64),
        left_mode=left_mode,
        right_mode=right_mode,
    )


class EpisodeReplayPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.episode = Path(self.temporary.name) / "episode_0000"
        self.episode.mkdir()

    def write(self, frames, status="complete", outcome="success", robot="R1_A7", ee="linker_o6"):
        manifest = {
            "schema": "xr_teleop_episode_v2",
            "status": status,
            "outcome": outcome,
            "episode_id": 0,
            "frame_count": len(frames),
            "frames": "frames.jsonl",
            "text": {},
            "info": {"robot": robot, "end_effector": ee, "frequency": 40},
        }
        (self.episode / "episode.json").write_text(json.dumps(manifest))
        (self.episode / "frames.jsonl").write_text(
            "".join(json.dumps(frame) + "\n" for frame in frames)
        )

    def published_frame(self, idx, seconds, arm=0.1, waist=0.0, unpublished=False):
        mono = 10_000_000_000 + round(seconds * 1e9)
        published = None if unpublished else {
            "arm_q": [arm] * 14,
            "arm_tau": [0.0] * 14,
            "head_q": [0.01, -0.02],
            "waist_q": waist,
            "monotonic_ns": mono,
            "sequence": idx + 1,
        }
        hands = None if unpublished else {
            side: {"q": [0.2] * 6, "mode": 1, "monotonic_ns": mono, "sequence": idx + 1}
            for side in ("left", "right")
        }
        return {
            "idx": idx,
            "colors": {},
            "states": {},
            "actions": {},
            "sample": {
                "timestamp_ns": 1_000_000_000_000 + round(seconds * 1e9),
                "monotonic_ns": mono,
                "mode": "following",
                "commands": {"arm": {"published": published}, "hands": {"published": hands}},
            },
        }

    def test_loads_published_commands_and_uses_real_timestamps(self):
        self.write([
            self.published_frame(0, 0.0, arm=0.1),
            self.published_frame(1, 0.2, arm=0.2),
            self.published_frame(2, 0.5, arm=0.3),
        ])
        plan = load_replay_plan(self.episode)
        self.assertEqual(len(plan.waypoints), 3)
        self.assertAlmostEqual(plan.duration_s, 0.5)
        self.assertAlmostEqual(plan.max_gap_ms, 300.0)
        self.assertEqual(plan.waypoints[1].monotonic_ns - plan.waypoints[0].monotonic_ns, 200_000_000)
        sampled = sample_at_time(plan.waypoints, plan.waypoints[0].monotonic_ns + 250_000_000)
        np.testing.assert_allclose(sampled.arm_q, np.full(14, 0.2))
        future = sample_at_time(plan.waypoints, plan.waypoints[1].monotonic_ns)
        np.testing.assert_allclose(future.arm_q, np.full(14, 0.2))

    def test_skips_leading_unpublished_and_holds_later_holes(self):
        self.write([
            self.published_frame(0, 0.0, unpublished=True),
            self.published_frame(1, 0.1, arm=0.4, waist=0.2),
            self.published_frame(2, 0.2, unpublished=True),
        ])
        plan = load_replay_plan(self.episode)
        self.assertEqual(plan.skipped_leading, 1)
        self.assertEqual(plan.held_unpublished, 1)
        self.assertEqual(len(plan.waypoints), 2)
        np.testing.assert_allclose(plan.waypoints[1].arm_q, np.full(14, 0.4))
        self.assertEqual(plan.waypoints[1].waist_q, 0.2)

    def test_rejects_incomplete_discarded_and_wrong_robot(self):
        frames = [self.published_frame(0, 0.0), self.published_frame(1, 0.1)]
        self.write(frames, status="incomplete")
        with self.assertRaisesRegex(ReplayError, "not complete"):
            load_replay_plan(self.episode)
        self.write(frames, status="complete", outcome="discarded")
        with self.assertRaisesRegex(ReplayError, "discarded"):
            load_replay_plan(self.episode)
        self.write(frames, robot="G1_29")
        with self.assertRaisesRegex(ReplayError, "R1_A7"):
            load_replay_plan(self.episode)

    def test_approach_does_not_snap_to_the_first_frame(self):
        current = pose(arm=0.0, waist=0.0, hand=0.0)
        target = pose(arm=0.8, waist=0.4, hand=0.8)
        duration, deltas = approach_duration(current, target, max_seconds=8.0)
        self.assertGreater(duration, 1.0)
        self.assertAlmostEqual(deltas["arm"], 0.8)
        poses = build_approach_poses(current, target, duration, hz=40.0)
        self.assertGreater(len(poses), 1)
        self.assertGreater(float(np.max(np.abs(poses[0].arm_q - current.arm_q))), 0.0)
        self.assertLess(float(np.max(np.abs(poses[0].arm_q - target.arm_q))), 0.8)
        np.testing.assert_allclose(poses[-1].arm_q, target.arm_q)
        np.testing.assert_allclose(poses[-1].head_q, target.head_q)
        self.assertAlmostEqual(poses[-1].waist_q, target.waist_q)

    def test_far_start_pose_is_refused_instead_of_snapping(self):
        current = pose(arm=0.0)
        target = pose(arm=20.0)
        with self.assertRaisesRegex(ReplayError, "too far"):
            approach_duration(current, target, max_seconds=2.0)

    def test_playback_speed_cannot_exceed_realtime(self):
        with self.assertRaisesRegex(ReplayError, "speeding up"):
            playback_clock_ns(0, 1.0, speed=1.5)
        self.assertEqual(playback_clock_ns(1000, 0.5, speed=0.5), 1000 + 250_000_000)

    def test_check_only_cli_does_not_mention_sending_commands(self):
        self.write([
            self.published_frame(0, 0.0),
            self.published_frame(1, 0.1),
        ])
        process = subprocess.run(
            [sys.executable, str(ROOT / "tools/replay_r1_episode_on_robot.py"),
             str(self.episode), "--check-only"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        payload = json.loads(process.stdout.split("[READY]")[0])
        self.assertFalse(payload["sends_robot_commands"])
        self.assertEqual(payload["waypoints"], 2)
        self.assertIn("no robot commands sent", process.stdout)
        self.assertNotIn("unitree_sdk2py", process.stdout)
        self.assertNotIn("Enter_Debug_Mode", process.stdout)


def _load_replay_tool():
    import importlib.util
    path = ROOT / "tools/replay_r1_episode_on_robot.py"
    spec = importlib.util.spec_from_file_location("replay_r1_episode_on_robot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReplayTtyRestoreTests(unittest.TestCase):
    def setUp(self):
        self.replay = _load_replay_tool()

    def _pty(self):
        import os
        import pty
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        return master, slave

    def test_restore_tty_reenables_echo_and_canonical(self):
        import termios
        _, slave = self._pty()
        snapshot = (slave, termios.tcgetattr(slave))
        attrs = termios.tcgetattr(slave)
        attrs[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(slave, termios.TCSADRAIN, attrs)
        broken = termios.tcgetattr(slave)
        self.assertFalse(broken[3] & termios.ECHO)
        self.assertFalse(broken[3] & termios.ICANON)
        self.assertTrue(self.replay.restore_tty(snapshot))
        restored = termios.tcgetattr(slave)
        self.assertTrue(restored[3] & termios.ECHO)
        self.assertTrue(restored[3] & termios.ICANON)

    def test_stop_keyboard_on_q_stops_listener_and_restores_tty(self):
        import sys
        import termios
        from unittest import mock

        _, slave = self._pty()
        snapshot = (slave, termios.tcgetattr(slave))
        attrs = termios.tcgetattr(slave)
        attrs[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(slave, termios.TCSADRAIN, attrs)

        class FakeListener:
            def __init__(self):
                self.joined = False

            def join(self, timeout=None):
                self.joined = True

        listener = FakeListener()
        fake_sshkeyboard = mock.Mock()
        with mock.patch.dict(sys.modules, {"sshkeyboard": fake_sshkeyboard}):
            self.replay.stop_keyboard(listener, snapshot)
        fake_sshkeyboard.stop_listening.assert_called_once()
        self.assertTrue(listener.joined)
        restored = termios.tcgetattr(slave)
        self.assertTrue(restored[3] & termios.ECHO)
        self.assertTrue(restored[3] & termios.ICANON)
