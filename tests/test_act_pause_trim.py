import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from teleop.utils.act_contract import GROUPS, UNITS
from teleop.utils.act_dataset import (BalancedBatchSampler, R1ACTDataset, pause_speed_limits, pause_trim_rows,
                                      still_rows)

FPS = 40
LIMITS = pause_speed_limits(GROUPS, UNITS)
DEX1 = len(UNITS) - 4  # left arm, right arm, left O6, then the Dex1 value and the body


def timeline(*parts):
    """parts: (seconds, joint-0 speed in rad/s). Returns times_ns and 24-D actions at 40 Hz."""
    times, actions, t, value = [], [], 0.0, 0.0
    for seconds, speed in parts:
        for _ in range(int(round(seconds * FPS))):
            times.append(int(round(t * 1e9)))
            row = np.zeros(len(UNITS))
            row[0] = value
            row[DEX1] = 4.5
            actions.append(row)
            t += 1 / FPS
            value += speed / FPS
    return np.asarray(times, dtype=np.int64), np.asarray(actions)


class StillnessTests(unittest.TestCase):
    def test_limits_follow_the_contract_units(self):
        self.assertEqual(len(LIMITS), 24)
        self.assertEqual(LIMITS[DEX1], 0.25)
        self.assertTrue(np.all(LIMITS[14:20] == 0.05))
        self.assertTrue(np.all(LIMITS[:14] == 0.03) and np.all(LIMITS[21:] == 0.03))

    def test_long_pause_keeps_both_ends_and_drops_the_middle(self):
        times, actions = timeline((3, 0.2), (10, 0.0), (3, 0.2))
        drop, stretches = pause_trim_rows(times, actions, LIMITS, 4.0)
        t = times / 1e9
        self.assertEqual(stretches, 1)
        self.assertFalse(drop[t < 5.4].any())       # stops at 3 s; still from about 3.5 s; keeps 2 s
        self.assertFalse(drop[t > 11.2].any())      # leaves at 13 s; keeps the last 2 s
        self.assertTrue(drop[(t > 5.6) & (t < 11.0)].all())

    def test_short_pause_is_untouched(self):
        times, actions = timeline((3, 0.2), (3, 0.0), (3, 0.2))
        drop, stretches = pause_trim_rows(times, actions, LIMITS, 4.0)
        self.assertFalse(drop.any())
        self.assertEqual(stretches, 0)

    def test_small_finger_and_gripper_jitter_still_counts_as_still(self):
        times, actions = timeline((12, 0.0))
        t = times / 1e9
        actions[:, 14] = 0.4 + 0.005 * np.sin(2 * np.pi * 0.5 * t)    # 0.016/s peak, under 0.05/s
        actions[:, DEX1] = 4.5 + 0.02 * np.sin(2 * np.pi * 0.5 * t)  # 0.06 rad/s peak, under 0.25
        self.assertTrue(still_rows(times, actions, LIMITS)[t > 0.5].all())

    def test_a_closing_hand_is_not_a_pause(self):
        times, actions = timeline((12, 0.0))
        actions[:, 15] = np.linspace(0.2, 0.9, len(times))  # 0.06/s: the O6 is moving
        drop, _ = pause_trim_rows(times, actions, LIMITS, 4.0)
        self.assertFalse(still_rows(times, actions, LIMITS).any())
        self.assertFalse(drop.any())

    def test_backward_clock_is_refused(self):
        times, actions = timeline((2, 0.0))
        times[5] = times[4] - 1
        with self.assertRaises(ValueError):
            still_rows(times, actions, LIMITS)


def write_export(root, episodes):
    """A minimal r1_act_hdf5_v4 export. episodes: list of (source_id, split, times_ns, actions, anchor_rows)."""
    meta = {"schema": "r1_act_hdf5_v4", "status": "complete", "fps": FPS, "camera_names": ["cam"],
            "joint_groups": GROUPS, "units": UNITS, "episodes": [], "splits": {"train": [], "val": [], "test": []},
            "normalization": {key: ([0.0] if key.endswith("mean") else [1.0]) * 24
                              for key in ("qpos_mean", "qpos_std", "action_mean", "action_std")}}
    for episode_id, (source, split, times, actions, rows) in enumerate(episodes):
        rows = np.asarray(rows)
        with h5py.File(root / f"episode_{episode_id}.hdf5", "w") as h:
            h["observations/images/cam"] = np.zeros((len(rows), 4, 4, 3), np.uint8)
            qpos = np.zeros((len(rows), 24), np.float32)
            qpos[:, 0] = rows  # each observation remembers its row
            h["observations/qpos"] = qpos
            h["observations/context"] = np.zeros((len(rows), 8), np.float32)
            h["phase_complete"] = np.zeros(len(rows), np.float32)
            h["source/phase"] = np.full(len(rows), -1, np.int64)
            h["source/anchor_index"] = rows
            h["source/action_anchor_ns"] = times
            h["source/monotonic_ns"] = times
            h["source/action_phase"] = np.full(len(times), -1, np.int64)
            h["action"] = actions.astype(np.float32)
        meta["episodes"].append({"episode_id": episode_id, "file": f"episode_{episode_id}.hdf5",
                                 "source_id": source, "anchors": len(rows)})
        meta["splits"][split].append(episode_id)
    (root / "dataset.json").write_text(json.dumps(meta))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        long_times, long_actions = timeline((3, 0.2), (10, 0.0), (3, 0.2))
        short_times, short_actions = timeline((3, 0.2), (3, 0.0), (3, 0.2))
        all_still_times, all_still_actions = timeline((12, 0.0))
        self.long_rows = np.arange(0, len(long_times), 2)  # every other row has a usable observation
        write_export(self.root, [
            (0, "train", long_times, long_actions, self.long_rows),
            (1, "train", short_times, short_actions, np.arange(len(short_times))),
            (2, "train", all_still_times, all_still_actions, np.arange(len(all_still_times))),
            (3, "val", long_times, long_actions, self.long_rows),
        ])
        self.long_t = long_times / 1e9

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_keeps_every_observation(self):
        data = R1ACTDataset(self.root, "train", chunk_size=40)
        self.assertIsNone(data.pause_trim)
        self.assertEqual(len(data), sum(len(f) for f in data.frames))
        self.assertEqual([len(f) for f in data.frames], [e["anchors"] for e in data.episodes])

    def test_trimmed_samples_come_only_from_kept_rows(self):
        data = R1ACTDataset(self.root, "train", chunk_size=40, max_pause_s=4.0)
        full = R1ACTDataset(self.root, "train", chunk_size=40)
        self.assertLess(len(data), len(full))
        self.assertEqual(len(data.frames[1]), full.episodes[1]["anchors"])  # 3 s pause untouched
        rows = [int(data[i][1][0]) for i in range(len(data.frames[0]))]  # qpos[0] = source row
        kept_t = self.long_t[rows]
        self.assertFalse(((kept_t > 5.6) & (kept_t < 11.0)).any())
        self.assertTrue((kept_t < 5.4).any() and (kept_t > 11.2).any())
        trim = data.pause_trim
        self.assertEqual(trim["observations"], len(full))
        self.assertEqual(trim["dropped"], len(full) - len(data))
        self.assertEqual(trim["stretches"], 2)  # the long pause, and the 12 s standstill
        json.dumps(trim)  # stored in the checkpoint: plain types only
        for value in trim.values():
            self.assertNotIsInstance(value, np.generic)

    def test_sampler_draws_only_existing_samples_and_skips_emptied_episodes(self):
        data = R1ACTDataset(self.root, "train", chunk_size=40, max_pause_s=4.0)
        batches = list(BalancedBatchSampler(data, 4, 0, 50, 42))
        drawn = {i for batch in batches for i in batch}
        self.assertTrue(all(0 <= i < len(data) for i in drawn))
        self.assertEqual(len(data.frames[2]) > 0, len(BalancedBatchSampler(data, 4, 0, 1, 42).groups) == 3)

    def test_untrimmed_sampler_draws_the_same_batches_as_before(self):
        data = R1ACTDataset(self.root, "train", chunk_size=40)
        legacy = SimpleNamespace(episodes=data.episodes)  # no frames attribute: the old behaviour
        self.assertEqual(list(BalancedBatchSampler(data, 3, 0, 20, 7)), list(BalancedBatchSampler(legacy, 3, 0, 20, 7)))

    def test_too_short_a_pause_limit_is_refused(self):
        with self.assertRaises(ValueError):
            R1ACTDataset(self.root, "train", chunk_size=40, max_pause_s=1.5)  # chunk horizon is 1 s
        with self.assertRaises(ValueError):
            R1ACTDataset(self.root, "train", chunk_size=40, max_pause_s=float("nan"))


@unittest.skipUnless(importlib.util.find_spec("torchvision"), "training needs torchvision (.venv-train)")
class TrainingArgumentTests(unittest.TestCase):
    def test_flag_is_optional_and_joins_resume_settings_only_when_used(self):
        import train_r1_act
        off = train_r1_act.parse_args(["data", "--output", "run"])
        on = train_r1_act.parse_args(["data", "--output", "run", "--max-pause-seconds", "4"])
        self.assertIsNone(off.max_pause_seconds)
        self.assertNotIn("max_pause_seconds", train_r1_act.training_settings(off))
        self.assertEqual(train_r1_act.training_settings(on)["max_pause_seconds"], 4.0)
        with self.assertRaises(SystemExit):
            train_r1_act.parse_args(["data", "--output", "run", "--max-pause-seconds", "-1"])


if __name__ == "__main__":
    unittest.main()
