import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# Pause trimming (--max-pause-seconds in tools/train_r1_act.py). A row is "still" when, over the
# trailing PAUSE_WINDOW_S, every commanded value moved slower than its limit: arms, waist and head
# in rad/s, O6 fingers in normalized units per second, the Dex1 gripper in rad/s.
PAUSE_WINDOW_S = 0.5
PAUSE_JOINT_SPEED = 0.03
PAUSE_HAND_SPEED = 0.05
PAUSE_GRIPPER_SPEED = 0.25


def pause_speed_limits(joint_groups, units):
    """Per-dimension speed below which a commanded value counts as still."""
    if not joint_groups or not units:
        raise ValueError("Pause trimming needs joint_groups and units in dataset.json")
    limits = []
    for group, size in joint_groups.items():
        for _ in range(size):
            unit = units[len(limits)]
            if unit == "normalized_0_to_1":
                limits.append(PAUSE_HAND_SPEED)
            elif group.endswith("_ee"):
                limits.append(PAUSE_GRIPPER_SPEED)
            else:
                limits.append(PAUSE_JOINT_SPEED)
    if len(limits) != len(units):
        raise ValueError("joint_groups and units disagree on the action size")
    return np.asarray(limits, dtype=np.float64)


def still_rows(times_ns, actions, limits, window_s=PAUSE_WINDOW_S):
    """True where every commanded value moved slower than its limit over the trailing window."""
    times = np.asarray(times_ns, dtype=np.int64)
    if np.any(np.diff(times) < 0):
        raise ValueError("Action request time goes backwards")
    actions = np.asarray(actions, dtype=np.float64)
    lo = np.searchsorted(times, times - int(round(window_s * 1e9)), side="left")
    dt = (times - times[lo]) / 1e9
    rate = np.abs(actions - actions[lo]) / np.maximum(dt, 1e-9)[:, None]
    return (dt >= window_s / 2) & np.all(rate < limits, axis=1)


def pause_trim_rows(times_ns, actions, limits, max_pause_s, window_s=PAUSE_WINDOW_S):
    """Rows deep inside still stretches longer than max_pause_s (True = drop as a training sample).

    Each such stretch keeps its first and last max_pause_s / 2 seconds, so the policy still learns
    to stop there and to leave; only the long middle, where every action chunk says "stay", goes.
    Rows stay in the action arrays either way; this only decides which observations are trained from.
    Returns (drop, stretches) where stretches counts the trimmed stretches.
    """
    times = np.asarray(times_ns, dtype=np.int64)
    still = still_rows(times, actions, limits, window_s)
    drop = np.zeros(len(times), dtype=bool)
    half = max_pause_s / 2 * 1e9
    stretches = 0
    i, n = 0, len(times)
    while i < n:
        if not still[i]:
            i += 1
            continue
        k = i
        while k + 1 < n and still[k + 1]:
            k += 1
        if times[k] - times[i] > max_pause_s * 1e9:
            span = times[i:k + 1]
            drop[i:k + 1] = (span - times[i] > half) & (times[k] - span > half)
            stretches += 1
        i = k + 1
    return drop, stretches


class R1ACTDataset(Dataset):
    """ACT tensors with explicit source-episode splits and no real-robot -1 shift.

    One sample per usable observation. Its action chunk runs on through every later
    row of the segment, including rows whose own observation was not usable.
    With max_pause_s, observations deep inside long still stretches are not used as
    samples (see pause_trim_rows); their actions still fill earlier chunks.
    """

    def __init__(self, root, split="train", chunk_size=40, max_pause_s=None):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "dataset.json").read_text())
        if self.metadata.get("schema") != "r1_act_hdf5_v4" or self.metadata.get("status") != "complete":
            raise ValueError("Dataset export is not complete or has an unknown schema")
        if split not in ("train", "val", "test") or chunk_size <= 0:
            raise ValueError("Use train/val/test and a positive chunk_size")
        self.chunk_size = chunk_size
        selected = set(self.metadata["splits"][split])
        self.episodes = [e for e in self.metadata["episodes"] if e["episode_id"] in selected]
        self.frames = self._sample_frames(max_pause_s)
        self.ends = np.cumsum([len(f) for f in self.frames])
        stats = self.metadata["normalization"]
        self.stats = {key: np.asarray(stats[key], dtype=np.float32)
                      for key in ("qpos_mean", "qpos_std", "action_mean", "action_std")}
        self._files = {}

    def _sample_frames(self, max_pause_s):
        """Per episode, the observation rows (positions in observations/*) used as samples."""
        self.pause_trim = None
        if max_pause_s is None:
            return [np.arange(e["anchors"]) for e in self.episodes]
        horizon = self.chunk_size / self.metadata["fps"]
        if not math.isfinite(max_pause_s) or max_pause_s < 2 * horizon:
            raise ValueError(f"max_pause_s must be at least twice the chunk horizon ({2 * horizon:.2f} s), "
                             "so the kept end of a pause still holds every chunk that leaves it")
        limits = pause_speed_limits(self.metadata.get("joint_groups"), self.metadata.get("units"))
        frames, dropped, stretches = [], 0, 0
        for episode in self.episodes:
            with h5py.File(self.root / episode["file"], "r") as h:
                times = h["source/action_anchor_ns"][:] if "source/action_anchor_ns" in h else h["source/monotonic_ns"][:]
                drop, count = pause_trim_rows(times, h["action"][:], limits, max_pause_s)
                anchors = h["source/anchor_index"][:]
            keep = np.flatnonzero(~drop[anchors])
            frames.append(keep)
            dropped += len(anchors) - len(keep)
            stretches += count
        total = sum(e["anchors"] for e in self.episodes)
        self.pause_trim = {"max_pause_s": float(max_pause_s), "window_s": PAUSE_WINDOW_S,
                           "speed_limits": {"joint_rad_s": PAUSE_JOINT_SPEED, "o6_per_s": PAUSE_HAND_SPEED,
                                            "gripper_rad_s": PAUSE_GRIPPER_SPEED},
                           "observations": int(total), "dropped": int(dropped), "stretches": int(stretches)}
        return frames

    def __getstate__(self):
        # h5py handles do not survive a DataLoader worker; each process opens its own.
        return {**self.__dict__, "_files": {}}

    def _file(self, episode):
        handle = self._files.get(episode["episode_id"])
        if handle is None:
            handle = self._files[episode["episode_id"]] = h5py.File(self.root / episode["file"], "r")
        return handle

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.ends, index, side="right"))
        episode = self.episodes[episode_index]
        local = index - (int(self.ends[episode_index - 1]) if episode_index else 0)
        frame_index = int(self.frames[episode_index][local])
        h = self._file(episode)
        images = np.stack([h["observations/images/" + name][frame_index]
                           for name in self.metadata["camera_names"]])
        qpos = h["observations/qpos"][frame_index]
        row = int(h["source/anchor_index"][frame_index])
        times = h["source/action_anchor_ns"][:] if "source/action_anchor_ns" in h else h["source/monotonic_ns"][:]
        indices, is_pad = action_chunk_indices(times, row, self.chunk_size, self.metadata["fps"], int(h["source/monotonic_ns"][-1]))
        actions = h["action"][:][indices]
        phase_ids = h["source/action_phase"][:]
        is_pad |= phase_ids[indices] != phase_ids[row]
        context = h["observations/context"][frame_index]
        complete = h["phase_complete"][frame_index]
        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        padded = (actions - self.stats["action_mean"]) / self.stats["action_std"]
        padded[is_pad] = 0
        return (torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255,
                torch.from_numpy(qpos), torch.from_numpy(padded), torch.from_numpy(is_pad),
                torch.from_numpy(context), torch.tensor(complete, dtype=torch.float32))


def load_act_data(root, batch_size=8, chunk_size=40, num_workers=0, pin_memory=False, max_pause_s=None):
    train = R1ACTDataset(root, "train", chunk_size, max_pause_s)
    val = R1ACTDataset(root, "val", chunk_size, max_pause_s)
    options = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory,
               "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train, shuffle=True, drop_last=len(train) >= batch_size, **options)
    val_loader = DataLoader(val, shuffle=False, **options) if len(val) else None
    return train_loader, val_loader, train.stats, train.metadata


def action_chunk_indices(times, row, chunk_size, fps, recorded_until_ns):
    """Causal targets within one continuity-checked export segment, including measured holds."""
    times = np.asarray(times, dtype=np.int64)
    if np.any(np.diff(times) < 0):
        raise ValueError("Action request time goes backwards")
    query = times[row] + np.rint(np.arange(chunk_size) * 1e9 / fps).astype(np.int64)
    indices = np.maximum(row, np.searchsorted(times, query, side="right") - 1)
    indices[0] = row
    # The exporter already splits missing control ticks. Slower hand updates alone are not gaps.
    return indices, query > recorded_until_ns


class BalancedBatchSampler(Sampler):
    """Each batch has its own seed, so restart does not depend on worker prefetch."""

    def __init__(self, dataset, batch_size, start_batch, stop_batch, seed, phases=False):
        self.batch_size, self.start, self.stop, self.seed = batch_size, start_batch, stop_batch, seed
        groups = {}
        offset = 0
        frames = getattr(dataset, "frames", None)
        for index, episode in enumerate(dataset.episodes):
            kept = np.arange(episode["anchors"]) if frames is None else np.asarray(frames[index])
            ids = np.arange(offset, offset + len(kept))
            source = episode["source_id"]
            if phases:
                with h5py.File(dataset.root / episode["file"], "r") as h:
                    labels = h["source/phase"][:][kept]
                for phase in np.unique(labels):
                    groups.setdefault((source, int(phase)), []).extend(ids[labels == phase].tolist())
            else:
                groups.setdefault((source, -1), []).extend(ids.tolist())
            offset += len(kept)
        # An episode whose samples were all trimmed has nothing to draw from.
        self.groups = [np.asarray(values) for values in groups.values() if values]

    def __iter__(self):
        for batch in range(self.start, self.stop):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, batch]))
            yield [int(rng.choice(self.groups[int(rng.integers(len(self.groups)))])) for _ in range(self.batch_size)]

    def __len__(self):
        return self.stop - self.start
