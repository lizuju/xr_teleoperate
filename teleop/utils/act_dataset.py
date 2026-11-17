import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class R1ACTDataset(Dataset):
    """ACT tensors with explicit source-episode splits and no real-robot -1 shift.

    One sample per usable observation. Its action chunk runs on through every later
    row of the segment, including rows whose own observation was not usable.
    """

    def __init__(self, root, split="train", chunk_size=40):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "dataset.json").read_text())
        if self.metadata.get("schema") != "r1_act_hdf5_v4" or self.metadata.get("status") != "complete":
            raise ValueError("Dataset export is not complete or has an unknown schema")
        if split not in ("train", "val", "test") or chunk_size <= 0:
            raise ValueError("Use train/val/test and a positive chunk_size")
        self.chunk_size = chunk_size
        selected = set(self.metadata["splits"][split])
        self.episodes = [e for e in self.metadata["episodes"] if e["episode_id"] in selected]
        self.ends = np.cumsum([e["anchors"] for e in self.episodes])
        stats = self.metadata["normalization"]
        self.stats = {key: np.asarray(stats[key], dtype=np.float32)
                      for key in ("qpos_mean", "qpos_std", "action_mean", "action_std")}
        self._files = {}

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
        frame_index = index - (int(self.ends[episode_index - 1]) if episode_index else 0)
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


def load_act_data(root, batch_size=8, chunk_size=40, num_workers=0, pin_memory=False):
    train = R1ACTDataset(root, "train", chunk_size)
    val = R1ACTDataset(root, "val", chunk_size)
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
        for episode in dataset.episodes:
            ids = np.arange(offset, offset + episode["anchors"])
            source = episode["source_id"]
            if phases:
                with h5py.File(dataset.root / episode["file"], "r") as h:
                    labels = h["source/phase"][:]
                for phase in np.unique(labels):
                    groups.setdefault((source, int(phase)), []).extend(ids[labels == phase].tolist())
            else:
                groups.setdefault((source, -1), []).extend(ids.tolist())
            offset += episode["anchors"]
        self.groups = [np.asarray(values) for values in groups.values()]

    def __iter__(self):
        for batch in range(self.start, self.stop):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, batch]))
            yield [int(rng.choice(self.groups[int(rng.integers(len(self.groups)))])) for _ in range(self.batch_size)]

    def __len__(self):
        return self.stop - self.start
