import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class R1ACTDataset(Dataset):
    """ACT tensors with explicit source-episode splits and no real-robot -1 shift."""

    def __init__(self, root, split="train", chunk_size=40):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "dataset.json").read_text())
        if self.metadata.get("schema") != "r1_act_hdf5_v1" or self.metadata.get("status") != "complete":
            raise ValueError("Dataset export is not complete or has an unknown schema")
        if split not in ("train", "val") or chunk_size <= 0:
            raise ValueError("Use train/val and a positive chunk_size")
        self.chunk_size = chunk_size
        selected = set(self.metadata["splits"][split])
        self.episodes = [e for e in self.metadata["episodes"] if e["episode_id"] in selected]
        self.ends = np.cumsum([e["frames"] for e in self.episodes])
        stats = self.metadata["normalization"]
        self.stats = {key: np.asarray(stats[key], dtype=np.float32)
                      for key in ("qpos_mean", "qpos_std", "action_mean", "action_std")}

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.ends, index, side="right"))
        episode = self.episodes[episode_index]
        frame_index = index - (int(self.ends[episode_index - 1]) if episode_index else 0)
        with h5py.File(self.root / episode["file"], "r") as h:
            images = np.stack([h["observations/images/" + name][frame_index]
                               for name in self.metadata["camera_names"]])
            qpos = h["observations/qpos"][frame_index]
            actions = h["action"][frame_index:frame_index + self.chunk_size]
        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        padded = np.zeros((self.chunk_size, self.metadata["action_dim"]), np.float32)
        padded[:len(actions)] = (actions - self.stats["action_mean"]) / self.stats["action_std"]
        is_pad = np.arange(self.chunk_size) >= len(actions)
        return (torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255,
                torch.from_numpy(qpos), torch.from_numpy(padded), torch.from_numpy(is_pad))


def load_act_data(root, batch_size=8, chunk_size=40, num_workers=0):
    train = R1ACTDataset(root, "train", chunk_size)
    val = R1ACTDataset(root, "val", chunk_size)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers) if len(val) else None
    return train_loader, val_loader, train.stats, train.metadata
