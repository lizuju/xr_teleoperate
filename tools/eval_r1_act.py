#!/usr/bin/env python3
"""Offline evaluation of an ACT checkpoint, in real units and against trivial baselines.

Low error here is necessary, not sufficient: it says nothing about closed-loop success.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop.utils.act_dataset import R1ACTDataset
from teleop.utils.act_policy import load_policy
from teleop.utils.act_runtime import GROUP_SLICES

UNITS = {"left_arm": "rad", "right_arm": "rad", "left_ee": "0-1", "right_ee": "0-1", "body": "rad"}


class ErrorTable:
    """Mean absolute error per joint and per horizon step, ignoring padded steps."""

    def __init__(self, chunk_size, action_dim):
        self.sum = np.zeros((chunk_size, action_dim))
        self.count = np.zeros(chunk_size)

    def add(self, pred, target, is_pad):
        valid = ~is_pad
        self.sum += (np.abs(pred - target) * valid[..., None]).sum(axis=0)
        self.count += valid.sum(axis=0)

    def joints(self):
        return self.sum.sum(axis=0) / max(self.count.sum(), 1)

    def at(self, index):
        return self.sum[index] / max(self.count[index], 1)


def evaluate(checkpoint, dataset_root=None, split="val", batch_size=16, num_workers=2, device="auto"):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model, data = load_policy(checkpoint, device)
    info = data["dataset"]
    root = Path(info["path"] if dataset_root is None else dataset_root)
    dataset = R1ACTDataset(root, split, chunk_size=data["config"]["chunk_size"])
    if dataset.metadata["joint_names"] != info["joint_names"] or dataset.metadata["camera_names"] != info["camera_names"]:
        raise ValueError("Dataset joints or cameras do not match this checkpoint")
    if len(dataset) == 0:
        raise ValueError(f"Dataset has no {split} frames")
    stats = data["normalization"]
    action_mean, action_std = (np.asarray(stats[key], dtype=np.float64) for key in ("action_mean", "action_std"))
    qpos_mean, qpos_std = (np.asarray(stats[key], dtype=np.float64) for key in ("qpos_mean", "qpos_std"))
    chunk_size, action_dim = data["config"]["chunk_size"], data["config"]["action_dim"]
    tables = {name: ErrorTable(chunk_size, action_dim) for name in ("policy", "hold_current", "train_mean")}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.inference_mode():
        for images, qpos, actions, is_pad in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                pred = model(images.to(device), qpos.to(device))[0]
            pad = is_pad.numpy()
            target = actions.numpy() * action_std + action_mean
            current = qpos.numpy() * qpos_std + qpos_mean
            tables["policy"].add(pred.float().cpu().numpy() * action_std + action_mean, target, pad)
            tables["hold_current"].add(np.repeat(current[:, None], chunk_size, axis=1), target, pad)
            tables["train_mean"].add(np.broadcast_to(action_mean, target.shape), target, pad)
    fps = info["fps"]
    horizons = {f"{index / fps:.3f}s": index for index in sorted({0, min(int(fps * .25), chunk_size - 1),
                                                                   min(int(fps * .5), chunk_size - 1), chunk_size - 1})}
    groups = {}
    for name, part in GROUP_SLICES.items():
        groups[name] = {"unit": UNITS[name],
                        **{kind: float(table.joints()[part].mean()) for kind, table in tables.items()},
                        "horizon": {label: float(tables["policy"].at(index)[part].mean()) for label, index in horizons.items()}}
        groups[name]["beats_hold_current"] = groups[name]["policy"] < groups[name]["hold_current"]
    joint_names = info["joint_names"]
    report = {"checkpoint": str(checkpoint), "step": data["step"], "dataset": str(root), "split": split,
              "frames": len(dataset), "independent": split == "val" and info["independent_validation"],
              "groups": groups, "joints": dict(zip(joint_names, tables["policy"].joints().tolist()))}
    return report


def print_report(report):
    print(f"[EVAL] step={report['step']} split={report['split']} frames={report['frames']} independent={report['independent']}")
    if not report["independent"]:
        print("[EVAL] 警告：这不是独立验证，只能说明模型能拟合训练数据，不能说明泛化能力。")
    print(f"{'group':<10} {'unit':<5} {'policy':>8} {'hold':>8} {'mean':>8}  horizon")
    for name, group in report["groups"].items():
        horizon = " ".join(f"{label}={value:.3f}" for label, value in group["horizon"].items())
        print(f"{name:<10} {group['unit']:<5} {group['policy']:8.4f} {group['hold_current']:8.4f} "
              f"{group['train_mean']:8.4f}  {horizon}")
    print("[EVAL] hold=一直保持当前姿态；mean=一直输出训练均值。策略应明显好于这两个基线。离线误差不等于闭环成功率。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, help="Default: the dataset recorded in the checkpoint")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="Default: eval_<split>.json beside the checkpoint")
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.dataset, args.split, args.batch_size, args.num_workers, args.device)
    print_report(report)
    output = args.output or args.checkpoint.parent / f"eval_{args.split}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
