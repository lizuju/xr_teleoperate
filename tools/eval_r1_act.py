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
from teleop.utils.act_contract import GROUP_SLICES, validate_contract

UNITS = {"left_arm": "rad", "right_arm": "rad", "left_ee": "0-1", "right_ee": "rad", "body": "rad"}


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


def evaluation_independence(info, metadata, split):
    if split not in ("val", "test") or not info.get("source_splits"):
        return False
    used = info["source_splits"].get("train", [])
    if split == "test":
        used = used + info["source_splits"].get("val", [])
    used_hashes = {s["frames_sha256"] for s in used}
    used_paths = {s["episode"] for s in used}
    selected = set(metadata["splits"][split])
    source_ids = {e["source_id"] for e in metadata["episodes"] if e["episode_id"] in selected}
    sources = [s for s in metadata["sources"] if s["source_id"] in source_ids]
    return bool(sources) and all(s["frames_sha256"] not in used_hashes and s["episode"] not in used_paths
                                 for s in sources)


def evaluate(checkpoint, dataset_root=None, split="val", batch_size=16, num_workers=2, device="auto"):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model, data = load_policy(checkpoint, device)
    info = data["dataset"]
    validate_contract(info)
    root = Path(info["path"] if dataset_root is None else dataset_root)
    dataset = R1ACTDataset(root, split, chunk_size=data["config"]["chunk_size"])
    validate_contract(dataset.metadata)
    if dataset.metadata["joint_names"] != info["joint_names"] or dataset.metadata["camera_names"] != info["camera_names"]:
        raise ValueError("Dataset joints or cameras do not match this checkpoint")
    if len(dataset) == 0:
        raise ValueError(f"Dataset has no {split} frames")
    for key in ("image_size_hw", "source_image_sizes", "gripper_position_limits", "controller_config", "image_color_profile"):
        if dataset.metadata[key] != info[key]:
            raise ValueError(f"Evaluation {key} must match the checkpoint")
    if data["config"]["context_dim"] and not (dataset.metadata.get("phase_annotations") or {}).get("reviewed"):
        raise ValueError("Conditioned evaluation requires reviewed phase annotations")
    stats = data["normalization"]
    dataset.stats = {k: np.asarray(stats[k], dtype=np.float32) for k in ("qpos_mean", "qpos_std", "action_mean", "action_std")}
    action_mean, action_std = (np.asarray(stats[key], dtype=np.float64) for key in ("action_mean", "action_std"))
    qpos_mean, qpos_std = (np.asarray(stats[key], dtype=np.float64) for key in ("qpos_mean", "qpos_std"))
    chunk_size, action_dim = data["config"]["chunk_size"], data["config"]["action_dim"]
    tables = {name: ErrorTable(chunk_size, action_dim) for name in ("policy", "hold_current", "train_mean")}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    phase_tables = {}
    regime_tables = {group: {regime: {kind: ErrorTable(chunk_size, action_dim) for kind in tables}
                             for regime in ("moving", "holding")} for group in GROUP_SLICES}
    completion_counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    with torch.inference_mode():
        for images, qpos, actions, is_pad, context, complete in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                output = model(images.to(device), qpos.to(device), context=context.to(device))
                pred = output[0]
            pad = is_pad.numpy()
            target = actions.numpy() * action_std + action_mean
            current = qpos.numpy() * qpos_std + qpos_mean
            predictions = {"policy": pred.float().cpu().numpy() * action_std + action_mean,
                           "hold_current": np.repeat(current[:, None], chunk_size, axis=1),
                           "train_mean": np.broadcast_to(action_mean, target.shape)}
            for kind, values in predictions.items():
                tables[kind].add(values, target, pad)
            for group, part in GROUP_SLICES.items():
                threshold = .03 if group == "left_ee" else .02
                moving = np.abs(target[..., part] - target[:, :1, part]).max(axis=-1) > threshold
                for regime, selected in (("moving", moving), ("holding", ~moving)):
                    for kind, values in predictions.items():
                        regime_tables[group][regime][kind].add(values, target, pad | ~selected)
            if output[3] is not None:
                prediction = output[3].float().sigmoid().cpu().numpy() >= .9
                truth = complete.numpy() > .5
                completion_counts["true_positive"] += int((prediction & truth).sum())
                completion_counts["false_positive"] += int((prediction & ~truth).sum())
                completion_counts["false_negative"] += int((~prediction & truth).sum())
                phases = context[:, :5].argmax(dim=1).numpy()
                for phase in np.unique(phases):
                    table = phase_tables.setdefault(int(phase), ErrorTable(chunk_size, action_dim))
                    selected = phases == phase
                    table.add(pred.float().cpu().numpy()[selected] * action_std + action_mean, target[selected], pad[selected])
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
              "frames": len(dataset), "independent": evaluation_independence(info, dataset.metadata, split),
              "groups": groups, "joints": dict(zip(joint_names, tables["policy"].joints().tolist())),
              "phase_joint_mae": {str(k): dict(zip(joint_names, v.joints().tolist())) for k, v in phase_tables.items()},
              "motion_definition": "Per group: accepted target differs from first chunk target by >0.02 rad (left O6 >0.03 normalized).",
              "motion_regimes": {group: {regime: {
                  "valid_steps": int(kind_tables["policy"].count.sum()),
                  **{kind: float(table.joints()[GROUP_SLICES[group]].mean()) if table.count.sum() else None
                     for kind, table in kind_tables.items()}}
                  for regime, kind_tables in regimes.items()} for group, regimes in regime_tables.items()},
              "completion_threshold": .9, "completion_counts": completion_counts}
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
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
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
