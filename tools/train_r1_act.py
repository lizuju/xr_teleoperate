#!/usr/bin/env python3
"""Train ACT on an exported R1/O6 dataset. Never connects to the robot.

Run with .venv-train (GPU), and not while teleoperating: it loads the whole machine.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop.utils.act_dataset import load_act_data
from teleop.utils.act_policy import (ACTPolicy, act_loss, augment_images, masked_l1, read_checkpoint,
                                     save_checkpoint)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory holding dataset.json (r1_act_hdf5_v3)")
    parser.add_argument("--output", type=Path, required=True, help="New run directory; existing only with --resume")
    parser.add_argument("--resume", action="store_true", help="Continue from OUTPUT/latest.pt")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=40, help="Prediction horizon in ticks; 40 is 1 s at 40 Hz")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dim-feedforward", type=int, default=3200)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not start the backbone from ImageNet weights")
    parser.add_argument("--no-augment", action="store_true", help="Disable image shifts and colour jitter")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    return parser.parse_args(argv)


def forever(loader):
    while True:
        yield from loader


def dataset_info(root, metadata, train_loader, val_loader):
    return {"path": str(Path(root).resolve()), "schema": metadata["schema"],
            "manifest_sha256": hashlib.sha256((Path(root) / "dataset.json").read_bytes()).hexdigest(),
            "fps": metadata["fps"], "task": metadata.get("task"), "joint_names": metadata["joint_names"],
            "camera_names": metadata["camera_names"], "image_size_hw": metadata["image_size_hw"],
            "source_image_sizes": metadata.get("source_image_sizes"),
            "train_frames": len(train_loader.dataset), "val_frames": 0 if val_loader is None else len(val_loader.dataset),
            "independent_validation": val_loader is not None}


def to_device(batch, device):
    return [item.to(device, non_blocking=True) for item in batch]


@torch.no_grad()
def validation_l1(model, loader, device, amp):
    """Normalized L1 at z=0, exactly how the policy runs when deployed."""
    model.eval()
    total, weight = 0.0, 0
    for batch in loader:
        images, qpos, actions, is_pad = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = model(images, qpos)[0]
        count = int((~is_pad).sum())
        total += float(masked_l1(pred.float(), actions, is_pad)) * count
        weight += count
    model.train()
    return total / max(weight, 1)


def train(args):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    amp = device.startswith("cuda")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"{args.output} already exists; pass --resume or choose a new directory")
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    train_loader, val_loader, _, metadata = load_act_data(
        args.dataset, batch_size=args.batch_size, chunk_size=args.chunk_size, num_workers=args.num_workers,
        pin_memory=amp)
    if len(train_loader.dataset) < args.batch_size:
        raise SystemExit(f"Only {len(train_loader.dataset)} training frames; collect more demonstrations")
    info = dataset_info(args.dataset, metadata, train_loader, val_loader)
    if val_loader is None:
        print("[TRAIN] 警告：数据集没有独立验证集（只有一条来源录像）。训练误差不能说明泛化能力。", flush=True)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    latest, best_path = args.output / "latest.pt", args.output / "best.pt"
    start_step, best = 0, float("inf")
    resume = read_checkpoint(latest) if args.resume else None
    if resume is not None:
        if resume["dataset"]["manifest_sha256"] != info["manifest_sha256"]:
            raise SystemExit("--resume: this run was trained on a different dataset export")
        model = ACTPolicy(**resume["config"], pretrained=False)
        model.load_state_dict(resume["model"])
        start_step = resume["step"]
        best = resume["metrics"].get("best", best)
    else:
        model = ACTPolicy(metadata["state_dim"], metadata["action_dim"], len(metadata["camera_names"]),
                          chunk_size=args.chunk_size, hidden_dim=args.hidden_dim,
                          dim_feedforward=args.dim_feedforward, encoder_layers=args.encoder_layers,
                          dropout=args.dropout, pretrained=not args.no_pretrained)
        (args.output / "run.json").write_text(json.dumps(
            {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
             "dataset": info}, indent=2, ensure_ascii=False) + "\n")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameter_groups(args.lr, args.lr_backbone), weight_decay=args.weight_decay)
    if resume is not None and resume["optimizer"] is not None:
        optimizer.load_state_dict(resume["optimizer"])
    normalization = metadata["normalization"]
    metrics_file = (args.output / "metrics.jsonl").open("a", encoding="utf-8")

    def checkpoint(path, step, with_optimizer):
        save_checkpoint(path, model, normalization, info, step, optimizer if with_optimizer else None, {"best": best})

    print(f"[TRAIN] device={device} frames train={info['train_frames']} val={info['val_frames']} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M steps={start_step}->{args.steps}", flush=True)
    batches = forever(train_loader)
    started, window_start, running = time.monotonic(), start_step, []
    step = start_step
    try:
        while step < args.steps:
            images, qpos, actions, is_pad = to_device(next(batches), device)
            if not args.no_augment:
                images = augment_images(images)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred, mu, logvar = model(images, qpos, actions, is_pad)
                loss, l1, kl = act_loss(pred.float(), actions, is_pad, mu.float(), logvar.float(), args.kl_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}; stopping without saving this state")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            step += 1
            running.append((float(loss), float(l1), float(kl)))
            evaluate = val_loader is not None and (step % args.eval_every == 0 or step == args.steps)
            if step % args.log_every == 0 or evaluate or step == args.steps:
                loss_mean, l1_mean, kl_mean = np.mean(running, axis=0)
                running = []
                entry = {"step": step, "loss": float(loss_mean), "l1": float(l1_mean), "kl": float(kl_mean),
                         "steps_per_s": (step - window_start) / max(time.monotonic() - started, 1e-6)}
                if evaluate:
                    entry["val_l1"] = validation_l1(model, val_loader, device, amp)
                # Without a validation set the training L1 is the only signal; the eval tool says so.
                score = entry.get("val_l1", entry["l1"] if val_loader is None else None)
                if score is not None and score < best:
                    best = score
                    checkpoint(best_path, step, False)
                    entry["best"] = True
                metrics_file.write(json.dumps(entry) + "\n")
                metrics_file.flush()
                text = " ".join(f"{key}={value:.4f}" for key, value in entry.items() if isinstance(value, float))
                print(f"[TRAIN] step={step} {text}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                checkpoint(latest, step, True)
    except KeyboardInterrupt:
        checkpoint(latest, step, True)
        print(f"[TRAIN] interrupted; saved {latest} at step {step}", flush=True)
        return 130
    metrics_file.close()
    print(f"[TRAIN] done: {latest} and {best_path}; best={'n/a' if best == float('inf') else f'{best:.4f}'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(train(parse_args()))
