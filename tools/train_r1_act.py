#!/usr/bin/env python3
"""Train the 24D left-O6/right-Dex1 policy offline in .venv-train."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop.utils.act_contract import validate_contract
from teleop.utils.act_dataset import R1ACTDataset, BalancedBatchSampler, add_qpos_noise, qpos_noise_scale
from teleop.utils.act_policy import (ACTPolicy, act_loss, augment_images, masked_l1, masked_mse, read_checkpoint,
                                     save_checkpoint)
from teleop.utils.icecream_task import CONTEXT_DIM


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--steps', type=int, default=50_000)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--chunk-size', type=int, default=40)
    p.add_argument('--conditioned', action='store_true', help='Use reviewed phase labels and learn phase completion')
    p.add_argument('--max-pause-seconds', type=float, default=None,
                   help='Train less on standing still: in a still stretch longer than this, keep only the first and '
                        'last half as samples (train and val alike). At least twice the chunk horizon. Default: off')
    p.add_argument('--qpos-noise', type=float, default=None,
                   help='Add Gaussian noise with this standard deviation (rad) to the arm, waist and head joint inputs '
                        'of each training sample, so the policy learns to steer back when its pose drifts. Targets, '
                        'O6 and Dex1 inputs and validation stay exact. Default: off')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-backbone', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--kl-weight', type=float, default=10.0)
    p.add_argument('--head', choices=('act', 'flow'), default='act',
                   help='act: CVAE + L1 decoder (default). flow: rectified-flow (flow matching) decoder, sampled '
                        'with --flow-steps Euler steps from a fixed noise draw at inference')
    p.add_argument('--decoder-layers', type=int, default=None, help='Decoder layers (default: 1 for act, 4 for flow)')
    p.add_argument('--flow-steps', type=int, default=10, help='Euler steps of a flow policy at inference')
    p.add_argument('--hidden-dim', type=int, default=512)
    p.add_argument('--dim-feedforward', type=int, default=3200)
    p.add_argument('--encoder-layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--patience', type=int, default=10, help='Stop after this many validation checks without improvement')
    p.add_argument('--ema-decay', type=float, default=None,
                   help='Keep an exponential moving average of the weights with this decay per optimizer step (for '
                        'example 0.9995). Validation, best.pt and the snapshots then use the averaged weights, and '
                        'snapshots also keep the raw weights as stepNNNNN_raw.pt. Default: off')
    p.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant',
                   help='constant (default), or cosine: decay from the base rates to --min-lr-ratio of them at --steps')
    p.add_argument('--warmup-steps', type=int, default=0,
                   help='Ramp the learning rates up linearly over this many optimizer steps first. Default: 0')
    p.add_argument('--min-lr-ratio', type=float, default=0.1, help='Cosine schedule: final fraction of the base rates')
    p.add_argument('--image-shift', type=int, default=0,
                   help='Shift each training image by up to this many pixels in x and y (border padding), on top of '
                        'the brightness/contrast jitter. Default: 0')
    p.add_argument('--snapshot-every', type=int, default=0,
                   help='Also save model-only stepNNNNN.pt every this many steps (0 = off). Does not change training')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    args = p.parse_args(argv)
    for key in ('steps', 'batch_size', 'accumulate', 'chunk_size', 'eval_every', 'save_every', 'log_every', 'patience',
                'flow_steps'):
        if getattr(args, key) < 1:
            p.error(f'{key} must be positive')
    if args.decoder_layers is None:
        args.decoder_layers = 4 if args.head == 'flow' else 1
    elif args.decoder_layers < 1:
        p.error('decoder_layers must be positive')
    if args.max_pause_seconds is not None and not (math.isfinite(args.max_pause_seconds) and args.max_pause_seconds > 0):
        p.error('max_pause_seconds must be a positive number of seconds')
    if args.qpos_noise is not None and not (math.isfinite(args.qpos_noise) and args.qpos_noise > 0):
        p.error('qpos_noise must be a positive number of radians')
    if args.ema_decay is not None and not (math.isfinite(args.ema_decay) and 0 < args.ema_decay < 1):
        p.error('ema_decay must be between 0 and 1')
    if not 0 <= args.warmup_steps < args.steps:
        p.error('warmup_steps must be at least 0 and fewer than steps')
    if not (math.isfinite(args.min_lr_ratio) and 0 <= args.min_lr_ratio <= 1):
        p.error('min_lr_ratio must be between 0 and 1')
    for key in ('image_shift', 'snapshot_every'):
        if getattr(args, key) < 0:
            p.error(f'{key} must be at least 0')
    return args


def lr_factor(step, total, schedule='constant', warmup=0, min_ratio=0.1):
    """Learning-rate multiplier for the optimizer step after `step` completed ones, out of `total`. It depends on the
    step alone, so a resumed run continues the same schedule."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == 'constant':
        return 1.0
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def ema_update(ema, model, decay, step):
    """ema = d ema + (1 - d) model after optimizer step `step` (1-based). d ramps up as (1 + step) / (10 + step) to
    `decay`, so the average is not anchored to the starting weights. Buffers are copied."""
    d = min(decay, (1 + step) / (10 + step))
    for averaged, current in zip(ema.parameters(), model.parameters()):
        averaged.mul_(d).add_(current.detach(), alpha=1 - d)
    for averaged, current in zip(ema.buffers(), model.buffers()):
        averaged.copy_(current)
    return d


def training_settings(args):
    """Settings a resumed run must share. Pause trimming joins only when used, so older runs still resume."""
    settings = {k: getattr(args, k) for k in ('batch_size', 'accumulate', 'chunk_size', 'conditioned', 'seed', 'no_augment', 'kl_weight')}
    if getattr(args, 'max_pause_seconds', None) is not None:
        settings['max_pause_seconds'] = args.max_pause_seconds
    if getattr(args, 'qpos_noise', None) is not None:
        settings['qpos_noise'] = args.qpos_noise
    if getattr(args, 'head', 'act') != 'act':
        settings['head'] = args.head
    # Run 12 options join only when used, so runs 1-11 still resume.
    if getattr(args, 'ema_decay', None) is not None:
        settings['ema_decay'] = args.ema_decay
    if getattr(args, 'lr_schedule', 'constant') != 'constant' or getattr(args, 'warmup_steps', 0):
        settings['lr_schedule'] = {'schedule': args.lr_schedule, 'warmup_steps': args.warmup_steps,
                                   'min_lr_ratio': args.min_lr_ratio, 'steps': args.steps}
    if getattr(args, 'image_shift', 0):
        settings['image_shift'] = args.image_shift
    return settings


def dataset_info(root, metadata, train_loader, val_loader):
    info = {key: metadata[key] for key in ('schema', 'fps', 'task', 'joint_names', 'camera_names', 'image_size_hw',
            'source_image_sizes', 'state_dim', 'action_dim', 'joint_groups', 'end_effectors', 'units',
            'gripper_position_limits', 'controller_config', 'image_transform_kind', 'image_color_profile', 'phase_annotations') if key in metadata}
    info.update(path=str(Path(root).resolve()),
                manifest_sha256=hashlib.sha256((Path(root) / 'dataset.json').read_bytes()).hexdigest(),
                train_frames=len(train_loader.dataset), val_frames=0 if val_loader is None else len(val_loader.dataset),
                independent_validation=val_loader is not None)
    by_id = {e['episode_id']: e for e in metadata.get('episodes', [])}
    sources = {e['source_id']: e for e in metadata.get('sources', [])}
    info['source_splits'] = {
        split: [sources[i] for i in sorted({by_id[e]['source_id'] for e in ids})]
        for split, ids in metadata.get('splits', {}).items()
    }
    return info


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def validation_l1(model, loader, device, amp):
    model.eval()
    total, weight, complete_sum = 0.0, 0, 0.0
    for batch in loader:
        images, qpos, actions, pad, context, complete = [x.to(device) for x in batch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            pred, _, _, logits = model(images, qpos, context=context)
        count = int((~pad).sum())
        total += float(masked_l1(pred.float(), actions, pad)) * count
        weight += count
        if logits is not None:
            complete_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), complete)) * len(images)
    model.train()
    return total / max(weight, 1), complete_sum / len(loader.dataset)


def train(args):
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    amp = device.startswith('cuda')
    if args.output.exists() and not args.resume:
        raise ValueError('Output exists; choose a new run or use --resume')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    train_data = R1ACTDataset(args.dataset, 'train', args.chunk_size, args.max_pause_seconds)
    val_data = R1ACTDataset(args.dataset, 'val', args.chunk_size, args.max_pause_seconds)
    meta = train_data.metadata
    validate_contract(meta)
    if not len(train_data) or not len(val_data):
        raise ValueError('Training requires separate nonempty train and validation source episodes')
    if args.conditioned and not (meta.get('phase_annotations') or {}).get('reviewed'):
        raise ValueError('Conditioned training requires reviewed within-episode phase annotations')
    noise = None
    if args.qpos_noise is not None:
        scale = qpos_noise_scale(meta.get('joint_groups'), meta.get('units'), train_data.stats['qpos_std'], args.qpos_noise)
        noise = torch.as_tensor(scale, device=device)
        print(f"[TRAIN] qpos noise: sigma {args.qpos_noise:g} rad on {int((scale > 0).sum())} joint inputs "
              "(arms, waist, head); O6, Dex1, targets and validation unchanged", flush=True)
    settings = training_settings(args)
    resume = read_checkpoint(args.output / 'latest.pt') if args.resume else None
    start = 0 if resume is None else resume['step']
    sampler = BalancedBatchSampler(train_data, args.batch_size, start * args.accumulate,
                                   args.steps * args.accumulate, args.seed, phases=args.conditioned)
    options = dict(num_workers=args.num_workers, pin_memory=amp, persistent_workers=args.num_workers > 0,
                   generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_data, batch_sampler=sampler, **options)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **options)
    info = dataset_info(args.dataset, meta, train_loader, val_loader)
    if args.max_pause_seconds is not None:
        info['pause_trim'] = {'train': train_data.pause_trim, 'val': val_data.pause_trim}
        for split, trim in info['pause_trim'].items():
            print(f"[TRAIN] pause trim {split}: {trim['dropped']} of {trim['observations']} samples dropped from "
                  f"{trim['stretches']} still stretches longer than {trim['max_pause_s']:g} s", flush=True)
    hashes = {split: {source['frames_sha256'] for source in sources}
              for split, sources in info['source_splits'].items()}
    if any(hashes[a] & hashes[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Source contents overlap between train, validation or test')
    if resume is not None:
        if resume['dataset']['manifest_sha256'] != info['manifest_sha256']:
            raise ValueError('Resume dataset differs from the saved dataset')
        if resume['training_state']['settings'] != settings:
            raise ValueError('Resume batch, context, augmentation, seed and horizon must match')
        model = ACTPolicy(**resume['config'], pretrained=False)
        model.load_state_dict(resume['model'])
    else:
        model = ACTPolicy(24, 24, 4, chunk_size=args.chunk_size, hidden_dim=args.hidden_dim,
                          dim_feedforward=args.dim_feedforward, encoder_layers=args.encoder_layers,
                          decoder_layers=args.decoder_layers, dropout=args.dropout, pretrained=not args.no_pretrained,
                          context_dim=CONTEXT_DIM if args.conditioned else 0, head=args.head,
                          flow_steps=args.flow_steps)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameter_groups(args.lr, args.lr_backbone), weight_decay=args.weight_decay)
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    ema_model = None
    if args.ema_decay is not None:
        ema_model = copy.deepcopy(model).eval()
        ema_model.requires_grad_(False)
        if resume is not None:
            ema_model.load_state_dict(resume['training_state']['ema'])
    best = float('inf') if resume is None else resume['metrics']['best']
    bad_checks = 0 if resume is None else resume['training_state']['bad_checks']
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        restore_rng(resume['training_state']['rng'])
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if resume is None:
        sources = ['tools/train_r1_act.py', 'teleop/utils/act_policy.py', 'teleop/utils/act_dataset.py',
                   'teleop/utils/act_contract.py', 'teleop/utils/act_runtime.py', 'teleop/utils/camera_color.py']
        record = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'dataset': info,
                  'torch': torch.__version__, 'device': device,
                  'gpu': torch.cuda.get_device_name() if amp else None,
                  'code_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in sources}}
        (args.output / 'run.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    def checkpoint(name, step, optimizer_state, weights=None):
        state = {'settings': settings, 'rng': rng_state(), 'bad_checks': bad_checks}
        if ema_model is not None and optimizer_state:
            state['ema'] = {key: value.detach().cpu() for key, value in ema_model.state_dict().items()}
        metrics = {'best': best}
        if weights is not None and weights is ema_model:
            metrics['weights'] = f'ema {args.ema_decay:g}'
        save_checkpoint(args.output / name, model if weights is None else weights, meta['normalization'], info, step,
                        optimizer if optimizer_state else None, metrics, state)
    flow = model.head == 'flow'
    names = ('loss', 'flow_mse', 'kl', 'phase_bce') if flow else ('loss', 'l1', 'kl', 'phase_bce')
    print(f'[TRAIN] {device} 24D head={model.head} train={len(train_data)} val={len(val_data)} '
          f'effective_batch={args.batch_size * args.accumulate}', flush=True)
    stream, step = iter(train_loader), start
    began = time.monotonic()
    log = (args.output / 'metrics.jsonl').open('a')
    try:
        while step < args.steps:
            optimizer.zero_grad(set_to_none=True)
            factor = lr_factor(step, args.steps, args.lr_schedule, args.warmup_steps, args.min_lr_ratio)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group['lr'] = base_lr * factor
            losses = []
            for _ in range(args.accumulate):
                images, qpos, actions, pad, context, complete = [x.to(device, non_blocking=True) for x in next(stream)]
                if not args.no_augment:
                    images = augment_images(images, max_shift=args.image_shift)
                if noise is not None:
                    qpos = add_qpos_noise(qpos, noise)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    if flow:
                        velocity, target, logits = model.flow_targets(images, qpos, actions, pad, context)
                        l1 = masked_mse(velocity.float(), target.float(), pad)
                        loss, kl = l1, l1.new_tensor(0.)
                    else:
                        pred, mu, logvar, logits = model(images, qpos, actions, pad, context)
                        loss, l1, kl = act_loss(pred.float(), actions, pad, mu.float(), logvar.float(), args.kl_weight)
                    phase_loss = loss.new_tensor(0.) if logits is None else torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.float(), complete, pos_weight=loss.new_tensor(10.))
                    loss = loss + phase_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss at step {step}')
                (loss / args.accumulate).backward()
                losses.append([float(loss), float(l1), float(kl), float(phase_loss)])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step(); step += 1
            if ema_model is not None:
                ema_update(ema_model, model, args.ema_decay, step)
            evaluate = step % args.eval_every == 0 or step == args.steps
            if step % args.log_every == 0 or evaluate:
                entry = dict(zip(names, np.mean(losses, axis=0).tolist()))
                entry.update(step=step, steps_per_s=(step-start) / max(time.monotonic()-began, 1e-6),
                             lr=optimizer.param_groups[0]['lr'])
                if amp:
                    entry['peak_gpu_mb'] = torch.cuda.max_memory_allocated() / 2**20
                if evaluate:
                    scored = model if ema_model is None else ema_model
                    entry['val_l1'], entry['val_phase_bce'] = validation_l1(scored, val_loader, device, amp)
                    if ema_model is not None:
                        ema_model.eval()
                    score = entry['val_l1'] + (entry['val_phase_bce'] * .1 if args.conditioned else 0.)
                    if score < best:
                        best, bad_checks = score, 0
                        checkpoint('best.pt', step, False, ema_model)
                    else:
                        bad_checks += 1
                log.write(json.dumps(entry) + '\n'); log.flush()
                print('[TRAIN] ' + json.dumps(entry), flush=True)
            if args.snapshot_every and step % args.snapshot_every == 0:
                checkpoint(f'step{step}.pt', step, False, ema_model)
                if ema_model is not None:
                    checkpoint(f'step{step}_raw.pt', step, False)
            if step % args.save_every == 0 or step == args.steps or bad_checks >= args.patience:
                checkpoint('latest.pt', step, True)
            if bad_checks >= args.patience:
                print('[TRAIN] validation patience reached', flush=True)
                break
    except KeyboardInterrupt:
        # Preserve the last completed checkpoint: a partial optimizer step cannot be resumed exactly.
        print('[TRAIN] interrupted; resume from the last completed latest.pt checkpoint', flush=True)
        return 130
    finally:
        log.close()
    print(f'[TRAIN] completed step={step}, best_validation={best:.6f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(train(parse_args()))
