#!/usr/bin/env python3
"""Run a trained ACT policy against the live R1_A7 + O6.

Default is shadow mode: it reads feedback and cameras and logs what the policy would
command, and never activates or publishes anything. Only --execute can move the robot,
and only after r is pressed. q / Ctrl+C stop the program; they are not an e-stop.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from replay_r1_episode_on_robot import (NETWORK_INTERFACE, current_pose, hold_until_quit, send_pose, shutdown,
                                        start_keyboard, stop_keyboard, teleop_pids)
from teleop.utils.act_runtime import (ACTION_DIM, ActionEnsemble, GravityModel, ObservationBuilder, PolicyClient,
                                      PolicyGuard, group_errors, join_action, split_action, summarize)
from teleop.utils.frame_sync import CameraSynchronizer
from teleop.utils.r1_episode_replay import ReplayPose

DEFAULT_WORKER_PYTHON = ROOT.parent / ".venv-train/bin/python"
DEFAULT_LOG_ROOT = ROOT.parent / "policy-runs"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--execute", action="store_true", help="Actually command the robot after r. Default: shadow")
    parser.add_argument("--shadow-seconds", type=float, help="Stop shadow mode after this long. Default: until q")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Hold after this long of execution")
    parser.add_argument("--blend-seconds", type=float, default=2.0, help="Blend from the live pose to the policy over this long")
    parser.add_argument("--max-start-delta", type=float, default=0.35, help="Refuse to start if the first arm target is further than this (rad)")
    parser.add_argument("--allow-out-of-range-start", action="store_true", help="Start even if the live pose is outside the training range")
    parser.add_argument("--max-tracking-error", type=float, default=0.5, help="Hold if arm error stays above this (rad)")
    parser.add_argument("--max-chunk-age", type=float, default=0.5, help="Hold if the policy has been silent this long, and drop results older than this (s)")
    parser.add_argument("--ensemble-k", type=float, default=0.01)
    # Teleop's shaper settings, which the training actions already went through. 0 disables the shaper.
    parser.add_argument("--arm-target-velocity-limit", type=float, default=6.0)
    parser.add_argument("--arm-target-accel-limit", type=float, default=40.0)
    parser.add_argument("--arm-velocity-limit", type=float, default=30.0)
    parser.add_argument("--arm-dq-limit", type=float, default=6.0)
    parser.add_argument("--img-server-ip", default="192.168.110.20")
    parser.add_argument("--network-interface", default=NETWORK_INTERFACE)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--device", default="auto", help="Inference device: auto, cuda or cpu")
    parser.add_argument("--log-dir", type=Path, help=f"Default: a new directory under {DEFAULT_LOG_ROOT}")
    return parser.parse_args(argv)


class PolicyRun:
    """One control tick's worth of reading, inference plumbing and logging. Commands nothing."""

    def __init__(self, arm, hand, cameras, client, builder, ensemble, guard, log_path):
        self.arm, self.hand, self.cameras, self.client = arm, hand, cameras, client
        self.builder, self.ensemble, self.guard = builder, ensemble, guard
        self.tick = 0
        self.ignore_before_tick = 0
        self.last_chunk_at = None
        self.observation_failures = {}
        self.stale_chunks = 0
        self.sent = 0
        self.inference_ms, self.round_trip_ms = [], []
        self.target_error = {name: [] for name in ("left_arm", "right_arm", "left_ee", "right_ee", "body")}
        self.log = log_path.open("a", encoding="utf-8")

    def live_action(self, pose):
        return join_action(pose.arm_q, pose.left_q, pose.right_q, pose.waist_q, pose.head_q)

    def step(self):
        """Advance one tick. Returns (policy target or None, live pose)."""
        head, wrists = self.cameras()
        self.builder.observe(head, wrists)
        response = self.client.poll()
        now = time.monotonic()
        if response is not None:
            (tick, sent_at), inference_ms, chunk = response
            if tick >= self.ignore_before_tick:
                # Age counts from the request: a late result still comes from the old observation.
                if now - sent_at <= self.guard.max_chunk_age_s:
                    self.ensemble.add(tick, chunk)
                    self.last_chunk_at = now
                else:
                    self.stale_chunks += 1
            self.inference_ms.append(inference_ms)
            self.round_trip_ms.append((now - sent_at) * 1e3)
        if self.client.idle:
            qpos, images, reason = self.builder.build(time.monotonic_ns())
            if reason is None:
                self.client.send(self.tick, time.monotonic(), qpos, images)
                self.sent += 1
            else:
                self.observation_failures[reason] = self.observation_failures.get(reason, 0) + 1
        target = self.ensemble.get(self.tick)
        pose = current_pose(self.arm, self.hand)
        if target is not None:
            errors = group_errors(target - self.live_action(pose))
            for name, value in errors.items():
                self.target_error[name].append(value)
        self.tick += 1
        return target, pose

    def write(self, **entry):
        self.log.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def summary(self):
        return {"ticks": self.tick, "requests": self.sent, "observation_failures": self.observation_failures,
                "stale_chunks": self.stale_chunks,
                "inference_ms": summarize(self.inference_ms), "round_trip_ms": summarize(self.round_trip_ms),
                "target_minus_live_max": {name: summarize(values) for name, values in self.target_error.items()},
                "clamped_ticks": self.guard.clamped_ticks}


def pace(start, period):
    time.sleep(max(0.0, period - (time.monotonic() - start)))


def run_shadow(run, period, stop, seconds=None, until=lambda: False):
    """Reads and logs only. Nothing here can reach a publisher."""
    began = time.monotonic()
    report_at = began + 1.0
    while not stop[0] and not until() and (seconds is None or time.monotonic() - began < seconds):
        started = time.monotonic()
        target, pose = run.step()
        if run.client.failed:
            raise RuntimeError(f"inference worker failed: {run.client.failed}")
        if target is not None:
            run.write(t=started, tick=run.tick, mode="shadow", target=target.tolist(), live=run.live_action(pose).tolist())
        if started >= report_at:
            report_at = started + 1.0
            latest = run.inference_ms[-1] if run.inference_ms else None
            print(f"[SHADOW] tick={run.tick} requests={run.sent} inference_ms={latest} "
                  f"observation_failures={run.observation_failures}", flush=True)
        pace(started, period)


def pose_from_action(action, gravity):
    parts = split_action(action)
    return ReplayPose(arm_q=parts["arm_q"], arm_tau=gravity(parts["arm_q"]), head_q=parts["head_q"],
                      waist_q=parts["waist_q"], left_q=parts["left_q"], right_q=parts["right_q"],
                      left_mode=1, right_mode=1)


def blended(start, target, alpha):
    return start + min(max(alpha, 0.0), 1.0) * (target - start)


def execute_loop(run, arm, hand, guard, gravity, period, stop, args):
    """Commands the robot until q, the time limit, or a guard trip. Returns the pose to hold."""
    live = current_pose(arm, hand)
    send_pose(arm, hand, live)
    start_action = run.live_action(live)
    # Activation blocks for a while, so anything already in flight saw an older pose.
    run.ensemble.reset()
    run.ignore_before_tick = run.tick
    run.last_chunk_at = None
    deadline = time.monotonic() + 2.0
    first = None
    while first is None:
        started = time.monotonic()
        if stop[0]:
            return live
        first, _ = run.step()
        if run.client.failed or (first is None and started > deadline):
            print("[POLICY] no policy output after activation; holding.", flush=True)
            return live
        send_pose(arm, hand, live)
        arm.hold_targets()
        pace(started, period)
    delta = float(np.abs(split_action(first)["arm_q"] - live.arm_q).max())
    if delta > args.max_start_delta:
        print(f"[POLICY] first arm target is {delta:.3f} rad from the live pose (limit {args.max_start_delta:.3f}); holding.", flush=True)
        return live
    print(f"[POLICY] commanding. Blend {args.blend_seconds:.1f}s, limit {args.max_seconds:.0f}s. "
          "q exits and is not an e-stop; the hardware e-stop is the robot button.", flush=True)
    began = time.monotonic()
    hold = live
    while not stop[0] and time.monotonic() - began < args.max_seconds:
        started = time.monotonic()
        target, pose = run.step()
        snapshot = arm.get_recording_snapshot()
        tracking = float(np.abs(np.asarray(snapshot["requested"]["arm_q"]) - np.asarray(snapshot["state"]["q"])).max())
        # Not started: a stall inside step() must stop this tick, not the next one.
        reason = guard.check(time.monotonic(), run.last_chunk_at, tracking, run.client.failed)
        if reason is None and target is None:
            reason = guard.tripped = "the last policy chunk ran out"
        if reason is not None:
            print(f"[POLICY] holding: {reason}", flush=True)
            return hold
        command = blended(start_action, guard.clamp(target), (started - began) / args.blend_seconds)
        hold = pose_from_action(command, gravity)
        send_pose(arm, hand, hold)
        run.write(t=started, tick=run.tick, mode="execute", target=target.tolist(), command=command.tolist(),
                  live=run.live_action(pose).tolist(), tracking=tracking)
        pace(started, period)
    print("[POLICY] time limit reached." if not stop[0] else "[POLICY] stop requested.", flush=True)
    return hold


def check_policy_metadata(meta):
    dataset = meta["dataset"]
    if meta["action_dim"] != ACTION_DIM or meta["state_dim"] != ACTION_DIM:
        raise RuntimeError("Policy is not the 29-dimension R1_A7 + O6 layout")
    if dataset["camera_names"] != ["head_left", "head_right", "left_wrist", "right_wrist"]:
        raise RuntimeError(f"Unexpected policy cameras: {dataset['camera_names']}")


def check_policy_cameras(dataset, camera_config):
    """A policy is only valid on the head images it trained on."""
    height, width = camera_config["head_camera"]["image_shape"]
    trained = dataset.get("source_image_sizes", {}).get("color_0")
    if trained != [width // 2, height]:
        raise RuntimeError(f"Policy trained on head eye size {trained}, but the stream is "
                           f"{[width // 2, height]}. Export and train again from recordings of this stream.")


def main(argv=None):
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        print(f"[POLICY] no checkpoint: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.worker_python.is_file():
        print(f"[POLICY] no training environment: {args.worker_python}", file=sys.stderr)
        return 2
    running = teleop_pids()
    if running and args.execute:
        print(f"[POLICY] teleoperation or replay already running as PID {', '.join(running)}", file=sys.stderr)
        return 2
    log_dir = args.log_dir or DEFAULT_LOG_ROOT / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    client = PolicyClient([str(args.worker_python), str(ROOT / "tools/act_inference_worker.py"),
                           str(args.checkpoint), "--device", args.device])
    arm = hand = img_client = listener = tty_snapshot = None
    run = None
    try:
        meta = client.meta
        check_policy_metadata(meta)
        fps = float(meta["dataset"]["fps"])
        period = 1.0 / fps
        print(f"[POLICY] mode={'EXECUTE' if args.execute else 'shadow'} device={meta['device']} step={meta['step']} "
              f"task={meta['dataset'].get('task')} fps={fps:g}", flush=True)

        # First: this module configures logging_mp, which must happen before any getLogger().
        from teleop.teleop_hand_and_arm import TeleImagerCameraClient, correct_palm_frame
        from teleop.robot_control.robot_arm import R1_A7_ArmController
        from teleop.robot_control.robot_hand_linker_o6 import LinkerO6Controller
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(0, networkInterface=args.network_interface)
        arm = R1_A7_ArmController(deferred_activation=True, arm_velocity_limit=args.arm_velocity_limit,
                                  dq_feedforward=True, dq_feedforward_limit=args.arm_dq_limit, dq_feedforward_filter=0.5,
                                  target_velocity_limit=args.arm_target_velocity_limit,
                                  target_accel_limit=args.arm_target_accel_limit)
        hand = LinkerO6Controller()
        hand.wait_until_ready(timeout=3.0)
        img_client = TeleImagerCameraClient(server_host=args.img_server_ip, request_bgr=True)
        check_policy_cameras(meta["dataset"], img_client.get_cam_config())

        def cameras():
            return img_client.get_head_frame(), {"left": correct_palm_frame(img_client.get_left_wrist_frame()),
                                                 "right": correct_palm_frame(img_client.get_right_wrist_frame())}

        sync = CameraSynchronizer(["head", "left_wrist", "right_wrist"], anchor="head")
        guard = PolicyGuard(meta["normalization"], max_tracking_error=args.max_tracking_error,
                            max_chunk_age_s=args.max_chunk_age)
        run = PolicyRun(arm, hand, cameras, client, ObservationBuilder(arm, hand, sync, image_hw=tuple(meta["dataset"]["image_size_hw"])),
                        ActionEnsemble(client.chunk_size, args.ensemble_k), guard, log_dir / "ticks.jsonl")
        stop = [False]
        started = threading.Event()
        if sys.stdin.isatty():
            listener, tty_snapshot = start_keyboard(stop, started)

        if not args.execute:
            print("[POLICY] shadow mode: nothing is activated or published. q exits.", flush=True)
            run_shadow(run, period, stop, args.shadow_seconds)
            return 0

        if listener is None:
            print("[POLICY] --execute needs a terminal for r and q.", file=sys.stderr)
            return 2
        print("[POLICY] previewing in shadow mode. Press r to start commanding, q to exit. "
              "Hardware e-stop is the robot button.", flush=True)
        run_shadow(run, period, stop, until=started.is_set)
        if stop[0]:
            return 0
        pose = current_pose(arm, hand)
        outside = guard.out_of_training_range(run.live_action(pose))
        if outside and not args.allow_out_of_range_start:
            names = [meta["dataset"]["joint_names"][index] for index in outside]
            print(f"[POLICY] refusing to start: live pose is outside the training range at {names}", flush=True)
            return 2

        from teleop.utils.motion_switcher import MotionSwitcher
        status, result = MotionSwitcher().Enter_Debug_Mode()
        if status != 0:
            raise RuntimeError(f"failed to enter debug mode: status={status}, result={result}")
        hand.activate()
        arm.defer_publishing()
        try:
            arm.activate(cancel_requested=lambda: stop[0], home_head_waist=False)
        except InterruptedError:
            print("[POLICY] cancelled during activation; nothing was commanded.", flush=True)
            return 0
        pose = current_pose(arm, hand)
        arm.reset_target_shaper(pose.arm_q)
        arm.start_publishing()
        last = execute_loop(run, arm, hand, guard, GravityModel(), period, stop, args)
        hold_until_quit(arm, hand, last, stop)
        return 0
    finally:
        if run is not None:
            run.log.close()
            (log_dir / "summary.json").write_text(json.dumps(run.summary(), indent=2, ensure_ascii=False) + "\n")
            print(f"[POLICY] summary: {json.dumps(run.summary(), ensure_ascii=False)}", flush=True)
            print(f"[POLICY] log: {log_dir}", flush=True)
        if listener is not None:
            stop_keyboard(listener, tty_snapshot)
        client.close()
        if img_client is not None:
            img_client.close()
        shutdown(arm, hand)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[POLICY] interrupted. q / Ctrl+C are not an e-stop.", file=sys.stderr)
        sys.exit(130)
