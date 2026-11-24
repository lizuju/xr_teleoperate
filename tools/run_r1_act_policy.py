#!/usr/bin/env python3
"""Run a trained ACT policy against the live R1_A7 + left O6 + right Dex1.

Default is shadow mode: it reads feedback and cameras and logs what the policy would
command, and never activates or publishes anything. Only --execute can move the robot,
and only after r is pressed. q / Ctrl+C stop the program; they are not an e-stop.

With --start-pose EPISODE, --execute first eases the robot to that recording's start pose
(first r), holds it, and starts the policy only on the second r. --start-pose standard (or a
standard pose .json) uses the teleop o-key start pose instead, so policy runs and new
recordings start from the same place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
                                        stop_keyboard, teleop_pids)
from teleop.robot_control.r1_standard_pose import DEFAULT_POSE_PATH, standard_start_pose
from teleop.utils.act_runtime import (ACTION_DIM, WAIST_INDEX, ActionEnsemble, GravityModel, ObservationBuilder,
                                      PolicyClient, PolicyGuard, WaistHold, group_errors, join_action, split_action,
                                      summarize)
from teleop.utils.frame_sync import CameraSynchronizer
from teleop.utils.act_contract import validate_contract
from teleop.utils.icecream_task import IcecreamTask, PHASES
from teleop.utils.r1_episode_replay import (APPROACH_HAND_VEL, APPROACH_WAIST_VEL, ReplayError, ReplayPose,
                                           approach_duration, approach_limits, build_approach_poses,
                                           recorded_start_pose)

DEV_ROOT = Path(os.environ.get("R1_DEV_ROOT", ROOT.parent))
DEFAULT_WORKER_PYTHON = DEV_ROOT / ".venv-train/bin/python"
DEFAULT_LOG_ROOT = DEV_ROOT / "policy-runs"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--execute", action="store_true", help="Actually command the robot after r. Default: shadow")
    parser.add_argument("--shadow-seconds", type=float, help="Stop shadow mode after this long. Default: until q")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Whole-run wall-clock limit, including phase changes and pauses")
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
    parser.add_argument("--img-server-ip", default="192.168.50.165")
    parser.add_argument("--network-interface", default=NETWORK_INTERFACE)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--device", default="auto", help="Inference device: auto, cuda or cpu")
    parser.add_argument("--log-dir", type=Path, help=f"Default: a new directory under {DEFAULT_LOG_ROOT}")
    parser.add_argument("--auto-stage", action="store_true", help="Use learned completion plus joint feedback; reviewed phase model required")
    parser.add_argument("--dispense-min-phase-seconds", type=float, default=3.0,
                        help="Minimum dispensing-phase duration; not a measured flow timer")
    parser.add_argument("--start-pose", type=Path,
                        help="Where execution begins: a recorded episode (its --start-frame), or 'standard' / a "
                             "standard pose .json (the teleop o-key pose: mean start of the round 2 demos, waist at "
                             "0). With --execute the first r eases the robot there and holds it, the second r starts "
                             "the policy")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame of --start-pose to use (default the first)")
    parser.add_argument("--start-speed", type=float, default=0.3, help="Arm and head speed of the start-pose approach (rad/s)")
    parser.add_argument("--max-start-approach-seconds", type=float, default=15.0,
                        help="Refuse a start-pose approach that would take longer than this")
    parser.add_argument("--waist-snap-deg", type=float, default=5.0,
                        help="While the waist holds a teleop waist stop (0, +-90 or +-148 deg, as the key-turn demos did), "
                             "command policy waist targets within this many degrees of it as exactly that stop. A "
                             "target further out ends the hold and passes unchanged, so turns start as the policy "
                             "begins them; the arm controller still slews the waist at its 40 deg/s limit. "
                             "0 disables. Default: 5")
    parser.add_argument("--waist-capture-deg", type=float, default=2.0,
                        help="Start holding a waist stop when the policy waist target, or the waist at the start, "
                             "comes within this many degrees of it. At most --waist-snap-deg. Default: 2")
    args = parser.parse_args(argv)
    if not (math.isfinite(args.start_speed) and 0.0 < args.start_speed <= 1.0):
        parser.error("--start-speed must be in (0, 1] rad/s")
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    if args.start_pose == Path("standard"):
        args.start_pose = DEFAULT_POSE_PATH
    if args.start_pose is not None and args.start_pose.suffix == ".json" and args.start_frame != 0:
        parser.error("--start-frame applies to a recorded episode, not a standard pose file")
    if not (math.isfinite(args.max_start_approach_seconds) and args.max_start_approach_seconds > 0.0):
        parser.error("--max-start-approach-seconds must be positive")
    if not (math.isfinite(args.waist_snap_deg) and 0.0 <= args.waist_snap_deg < 45.0):
        parser.error("--waist-snap-deg must be at least 0 and under 45")
    if not (math.isfinite(args.waist_capture_deg) and args.waist_capture_deg >= 0.0
            and (args.waist_snap_deg == 0.0 or args.waist_capture_deg <= args.waist_snap_deg)):
        parser.error("--waist-capture-deg must be at least 0 and no more than --waist-snap-deg")
    for key in ("max_seconds", "max_chunk_age", "max_tracking_error"):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) <= 0:
            parser.error(f"{key} must be finite and positive")
    for key in ("blend_seconds", "max_start_delta", "ensemble_k", "dispense_min_phase_seconds",
                "arm_target_velocity_limit", "arm_target_accel_limit", "arm_velocity_limit", "arm_dq_limit"):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) < 0:
            parser.error(f"{key} must be finite and nonnegative")
    return args


class PolicyRun:
    """One control tick's worth of reading, inference plumbing and logging. Commands nothing."""

    def __init__(self, arm, hand, cameras, client, builder, ensemble, guard, log_path):
        self.arm, self.hand, self.cameras, self.client = arm, hand, cameras, client
        self.builder, self.ensemble, self.guard = builder, ensemble, guard
        self.tick = 0
        self.clock_origin = time.monotonic()
        self.fps = 40.0
        self.task = None
        self.completion = None
        self.paused = False
        self.finished = False
        self.execution_deadline = None
        self.advance_requested = False
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
        self.tick = max(self.tick, int((time.monotonic() - self.clock_origin) * self.fps))
        head, wrists = self.cameras()
        self.builder.observe(head, wrists)
        response = self.client.poll()
        now = time.monotonic()
        if response is not None:
            (tick, sent_at), inference_ms, chunk, completion = response
            if tick >= self.ignore_before_tick:
                # Age counts from the request: a late result still comes from the old observation.
                if now - sent_at <= self.guard.max_chunk_age_s:
                    self.ensemble.add(tick, chunk)
                    self.last_chunk_at = sent_at
                    self.completion = completion
                else:
                    self.stale_chunks += 1
            self.inference_ms.append(inference_ms)
            self.round_trip_ms.append((now - sent_at) * 1e3)
        if self.client.idle:
            qpos, images, reason = self.builder.build(time.monotonic_ns())
            if reason is None:
                self.client.send(self.tick, time.monotonic(), qpos, images,
                                 None if self.task is None else self.task.context(time.monotonic()))
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

    def reset_predictions(self):
        self.ensemble.reset()
        self.ignore_before_tick = self.tick
        self.last_chunk_at = self.completion = None

    def write(self, **entry):
        if self.task is not None:
            entry.update(phase=PHASES[self.task.phase], phase_complete=self.completion,
                         task_done=self.task.done)
        self.log.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self.log.flush()

    def summary(self):
        return {"ticks": self.tick, "requests": self.sent, "observation_failures": self.observation_failures,
                "stale_chunks": self.stale_chunks,
                "inference_ms": summarize(self.inference_ms), "round_trip_ms": summarize(self.round_trip_ms),
                "target_minus_live_max": {name: summarize(values) for name, values in self.target_error.items()},
                "clamped_ticks": self.guard.clamped_ticks,
                "task_events": [] if self.task is None else self.task.events,
                "task_done": False if self.task is None else self.task.done}


def pace(start, period):
    time.sleep(max(0.0, period - (time.monotonic() - start)))


def run_shadow(run, period, stop, seconds=None, until=lambda: False, phase_preview=False):
    """Reads and logs only. Nothing here can reach a publisher."""
    began = time.monotonic()
    report_at = began + 1.0
    while not stop[0] and not until() and (seconds is None or time.monotonic() - began < seconds):
        started = time.monotonic()
        target, pose = run.step()
        if phase_preview and run.task is not None:
            if run.task.started is None:
                run.task.start(started, run.live_action(pose))
            if run.advance_requested:
                run.advance_requested = False
                run.task.advance(started, run.live_action(pose), "shadow_operator_preview")
                run.reset_predictions()
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


def ease_to_start(arm, hand, target, args, period, stop):
    """Ease the activated robot from its live pose to a recorded start pose.

    Arm and head move at no more than --start-speed and the waist no faster than the replay
    approach. Returns the pose to hold, or None if q stopped the move.
    """
    live = current_pose(arm, hand)
    limits = approach_limits(arm=args.start_speed, head=args.start_speed,
                             waist=min(APPROACH_WAIST_VEL, args.start_speed), hand=APPROACH_HAND_VEL)
    seconds, deltas = approach_duration(live, target, limits=limits, min_seconds=2.0,
                                        max_seconds=args.max_start_approach_seconds)
    print(f"[POLICY] easing to the recorded start pose over {seconds:.1f}s (arm {deltas['arm']:.3f} rad, "
          f"head {deltas['head']:.3f} rad, waist {deltas['waist']:.3f} rad). "
          "Hardware e-stop is the robot button.", flush=True)
    for pose in build_approach_poses(live, target, seconds, hz=1.0 / period):
        if stop[0]:
            print("[POLICY] start-pose approach stopped.", flush=True)
            return None
        send_pose(arm, hand, pose)
        time.sleep(period)
    return target


def execute_loop(run, arm, hand, guard, gravity, period, stop, args, initial_hold=None, blend=True):
    """Commands the robot until q, the time limit, or a guard trip. Returns the pose to hold."""
    live = current_pose(arm, hand)
    hold = live if initial_hold is None else initial_hold
    send_pose(arm, hand, hold)
    start_action = run.live_action(hold)
    # A resumed run starts from the last command, so a stop it was holding is held again.
    waist = WaistHold(math.radians(getattr(args, "waist_snap_deg", 0.0)),
                      math.radians(getattr(args, "waist_capture_deg", 0.0)), start_action[WAIST_INDEX])
    if run.execution_deadline is None:
        run.execution_deadline = time.monotonic() + args.max_seconds
    # Activation blocks for a while, so anything already in flight saw an older pose.
    run.ensemble.reset()
    run.ignore_before_tick = run.tick
    run.last_chunk_at = None
    if run.task is not None and run.task.started is None:
        run.task.start(time.monotonic(), start_action)
    deadline = time.monotonic() + 2.0
    first = None
    while first is None:
        started = time.monotonic()
        if stop[0] or run.paused:
            return hold
        if started >= run.execution_deadline:
            run.finished = True
            return hold
        first, _ = run.step()
        if run.client.failed or (first is None and started > deadline):
            guard.tripped = "no policy output after activation"
            print("[POLICY] no policy output after activation; holding.", flush=True)
            return hold
        send_pose(arm, hand, hold)
        arm.hold_targets()
        pace(started, period)
    if run.task is not None:
        first = run.task.apply(first)
    delta = float(np.abs(split_action(first)["arm_q"] - live.arm_q).max())
    if delta > args.max_start_delta:
        guard.tripped = "first arm target too far from current pose"
        print(f"[POLICY] first arm target is {delta:.3f} rad from the live pose (limit {args.max_start_delta:.3f}); holding.", flush=True)
        return hold
    print(f"[POLICY] commanding. Blend {args.blend_seconds:.1f}s, limit {args.max_seconds:.0f}s, "
          f"waist hold {math.degrees(waist.release):g} deg (capture {math.degrees(waist.capture):g}). "
          "q exits and is not an e-stop; the hardware e-stop is the robot button.", flush=True)
    began = time.monotonic()
    while not stop[0] and time.monotonic() < run.execution_deadline:
        started = time.monotonic()
        target, pose = run.step()
        if run.paused:
            run.write(t=started, mode="paused", live=run.live_action(pose).tolist())
            return hold
        snapshot = arm.get_recording_snapshot()
        tracking = float(np.abs(np.asarray(snapshot["requested"]["arm_q"]) - np.asarray(snapshot["state"]["q"])).max())
        # Not started: a stall inside step() must stop this tick, not the next one.
        reason = guard.check(time.monotonic(), run.last_chunk_at, tracking, run.client.failed)
        if reason is None and target is None:
            reason = guard.tripped = "the last policy chunk ran out"
        if reason is not None:
            print(f"[POLICY] holding: {reason}", flush=True)
            return hold
        if run.task is not None:
            changed = run.advance_requested
            run.advance_requested = False
            if changed:
                run.task.advance(started, run.live_action(hold))
            else:
                changed = run.task.observe(started, run.live_action(pose), run.completion)
            if run.task.done:
                run.write(t=started, mode="completed", live=run.live_action(pose).tolist())
                return hold
            if changed:
                if run.task.phase == 2:
                    run.task.hold_pose = run.live_action(hold).copy()
                run.reset_predictions()
                return hold
            target = run.task.apply(target)
        alpha = (started - began) / args.blend_seconds if blend and args.blend_seconds > 0 else 1.0
        # The log keeps the policy's own target; the command carries the held waist.
        command = blended(start_action, guard.clamp(waist.apply(target)), alpha)
        hold = pose_from_action(command, gravity)
        send_pose(arm, hand, hold)
        run.write(t=started, tick=run.tick, mode="execute", target=target.tolist(), command=command.tolist(),
                  live=run.live_action(pose).tolist(), tracking=tracking)
        pace(started, period)
    run.finished = True
    print("[POLICY] time limit reached." if not stop[0] else "[POLICY] stop requested.", flush=True)
    return hold


def check_policy_metadata(meta):
    validate_contract(meta["dataset"])
    if meta["action_dim"] != ACTION_DIM or meta["state_dim"] != ACTION_DIM:
        raise RuntimeError("Policy is not the 24-dimension R1_A7 + left O6 + right Dex1 layout")


def check_policy_cameras(dataset, camera_config):
    sizes = dataset.get("source_image_sizes", {})
    for key, camera in (("color_0", "head_camera"), ("color_1", "head_camera"),
                        ("color_2", "left_wrist_camera"), ("color_3", "right_wrist_camera")):
        shape = (camera_config.get(camera) or {}).get("image_shape")
        actual = None if not shape else [shape[1] // (2 if camera == "head_camera" else 1), shape[0]]
        if actual is None or sizes.get(key) != actual:
            raise RuntimeError(f"Policy trained on {key} size {sizes.get(key)}, but {camera} is {actual}. "
                               "Export and train again from recordings of this stream.")


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
        (log_dir / "run.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "dataset": meta["dataset"], "mode": "execute" if args.execute else "shadow",
            "started_at": time.time(), "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        }, indent=2, ensure_ascii=False) + "\n")
        check_policy_metadata(meta)
        start_target = None
        if args.start_pose is not None:
            if args.start_pose.suffix == ".json":
                start_target, start_sides = standard_start_pose(args.start_pose)
            else:
                start_target, start_sides = recorded_start_pose(args.start_pose, args.start_frame)
            if start_sides != meta["dataset"].get("end_effectors"):
                raise ValueError(f"--start-pose was recorded with {start_sides}, "
                                 f"but the policy drives {meta['dataset'].get('end_effectors')}")
        if args.auto_stage and not meta.get("context_dim"):
            raise ValueError("--auto-stage requires a phase-conditioned model")
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
        from teleop.robot_control.robot_hand_dex1_r1 import Dex1LowCmdGripper, O6Dex1Hands
        gripper_low, gripper_high = validate_contract(meta["dataset"])
        hardware = meta["dataset"]["controller_config"]
        hand = O6Dex1Hands(LinkerO6Controller(sides=("left",)), Dex1LowCmdGripper(
            arm, slot=hardware["dex1_slot"], kp=hardware["dex1_kp"], kd=hardware["dex1_kd"],
            grip_delta=hardware["dex1_grip_delta"], closed_q=gripper_low, open_q=gripper_high))
        hand.wait_until_ready(timeout=3.0)
        img_client = TeleImagerCameraClient(server_host=args.img_server_ip, request_bgr=True)
        check_policy_cameras(meta["dataset"], img_client.get_cam_config())

        def cameras():
            return img_client.get_head_frame(), {"left": correct_palm_frame(img_client.get_left_wrist_frame(), "linker_o6"),
                                                 "right": correct_palm_frame(img_client.get_right_wrist_frame(), "dex1_1")}

        sync = CameraSynchronizer(["head", "left_wrist", "right_wrist"], anchor="head")
        guard = PolicyGuard(meta["normalization"], max_tracking_error=args.max_tracking_error,
                            max_chunk_age_s=args.max_chunk_age, gripper_limits=(gripper_low, gripper_high))
        run = PolicyRun(arm, hand, cameras, client, ObservationBuilder(arm, hand, sync, image_hw=tuple(meta["dataset"]["image_size_hw"])),
                        ActionEnsemble(client.chunk_size, args.ensemble_k), guard, log_dir / "ticks.jsonl")
        run.fps = fps
        if meta.get("context_dim"):
            run.task = IcecreamTask((gripper_low, gripper_high), args.auto_stage, args.dispense_min_phase_seconds)
        if start_target is not None:
            outside = guard.out_of_training_range(run.live_action(start_target))
            if outside:
                names = [meta["dataset"]["joint_names"][index] for index in outside]
                print(f"[POLICY] refusing --start-pose: it is outside the training range at {names}", flush=True)
                return 2
            away = float(np.abs(start_target.arm_q - current_pose(arm, hand).arm_q).max())
            where = args.start_pose if args.start_pose.suffix == ".json" else f"{args.start_pose} frame {args.start_frame}"
            print(f"[POLICY] start pose: {where}; "
                  f"the arm is {away:.3f} rad away from it now.", flush=True)
        stop = [False]
        started = threading.Event()
        if sys.stdin.isatty():
            from replay_r1_episode_on_robot import snapshot_tty
            from sshkeyboard import listen_keyboard
            tty_snapshot = snapshot_tty()
            def policy_key(key):
                if key == "q":
                    stop[0] = True
                elif key == "p":
                    run.paused = True
                elif key == "r":
                    run.paused = False
                    started.set()
                elif key == "n":
                    run.advance_requested = True
            listener = threading.Thread(target=listen_keyboard,
                                        kwargs={"on_press": policy_key, "until": None, "sequential": True}, daemon=True)
            listener.start()

        if not args.execute:
            print("[POLICY] shadow mode: nothing is activated or published. q exits.", flush=True)
            run_shadow(run, period, stop, args.shadow_seconds, phase_preview=True)
            return 0

        if listener is None:
            print("[POLICY] --execute needs a terminal for r and q.", file=sys.stderr)
            return 2
        if start_target is not None:
            print("[POLICY] --start-pose: the first r eases the robot to the recorded start pose and holds it; "
                  "the second r starts the policy.", flush=True)
        print("[POLICY] previewing in shadow mode. r starts/resumes, p pauses, n confirms the next phase, q exits. "
              "Hardware e-stop is the robot button.", flush=True)
        run_shadow(run, period, stop, until=started.is_set)
        if stop[0]:
            return 0
        pose = current_pose(arm, hand)
        outside = guard.out_of_training_range(run.live_action(pose))
        # With --start-pose the live pose is checked once the robot is held at the start pose.
        if outside and start_target is None and not args.allow_out_of_range_start:
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
        gravity = GravityModel()
        last, blend_next = None, True
        if start_target is not None:
            try:
                last = ease_to_start(arm, hand, start_target, args, period, stop)
            except ReplayError as exc:
                print(f"[POLICY] not moving to the start pose: {exc}", flush=True)
                hold_until_quit(arm, hand, current_pose(arm, hand), stop)
                return 2
            if last is None:
                return 0
            started.clear()
            print("[POLICY] holding the recorded start pose. r starts the policy; q exits (not an e-stop).", flush=True)
            while not started.is_set() and not stop[0]:
                send_pose(arm, hand, last)
                arm.hold_targets()
                time.sleep(period)
            if stop[0]:
                return 0
            outside = guard.out_of_training_range(run.live_action(current_pose(arm, hand)))
            if outside and not args.allow_out_of_range_start:
                names = [meta["dataset"]["joint_names"][index] for index in outside]
                print(f"[POLICY] refusing to start: the held pose is outside the training range at {names}", flush=True)
                hold_until_quit(arm, hand, last, stop)
                return 2
        while not stop[0]:
            started.clear()
            last = execute_loop(run, arm, hand, guard, gravity, period, stop, args, last, blend_next)
            blend_next = False
            if run.finished or guard.tripped or (run.task is not None and run.task.done):
                hold_until_quit(arm, hand, last, stop)
                break
            if run.paused:
                print("[POLICY] paused; r resumes, q exits. Corrections must be recorded by teleop after this writer exits.", flush=True)
                pause_started = time.monotonic()
                while run.paused and not stop[0]:
                    send_pose(arm, hand, last)
                    if time.monotonic() >= run.execution_deadline:
                        run.finished = True
                        break
                    time.sleep(period)
                if run.task is not None:
                    run.task.started += time.monotonic() - pause_started
                run.reset_predictions()
                blend_next = True
                if run.finished:
                    hold_until_quit(arm, hand, last, stop)
                    break
            elif run.task is None:
                hold_until_quit(arm, hand, last, stop)
                break
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
