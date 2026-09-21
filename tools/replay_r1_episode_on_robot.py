#!/usr/bin/env python3
"""Replay a recorded R1_A7 + Linker O6 episode onto the real robot.

By default this only validates the episode and prints a plan. It does not
send motion until you run without --check-only and then press r. q / Ctrl+C
stop the program; they are not an e-stop.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop.utils.r1_episode_replay import (
    DEFAULT_APPROACH_HZ,
    DEFAULT_MAX_APPROACH_S,
    MAX_PLAYBACK_SPEED,
    ReplayError,
    ReplayPose,
    approach_duration,
    build_approach_poses,
    load_replay_plan,
    plan_summary,
    playback_clock_ns,
    sample_at_time,
)


NETWORK_INTERFACE = "eno1"
CONTROL_HZ = 40.0


def teleop_pids():
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        names = [Path(os.fsdecode(arg)).name for arg in args if arg]
        if "teleop_hand_and_arm.py" in names or "replay_r1_episode_on_robot.py" in names:
            pid = process.name
            if pid != str(os.getpid()):
                found.append(pid)
    return found


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay a recorded R1 teleop episode onto the robot. "
                    "q and Ctrl+C are not an e-stop."
    )
    parser.add_argument("episode", type=Path, help="episode_* directory or episode.json")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate and print the plan; do not connect or move")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed in (0, 1]; cannot speed up past realtime")
    parser.add_argument("--max-approach-seconds", type=float, default=DEFAULT_MAX_APPROACH_S,
                        help="Refuse to start if ramping from the live pose would take longer")
    parser.add_argument("--network-interface", default=NETWORK_INTERFACE)
    parser.add_argument("--arm-velocity-limit", type=float, default=30.0)
    parser.add_argument("--arm-dq-feedforward", choices=["on", "off"], default="on")
    parser.add_argument("--arm-dq-limit", type=float, default=6.0)
    parser.add_argument("--arm-target-velocity-limit", type=float, default=6.0)
    parser.add_argument("--arm-target-accel-limit", type=float, default=40.0)
    return parser.parse_args(argv)


def validate_speed(speed):
    if not np.isfinite(speed) or speed <= 0.0:
        raise ReplayError("playback speed must be positive")
    if speed > MAX_PLAYBACK_SPEED:
        raise ReplayError(f"playback speed must be <= {MAX_PLAYBACK_SPEED} (no speeding up)")
    return float(speed)


def current_pose(arm_ctrl, hand_ctrl):
    snapshot = arm_ctrl.get_recording_snapshot()["state"]
    left, right = hand_ctrl.get_state()
    return ReplayPose(
        arm_q=np.asarray(snapshot["q"], dtype=np.float64),
        arm_tau=np.zeros(14, dtype=np.float64),
        head_q=np.asarray(snapshot["head_q"], dtype=np.float64),
        waist_q=float(snapshot["waist_q"]),
        left_q=np.asarray(left, dtype=np.float64),
        right_q=np.asarray(right, dtype=np.float64),
        left_mode=0,
        right_mode=0,
    )


def send_pose(arm_ctrl, hand_ctrl, pose):
    arm_ctrl.ctrl_dual_arm_and_head(
        np.asarray(pose.arm_q, dtype=np.float64),
        np.asarray(pose.arm_tau, dtype=np.float64),
        np.asarray(pose.head_q, dtype=np.float64),
        waist_yaw_target=float(pose.waist_q),
    )
    hand_ctrl.update(
        pose.left_q,
        pose.right_q,
        tracking_fresh=(pose.left_mode == 1, pose.right_mode == 1),
    )


def stream_poses(arm_ctrl, hand_ctrl, poses, hz, stop):
    period = 1.0 / hz
    for pose in poses:
        if stop():
            return False
        send_pose(arm_ctrl, hand_ctrl, pose)
        time.sleep(period)
    return True


def start_keyboard(stop, started):
    def on_press(key):
        if key in ("q", "Q"):
            stop[0] = True
        elif key in ("r", "R"):
            started.set()

    from sshkeyboard import listen_keyboard
    listener = threading.Thread(
        target=listen_keyboard,
        kwargs={"on_press": on_press, "until": None, "sequential": False},
        daemon=True,
    )
    listener.start()
    return listener


def wait_for_start(stop, started):
    print("[REPLAY] press r to approach the recorded start and play. "
          "q exits without an e-stop. Hardware e-stop is the robot button.",
          flush=True)
    while not started.is_set() and not stop[0]:
        time.sleep(0.05)
    return started.is_set() and not stop[0]


def hold_until_quit(arm_ctrl, hand_ctrl, pose, stop):
    print("[REPLAY] holding last pose. Press q to exit (not an e-stop).", flush=True)
    while not stop[0]:
        send_pose(arm_ctrl, hand_ctrl, pose)
        arm_ctrl.hold_targets()
        time.sleep(1.0 / CONTROL_HZ)


def run_robot(args, plan):
    speed = validate_speed(args.speed)
    pids = teleop_pids()
    if pids:
        raise ReplayError(f"teleoperation or replay already running as PID {', '.join(pids)}")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from teleop.robot_control.robot_arm import R1_A7_ArmController
    from teleop.robot_control.robot_hand_linker_o6 import LinkerO6Controller
    from teleop.utils.motion_switcher import MotionSwitcher

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    arm_ctrl = R1_A7_ArmController(
        deferred_activation=True,
        arm_velocity_limit=args.arm_velocity_limit,
        dq_feedforward=args.arm_dq_feedforward == "on",
        dq_feedforward_limit=args.arm_dq_limit,
        dq_feedforward_filter=0.5,
        target_velocity_limit=args.arm_target_velocity_limit,
        target_accel_limit=args.arm_target_accel_limit,
    )
    hand_ctrl = LinkerO6Controller()
    stop = [False]
    started = threading.Event()
    start_keyboard(stop, started)
    try:
        hand_ctrl.wait_until_ready(timeout=3.0)
        live = current_pose(arm_ctrl, hand_ctrl)
        needed, deltas = approach_duration(
            live, plan.first, max_seconds=args.max_approach_seconds,
        )
        print(json.dumps({
            **plan_summary(plan),
            "live_to_first": deltas,
            "approach_s": needed,
            "speed": speed,
            "sends_robot_commands": True,
        }, indent=2, ensure_ascii=False), flush=True)

        if not wait_for_start(stop, started):
            print("[REPLAY] cancelled before motion.", flush=True)
            return 0

        motion_switcher = MotionSwitcher()
        status, result = motion_switcher.Enter_Debug_Mode()
        if status != 0:
            raise ReplayError(f"failed to enter debug mode: status={status}, result={result}")
        hand_ctrl.activate()
        arm_ctrl.defer_publishing()
        try:
            arm_ctrl.activate(cancel_requested=lambda: stop[0], home_head_waist=False)
        except InterruptedError:
            print("[REPLAY] cancelled during activation; no playback.", flush=True)
            return 0
        live = current_pose(arm_ctrl, hand_ctrl)
        arm_ctrl.reset_target_shaper(live.arm_q)
        arm_ctrl.start_publishing()
        send_pose(arm_ctrl, hand_ctrl, live)
        needed, deltas = approach_duration(
            live, plan.first, max_seconds=args.max_approach_seconds,
        )
        approach = build_approach_poses(live, plan.first, needed, hz=DEFAULT_APPROACH_HZ)
        print(
            f"[REPLAY] approaching first frame over {needed:.2f}s "
            f"(arm {deltas['arm']:.3f} rad, waist {deltas['waist']:.3f} rad). "
            "Hardware e-stop is the robot button.",
            flush=True,
        )
        if not stream_poses(arm_ctrl, hand_ctrl, approach, DEFAULT_APPROACH_HZ, lambda: stop[0]):
            hold_until_quit(arm_ctrl, hand_ctrl, live, stop)
            return 0

        start_ns = plan.first.monotonic_ns
        last_ns = plan.last.monotonic_ns
        wall0 = time.monotonic()
        print(f"[REPLAY] playing {plan.duration_s:.2f}s at speed {speed:g}.", flush=True)
        last = plan.first
        while not stop[0]:
            elapsed = time.monotonic() - wall0
            query_ns = playback_clock_ns(start_ns, elapsed, speed=speed)
            if query_ns >= last_ns:
                last = plan.last
                send_pose(arm_ctrl, hand_ctrl, last)
                break
            last = sample_at_time(plan.waypoints, query_ns)
            send_pose(arm_ctrl, hand_ctrl, last)
            time.sleep(1.0 / CONTROL_HZ)
        hold_until_quit(arm_ctrl, hand_ctrl, last, stop)
        return 0
    finally:
        shutdown(arm_ctrl, hand_ctrl)


def shutdown(arm_ctrl, hand_ctrl):
    errors = []
    if hand_ctrl is not None:
        try:
            hand_ctrl.stop()
        except Exception as exc:
            errors.append(f"hand stop: {exc}")
    if arm_ctrl is not None:
        try:
            arm_ctrl.stop()
        except Exception as exc:
            errors.append(f"arm stop: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = load_replay_plan(args.episode)
        summary = plan_summary(plan)
        validate_speed(args.speed)
        if args.check_only:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print("[READY] episode can be replayed; no robot commands sent", flush=True)
            return 0
        return run_robot(args, plan)
    except ReplayError as exc:
        print(f"[REPLAY] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[REPLAY] interrupted. q / Ctrl+C are not an e-stop.", file=sys.stderr)
        sys.exit(130)
