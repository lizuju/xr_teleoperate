#!/usr/bin/env python3
"""Replay the taught right-lever ice-cream sequence onto the real R1.

``--check-only`` never imports DDS publishers: it only loads waypoints / grip
cap, prints the plan (grasp waist is used through withdraw even if the stored
withdraw waist differs), and does not move.

Without ``--check-only``: connect read-only first, print live-to-grasp deltas,
then wait for key ``r`` before any lowcmd or hand close. ``q`` / Ctrl+C are a
software e-stop for this tool: stop the trajectory, freeze at LIVE joints
(dq≈0), release the right hand (left hand stays on live q), stop the arm
publisher, and exit debug mode when the API allows. This is NOT the hardware
power cut on the robot button.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop.utils.r1_icecream_lever_replay import (
    DEFAULT_MAX_APPROACH_S,
    DEFAULT_SLOT,
    IcecreamLeverError,
    LEVER_SLOTS,
    LeverPose,
    build_arm14,
    build_motion_segments,
    default_grip_cap_path,
    default_waypoints_path,
    extract_lever_slot,
    format_check_only,
    live_to_grasp_deltas,
    load_right_max_close_q,
    load_waypoints,
)


NETWORK_INTERFACE = "eno1"
CONTROL_HZ = 40.0
HAND_DOF = 6


def teleop_pids():
    found = []
    self_pid = str(os.getpid())
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        names = [Path(os.fsdecode(arg)).name for arg in args if arg]
        blockers = (
            "teleop_hand_and_arm.py",
            "replay_r1_episode_on_robot.py",
            "replay_r1_icecream_lever.py",
        )
        if any(name in names for name in blockers) and process.name != self_pid:
            found.append(process.name)
    return found


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay taught right-lever ice-cream sequence. "
        "q / Ctrl+C = software e-stop for this tool (not the hardware button)."
    )
    parser.add_argument(
        "--slot",
        default=DEFAULT_SLOT,
        choices=sorted(LEVER_SLOTS),
        help=f"lever slot (default {DEFAULT_SLOT})",
    )
    parser.add_argument(
        "--waypoints",
        type=Path,
        default=None,
        help="JSON path (default ~/.config/xr_teleoperate/icecream_waypoints.json)",
    )
    parser.add_argument(
        "--grip-cap",
        type=Path,
        default=None,
        help="JSON path (default ~/.config/xr_teleoperate/o6_grip_cap.json)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print the plan; do not connect DDS or move",
    )
    parser.add_argument(
        "--max-approach-seconds",
        type=float,
        default=DEFAULT_MAX_APPROACH_S,
        help="Refuse if live→grasp (or final waist turn) ramp would exceed this",
    )
    parser.add_argument("--network-interface", default=NETWORK_INTERFACE)
    parser.add_argument("--arm-velocity-limit", type=float, default=30.0)
    parser.add_argument("--arm-dq-feedforward", choices=["on", "off"], default="on")
    parser.add_argument("--arm-dq-limit", type=float, default=6.0)
    parser.add_argument("--arm-target-velocity-limit", type=float, default=6.0)
    parser.add_argument("--arm-target-accel-limit", type=float, default=40.0)
    return parser.parse_args(argv)


def snapshot_tty(stream=None):
    stream = sys.stdin if stream is None else stream
    if not hasattr(stream, "isatty") or not stream.isatty():
        return None
    import termios

    fd = stream.fileno()
    try:
        return (fd, termios.tcgetattr(fd))
    except termios.error:
        return None


def restore_tty(snapshot):
    if snapshot is None:
        return False
    fd, attrs = snapshot
    import termios

    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        return True
    except termios.error:
        return False


def start_keyboard(stop, started):
    snapshot = snapshot_tty()

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
    return listener, snapshot


def stop_keyboard(listener, snapshot):
    try:
        from sshkeyboard import stop_listening

        stop_listening()
    except Exception:
        pass
    if listener is not None:
        listener.join(timeout=1.0)
    restore_tty(snapshot)


def wait_for_start(stop, started):
    print(
        "[ICECREAM] press r to approach grasp and play. "
        "q / Ctrl+C = software e-stop for this tool "
        "(stop trajectory, freeze LIVE, release right hand). "
        "Hardware power cut is still the robot button.",
        flush=True,
    )
    while not started.is_set() and not stop[0]:
        time.sleep(0.05)
    return started.is_set() and not stop[0]


def current_live(arm_ctrl, hand_ctrl):
    snapshot = arm_ctrl.get_recording_snapshot()["state"]
    arm_q = np.asarray(snapshot["q"], dtype=np.float64).reshape(14)
    head_q = np.asarray(snapshot["head_q"], dtype=np.float64).reshape(2)
    waist_q = float(snapshot["waist_q"])
    left_hand, right_hand = hand_ctrl.get_state()
    return {
        "left7": arm_q[:7].copy(),
        "right7": arm_q[7:14].copy(),
        "head2": head_q.copy(),
        "waist": waist_q,
        "left_hand": np.asarray(left_hand, dtype=np.float64).copy(),
        "right_hand": np.asarray(right_hand, dtype=np.float64).copy(),
    }


def send_pose(arm_ctrl, hand_ctrl, pose):
    arm_ctrl.ctrl_dual_arm_and_head(
        np.asarray(pose.arm_q, dtype=np.float64),
        np.zeros(14, dtype=np.float64),
        np.asarray(pose.head_q, dtype=np.float64),
        waist_yaw_target=float(pose.waist_q),
    )
    hand_ctrl.update(
        pose.left_q,
        pose.right_q,
        tracking_fresh=(bool(pose.left_fresh), bool(pose.right_fresh)),
    )


def stream_poses(arm_ctrl, hand_ctrl, poses, hz, stop):
    period = 1.0 / hz
    last = None
    for pose in poses:
        if stop():
            return False, last
        send_pose(arm_ctrl, hand_ctrl, pose)
        last = pose
        time.sleep(period)
    return True, last


def freeze_live_release_right(arm_ctrl, hand_ctrl):
    """Command LIVE joints (dq≈0 via shaper reset) and release right hand only."""
    live = current_live(arm_ctrl, hand_ctrl)
    arm_ctrl.reset_target_shaper(np.concatenate([live["left7"], live["right7"]]))
    open_right = np.zeros(HAND_DOF, dtype=np.float64)
    freeze = LeverPose(
        arm_q=build_arm14(live["left7"], live["right7"]),
        head_q=np.asarray(live["head2"], dtype=np.float64),
        waist_q=float(live["waist"]),
        left_q=np.asarray(live["left_hand"], dtype=np.float64),
        right_q=open_right,
        left_fresh=True,
        right_fresh=False,  # mode 0 on right; left keeps tracking live q
    )
    # A few cycles so the freeze + right release land before we stop publishing.
    for _ in range(max(3, int(CONTROL_HZ * 0.1))):
        send_pose(arm_ctrl, hand_ctrl, freeze)
        time.sleep(1.0 / CONTROL_HZ)
    return freeze


def close_hand_publishers_without_dual_release(hand_ctrl):
    """Close hand DDS endpoints without writing mode (0,0) to both sides.

    LinkerO6Controller.stop() releases both hands; that can drop a left-hand cup.
    After a hold-left / release-right e-stop we only tear down endpoints.
    """
    if hand_ctrl is None or getattr(hand_ctrl, "closed", False):
        return
    hand_ctrl.closed = True
    hand_ctrl.active = False
    errors = []
    for name in ("left_publisher", "right_publisher", "left_subscriber", "right_subscriber"):
        endpoint = getattr(hand_ctrl, name, None)
        if endpoint is None:
            continue
        try:
            endpoint.Close()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
        else:
            setattr(hand_ctrl, name, None)
    if errors:
        raise RuntimeError("; ".join(errors))


def software_estop(arm_ctrl, hand_ctrl, motion_switcher=None, *, publishing=False, entered_debug=False):
    """Immediate software e-stop for this playback tool (not hardware power cut)."""
    print(
        "[ICECREAM] software e-stop: stop trajectory, freeze LIVE joints, "
        "release right hand (hold left), stop arm publisher"
        + (", exit debug mode" if entered_debug else "")
        + ". Hardware power cut remains the robot button.",
        flush=True,
    )
    freeze_ok = False
    if arm_ctrl is not None and hand_ctrl is not None and publishing:
        try:
            freeze_live_release_right(arm_ctrl, hand_ctrl)
            freeze_ok = True
        except Exception as exc:
            print(f"[ICECREAM] freeze/release-right failed: {exc}", flush=True)
            try:
                left_q, _right_q = hand_ctrl.get_state()
                hand_ctrl.update(
                    np.asarray(left_q, dtype=np.float64),
                    np.zeros(HAND_DOF, dtype=np.float64),
                    tracking_fresh=(True, False),
                )
            except Exception as hand_exc:
                print(f"[ICECREAM] right-hand release fallback failed: {hand_exc}", flush=True)

    errors = []
    if arm_ctrl is not None:
        try:
            arm_ctrl.stop()
        except Exception as exc:
            errors.append(f"arm stop: {exc}")
    if hand_ctrl is not None:
        try:
            if freeze_ok or publishing:
                close_hand_publishers_without_dual_release(hand_ctrl)
            else:
                hand_ctrl.stop()
        except Exception as exc:
            errors.append(f"hand stop: {exc}")
    if entered_debug and motion_switcher is not None:
        try:
            status, result = motion_switcher.Exit_Debug_Mode()
            print(
                f"[ICECREAM] Exit_Debug_Mode status={status} result={result}",
                flush=True,
            )
        except Exception as exc:
            errors.append(f"exit debug: {exc}")
            print(
                f"[ICECREAM] Exit_Debug_Mode failed ({exc}); "
                "arm publisher already stopped — not chasing playback targets.",
                flush=True,
            )
    if errors:
        print(f"[ICECREAM] e-stop cleanup issues: {'; '.join(errors)}", flush=True)
    return 0


def shutdown(arm_ctrl, hand_ctrl, *, prefer_hold_left=False):
    errors = []
    if hand_ctrl is not None:
        try:
            if prefer_hold_left:
                close_hand_publishers_without_dual_release(hand_ctrl)
            else:
                hand_ctrl.stop()
        except Exception as exc:
            errors.append(f"hand stop: {exc}")
    if arm_ctrl is not None:
        try:
            # Idempotent if software_estop already stopped publishing.
            if getattr(arm_ctrl, "publish_running", False) or getattr(arm_ctrl, "active", False):
                arm_ctrl.stop()
        except Exception as exc:
            errors.append(f"arm stop: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def run_robot(args, shell_plan):
    pids = teleop_pids()
    if pids:
        raise IcecreamLeverError(
            f"teleoperation or replay already running as PID {', '.join(pids)}"
        )

    # Imports that construct publishers live only on the motion path.
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
    # Never apply the left grip cap during this playback; we command the right
    # close target explicitly from the saved right max_close_q.
    hand_ctrl = LinkerO6Controller(apply_grip_cap=False)
    stop = [False]
    started = threading.Event()
    listener, tty_snapshot = start_keyboard(stop, started)
    motion_switcher = None
    entered_debug = False
    publishing = False
    estop_done = False
    try:
        hand_ctrl.wait_until_ready(timeout=3.0)
        live = current_live(arm_ctrl, hand_ctrl)
        deltas = live_to_grasp_deltas(
            live["right7"], live["waist"], shell_plan.grasp_right_arm, shell_plan.grasp_waist_rad
        )
        print(format_check_only(shell_plan), flush=True)
        print(
            "[ICECREAM] live→grasp: "
            f"right_arm_max|Δq|={deltas['right_arm_max_abs_dq']:.4f} rad, "
            f"waist Δ={deltas['waist_delta_deg']:.2f}° "
            f"(grasp waist {shell_plan.grasp_waist_deg:.2f}°, "
            f"turn {shell_plan.turn_waist_yaw_deg:.2f}°). "
            "No lowcmd yet — waiting for r.",
            flush=True,
        )
        # Pre-validate approach duration before the operator presses r.
        from teleop.utils.r1_icecream_lever_replay import approach_duration_right_waist

        approach_duration_right_waist(
            live["right7"],
            live["waist"],
            shell_plan.grasp_right_arm,
            shell_plan.grasp_waist_rad,
            max_seconds=args.max_approach_seconds,
        )

        if not wait_for_start(stop, started):
            print("[ICECREAM] cancelled before motion (no lowcmd sent).", flush=True)
            return 0

        motion_switcher = MotionSwitcher()
        status, result = motion_switcher.Enter_Debug_Mode()
        if status != 0:
            raise IcecreamLeverError(
                f"failed to enter debug mode: status={status}, result={result}"
            )
        entered_debug = True
        hand_ctrl.activate()
        arm_ctrl.defer_publishing()
        try:
            arm_ctrl.activate(cancel_requested=lambda: stop[0], home_head_waist=False)
        except InterruptedError:
            print("[ICECREAM] cancelled during activation; no playback.", flush=True)
            estop_done = True
            return software_estop(
                arm_ctrl,
                hand_ctrl,
                motion_switcher,
                publishing=False,
                entered_debug=entered_debug,
            )

        if stop[0]:
            estop_done = True
            return software_estop(
                arm_ctrl,
                hand_ctrl,
                motion_switcher,
                publishing=False,
                entered_debug=entered_debug,
            )

        live = current_live(arm_ctrl, hand_ctrl)
        plan, deltas, approach_s = build_motion_segments(
            shell_plan,
            live_left7=live["left7"],
            live_right7=live["right7"],
            live_head2=live["head2"],
            live_waist=live["waist"],
            live_left_hand=live["left_hand"],
            live_right_hand=live["right_hand"],
            max_approach_s=args.max_approach_seconds,
            hz=CONTROL_HZ,
        )
        plan.sends_robot_commands = True

        arm_ctrl.reset_target_shaper(np.concatenate([live["left7"], live["right7"]]))
        arm_ctrl.start_publishing()
        publishing = True
        # Seed with live hold (left fresh) before the first ramp sample.
        seed = LeverPose(
            arm_q=build_arm14(live["left7"], live["right7"]),
            head_q=np.asarray(live["head2"], dtype=np.float64),
            waist_q=float(live["waist"]),
            left_q=np.asarray(live["left_hand"], dtype=np.float64),
            right_q=np.asarray(live["right_hand"], dtype=np.float64),
            left_fresh=True,
            right_fresh=True,
        )
        send_pose(arm_ctrl, hand_ctrl, seed)
        print(
            f"[ICECREAM] starting approach ({approach_s:.2f}s) then lever sequence. "
            f"Waist will hold {plan.grasp_waist_deg:.2f}° then ramp to "
            f"{plan.turn_waist_yaw_deg:.2f}° (Δ≈{plan.grasp_waist_deg - plan.turn_waist_yaw_deg:.1f}°). "
            "q / Ctrl+C = software e-stop. Hardware power cut is the robot button.",
            flush=True,
        )

        for segment in plan.segments:
            if stop[0]:
                estop_done = True
                return software_estop(
                    arm_ctrl,
                    hand_ctrl,
                    motion_switcher,
                    publishing=publishing,
                    entered_debug=entered_debug,
                )
            print(
                f"[ICECREAM] segment {segment.name} ({segment.duration_s:.2f}s, "
                f"{len(segment.poses)} cmds)",
                flush=True,
            )
            ok, _last_pose = stream_poses(
                arm_ctrl, hand_ctrl, segment.poses, CONTROL_HZ, lambda: stop[0]
            )
            if not ok:
                estop_done = True
                return software_estop(
                    arm_ctrl,
                    hand_ctrl,
                    motion_switcher,
                    publishing=publishing,
                    entered_debug=entered_debug,
                )

        print(
            "[ICECREAM] sequence complete. Press q for software e-stop / exit "
            "(freeze LIVE + release right hand). Hardware button still cuts power.",
            flush=True,
        )
        while not stop[0]:
            # Hold final pose only until q; then e-stop path (not forever-chase).
            if plan.segments and plan.segments[-1].poses:
                send_pose(arm_ctrl, hand_ctrl, plan.segments[-1].poses[-1])
                arm_ctrl.hold_targets()
            time.sleep(1.0 / CONTROL_HZ)
        estop_done = True
        return software_estop(
            arm_ctrl,
            hand_ctrl,
            motion_switcher,
            publishing=publishing,
            entered_debug=entered_debug,
        )
    except KeyboardInterrupt:
        stop[0] = True
        estop_done = True
        return software_estop(
            arm_ctrl,
            hand_ctrl,
            motion_switcher,
            publishing=publishing,
            entered_debug=entered_debug,
        )
    finally:
        stop_keyboard(listener, tty_snapshot)
        if not estop_done:
            # Normal / early cancel before publishing: tear down without dual cup drop
            # when we never closed the right hand for playback.
            shutdown(
                arm_ctrl,
                hand_ctrl,
                prefer_hold_left=publishing or entered_debug,
            )
            if entered_debug and motion_switcher is not None:
                try:
                    motion_switcher.Exit_Debug_Mode()
                except Exception:
                    pass


def main(argv=None):
    args = parse_args(argv)
    waypoints_path = args.waypoints or default_waypoints_path()
    grip_path = args.grip_cap or default_grip_cap_path()
    waypoints = load_waypoints(waypoints_path)
    right_q = load_right_max_close_q(grip_path)
    plan = extract_lever_slot(waypoints, right_q, slot_name=args.slot)
    if args.check_only:
        print(format_check_only(plan))
        print("[READY] check-only complete; no robot commands sent", flush=True)
        return 0
    return run_robot(args, plan)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IcecreamLeverError as error:
        print(f"[ICECREAM] {error}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print(
            "\n[ICECREAM] interrupted — software e-stop path (not hardware power cut).",
            file=sys.stderr,
        )
        raise SystemExit(130)
