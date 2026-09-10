#!/usr/bin/env python3
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from teleop.robot_control.linker_o6_retargeting import (
    DualLinkerO6Retargeter,
    is_tracking_fresh,
)


HAND_SCHEMA = "linker_o6_target_v1"
ARM_SCHEMA = "r1_a7_arm_ik_target_v1"
ARM_MAPPING = "r1_a7_visionpro_cartesian_v1"
ARM_REFERENCE = "armed_actual_wrist_yaw_pose"
ARM_FRAME = "r1_waist_yaw"
HAND_AXIS_ORDER = ["thumb_pitch", "thumb_yaw", "index", "middle", "ring", "pinky"]
HAND_ORDER = ["left", "right"]
ARM_DEADBAND_M = 0.002
MAX_WRIST_TRACKING_SPEED_M_S = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description="Vision Pro to isolated R1-A7 + O6 simulation snapshots")
    parser.add_argument("--hand-live-state", default="/run/user/1000/unitree-o6-live/target.json")
    parser.add_argument("--arm-live-state", default="/run/user/1000/unitree-r1-arm-live/target.json")
    parser.add_argument("--linker-o6-urdf-root", required=True)
    parser.add_argument("--linker-o6-method", choices=("vector", "position", "dexpilot"), default="vector")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--tracking-timeout", type=float, default=0.25)
    return parser.parse_args()


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def acquire_writer_lock(path, writer_name):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.parent / "writer.lock"
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(f"Another live-state writer holds {lock_path}") from error
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{writer_name} pid={os.getpid()}\n")
    lock_file.flush()
    return lock_file


def read_available_keys():
    keys = []
    while select.select([sys.stdin.fileno()], [], [], 0.0)[0]:
        value = os.read(sys.stdin.fileno(), 1)
        if not value:
            break
        keys.append(value.decode(errors="ignore").lower())
    return keys


def radial_deadband(offset):
    result = []
    for base in (0, 3):
        values = np.asarray(offset[base : base + 3], dtype=np.float64)
        if float(np.linalg.norm(values)) < ARM_DEADBAND_M:
            values[:] = 0.0
        result.extend(values.tolist())
    return result


def disarmed_payloads(sequence, retargeter, reason):
    timestamp_ns = time.monotonic_ns()
    common = {
        "armed": False,
        "reason": reason,
        "sequence": sequence,
        "monotonic_timestamp": timestamp_ns / 1_000_000_000.0,
        "published_monotonic_ns": timestamp_ns,
    }
    return (
        {"schema": HAND_SCHEMA, "mapping": retargeter.mapping_name, "retargeting_method": retargeter.method, **common},
        {"schema": ARM_SCHEMA, "mapping": ARM_MAPPING, **common},
    )


def main():
    args = parse_args()
    from televuer import TeleVuerWrapper

    if not math.isfinite(args.frequency) or args.frequency <= 0.0:
        raise ValueError("--frequency must be positive and finite")
    if not math.isfinite(args.tracking_timeout) or args.tracking_timeout <= 0.0:
        raise ValueError("--tracking-timeout must be positive and finite")

    hand_path = Path(args.hand_live_state)
    arm_path = Path(args.arm_live_state)
    hand_lock = acquire_writer_lock(hand_path, "visionpro_r1_a7_o6_sim:hand")
    try:
        arm_lock = acquire_writer_lock(arm_path, "visionpro_r1_a7_o6_sim:arm")
    except Exception:
        hand_lock.close()
        raise

    retargeter = DualLinkerO6Retargeter(args.linker_o6_urdf_root, method=args.linker_o6_method)
    wrapper = TeleVuerWrapper(
        use_hand_tracking=True,
        binocular=False,
        img_shape=(480, 640),
        display_mode="pass-through",
        zmq=False,
        webrtc=False,
        arm_reference_mode="head_yaw",
    )
    old_terminal = termios.tcgetattr(sys.stdin.fileno()) if sys.stdin.isatty() else None
    if old_terminal is not None:
        tty.setcbreak(sys.stdin.fileno())

    sequence = 0
    baseline = None
    previous_wrist_position = None
    previous_motion_timestamp = None
    fresh_frames = 0
    rearm_requested = False
    next_tick = time.monotonic()
    next_status = next_tick
    running = True
    print("[VISIONPRO SIM] pass-through server ready on https://HOST:8012", flush=True)
    print("[VISIONPRO SIM] put both wrists at the neutral pose, then press r; q stops", flush=True)
    print("[VISIONPRO SIM] simulation snapshots only; DDS/network robot control/serial are not used", flush=True)
    print(f"[VISIONPRO SIM] hand method={retargeter.method} mapping={retargeter.mapping_name}", flush=True)

    try:
        while running:
            for key in read_available_keys():
                if key in ("q", "\x03"):
                    running = False
                    break
                if key == "r":
                    retargeter.reset()
                    rearm_requested = True
                    baseline = None
                    previous_wrist_position = None
                    previous_motion_timestamp = None
                    fresh_frames = 0
                    print("[VISIONPRO SIM] rearm requested", flush=True)
            if not running:
                break

            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.01))
                continue
            next_tick = max(next_tick + 1.0 / args.frequency, now)
            tele_data = wrapper.get_tele_data()
            fresh = is_tracking_fresh(
                tele_data.motion_data_ready,
                tele_data.motion_data_timestamp,
                args.tracking_timeout,
                now=now,
            )
            sequence += 1

            if not fresh:
                retargeter.reset()
                baseline = None
                previous_wrist_position = None
                previous_motion_timestamp = None
                fresh_frames = 0
                rearm_requested = False
                hand_payload, arm_payload = disarmed_payloads(sequence, retargeter, "stale")
            else:
                fresh_frames += 1
                wrist_position = np.concatenate(
                    (tele_data.left_wrist_pose[:3, 3], tele_data.right_wrist_pose[:3, 3])
                )
                tracking_jump = False
                if (
                    previous_wrist_position is not None
                    and tele_data.motion_data_timestamp > previous_motion_timestamp
                ):
                    sample_dt = tele_data.motion_data_timestamp - previous_motion_timestamp
                    wrist_speed = max(
                        float(
                            np.linalg.norm(
                                wrist_position[:3] - previous_wrist_position[:3]
                            )
                        ),
                        float(
                            np.linalg.norm(
                                wrist_position[3:] - previous_wrist_position[3:]
                            )
                        ),
                    ) / sample_dt
                    tracking_jump = wrist_speed > MAX_WRIST_TRACKING_SPEED_M_S
                previous_wrist_position = wrist_position.copy()
                previous_motion_timestamp = tele_data.motion_data_timestamp

                if tracking_jump:
                    retargeter.reset()
                    baseline = None
                    fresh_frames = 0
                    rearm_requested = False
                    hand_payload, arm_payload = disarmed_payloads(
                        sequence, retargeter, "tracking_jump"
                    )
                    print("[VISIONPRO SIM] tracking jump; disarmed, press r again", flush=True)
                else:
                    if rearm_requested and fresh_frames >= 2:
                        baseline = wrist_position.copy()
                        rearm_requested = False
                        print("[VISIONPRO SIM] armed; dual-wrist zero baseline captured", flush=True)

                if tracking_jump or baseline is None:
                    if not tracking_jump:
                        hand_payload, arm_payload = disarmed_payloads(
                            sequence, retargeter, "disarmed"
                        )
                else:
                    left_target, right_target = retargeter.retarget(
                        tele_data.left_hand_pos, tele_data.right_hand_pos
                    )
                    arm_offset = radial_deadband((wrist_position - baseline).tolist())
                    timestamp_ns = time.monotonic_ns()
                    common = {
                        "armed": True,
                        "sequence": sequence,
                        "monotonic_timestamp": tele_data.motion_data_timestamp,
                        "published_monotonic_ns": timestamp_ns,
                    }
                    hand_payload = {
                        "schema": HAND_SCHEMA,
                        "mapping": retargeter.mapping_name,
                        "retargeting_method": retargeter.method,
                        "target_units": "normalized_0_1",
                        "hardware_axis_order": HAND_AXIS_ORDER,
                        "target_hand_order": HAND_ORDER,
                        "target_12": left_target.tolist() + right_target.tolist(),
                        **common,
                    }
                    arm_payload = {
                        "schema": ARM_SCHEMA,
                        "mapping": ARM_MAPPING,
                        "reference": ARM_REFERENCE,
                        "frame": ARM_FRAME,
                        "target_units": "m",
                        "target_hand_order": HAND_ORDER,
                        "position_offset_m": arm_offset,
                        **common,
                    }

            atomic_write_json(hand_path, hand_payload)
            atomic_write_json(arm_path, arm_payload)
            if now >= next_status:
                print(
                    f"[VISIONPRO SIM] sequence={sequence} fresh={fresh} "
                    f"armed={arm_payload['armed']} offset_m="
                    f"{[round(value, 4) for value in arm_payload.get('position_offset_m', [0.0] * 6)]}",
                    flush=True,
                )
                next_status = now + 2.0
    finally:
        sequence += 1
        hand_payload, arm_payload = disarmed_payloads(sequence, retargeter, "disarmed")
        atomic_write_json(hand_path, hand_payload)
        atomic_write_json(arm_path, arm_payload)
        if old_terminal is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_terminal)
        wrapper.close()
        arm_lock.close()
        hand_lock.close()


if __name__ == "__main__":
    main()
