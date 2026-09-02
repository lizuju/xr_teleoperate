#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC


SCHEMA = "r1_a7_lowstate_shadow_v1"
MOTOR_COUNT = 35
WAIST_INDEX = 13
LEFT_ARM_INDICES = list(range(15, 22))
RIGHT_ARM_INDICES = list(range(22, 29))
HEAD_INDICES = [29, 30]
ARM_INDICES = LEFT_ARM_INDICES + RIGHT_ARM_INDICES


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only R1-A7 lowstate JSON shadow")
    parser.add_argument("--interface", default="eno1")
    parser.add_argument(
        "--output",
        default="/run/user/1000/unitree-r1-lowstate/state.json",
    )
    parser.add_argument("--frequency", type=float, default=100.0)
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=0.0)
    return parser.parse_args()


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def main():
    args = parse_args()
    for name in ("frequency", "startup_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.duration) or args.duration < 0.0:
        raise ValueError("--duration must be finite and non-negative")

    output_path = Path(args.output)
    lock = threading.Lock()
    first_sample = threading.Event()
    latest = None
    sequence = 0
    crc = CRC()

    def handle(message: LowState_):
        nonlocal latest, sequence
        if crc.Crc(message) != int(message.crc):
            return
        sample_ns = time.monotonic_ns()
        full_q = [float(message.motor_state[index].q) for index in range(MOTOR_COUNT)]
        full_dq = [float(message.motor_state[index].dq) for index in range(MOTOR_COUNT)]
        arm_q = [full_q[index] for index in ARM_INDICES]
        arm_dq = [full_dq[index] for index in ARM_INDICES]
        fixed_q = [
            float(message.motor_state[WAIST_INDEX].q),
            float(message.motor_state[HEAD_INDICES[0]].q),
            float(message.motor_state[HEAD_INDICES[1]].q),
        ]
        if not all(math.isfinite(value) for value in full_q + full_dq):
            return
        with lock:
            sequence += 1
            latest = {
                "schema": SCHEMA,
                "sequence": sequence,
                "sample_monotonic_ns": sample_ns,
                "mode_machine": int(message.mode_machine),
                "crc_valid": True,
                "motor_count": MOTOR_COUNT,
                "arm_indices": ARM_INDICES,
                "waist_index": WAIST_INDEX,
                "left_arm_indices": LEFT_ARM_INDICES,
                "right_arm_indices": RIGHT_ARM_INDICES,
                "head_indices": HEAD_INDICES,
                "full_q": full_q,
                "full_dq": full_dq,
                "arm_q": arm_q,
                "arm_dq": arm_dq,
                "waist_yaw_q": fixed_q[0],
                "head_q": fixed_q[1:],
            }
        first_sample.set()

    ChannelFactoryInitialize(0, args.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(handle, 10)
    if not first_sample.wait(args.startup_timeout):
        raise RuntimeError("No rt/lowstate sample received before startup timeout")

    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    next_tick = started
    next_print = started
    try:
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.01))
                continue
            next_tick = max(next_tick + 1.0 / args.frequency, now)
            with lock:
                payload = None if latest is None else dict(latest)
            if payload is None:
                continue
            payload["published_monotonic_ns"] = time.monotonic_ns()
            atomic_write_json(output_path, payload)
            if now >= next_print:
                print(
                    f"[R1 LOWSTATE SHADOW] sequence={payload['sequence']} "
                    f"mode_machine={payload['mode_machine']} "
                    f"waist_yaw={payload['waist_yaw_q']:+.4f}",
                    flush=True,
                )
                next_print = now + 2.0
    except KeyboardInterrupt:
        pass
    finally:
        print("[R1 LOWSTATE SHADOW] stopped; no command channel was created", flush=True)


if __name__ == "__main__":
    main()
