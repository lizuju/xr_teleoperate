#!/usr/bin/env python3
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import select
import signal
import struct
import time


DEADMAN_SCHEMA = "r1_a7_arm_deadman_v1"
EV_SYN = 0
EV_KEY = 1
SYN_DROPPED = 3
SYN_REPORT = 0
KEY_SPACE = 57
KEY_RELEASE = 0
KEY_PRESS = 1
KEY_REPEAT = 2
INPUT_EVENT = struct.Struct("llHHi")
EVIOCSCLOCKID = 0x400445A0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local USB keyboard hold-to-run heartbeat for the R1 arm shadow gate"
    )
    parser.add_argument(
        "--device",
        default=(
            "/dev/input/by-id/"
            "usb-DWF8851_Lenovo_Traditional_USB_Keyboard-event-kbd"
        ),
    )
    parser.add_argument(
        "--output",
        default="/run/user/1000/unitree-r1-arm-live/deadman.json",
    )
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--key-event-timeout", type=float, default=0.08)
    parser.add_argument("--first-repeat-timeout", type=float, default=0.75)
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


def acquire_writer_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.parent / "deadman-writer.lock"
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError(f"another deadman producer owns {lock_path}")
    os.fchmod(lock_file.fileno(), 0o600)
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"r1_a7_keyboard_deadman pid={os.getpid()}\n")
    lock_file.flush()
    return lock_file


class KeyboardDeadmanState:
    def __init__(self, key_event_timeout, first_repeat_timeout):
        self.key_event_timeout = key_event_timeout
        self.first_repeat_timeout = first_repeat_timeout
        self.ready_for_press = False
        self.physical_down = False
        self.repeat_seen = False
        self.dropping_until_syn_report = False
        self.press_started = None
        self.last_key_event = None
        self.status = "startup_requires_release"

    def require_release(self, status, now=None):
        self.ready_for_press = False
        self.physical_down = False
        self.repeat_seen = False
        self.press_started = None
        self.last_key_event = now
        self.status = status

    def consume(self, event_type, event_code, event_value, now):
        if self.dropping_until_syn_report:
            if event_type == EV_SYN and event_code == SYN_REPORT:
                self.dropping_until_syn_report = False
                self.status = "syn_dropped_requires_release"
            return
        if event_type == EV_SYN and event_code == SYN_DROPPED:
            self.dropping_until_syn_report = True
            self.ready_for_press = False
            self.physical_down = False
            self.repeat_seen = False
            self.press_started = None
            self.last_key_event = None
            self.status = "syn_dropped_requires_release"
            return
        if event_type != EV_KEY or event_code != KEY_SPACE:
            return
        if event_value == KEY_RELEASE:
            self.ready_for_press = True
            self.physical_down = False
            self.repeat_seen = False
            self.press_started = None
            self.last_key_event = now
            self.status = "released"
        elif event_value == KEY_PRESS:
            if self.ready_for_press:
                self.physical_down = True
                self.repeat_seen = False
                self.press_started = now
                self.last_key_event = now
                self.status = "waiting_for_repeat"
        elif event_value == KEY_REPEAT and self.physical_down:
            if (
                not self.repeat_seen
                and self.press_started is not None
                and now - self.press_started >= self.first_repeat_timeout
            ):
                self.ready_for_press = False
                self.physical_down = False
                self.press_started = None
                self.last_key_event = now
                self.status = "first_repeat_timeout_requires_release"
                return
            if (
                self.repeat_seen
                and self.last_key_event is not None
                and now - self.last_key_event >= self.key_event_timeout
            ):
                self.ready_for_press = False
                self.physical_down = False
                self.repeat_seen = False
                self.press_started = None
                self.last_key_event = now
                self.status = "key_event_timeout_requires_release"
                return
            self.repeat_seen = True
            self.last_key_event = now
            self.status = "held"

    def snapshot(self, now):
        age = None if self.last_key_event is None else now - self.last_key_event
        if (
            self.physical_down
            and not self.repeat_seen
            and age is not None
            and age >= self.first_repeat_timeout
        ):
            self.ready_for_press = False
            self.physical_down = False
            self.press_started = None
            self.status = "first_repeat_timeout_requires_release"
        if (
            self.repeat_seen
            and self.physical_down
            and age is not None
            and age >= self.key_event_timeout
        ):
            self.ready_for_press = False
            self.physical_down = False
            self.repeat_seen = False
            self.press_started = None
            self.status = "key_event_timeout_requires_release"
        held = self.ready_for_press and self.physical_down and self.repeat_seen
        return held, self.status, age


def stop_on_signal(_signum, _frame):
    raise KeyboardInterrupt


def validate_cli(args):
    for name in ("frequency", "key_event_timeout", "first_repeat_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.duration) or args.duration < 0.0:
        raise ValueError("--duration must be finite and non-negative")


def payload(sequence, held, status, age, device, updated_monotonic_ns):
    return {
        "schema": DEADMAN_SCHEMA,
        "sequence": sequence,
        "held": held,
        "status": status,
        "device": device,
        "key_code": KEY_SPACE,
        "key_name": "KEY_SPACE",
        "key_event_age_ms": None if age is None else round(age * 1000.0, 3),
        "updated_monotonic_ns": updated_monotonic_ns,
    }


def main():
    args = parse_args()
    validate_cli(args)
    output_path = Path(args.output)
    device_path = Path(args.device)
    writer_lock = acquire_writer_lock(output_path)
    sequence = max(1, time.monotonic_ns())
    atomic_write_json(
        output_path,
        payload(
            sequence,
            False,
            "startup_requires_release",
            None,
            str(device_path),
            time.monotonic_ns(),
        ),
    )
    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_signal)
    previous_sighup = signal.signal(signal.SIGHUP, stop_on_signal)
    descriptor = None
    state = KeyboardDeadmanState(
        args.key_event_timeout,
        args.first_repeat_timeout,
    )
    pending = bytearray()
    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    next_tick = started
    last_status = None
    print(
        "[R1 KEYBOARD DEADMAN] release Space once, then hold Space continuously; "
        "release/disconnect/stale events stop the heartbeat",
        flush=True,
    )
    try:
        descriptor = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
        fcntl.ioctl(
            descriptor,
            EVIOCSCLOCKID,
            struct.pack("i", time.CLOCK_MONOTONIC),
        )
        poller = select.poll()
        poller.register(
            descriptor,
            select.POLLIN | select.POLLERR | select.POLLHUP | select.POLLNVAL,
        )
        last_loop_started = time.monotonic()
        while deadline is None or time.monotonic() < deadline:
            loop_started = time.monotonic()
            discard_events = loop_started - last_loop_started >= args.key_event_timeout
            last_loop_started = loop_started
            if discard_events:
                state.require_release("loop_gap_requires_release", loop_started)
                pending.clear()
            wait_ms = max(
                0,
                min(20, math.ceil((next_tick - loop_started) * 1000.0)),
            )
            for _fd, event_mask in poller.poll(wait_ms):
                if event_mask & (select.POLLERR | select.POLLHUP | select.POLLNVAL):
                    raise RuntimeError("keyboard device disconnected")
                if event_mask & select.POLLIN:
                    while True:
                        try:
                            chunk = os.read(descriptor, INPUT_EVENT.size * 32)
                        except BlockingIOError:
                            break
                        if not chunk:
                            raise RuntimeError("keyboard device returned EOF")
                        pending.extend(chunk)
                    if time.monotonic() - loop_started >= args.key_event_timeout:
                        discard_events = True
                        state.require_release(
                            "loop_gap_requires_release",
                            time.monotonic(),
                        )
                    if discard_events:
                        pending.clear()
                    else:
                        while len(pending) >= INPUT_EVENT.size:
                            seconds, microseconds, event_type, event_code, event_value = (
                                INPUT_EVENT.unpack_from(pending)
                            )
                            del pending[: INPUT_EVENT.size]
                            event_time = seconds + microseconds / 1_000_000.0
                            state.consume(
                                event_type,
                                event_code,
                                event_value,
                                event_time,
                            )
            now = time.monotonic()
            if now - loop_started >= args.key_event_timeout:
                state.require_release("loop_gap_requires_release", now)
                pending.clear()
            if now < next_tick:
                continue
            next_tick = max(next_tick + 1.0 / args.frequency, now)
            publish_now = time.monotonic()
            if publish_now - loop_started >= args.key_event_timeout:
                state.require_release("loop_gap_requires_release", publish_now)
                pending.clear()
            held, status, age = state.snapshot(publish_now)
            sequence += 1
            atomic_write_json(
                output_path,
                payload(
                    sequence,
                    held,
                    status,
                    age,
                    str(device_path),
                    int(publish_now * 1_000_000_000),
                ),
            )
            if status != last_status:
                print(
                    f"[R1 KEYBOARD DEADMAN] status={status} held={held}",
                    flush=True,
                )
                last_status = status
    except KeyboardInterrupt:
        pass
    finally:
        sequence += 1
        atomic_write_json(
            output_path,
            payload(
                sequence,
                False,
                "stopped",
                None,
                str(device_path),
                time.monotonic_ns(),
            ),
        )
        if descriptor is not None:
            os.close(descriptor)
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGHUP, previous_sighup)
        writer_lock.close()
        print("[R1 KEYBOARD DEADMAN] stopped held=False", flush=True)


if __name__ == "__main__":
    main()
