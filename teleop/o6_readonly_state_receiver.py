#!/usr/bin/env python3
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from o6_readonly_contract import STATE_SCHEMA, validate_source_packet


def parse_args():
    parser = argparse.ArgumentParser(description="Validate PC2 FC04-only O6 NDJSON on stdin")
    parser.add_argument(
        "--output",
        default="/run/user/1000/unitree-o6-readonly/state.json",
    )
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
    lock_path = path.parent / "writer.lock"
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError(f"another O6 read-only receiver owns {lock_path}")
    os.fchmod(lock_file.fileno(), 0o600)
    return lock_file


def rejected_payload(reason):
    return {
        "schema": STATE_SCHEMA,
        "receiver_status": "rejected",
        "receiver_reason": reason,
        "received_monotonic_ns": time.monotonic_ns(),
        "actuation_enabled": False,
        "writes_enabled": False,
    }


def main():
    args = parse_args()
    output = Path(args.output)
    lock_file = acquire_writer_lock(output)
    previous_boot_id = None
    previous_sequence = None
    accepted = 0
    rejected = 0
    print("[O6 READONLY RECEIVER] stdin only; no device, DDS, or network output", flush=True)
    try:
        for line in sys.stdin:
            if len(line) > 32768:
                rejected += 1
                atomic_write_json(output, rejected_payload("line_too_large"))
                continue
            try:
                row = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                )
                validate_source_packet(row, previous_boot_id, previous_sequence)
            except (json.JSONDecodeError, UnicodeError, ValueError) as error:
                rejected += 1
                atomic_write_json(output, rejected_payload(str(error)))
                continue
            previous_boot_id = row["source_boot_id"]
            previous_sequence = row["sequence"]
            payload = dict(row)
            payload["schema"] = STATE_SCHEMA
            payload["receiver_status"] = "valid"
            payload["received_monotonic_ns"] = time.monotonic_ns()
            atomic_write_json(output, payload)
            accepted += 1
    finally:
        atomic_write_json(
            output,
            {
                "schema": STATE_SCHEMA,
                "receiver_status": "stopped",
                "received_monotonic_ns": time.monotonic_ns(),
                "actuation_enabled": False,
                "writes_enabled": False,
            },
        )
        lock_file.close()
        print(
            f"[O6 READONLY RECEIVER] stopped accepted={accepted} rejected={rejected}",
            flush=True,
        )


if __name__ == "__main__":
    main()
