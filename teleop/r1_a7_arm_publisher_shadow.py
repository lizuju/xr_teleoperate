#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty


LOWSTATE_SCHEMA = "r1_a7_lowstate_shadow_v1"
COMMAND_SCHEMA = "r1_a7_arm_command_intent_v1"
DEADMAN_SCHEMA = "r1_a7_arm_deadman_v1"
SHADOW_SCHEMA = "r1_a7_arm_publisher_shadow_v1"
MOTOR_COUNT = 35
ARM_INDICES = list(range(15, 29))
REQUIRED_PLATFORM_PROFILE = "r1_a7_dual_linker_o6_v1"
REQUIRED_COLLISION_MODEL = "r1_a7_linker_o6_open"
REQUIRED_COLLISION_MODEL_SHA256 = "5a64e7c2e6d4fa6b11e3243ffb4d7bf398e3537338cc0cf147c46ab1b8ef3bc5"
REQUIRED_O6_COLLISION_POSE = "fresh_verified_open"
MAX_O6_STATE_AGE = 0.25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shadow-only R1-A7 full-body takeover and arm command safety gate"
    )
    parser.add_argument(
        "--lowstate-shadow",
        default="/run/user/1000/unitree-r1-lowstate/state.json",
    )
    parser.add_argument(
        "--bridge-command",
        default="/run/user/1000/unitree-r1-arm-bridge/command.json",
    )
    parser.add_argument(
        "--deadman-state",
        default="/run/user/1000/unitree-r1-arm-live/deadman.json",
    )
    parser.add_argument(
        "--output",
        default="/run/user/1000/unitree-r1-arm-publisher/shadow.json",
    )
    parser.add_argument("--frequency", type=float, default=250.0)
    parser.add_argument("--lowstate-timeout", type=float, default=0.10)
    parser.add_argument("--command-timeout", type=float, default=0.10)
    parser.add_argument("--deadman-timeout", type=float, default=0.10)
    parser.add_argument("--sequence-freeze-timeout", type=float, default=0.25)
    parser.add_argument("--max-xr-age", type=float, default=0.25)
    parser.add_argument("--max-following-error-rad", type=float, default=0.05)
    parser.add_argument("--following-error-duration", type=float, default=0.10)
    parser.add_argument("--zero-hold-duration", type=float, default=10.0)
    parser.add_argument("--max-arm-velocity-rad-s", type=float, default=0.15)
    parser.add_argument("--max-non-arm-drift-rad", type=float, default=0.02)
    parser.add_argument("--max-takeover-velocity-rad-s", type=float, default=0.05)
    parser.add_argument("--max-non-arm-velocity-rad-s", type=float, default=0.05)
    parser.add_argument("--max-publish-gap", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=0.0)
    return parser.parse_args()


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def finite_vector(value, length):
    return (
        isinstance(value, list)
        and len(value) == length
        and all(finite_number(item) for item in value)
    )


def valid_positive_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= 2**63 - 1
    )


def validate_full_lowstate(row, now, timeout):
    if not isinstance(row, dict):
        return None, "invalid_document"
    if row.get("schema") != LOWSTATE_SCHEMA:
        return None, "schema_rejected"
    if (
        row.get("motor_count") != MOTOR_COUNT
        or row.get("waist_index") != 13
        or row.get("left_arm_indices") != list(range(15, 22))
        or row.get("right_arm_indices") != list(range(22, 29))
        or row.get("head_indices") != [29, 30]
    ):
        return None, "mapping_rejected"
    if row.get("crc_valid") is not True:
        return None, "crc_rejected"
    if not valid_positive_integer(row.get("sequence")):
        return None, "invalid_sequence"
    sample_ns = row.get("sample_monotonic_ns")
    published_ns = row.get("published_monotonic_ns")
    if not valid_positive_integer(sample_ns) or not valid_positive_integer(published_ns):
        return None, "invalid_timestamp"
    mode_machine = row.get("mode_machine")
    if not isinstance(mode_machine, int) or isinstance(mode_machine, bool):
        return None, "invalid_mode"
    if not finite_vector(row.get("full_q"), MOTOR_COUNT) or not finite_vector(
        row.get("full_dq"), MOTOR_COUNT
    ):
        return None, "invalid_state"
    sample_age = now - sample_ns / 1_000_000_000.0
    transport_age = now - published_ns / 1_000_000_000.0
    if sample_age < 0.0 or transport_age < 0.0 or published_ns < sample_ns:
        return None, "future_timestamp"
    if sample_age >= timeout or transport_age >= timeout:
        return None, "stale"
    return {
        "sequence": row["sequence"],
        "mode_machine": mode_machine,
        "full_q": [float(value) for value in row["full_q"]],
        "full_dq": [float(value) for value in row["full_dq"]],
        "sample_age": sample_age,
        "transport_age": transport_age,
    }, "valid"


def validate_bridge_candidate(row, now, timeout):
    if not isinstance(row, dict):
        return None, "invalid_document"
    if row.get("schema") != COMMAND_SCHEMA:
        return None, "schema_rejected"
    if row.get("platform_profile") != REQUIRED_PLATFORM_PROFILE:
        return None, "platform_profile_rejected"
    if row.get("physical_publishing_enabled") is not False:
        return None, "bridge_mode_rejected"
    if row.get("o6_enabled") is not False:
        return None, "o6_mode_rejected"
    if row.get("collision_model") != REQUIRED_COLLISION_MODEL:
        return None, "collision_model_rejected"
    if row.get("collision_model_sha256") != REQUIRED_COLLISION_MODEL_SHA256:
        return None, "collision_model_hash_rejected"
    if row.get("o6_collision_pose") != REQUIRED_O6_COLLISION_POSE:
        return None, "o6_collision_pose_rejected"
    if row.get("decision") not in ("new", "hold") or row.get("source_armed") is not True:
        return None, "bridge_not_ready"
    if not valid_positive_integer(row.get("source_sequence")):
        return None, "invalid_sequence"
    updated_ns = row.get("bridge_monotonic_ns")
    if not valid_positive_integer(updated_ns):
        return None, "invalid_timestamp"
    age = now - updated_ns / 1_000_000_000.0
    if age < 0.0:
        return None, "future_timestamp"
    if age >= timeout:
        return None, "stale"
    o6_sequence = row.get("o6_state_sequence")
    if not valid_positive_integer(o6_sequence):
        return None, "invalid_o6_sequence"
    o6_age_ms = row.get("o6_state_age_ms")
    if not finite_number(o6_age_ms) or float(o6_age_ms) < 0.0:
        return None, "invalid_o6_age"
    if float(o6_age_ms) / 1000.0 + age >= MAX_O6_STATE_AGE:
        return None, "o6_state_stale"
    o6_angles = row.get("o6_angles_raw")
    if not isinstance(o6_angles, dict) or set(o6_angles) != {"left", "right"}:
        return None, "invalid_o6_angles"
    if any(
        not isinstance(o6_angles[side], list)
        or len(o6_angles[side]) != 6
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 250
            or value > 255
            for value in o6_angles[side]
        )
        for side in ("left", "right")
    ):
        return None, "invalid_o6_angles"
    mode_machine = row.get("mode_machine")
    if not isinstance(mode_machine, int) or isinstance(mode_machine, bool):
        return None, "invalid_mode"
    tracking_age_ms = row.get("source_tracking_age_ms")
    if not finite_number(tracking_age_ms) or float(tracking_age_ms) < 0.0:
        return None, "invalid_tracking_age"
    if not finite_vector(row.get("candidate_arm_q"), len(ARM_INDICES)):
        return None, "invalid_candidate"
    return {
        "sequence": row["source_sequence"],
        "platform_profile": row["platform_profile"],
        "mode_machine": mode_machine,
        "candidate_q": [float(value) for value in row["candidate_arm_q"]],
        "tracking_age": float(tracking_age_ms) / 1000.0,
        "o6_sequence": o6_sequence,
        "o6_age": float(o6_age_ms) / 1000.0 + age,
        "age": age,
    }, "valid"


def validate_deadman(row, now, timeout):
    if not isinstance(row, dict):
        return None, "invalid_document"
    if row.get("schema") != DEADMAN_SCHEMA:
        return None, "schema_rejected"
    if not valid_positive_integer(row.get("sequence")):
        return None, "invalid_sequence"
    if not isinstance(row.get("held"), bool):
        return None, "invalid_held"
    updated_ns = row.get("updated_monotonic_ns")
    if not valid_positive_integer(updated_ns):
        return None, "invalid_timestamp"
    age = now - updated_ns / 1_000_000_000.0
    if age < 0.0:
        return None, "future_timestamp"
    if age >= timeout:
        return None, "stale"
    return {
        "sequence": row["sequence"],
        "held": row["held"],
        "age": age,
    }, "valid"


class ArmPublisherSafetyGate:
    def __init__(
        self,
        lowstate_timeout,
        command_timeout,
        deadman_timeout,
        sequence_freeze_timeout,
        max_xr_age,
        max_following_error_rad,
        following_error_duration,
        zero_hold_cycles,
        max_arm_velocity_rad_s=1.0,
        max_non_arm_drift_rad=0.02,
        max_takeover_velocity_rad_s=0.05,
        max_non_arm_velocity_rad_s=0.05,
        max_publish_gap=0.05,
    ):
        self.lowstate_timeout = lowstate_timeout
        self.command_timeout = command_timeout
        self.deadman_timeout = deadman_timeout
        self.sequence_freeze_timeout = sequence_freeze_timeout
        self.max_xr_age = max_xr_age
        self.max_following_error_rad = max_following_error_rad
        self.following_error_duration = following_error_duration
        self.zero_hold_cycles = zero_hold_cycles
        self.max_arm_velocity_rad_s = max_arm_velocity_rad_s
        self.max_non_arm_drift_rad = max_non_arm_drift_rad
        self.max_takeover_velocity_rad_s = max_takeover_velocity_rad_s
        self.max_non_arm_velocity_rad_s = max_non_arm_velocity_rad_s
        self.max_publish_gap = max_publish_gap
        self.state = "disarmed"
        self.fault_latched = False
        self.takeover_q = None
        self.last_full_q = None
        self.last_publish_time = None
        self.expected_mode = None
        self.zero_hold_count = 0
        self.following_error_started_at = None
        self.last_lowstate_sequence = None
        self.last_new_lowstate_time = None
        self.last_deadman_sequence = None
        self.last_new_deadman_time = None
        self.last_candidate_sequence = None
        self.last_new_candidate_time = None

    def _decision(self, status, should_publish=False, full_q=None, first=False):
        return {
            "status": status,
            "fault_latched": self.fault_latched,
            "should_publish": should_publish,
            "full_q": None if full_q is None else list(full_q),
            "first_command_matches_actual": first,
        }

    def _fault(self, status):
        self.state = "fault_latched"
        self.fault_latched = True
        return self._decision(status)

    def reset_fault(self, deadman_row, now):
        if not self.fault_latched:
            return False
        deadman, status = validate_deadman(deadman_row, now, self.deadman_timeout)
        if status != "valid" or deadman["held"]:
            return False
        self.state = "disarmed"
        self.fault_latched = False
        self.takeover_q = None
        self.last_full_q = None
        self.last_publish_time = None
        self.expected_mode = None
        self.zero_hold_count = 0
        self.following_error_started_at = None
        return True

    def _check_lowstate_sequence(self, sequence, now):
        if self.last_lowstate_sequence is None:
            self.last_lowstate_sequence = sequence
            self.last_new_lowstate_time = now
            return None
        if sequence < self.last_lowstate_sequence:
            return "lowstate_sequence_regression"
        if sequence > self.last_lowstate_sequence:
            self.last_lowstate_sequence = sequence
            self.last_new_lowstate_time = now
            return None
        if now - self.last_new_lowstate_time >= self.sequence_freeze_timeout:
            return "lowstate_sequence_frozen"
        return None

    def _check_deadman_sequence(self, sequence, now):
        if self.last_deadman_sequence is None:
            self.last_deadman_sequence = sequence
            self.last_new_deadman_time = now
            return None
        if sequence < self.last_deadman_sequence:
            return "deadman_sequence_regression"
        if sequence > self.last_deadman_sequence:
            self.last_deadman_sequence = sequence
            self.last_new_deadman_time = now
            return None
        if now - self.last_new_deadman_time >= self.sequence_freeze_timeout:
            return "deadman_sequence_frozen"
        return None

    def _check_candidate_sequence(self, sequence, now):
        if self.last_candidate_sequence is None:
            self.last_candidate_sequence = sequence
            self.last_new_candidate_time = now
            return None
        if sequence < self.last_candidate_sequence:
            return "sequence_regression"
        if sequence > self.last_candidate_sequence:
            self.last_candidate_sequence = sequence
            self.last_new_candidate_time = now
            return None
        if now - self.last_new_candidate_time >= self.sequence_freeze_timeout:
            return "sequence_frozen"
        return None

    def _check_following_error(self, actual_q, now):
        if self.last_full_q is None:
            return None
        error = max(
            abs(actual_q[index] - self.last_full_q[index]) for index in ARM_INDICES
        )
        if error > self.max_following_error_rad:
            if self.following_error_started_at is None:
                self.following_error_started_at = now
            if now - self.following_error_started_at >= self.following_error_duration:
                return "following_error"
        else:
            self.following_error_started_at = None
        return None

    def _check_non_arm_drift(self, actual_q):
        if self.takeover_q is None:
            return None
        drift = max(
            abs(actual_q[index] - self.takeover_q[index])
            for index in range(MOTOR_COUNT)
            if index not in ARM_INDICES
        )
        return "non_arm_drift" if drift > self.max_non_arm_drift_rad else None

    def _check_non_arm_velocity(self, actual_dq):
        velocity = max(
            abs(actual_dq[index])
            for index in range(MOTOR_COUNT)
            if index not in ARM_INDICES
        )
        return (
            "non_arm_moving"
            if velocity > self.max_non_arm_velocity_rad_s
            else None
        )

    def _active_target(self, candidate_q, now):
        full_q = list(self.takeover_q)
        elapsed = max(0.0, now - self.last_publish_time)
        max_step = self.max_arm_velocity_rad_s * elapsed
        for arm_offset, motor_index in enumerate(ARM_INDICES):
            previous = self.last_full_q[motor_index]
            delta = candidate_q[arm_offset] - previous
            if abs(delta) <= max_step + 1.0e-12:
                full_q[motor_index] = candidate_q[arm_offset]
            else:
                full_q[motor_index] = previous + math.copysign(max_step, delta)
        return full_q

    def step(self, lowstate_row, bridge_row, deadman_row, now, arm_request=False):
        if self.fault_latched:
            return self._decision("fault_latched")

        lowstate, lowstate_status = validate_full_lowstate(
            lowstate_row, now, self.lowstate_timeout
        )
        if lowstate_status != "valid":
            if self.state == "disarmed" and not arm_request:
                return self._decision(f"lowstate_{lowstate_status}")
            return self._fault(
                "lowstate_stale" if lowstate_status == "stale" else f"lowstate_{lowstate_status}"
            )
        lowstate_sequence_fault = self._check_lowstate_sequence(
            lowstate["sequence"], now
        )
        if lowstate_sequence_fault is not None and (
            self.state != "disarmed" or arm_request
        ):
            return self._fault(lowstate_sequence_fault)

        deadman, deadman_status = validate_deadman(
            deadman_row, now, self.deadman_timeout
        )
        if deadman_status != "valid":
            if self.state == "disarmed" and not arm_request:
                return self._decision(f"deadman_{deadman_status}")
            return self._fault(
                "deadman_stale" if deadman_status == "stale" else f"deadman_{deadman_status}"
            )
        deadman_sequence_fault = self._check_deadman_sequence(deadman["sequence"], now)
        if deadman_sequence_fault is not None and (
            self.state != "disarmed" or arm_request
        ):
            return self._fault(deadman_sequence_fault)
        if not deadman["held"]:
            if self.state == "disarmed":
                return self._decision("disarmed")
            return self._fault("deadman_released")

        candidate, candidate_status = validate_bridge_candidate(
            bridge_row, now, self.command_timeout
        )
        if candidate_status != "valid":
            if self.state == "disarmed" and not arm_request:
                return self._decision(f"command_{candidate_status}")
            return self._fault(
                "command_stale" if candidate_status == "stale" else f"command_{candidate_status}"
            )
        if candidate["tracking_age"] >= self.max_xr_age:
            return self._fault("xr_stale")
        if candidate["mode_machine"] != lowstate["mode_machine"]:
            return self._fault("mode_mismatch")

        sequence_fault = self._check_candidate_sequence(candidate["sequence"], now)
        if sequence_fault is not None and (self.state != "disarmed" or arm_request):
            return self._fault(sequence_fault)

        if self.state == "disarmed":
            if not arm_request:
                return self._decision("disarmed")
            if max(abs(value) for value in lowstate["full_dq"]) > self.max_takeover_velocity_rad_s:
                return self._fault("takeover_state_moving")
            self.state = "zero_hold"
            self.takeover_q = list(lowstate["full_q"])
            self.last_full_q = list(self.takeover_q)
            self.last_publish_time = now
            self.expected_mode = lowstate["mode_machine"]
            self.zero_hold_count = 1
            self.following_error_started_at = None
            return self._decision(
                "zero_hold",
                should_publish=True,
                full_q=self.takeover_q,
                first=True,
            )

        if lowstate["mode_machine"] != self.expected_mode:
            return self._fault("mode_changed")
        if now - self.last_publish_time >= self.max_publish_gap:
            return self._fault("publisher_gap")
        following_fault = self._check_following_error(lowstate["full_q"], now)
        if following_fault is not None:
            return self._fault(following_fault)
        drift_fault = self._check_non_arm_drift(lowstate["full_q"])
        if drift_fault is not None:
            return self._fault(drift_fault)
        velocity_fault = self._check_non_arm_velocity(lowstate["full_dq"])
        if velocity_fault is not None:
            return self._fault(velocity_fault)

        if self.state == "zero_hold" and self.zero_hold_count < self.zero_hold_cycles:
            self.zero_hold_count += 1
            self.last_publish_time = now
            self.last_full_q = list(self.takeover_q)
            return self._decision("zero_hold", should_publish=True, full_q=self.takeover_q)

        self.state = "active"
        full_q = self._active_target(candidate["candidate_q"], now)
        self.last_full_q = list(full_q)
        self.last_publish_time = now
        return self._decision("active", should_publish=True, full_q=full_q)


def load_json(path):
    try:
        row = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    return row if isinstance(row, dict) else None


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


def read_available_keys():
    keys = []
    while select.select([sys.stdin.fileno()], [], [], 0.0)[0]:
        value = os.read(sys.stdin.fileno(), 1)
        if not value:
            break
        keys.append(value.decode(errors="ignore").lower())
    return keys


def validate_cli(args):
    for name in (
        "frequency",
        "lowstate_timeout",
        "command_timeout",
        "deadman_timeout",
        "sequence_freeze_timeout",
        "max_xr_age",
        "max_following_error_rad",
        "following_error_duration",
        "zero_hold_duration",
        "max_arm_velocity_rad_s",
        "max_non_arm_drift_rad",
        "max_takeover_velocity_rad_s",
        "max_non_arm_velocity_rad_s",
        "max_publish_gap",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.duration) or args.duration < 0.0:
        raise ValueError("--duration must be finite and non-negative")


def main():
    args = parse_args()
    validate_cli(args)
    gate = ArmPublisherSafetyGate(
        lowstate_timeout=args.lowstate_timeout,
        command_timeout=args.command_timeout,
        deadman_timeout=args.deadman_timeout,
        sequence_freeze_timeout=args.sequence_freeze_timeout,
        max_xr_age=args.max_xr_age,
        max_following_error_rad=args.max_following_error_rad,
        following_error_duration=args.following_error_duration,
        zero_hold_cycles=max(1, math.ceil(args.zero_hold_duration * args.frequency)),
        max_arm_velocity_rad_s=args.max_arm_velocity_rad_s,
        max_non_arm_drift_rad=args.max_non_arm_drift_rad,
        max_takeover_velocity_rad_s=args.max_takeover_velocity_rad_s,
        max_non_arm_velocity_rad_s=args.max_non_arm_velocity_rad_s,
        max_publish_gap=args.max_publish_gap,
    )
    lowstate_path = Path(args.lowstate_shadow)
    bridge_path = Path(args.bridge_command)
    deadman_path = Path(args.deadman_state)
    output_path = Path(args.output)
    old_terminal = termios.tcgetattr(sys.stdin.fileno()) if sys.stdin.isatty() else None
    if old_terminal is not None:
        tty.setcbreak(sys.stdin.fileno())
    arm_request = False
    reset_request = False
    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    next_tick = started
    last_status = None
    print("[R1 ARM PUBLISHER SHADOW] NO ROBOT OUTPUT; press r to request shadow takeover", flush=True)
    print("[R1 ARM PUBLISHER SHADOW] press x with deadman released to reset a fault; q stops", flush=True)
    try:
        while deadline is None or time.monotonic() < deadline:
            for key in read_available_keys():
                if key in ("q", "\x03"):
                    return
                if key == "r":
                    arm_request = True
                if key == "x":
                    reset_request = True
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.002))
                continue
            next_tick = max(next_tick + 1.0 / args.frequency, now)
            lowstate_row = load_json(lowstate_path)
            bridge_row = load_json(bridge_path)
            deadman_row = load_json(deadman_path)
            reset_this_tick = reset_request
            reset_request = False
            if reset_this_tick and gate.reset_fault(deadman_row, now):
                arm_request = False
            request_this_tick = arm_request
            arm_request = False
            decision = gate.step(
                lowstate_row,
                bridge_row,
                deadman_row,
                now,
                arm_request=request_this_tick,
            )
            payload = {
                "schema": SHADOW_SCHEMA,
                "mode": "shadow",
                "physical_publishing_enabled": False,
                "required_platform_profile": REQUIRED_PLATFORM_PROFILE,
                "required_collision_model": REQUIRED_COLLISION_MODEL,
                "required_collision_model_sha256": REQUIRED_COLLISION_MODEL_SHA256,
                "required_o6_collision_pose": REQUIRED_O6_COLLISION_POSE,
                "would_publish": decision["should_publish"],
                **decision,
                "updated_monotonic_ns": time.monotonic_ns(),
            }
            atomic_write_json(output_path, payload)
            if decision["status"] != last_status:
                print(
                    f"[R1 ARM PUBLISHER SHADOW] status={decision['status']} "
                    f"fault_latched={decision['fault_latched']} "
                    f"would_publish={decision['should_publish']}",
                    flush=True,
                )
                last_status = decision["status"]
    finally:
        if old_terminal is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_terminal)
        atomic_write_json(
            output_path,
            {
                "schema": SHADOW_SCHEMA,
                "mode": "shadow",
                "physical_publishing_enabled": False,
                "required_platform_profile": REQUIRED_PLATFORM_PROFILE,
                "would_publish": False,
                "status": "stopped",
                "updated_monotonic_ns": time.monotonic_ns(),
            },
        )
        print("[R1 ARM PUBLISHER SHADOW] stopped; no robot command path exists", flush=True)


if __name__ == "__main__":
    main()
