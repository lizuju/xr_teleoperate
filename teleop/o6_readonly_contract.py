import math
from collections import deque


SOURCE_SCHEMA = "linker_o6_fc04_pair_v1"
STATE_SCHEMA = "linker_o6_readonly_state_v1"
CONFIGURATION_ID = "r1_a7_dual_linker_o6_v1"
AXIS_ORDER = [
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]
EXPECTED_HANDS = {
    "left": {"device_path": "/dev/ttyHAND0", "slave_id": 40, "direction_code": 76},
    "right": {"device_path": "/dev/ttyHAND1", "slave_id": 39, "direction_code": 82},
}
DRIVER_UPPER_RAD = {
    "left": [0.58, 1.30, 1.60, 1.60, 1.60, 1.60],
    "right": [0.58, 1.36, 1.60, 1.60, 1.60, 1.60],
}


def _integer(value, minimum=0, maximum=2**63 - 1):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= maximum
    )


def _register_array(value, maximum=65535):
    return (
        isinstance(value, list)
        and len(value) == 6
        and all(_integer(item, 0, maximum) for item in value)
    )


def validate_source_packet(row, previous_boot_id=None, previous_sequence=None):
    if not isinstance(row, dict) or row.get("schema") != SOURCE_SCHEMA:
        raise ValueError("schema_rejected")
    if row.get("configuration_id") != CONFIGURATION_ID:
        raise ValueError("configuration_rejected")
    if row.get("hardware_axis_order") != AXIS_ORDER:
        raise ValueError("axis_order_rejected")
    if row.get("actuation_enabled") is not False or row.get("writes_enabled") is not False:
        raise ValueError("actuation_contract_rejected")
    protocol = row.get("protocol")
    if protocol != {
        "transport": "modbus_rtu",
        "baudrate": 4_000_000,
        "function_code": 4,
        "start_register": 0,
        "register_count": 45,
    }:
        raise ValueError("protocol_rejected")
    boot_id = row.get("source_boot_id")
    if not isinstance(boot_id, str) or not boot_id or len(boot_id) > 128:
        raise ValueError("invalid_boot_id")
    if previous_boot_id is not None and boot_id != previous_boot_id:
        raise ValueError("source_boot_changed")
    sequence = row.get("sequence")
    if not _integer(sequence, 1):
        raise ValueError("invalid_sequence")
    if previous_sequence is not None and sequence <= previous_sequence:
        raise ValueError("sequence_nonincreasing")
    source_ns = row.get("source_monotonic_ns")
    if not _integer(source_ns, 1):
        raise ValueError("invalid_source_timestamp")
    pair_skew_ms = row.get("pair_skew_ms")
    if (
        not isinstance(pair_skew_ms, (int, float))
        or isinstance(pair_skew_ms, bool)
        or not math.isfinite(float(pair_skew_ms))
        or not 0.0 <= float(pair_skew_ms) <= 50.0
    ):
        raise ValueError("pair_skew_rejected")
    hands = row.get("hands")
    if not isinstance(hands, dict) or set(hands) != set(EXPECTED_HANDS):
        raise ValueError("hand_set_rejected")
    for side, expected in EXPECTED_HANDS.items():
        hand = hands.get(side)
        if not isinstance(hand, dict):
            raise ValueError(f"{side}_hand_rejected")
        for key, expected_value in expected.items():
            if hand.get(key) != expected_value:
                raise ValueError(f"{side}_{key}_rejected")
        if hand.get("read_ok") is not True or hand.get("crc_valid") is not True:
            raise ValueError(f"{side}_read_rejected")
        for key in ("read_started_monotonic_ns", "read_completed_monotonic_ns"):
            if not _integer(hand.get(key), 1):
                raise ValueError(f"{side}_{key}_rejected")
        if hand["read_completed_monotonic_ns"] < hand["read_started_monotonic_ns"]:
            raise ValueError(f"{side}_read_time_rejected")
        if hand.get("freedom") != 6 or hand.get("version") != 3:
            raise ValueError(f"{side}_identity_rejected")
        for key in (
            "angles_raw",
            "torques_raw",
            "speeds_raw",
            "temperatures_raw",
            "errors_raw",
        ):
            if not _register_array(hand.get(key), 255 if key == "angles_raw" else 65535):
                raise ValueError(f"{side}_{key}_rejected")
        for key in ("hardware_version", "software_version", "mechanical_version"):
            value = hand.get(key)
            if not isinstance(value, str) or not value or len(value) > 64:
                raise ValueError(f"{side}_{key}_rejected")
    return row


def validate_state_frame(row, now, timeout):
    if not isinstance(row, dict) or row.get("schema") != STATE_SCHEMA:
        return None, "schema_rejected", None
    if row.get("receiver_status") != "valid":
        return None, f"receiver_{row.get('receiver_status', 'invalid')}", None
    received_ns = row.get("received_monotonic_ns")
    if not _integer(received_ns, 1):
        return None, "invalid_received_timestamp", None
    age = now - received_ns / 1_000_000_000.0
    if age < 0.0:
        return None, "future_timestamp", age
    if age >= timeout:
        return None, "stale", age
    try:
        source_row = dict(row)
        source_row["schema"] = SOURCE_SCHEMA
        validate_source_packet(source_row)
    except ValueError as error:
        return None, str(error), age
    if any(value != 0 for side in EXPECTED_HANDS for value in row["hands"][side]["errors_raw"]):
        return None, "device_error", age
    return row, "valid", age


def o6_joint_positions(angles_by_side):
    positions = {}
    for side, prefix in (("left", "lh_"), ("right", "rh_")):
        raw = angles_by_side[side]
        closure = [(255 - value) / 255.0 for value in raw]
        driver = [value * upper for value, upper in zip(closure, DRIVER_UPPER_RAD[side])]
        positions[prefix + "thumb_cmc_pitch"] = driver[0]
        positions[prefix + "thumb_cmc_yaw"] = driver[1]
        positions[prefix + "thumb_ip"] = driver[0] * (2.29 if side == "left" else 1.86)
        for index, finger in enumerate(("index", "middle", "ring", "pinky"), start=2):
            positions[prefix + finger + "_mcp_pitch"] = driver[index]
            positions[prefix + finger + "_dip"] = driver[index] * 0.89
    return positions


class O6OpenPoseGate:
    def __init__(self, minimum_raw=250, stability_frames=3, maximum_span_raw=1):
        self.minimum_raw = minimum_raw
        self.stability_frames = stability_frames
        self.maximum_span_raw = maximum_span_raw
        self.samples = deque(maxlen=stability_frames)
        self.last_sequence = None
        self.last_boot_id = None
        self.verified = False
        self.angles_by_side = None

    def reset(self):
        self.samples.clear()
        self.last_sequence = None
        self.last_boot_id = None
        self.verified = False
        self.angles_by_side = None

    def evaluate(self, row, now, timeout):
        state, status, age = validate_state_frame(row, now, timeout)
        if status != "valid":
            self.reset()
            return None, status, age, None
        sequence = state["sequence"]
        boot_id = state["source_boot_id"]
        if self.last_boot_id is not None and boot_id != self.last_boot_id:
            self.reset()
            return None, "source_boot_changed", age, sequence
        if self.last_sequence is not None and sequence < self.last_sequence:
            self.reset()
            return None, "sequence_regression", age, sequence
        if sequence == self.last_sequence:
            if self.verified:
                return o6_joint_positions(self.angles_by_side), "fresh_verified_open", age, sequence
            return None, "waiting_for_stable_open", age, sequence

        angles = {
            side: list(state["hands"][side]["angles_raw"])
            for side in EXPECTED_HANDS
        }
        if any(value < self.minimum_raw for values in angles.values() for value in values):
            self.reset()
            return None, "not_open", age, sequence
        flattened = angles["left"] + angles["right"]
        self.samples.append(flattened)
        self.last_sequence = sequence
        self.last_boot_id = boot_id
        self.verified = False
        if len(self.samples) < self.stability_frames:
            return None, "waiting_for_stable_open", age, sequence
        for axis in range(12):
            values = [sample[axis] for sample in self.samples]
            if max(values) - min(values) > self.maximum_span_raw:
                return None, "open_pose_unstable", age, sequence
        self.verified = True
        self.angles_by_side = angles
        return o6_joint_positions(angles), "fresh_verified_open", age, sequence

    def matches_locked_pose(self, angles_by_side, maximum_delta_raw=1):
        if not self.verified or self.angles_by_side is None:
            return False
        return all(
            abs(current - locked) <= maximum_delta_raw
            for side in EXPECTED_HANDS
            for current, locked in zip(angles_by_side[side], self.angles_by_side[side])
        )
