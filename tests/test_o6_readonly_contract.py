from pathlib import Path
import unittest

from teleop import o6_readonly_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "pc2" / "o6_fc04_source.cpp"
RECEIVER_PATH = REPO_ROOT / "teleop" / "o6_readonly_state_receiver.py"


def hand(side, angles=None):
    expected = contract.EXPECTED_HANDS[side]
    return {
        **expected,
        "read_started_monotonic_ns": 1_000_000_000,
        "read_completed_monotonic_ns": 1_001_000_000,
        "read_ok": True,
        "crc_valid": True,
        "freedom": 6,
        "version": 3,
        "device_number": "1.2.3",
        "hardware_version": "3.0.0",
        "software_version": "3.0.4",
        "mechanical_version": "1.0.0",
        "angles_raw": angles or [254, 255, 254, 254, 254, 254],
        "torques_raw": [0] * 6,
        "speeds_raw": [0] * 6,
        "temperatures_raw": [25] * 6,
        "errors_raw": [0] * 6,
    }


def source_packet(sequence=1, angles=None):
    return {
        "schema": contract.SOURCE_SCHEMA,
        "configuration_id": contract.CONFIGURATION_ID,
        "source_boot_id": "boot-a",
        "sequence": sequence,
        "source_monotonic_ns": 1_000_000_000 + sequence,
        "pair_skew_ms": 1.0,
        "actuation_enabled": False,
        "writes_enabled": False,
        "protocol": {
            "transport": "modbus_rtu",
            "baudrate": 4_000_000,
            "function_code": 4,
            "start_register": 0,
            "register_count": 45,
        },
        "hardware_axis_order": contract.AXIS_ORDER,
        "hands": {
            "left": hand("left", angles),
            "right": hand("right", angles),
        },
    }


def state_frame(sequence=1, received=10.0, angles=None):
    row = source_packet(sequence, angles)
    row["schema"] = contract.STATE_SCHEMA
    row["receiver_status"] = "valid"
    row["received_monotonic_ns"] = int(received * 1_000_000_000)
    return row


class O6ReadonlyContractTest(unittest.TestCase):
    def test_exact_fc04_identity_and_mapping_contract(self):
        row = contract.validate_source_packet(source_packet())
        self.assertEqual(row["hands"]["left"]["slave_id"], 40)
        self.assertEqual(row["hands"]["right"]["slave_id"], 39)
        self.assertEqual(row["protocol"]["function_code"], 4)
        self.assertFalse(row["actuation_enabled"])
        self.assertFalse(row["writes_enabled"])

    def test_wrong_identity_mapping_crc_and_sequence_are_rejected(self):
        checks = []
        wrong_id = source_packet()
        wrong_id["hands"]["left"]["slave_id"] = 39
        checks.append((wrong_id, "left_slave_id_rejected"))
        wrong_direction = source_packet()
        wrong_direction["hands"]["right"]["direction_code"] = 76
        checks.append((wrong_direction, "right_direction_code_rejected"))
        wrong_crc = source_packet()
        wrong_crc["hands"]["left"]["crc_valid"] = False
        checks.append((wrong_crc, "left_read_rejected"))
        wrong_axis = source_packet()
        wrong_axis["hardware_axis_order"] = list(reversed(contract.AXIS_ORDER))
        checks.append((wrong_axis, "axis_order_rejected"))
        for row, expected in checks:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    contract.validate_source_packet(row)
        with self.assertRaisesRegex(ValueError, "sequence_nonincreasing"):
            contract.validate_source_packet(source_packet(2), "boot-a", 2)

    def test_host_arrival_time_is_the_freshness_clock(self):
        row = state_frame(received=10.0)
        self.assertEqual(contract.validate_state_frame(row, 10.249, 0.25)[1], "valid")
        self.assertEqual(contract.validate_state_frame(row, 10.25, 0.25)[1], "stale")
        future = state_frame(received=10.1)
        self.assertEqual(contract.validate_state_frame(future, 10.0, 0.25)[1], "future_timestamp")

    def test_open_pose_requires_three_new_stable_frames(self):
        gate = contract.O6OpenPoseGate()
        self.assertEqual(gate.evaluate(state_frame(1), 10.0, 0.25)[1], "waiting_for_stable_open")
        self.assertEqual(gate.evaluate(state_frame(1), 10.0, 0.25)[1], "waiting_for_stable_open")
        self.assertEqual(gate.evaluate(state_frame(2), 10.0, 0.25)[1], "waiting_for_stable_open")
        joint_q, status, _, sequence = gate.evaluate(state_frame(3), 10.0, 0.25)
        self.assertEqual(status, "fresh_verified_open")
        self.assertEqual(sequence, 3)
        self.assertEqual(len(joint_q), 22)
        self.assertAlmostEqual(joint_q["lh_thumb_cmc_pitch"], 0.58 / 255.0)
        self.assertAlmostEqual(joint_q["lh_index_dip"], 1.60 * 0.89 / 255.0)
        self.assertEqual(joint_q["rh_thumb_cmc_yaw"], 0.0)

    def test_not_open_unstable_stale_and_device_error_fail_closed(self):
        gate = contract.O6OpenPoseGate()
        self.assertEqual(
            gate.evaluate(state_frame(1, angles=[249, 255, 255, 255, 255, 255]), 10.0, 0.25)[1],
            "not_open",
        )
        error = state_frame(2)
        error["hands"]["right"]["errors_raw"][0] = 1
        self.assertEqual(gate.evaluate(error, 10.0, 0.25)[1], "device_error")
        self.assertEqual(gate.evaluate(state_frame(3, received=9.0), 10.0, 0.25)[1], "stale")

        gate = contract.O6OpenPoseGate()
        gate.evaluate(state_frame(1, angles=[250] * 6), 10.0, 0.25)
        gate.evaluate(state_frame(2, angles=[252] * 6), 10.0, 0.25)
        self.assertEqual(
            gate.evaluate(state_frame(3, angles=[250] * 6), 10.0, 0.25)[1],
            "open_pose_unstable",
        )

    def test_pc2_source_has_no_control_protocol_path(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        for token in (
            "0x10",
            "writeRegisters",
            "setControlState",
            "ChannelPublisher",
            "ChannelSubscriber",
            "rt/linker",
            "LowCmd",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, source)
        self.assertIn("0x04", source)
        self.assertIn('Fc04Port left("/dev/ttyHAND0", 40)', source)
        self.assertIn('Fc04Port right("/dev/ttyHAND1", 39)', source)

    def test_receiver_has_no_device_network_or_robot_output(self):
        source = RECEIVER_PATH.read_text(encoding="utf-8")
        for token in (
            "import socket",
            "import serial",
            "/dev/tty",
            "ChannelPublisher",
            "unitree_sdk2py",
            "subprocess",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, source)
        self.assertIn("for line in sys.stdin", source)


if __name__ == "__main__":
    unittest.main()
