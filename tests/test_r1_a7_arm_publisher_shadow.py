import importlib.util
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHER_PATH = REPO_ROOT / "teleop" / "r1_a7_arm_publisher_shadow.py"
spec = importlib.util.spec_from_file_location("r1_a7_arm_publisher_shadow", PUBLISHER_PATH)
publisher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = publisher
spec.loader.exec_module(publisher)


NOW = 100.0
ARM_INDICES = list(range(15, 29))


def lowstate(now=NOW, sequence=1, mode_machine=4, full_q=None, **updates):
    if full_q is None:
        full_q = [index / 100.0 for index in range(35)]
    row = {
        "schema": "r1_a7_lowstate_shadow_v1",
        "sequence": sequence,
        "sample_monotonic_ns": int(now * 1_000_000_000),
        "published_monotonic_ns": int(now * 1_000_000_000),
        "mode_machine": mode_machine,
        "crc_valid": True,
        "motor_count": 35,
        "waist_index": 13,
        "left_arm_indices": list(range(15, 22)),
        "right_arm_indices": list(range(22, 29)),
        "head_indices": [29, 30],
        "full_q": list(full_q),
        "full_dq": [0.0] * 35,
        "arm_q": [full_q[index] for index in ARM_INDICES],
        "arm_dq": [0.0] * 14,
        "waist_yaw_q": full_q[13],
        "head_q": [full_q[29], full_q[30]],
    }
    row.update(updates)
    return row


def bridge_candidate(
    now=NOW,
    sequence=1,
    candidate_q=None,
    tracking_age_ms=1.0,
    **updates,
):
    if candidate_q is None:
        candidate_q = [0.2 + index / 100.0 for index in range(14)]
    row = {
        "schema": "r1_a7_arm_command_intent_v1",
        "decision": "new",
        "platform_profile": "r1_a7_dual_linker_o6_v1",
        "physical_publishing_enabled": False,
        "o6_enabled": False,
        "source_armed": True,
        "collision_model": "r1_a7_linker_o6_open",
        "collision_model_sha256": publisher.REQUIRED_COLLISION_MODEL_SHA256,
        "o6_collision_pose": "fresh_verified_open",
        "o6_state_sequence": sequence,
        "o6_state_age_ms": 1.0,
        "o6_angles_raw": {
            "left": [254, 255, 254, 254, 254, 254],
            "right": [254, 255, 254, 254, 254, 254],
        },
        "source_sequence": sequence,
        "mode_machine": 4,
        "source_tracking_age_ms": tracking_age_ms,
        "candidate_arm_q": list(candidate_q),
        "bridge_monotonic_ns": int(now * 1_000_000_000),
    }
    row.update(updates)
    return row


def deadman(now=NOW, sequence=1, held=True, **updates):
    row = {
        "schema": "r1_a7_arm_deadman_v1",
        "sequence": sequence,
        "held": held,
        "updated_monotonic_ns": int(now * 1_000_000_000),
    }
    row.update(updates)
    return row


class ArmPublisherFaultInjectionTest(unittest.TestCase):
    def make_gate(self, **updates):
        options = {
            "lowstate_timeout": 0.10,
            "command_timeout": 0.10,
            "deadman_timeout": 0.10,
            "sequence_freeze_timeout": 0.12,
            "max_xr_age": 0.25,
            "max_following_error_rad": 0.01,
            "following_error_duration": 0.02,
            "zero_hold_cycles": 1,
        }
        options.update(updates)
        return publisher.ArmPublisherSafetyGate(**options)

    def assert_fault(self, decision, status):
        self.assertEqual(decision["status"], status)
        self.assertTrue(decision["fault_latched"])
        self.assertFalse(decision["should_publish"])
        self.assertIsNone(decision["full_q"])

    def arm_once(self, gate, actual_q=None):
        decision = gate.step(
            lowstate(full_q=actual_q),
            bridge_candidate(),
            deadman(),
            NOW,
            arm_request=True,
        )
        self.assertEqual(decision["status"], "zero_hold")
        self.assertTrue(decision["should_publish"])
        return decision

    def test_boundary_validators_require_exact_35_axis_contracts(self):
        normalized, status = publisher.validate_full_lowstate(lowstate(), NOW, 0.10)
        self.assertEqual(status, "valid")
        self.assertEqual(len(normalized["full_q"]), 35)
        self.assertEqual(len(normalized["full_dq"]), 35)

        normalized, status = publisher.validate_bridge_candidate(
            bridge_candidate(), NOW, 0.10
        )
        self.assertEqual(status, "valid")
        self.assertEqual(len(normalized["candidate_q"]), 14)

        normalized, status = publisher.validate_deadman(deadman(), NOW, 0.10)
        self.assertEqual(status, "valid")
        self.assertTrue(normalized["held"])

        invalid = lowstate(full_q=[0.0] * 34)
        self.assertEqual(
            publisher.validate_full_lowstate(invalid, NOW, 0.10)[1],
            "invalid_state",
        )
        invalid_mapping = lowstate(left_arm_indices=list(range(14, 21)))
        self.assertEqual(
            publisher.validate_full_lowstate(invalid_mapping, NOW, 0.10)[1],
            "mapping_rejected",
        )
        invalid_crc = lowstate(crc_valid=False)
        self.assertEqual(
            publisher.validate_full_lowstate(invalid_crc, NOW, 0.10)[1],
            "crc_rejected",
        )

        for field, value in (
            ("schema", "wrong"),
            ("platform_profile", "wrong"),
            ("decision", "fault"),
            ("physical_publishing_enabled", True),
            ("o6_enabled", True),
            ("collision_model", "r1_a7_urdf_without_o6"),
            ("collision_model_sha256", "0" * 64),
            ("o6_collision_pose", "not_integrated"),
            ("o6_state_sequence", 0),
            ("o6_state_age_ms", 250.0),
            ("o6_angles_raw", {"left": [249] * 6, "right": [255] * 6}),
        ):
            with self.subTest(field=field):
                rejected = bridge_candidate(**{field: value})
                normalized, status = publisher.validate_bridge_candidate(
                    rejected, NOW, 0.10
                )
                self.assertIsNone(normalized)
                self.assertNotEqual(status, "valid")

    def test_first_command_is_exact_fresh_actual_full_body_hold(self):
        actual_q = [(-0.17 + index / 100.0) for index in range(35)]
        decision = self.arm_once(self.make_gate(), actual_q)
        self.assertTrue(decision["first_command_matches_actual"])
        self.assertEqual(decision["full_q"], actual_q)
        self.assertEqual(len(decision["full_q"]), 35)
        self.assertNotEqual(
            [decision["full_q"][index] for index in ARM_INDICES],
            bridge_candidate()["candidate_arm_q"],
        )

    def test_every_zero_hold_cycle_replays_one_actual_snapshot(self):
        actual_q = [(-0.17 + index / 100.0) for index in range(35)]
        gate = self.make_gate(zero_hold_cycles=3)
        first = gate.step(
            lowstate(full_q=actual_q),
            bridge_candidate(candidate_q=[0.2] * 14),
            deadman(),
            NOW,
            arm_request=True,
        )
        self.assertEqual(first["status"], "zero_hold")
        self.assertEqual(first["full_q"], actual_q)

        for index in (2, 3):
            now = NOW + index * 0.01
            measured_q = list(actual_q)
            measured_q[15] += index * 0.001
            decision = gate.step(
                lowstate(now, sequence=index, full_q=measured_q),
                bridge_candidate(now, sequence=index, candidate_q=[0.3] * 14),
                deadman(now, sequence=index),
                now,
            )
            self.assertEqual(decision["status"], "zero_hold")
            self.assertEqual(decision["full_q"], actual_q)

    def test_active_command_changes_only_arm_indices(self):
        actual_q = [(-0.17 + index / 100.0) for index in range(35)]
        target = [actual_q[index] + 0.005 for index in ARM_INDICES]
        gate = self.make_gate(max_following_error_rad=0.02)
        self.arm_once(gate, actual_q)
        decision = gate.step(
            lowstate(NOW + 0.02, sequence=2, full_q=actual_q),
            bridge_candidate(NOW + 0.02, sequence=2, candidate_q=target),
            deadman(NOW + 0.02, sequence=2),
            NOW + 0.02,
        )
        self.assertEqual(decision["status"], "active")
        self.assertTrue(decision["should_publish"])
        for motor_index, expected in zip(ARM_INDICES, target):
            self.assertAlmostEqual(decision["full_q"][motor_index], expected)
        non_arm_indices = [index for index in range(35) if index not in ARM_INDICES]
        self.assertEqual(
            [decision["full_q"][index] for index in non_arm_indices],
            [actual_q[index] for index in non_arm_indices],
        )

    def test_xr_stale_latches_and_stops(self):
        gate = self.make_gate()
        self.arm_once(gate)
        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.02, sequence=2),
                bridge_candidate(
                    NOW + 0.02,
                    sequence=2,
                    tracking_age_ms=251.0,
                ),
                deadman(NOW + 0.02, sequence=2),
                NOW + 0.02,
            ),
            "xr_stale",
        )

    def test_command_age_latches_and_stops(self):
        gate = self.make_gate()
        self.arm_once(gate)
        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.11, sequence=2),
                bridge_candidate(NOW, sequence=2),
                deadman(NOW + 0.11, sequence=2),
                NOW + 0.11,
            ),
            "command_stale",
        )

    def test_lowstate_sample_age_latches_and_stops(self):
        gate = self.make_gate()
        self.arm_once(gate)
        stale = lowstate(NOW + 0.11, sequence=2)
        stale["sample_monotonic_ns"] = int(NOW * 1_000_000_000)
        self.assert_fault(
            gate.step(
                stale,
                bridge_candidate(NOW + 0.11, sequence=2),
                deadman(NOW + 0.11, sequence=2),
                NOW + 0.11,
            ),
            "lowstate_stale",
        )

    def test_sequence_freeze_cannot_be_hidden_by_rewriting_timestamp(self):
        gate = self.make_gate()
        self.arm_once(gate)
        decision = gate.step(
            lowstate(NOW + 0.05, sequence=2),
            bridge_candidate(NOW + 0.05, sequence=1),
            deadman(NOW + 0.05, sequence=2),
            NOW + 0.05,
        )
        self.assertFalse(decision["fault_latched"])
        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.13, sequence=3),
                bridge_candidate(NOW + 0.13, sequence=1),
                deadman(NOW + 0.13, sequence=3),
                NOW + 0.13,
            ),
            "sequence_frozen",
        )

    def test_frozen_sequence_cannot_arm_from_disarmed(self):
        gate = self.make_gate()
        decision = gate.step(
            lowstate(),
            bridge_candidate(),
            deadman(),
            NOW,
        )
        self.assertEqual(decision["status"], "disarmed")

        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.13, sequence=2),
                bridge_candidate(NOW + 0.13, sequence=1),
                deadman(NOW + 0.13, sequence=2),
                NOW + 0.13,
                arm_request=True,
            ),
            "sequence_frozen",
        )

    def test_lowstate_sequence_freeze_latches_even_if_timestamp_is_rewritten(self):
        gate = self.make_gate()
        self.arm_once(gate)
        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.13, sequence=1),
                bridge_candidate(NOW + 0.13, sequence=2),
                deadman(NOW + 0.13, sequence=2),
                NOW + 0.13,
            ),
            "lowstate_sequence_frozen",
        )

    def test_deadman_sequence_freeze_latches_even_if_timestamp_is_rewritten(self):
        gate = self.make_gate()
        self.arm_once(gate)
        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.13, sequence=2),
                bridge_candidate(NOW + 0.13, sequence=2),
                deadman(NOW + 0.13, sequence=1),
                NOW + 0.13,
            ),
            "deadman_sequence_frozen",
        )

    def test_all_input_sequence_regressions_latch(self):
        cases = (
            ("lowstate", "lowstate_sequence_regression"),
            ("deadman", "deadman_sequence_regression"),
            ("command", "sequence_regression"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                gate = self.make_gate()
                gate.step(
                    lowstate(sequence=2),
                    bridge_candidate(sequence=2),
                    deadman(sequence=2),
                    NOW,
                    arm_request=True,
                )
                lowstate_sequence = 1 if source == "lowstate" else 3
                command_sequence = 1 if source == "command" else 3
                deadman_sequence = 1 if source == "deadman" else 3
                self.assert_fault(
                    gate.step(
                        lowstate(NOW + 0.02, sequence=lowstate_sequence),
                        bridge_candidate(NOW + 0.02, sequence=command_sequence),
                        deadman(NOW + 0.02, sequence=deadman_sequence),
                        NOW + 0.02,
                    ),
                    expected,
                )

    def test_fault_reset_preserves_all_frozen_source_baselines(self):
        cases = (
            ("lowstate", "lowstate_sequence_frozen"),
            ("deadman", "deadman_sequence_frozen"),
            ("command", "sequence_frozen"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                gate = self.make_gate()
                self.arm_once(gate)
                gate.step(
                    lowstate(NOW + 0.13, sequence=1 if source == "lowstate" else 2),
                    bridge_candidate(NOW + 0.13, sequence=1 if source == "command" else 2),
                    deadman(NOW + 0.13, sequence=1 if source == "deadman" else 2),
                    NOW + 0.13,
                )
                reset_deadman_sequence = 1 if source == "deadman" else 3
                self.assertTrue(
                    gate.reset_fault(
                        deadman(
                            NOW + 0.14,
                            sequence=reset_deadman_sequence,
                            held=False,
                        ),
                        NOW + 0.14,
                    )
                )
                self.assert_fault(
                    gate.step(
                        lowstate(NOW + 0.15, sequence=1 if source == "lowstate" else 3),
                        bridge_candidate(
                            NOW + 0.15,
                            sequence=1 if source == "command" else 3,
                        ),
                        deadman(NOW + 0.15, sequence=1 if source == "deadman" else 4),
                        NOW + 0.15,
                        arm_request=True,
                    ),
                    expected,
                )

    def test_reset_is_rejected_when_no_fault_is_latched(self):
        gate = self.make_gate()
        self.assertFalse(gate.reset_fault(deadman(held=False), NOW))

    def test_takeover_requires_all_35_axes_to_be_stationary(self):
        for motor_index in range(35):
            with self.subTest(motor_index=motor_index):
                moving_dq = [0.0] * 35
                moving_dq[motor_index] = 0.051
                self.assert_fault(
                    self.make_gate().step(
                        lowstate(full_dq=moving_dq),
                        bridge_candidate(),
                        deadman(),
                        NOW,
                        arm_request=True,
                    ),
                    "takeover_state_moving",
                )

    def test_every_non_arm_joint_drift_latches(self):
        for motor_index in (index for index in range(35) if index not in ARM_INDICES):
            with self.subTest(motor_index=motor_index):
                gate = self.make_gate()
                actual_q = [0.0] * 35
                self.arm_once(gate, actual_q)
                drifted_q = list(actual_q)
                drifted_q[motor_index] = 0.021
                self.assert_fault(
                    gate.step(
                        lowstate(NOW + 0.02, sequence=2, full_q=drifted_q),
                        bridge_candidate(NOW + 0.02, sequence=2),
                        deadman(NOW + 0.02, sequence=2),
                        NOW + 0.02,
                    ),
                    "non_arm_drift",
                )

    def test_non_arm_motion_after_takeover_latches(self):
        gate = self.make_gate()
        self.arm_once(gate, [0.0] * 35)
        moving_dq = [0.0] * 35
        moving_dq[13] = 0.051
        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.02, sequence=2, full_q=[0.0] * 35, full_dq=moving_dq),
                bridge_candidate(NOW + 0.02, sequence=2),
                deadman(NOW + 0.02, sequence=2),
                NOW + 0.02,
            ),
            "non_arm_moving",
        )

    def test_publisher_loop_gap_latches_instead_of_allowing_a_large_step(self):
        gate = self.make_gate(max_publish_gap=0.05)
        self.arm_once(gate, [0.0] * 35)
        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.051, sequence=2, full_q=[0.0] * 35),
                bridge_candidate(
                    NOW + 0.051,
                    sequence=2,
                    candidate_q=[0.2] * 14,
                ),
                deadman(NOW + 0.051, sequence=2),
                NOW + 0.051,
            ),
            "publisher_gap",
        )

    def test_mode_change_after_takeover_latches_and_stops(self):
        gate = self.make_gate()
        self.arm_once(gate)
        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.02, sequence=2, mode_machine=5),
                bridge_candidate(NOW + 0.02, sequence=2, mode_machine=5),
                deadman(NOW + 0.02, sequence=2),
                NOW + 0.02,
            ),
            "mode_changed",
        )

    def test_persistent_following_error_latches_and_stops(self):
        actual_q = [0.0] * 35
        target = [0.02] * 14
        gate = self.make_gate(max_arm_velocity_rad_s=1.0)
        self.arm_once(gate, actual_q)

        first = gate.step(
            lowstate(NOW + 0.02, sequence=2, full_q=actual_q),
            bridge_candidate(NOW + 0.02, sequence=2, candidate_q=target),
            deadman(NOW + 0.02, sequence=2),
            NOW + 0.02,
        )
        self.assertFalse(first["fault_latched"])
        self.assertTrue(first["should_publish"])

        second = gate.step(
            lowstate(NOW + 0.04, sequence=3, full_q=actual_q),
            bridge_candidate(NOW + 0.04, sequence=3, candidate_q=target),
            deadman(NOW + 0.04, sequence=3),
            NOW + 0.04,
        )
        self.assertFalse(second["fault_latched"])

        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.061, sequence=4, full_q=actual_q),
                bridge_candidate(NOW + 0.061, sequence=4, candidate_q=target),
                deadman(NOW + 0.061, sequence=4),
                NOW + 0.061,
            ),
            "following_error",
        )

    def test_deadman_release_latches_and_stops(self):
        gate = self.make_gate()
        self.arm_once(gate)
        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.02, sequence=2),
                bridge_candidate(NOW + 0.02, sequence=2),
                deadman(NOW + 0.02, sequence=2, held=False),
                NOW + 0.02,
            ),
            "deadman_released",
        )

    def test_deadman_heartbeat_stale_latches_and_stops(self):
        gate = self.make_gate()
        self.arm_once(gate)
        self.assert_fault(
            gate.step(
                lowstate(NOW + 0.11, sequence=2),
                bridge_candidate(NOW + 0.11, sequence=2),
                deadman(NOW, sequence=2),
                NOW + 0.11,
            ),
            "deadman_stale",
        )

    def test_fault_remains_latched_without_a_new_arm_request(self):
        gate = self.make_gate()
        self.arm_once(gate)
        gate.step(
            lowstate(NOW + 0.02, sequence=2),
            bridge_candidate(NOW + 0.02, sequence=2),
            deadman(NOW + 0.02, sequence=2, held=False),
            NOW + 0.02,
        )
        decision = gate.step(
            lowstate(NOW + 0.04, sequence=3),
            bridge_candidate(NOW + 0.04, sequence=3),
            deadman(NOW + 0.04, sequence=3),
            NOW + 0.04,
            arm_request=True,
        )
        self.assert_fault(decision, "fault_latched")

    def test_shadow_module_contains_no_dds_or_device_io(self):
        source = PUBLISHER_PATH.read_text(encoding="utf-8")
        for token in (
            "unitree_sdk2py",
            "ChannelFactoryInitialize",
            "ChannelPublisher",
            "rt/lowcmd",
            "import socket",
            "import serial",
            "/dev/tty",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, source)
        self.assertLess(source.index("reset_request = False"), source.index("while deadline"))
        self.assertIn("reset_this_tick = reset_request", source)
        self.assertIn("request_this_tick = arm_request", source)
        self.assertIn("arm_request = False", source)


if __name__ == "__main__":
    unittest.main()
