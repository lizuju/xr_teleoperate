import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = REPO_ROOT / "teleop" / "r1_a7_arm_live_bridge.py"
LOWSTATE_PATH = REPO_ROOT / "teleop" / "r1_a7_lowstate_shadow.py"
PUBLISHER_PATH = REPO_ROOT / "teleop" / "r1_a7_arm_publisher_shadow.py"
spec = importlib.util.spec_from_file_location("r1_a7_arm_live_bridge", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)
publisher_spec = importlib.util.spec_from_file_location(
    "r1_a7_arm_publisher_shadow", PUBLISHER_PATH
)
publisher = importlib.util.module_from_spec(publisher_spec)
sys.modules[publisher_spec.name] = publisher
publisher_spec.loader.exec_module(publisher)


def frame(now, sequence=1, offset=None, **updates):
    row = {
        "schema": bridge.ARM_SCHEMA,
        "mapping": bridge.ARM_MAPPING,
        "reference": bridge.ARM_REFERENCE,
        "frame": bridge.ARM_FRAME,
        "target_units": "m",
        "target_hand_order": bridge.HAND_ORDER,
        "armed": True,
        "sequence": sequence,
        "monotonic_timestamp": now,
        "published_monotonic_ns": int(now * 1_000_000_000),
        "position_offset_m": offset or [0.0] * 6,
    }
    row.update(updates)
    return row


def lowstate_frame(now, sequence=1, **updates):
    full_q = [0.0] * bridge.MOTOR_COUNT
    full_dq = [0.0] * bridge.MOTOR_COUNT
    row = {
        "schema": bridge.LOWSTATE_SCHEMA,
        "sequence": sequence,
        "sample_monotonic_ns": int(now * 1_000_000_000),
        "published_monotonic_ns": int(now * 1_000_000_000),
        "mode_machine": 4,
        "crc_valid": True,
        "motor_count": bridge.MOTOR_COUNT,
        "arm_indices": bridge.ARM_INDICES,
        "waist_index": bridge.WAIST_INDEX,
        "left_arm_indices": bridge.LEFT_ARM_INDICES,
        "right_arm_indices": bridge.RIGHT_ARM_INDICES,
        "head_indices": bridge.HEAD_INDICES,
        "full_q": full_q,
        "full_dq": full_dq,
    }
    row.update(updates)
    return row


def command_intent(
    status,
    solution=None,
    last_safe=None,
    command_ready=False,
    latched=False,
    collision_verified=False,
):
    return bridge.build_command_intent(
        status=status,
        solution=solution,
        last_safe_arm_q=last_safe,
        command_ready=command_ready,
        gate_latched=latched,
        source_row=frame(10.0),
        sequence=1,
        tracking_age=0.01,
        transport_age=0.02,
        mode_machine=4,
        reference_arm_q=bridge.np.zeros(14),
        reference_fixed_q=bridge.np.zeros(3),
        fixed_joint_drift=0.0,
        collision_model=(
            bridge.VERIFIED_COLLISION_MODEL
            if collision_verified
            else bridge.UNVERIFIED_COLLISION_MODEL
        ),
        o6_collision_pose=(
            bridge.O6_VERIFIED_POSE if collision_verified else bridge.O6_UNVERIFIED_POSE
        ),
        o6_state_sequence=3 if collision_verified else None,
        o6_state_age=0.01 if collision_verified else None,
        o6_angles_raw={
            "left": [254, 255, 254, 254, 254, 254],
            "right": [254, 255, 254, 254, 254, 254],
        }
        if collision_verified
        else None,
        updated_monotonic_ns=10_000_000_000,
    )


class ArmBridgeValidationTest(unittest.TestCase):
    def make_gate(self):
        return bridge.SafetyGate(0.25, 8, 0.35)

    def disarm(self, gate, now=10.0):
        row = frame(now, armed=False, reason="disarmed")
        offset, status, _, _, _ = gate.evaluate(row, now)
        self.assertIsNone(offset)
        self.assertEqual(status, "disarmed")

    def test_requires_disarm_then_two_distinct_fresh_frames(self):
        gate = self.make_gate()
        row = frame(10.0)
        self.assertEqual(gate.evaluate(row, 10.0)[1], "startup_requires_source_disarm")
        self.disarm(gate)
        self.assertEqual(gate.evaluate(frame(10.01, sequence=2), 10.01)[1], "waiting_for_second_fresh_frame")
        offset, status, _, _, _ = gate.evaluate(frame(10.02, sequence=3), 10.02)
        self.assertEqual(status, "ready")
        self.assertEqual(offset.tolist(), [0.0] * 6)

    def test_stale_and_future_timestamps_are_rejected(self):
        row = frame(10.0)
        self.assertEqual(bridge.validate_arm_frame(row, 10.3, 0.25)[1], "stale")
        self.assertEqual(bridge.validate_arm_frame(row, 10.25, 0.25)[1], "stale")
        self.assertEqual(bridge.validate_arm_frame(row, 9.9, 0.25)[1], "future_timestamp")

    def test_only_a_fresh_valid_disarm_can_clear_the_fault_latch(self):
        for field, value, expected in (
            ("schema", "wrong", "schema_rejected"),
            ("mapping", "wrong", "mapping_rejected"),
            ("sequence", 0, "invalid_sequence"),
            ("reason", "unknown", "disarm_reason_rejected"),
        ):
            with self.subTest(field=field):
                gate = self.make_gate()
                gate.latch_fault()
                row = frame(10.0, armed=False, reason="disarmed")
                row[field] = value
                self.assertEqual(gate.evaluate(row, 10.0)[1], expected)
                self.assertTrue(gate.latched)
                self.assertFalse(gate.source_disarm_seen)
        gate = self.make_gate()
        gate.latch_fault()
        missing_reason = frame(10.0, armed=False)
        missing_reason.pop("reason", None)
        self.assertEqual(
            gate.evaluate(missing_reason, 10.0)[1],
            "disarm_reason_rejected",
        )
        self.assertTrue(gate.latched)
        self.assertFalse(gate.source_disarm_seen)
        gate = self.make_gate()
        gate.latch_fault()
        stale_disarm = frame(10.0, armed=False, reason="stale")
        self.assertEqual(gate.evaluate(stale_disarm, 10.3)[1], "stale")
        self.assertTrue(gate.latched)
        self.assertFalse(gate.source_disarm_seen)
        self.disarm(gate, 10.4)
        self.assertFalse(gate.latched)
        self.assertTrue(gate.source_disarm_seen)

    def test_schema_and_coordinate_contract_are_rejected_fail_closed(self):
        checks = {
            "schema": "schema_rejected",
            "mapping": "mapping_rejected",
            "reference": "reference_rejected",
            "frame": "frame_rejected",
            "target_units": "units_rejected",
            "target_hand_order": "hand_order_rejected",
        }
        for field, expected in checks.items():
            with self.subTest(field=field):
                row = frame(10.0)
                row[field] = "wrong"
                self.assertEqual(bridge.validate_arm_frame(row, 10.0, 0.25)[1], expected)

    def test_invalid_targets_are_rejected_without_a_fixed_cartesian_workspace(self):
        invalid = ([0.0] * 5, [0.0] * 5 + [math.nan], [0.0] * 5 + [True])
        for values in invalid:
            with self.subTest(values=values):
                self.assertEqual(
                    bridge.validate_arm_frame(frame(10.0, offset=values), 10.0, 0.25)[1],
                    "invalid_target",
                )
        self.assertEqual(
            bridge.validate_arm_frame(frame(10.0, offset=[0.12, 0.1, 0.0, 0.0, 0.0, 0.0]), 10.0, 0.25)[1],
            "valid",
        )

    def test_extreme_integers_are_rejected_without_overflow(self):
        row = frame(10.0)
        row["sequence"] = 10**10000
        self.assertEqual(bridge.validate_arm_frame(row, 10.0, 0.25)[1], "invalid_sequence")
        row = frame(10.0)
        row["published_monotonic_ns"] = 10**10000
        self.assertEqual(
            bridge.validate_arm_frame(row, 10.0, 0.25)[1],
            "invalid_published_timestamp",
        )
        row = frame(10.0)
        row["monotonic_timestamp"] = 10**10000
        self.assertEqual(
            bridge.validate_arm_frame(row, 10.0, 0.25)[1],
            "invalid_source_timestamp",
        )

    def test_duplicate_is_ignored_and_sequence_fault_latches(self):
        gate = self.make_gate()
        self.disarm(gate)
        gate.evaluate(frame(10.01, sequence=2), 10.01)
        self.assertEqual(gate.evaluate(frame(10.01, sequence=2), 10.01)[1], "duplicate")
        self.assertEqual(gate.evaluate(frame(10.02, sequence=1), 10.02)[1], "sequence_regression")
        self.assertEqual(
            gate.evaluate(frame(10.03, sequence=3), 10.03)[1],
            "fault_latched_waiting_for_source_disarm",
        )
        self.disarm(gate, 10.04)
        self.assertEqual(gate.evaluate(frame(10.05, sequence=4), 10.05)[1], "waiting_for_second_fresh_frame")

    def test_frozen_duplicate_becomes_stale_without_refreshing_freshness(self):
        gate = self.make_gate()
        self.disarm(gate)
        row = frame(10.01, sequence=2)
        gate.evaluate(row, 10.01)
        self.assertEqual(gate.evaluate(row, 10.02)[1], "duplicate")
        self.assertEqual(gate.evaluate(row, 10.27)[1], "stale")
        self.assertTrue(gate.latched)

    def test_new_sequence_with_same_source_sample_is_held_not_latched(self):
        gate = self.make_gate()
        self.disarm(gate)
        gate.evaluate(frame(10.01, sequence=2), 10.01)
        row = frame(10.01, sequence=3)
        row["published_monotonic_ns"] = int(10.02 * 1_000_000_000)
        self.assertEqual(gate.evaluate(row, 10.02)[1], "duplicate_source_sample")
        self.assertFalse(gate.latched)

    def test_target_jump_latches(self):
        gate = self.make_gate()
        self.disarm(gate)
        gate.evaluate(frame(10.01, sequence=2), 10.01)
        row = frame(10.02, sequence=3, offset=[-0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(gate.evaluate(row, 10.02)[1], "target_jump")

    def test_sequence_gap_boundary(self):
        gate = self.make_gate()
        self.disarm(gate)
        gate.evaluate(frame(10.01, sequence=2), 10.01)
        self.assertEqual(gate.evaluate(frame(10.02, sequence=10), 10.02)[1], "ready")
        self.assertEqual(gate.evaluate(frame(10.03, sequence=19), 10.03)[1], "sequence_gap")

    def test_status_write_rejects_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            bridge.atomic_write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            with self.assertRaises(ValueError):
                bridge.atomic_write_json(path, {"value": math.nan})

    def test_json_loader_rejects_non_object_and_nonstandard_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.json"
            path.write_text("[]", encoding="utf-8")
            self.assertEqual(bridge.load_json(path)[1], "invalid_document")
            path.write_text('{"value":NaN}', encoding="utf-8")
            self.assertEqual(bridge.load_json(path)[1], "unavailable")

    def test_unavailable_document_latches_until_a_valid_disarm(self):
        gate = self.make_gate()
        offset, status, _, _, _ = bridge.evaluate_loaded_frame(
            gate,
            None,
            "unavailable",
            10.0,
        )
        self.assertIsNone(offset)
        self.assertEqual(status, "unavailable")
        self.assertTrue(gate.latched)
        self.assertEqual(
            gate.evaluate(frame(10.01, sequence=2), 10.01)[1],
            "fault_latched_waiting_for_source_disarm",
        )

    def test_lowstate_contract_accepts_only_fresh_exact_a7_mapping(self):
        arm_q, arm_dq, fixed_q, status, sample_age, transport_age = (
            bridge.validate_lowstate_frame(lowstate_frame(10.0), 10.01, 0.10)
        )
        self.assertEqual(status, "valid")
        self.assertEqual(arm_q.shape, (14,))
        self.assertEqual(arm_dq.shape, (14,))
        self.assertEqual(fixed_q.tolist(), [0.0, 0.0, 0.0])
        self.assertAlmostEqual(sample_age, 0.01)
        self.assertAlmostEqual(transport_age, 0.01)

        stale = lowstate_frame(10.0)
        self.assertEqual(
            bridge.validate_lowstate_frame(stale, 10.101, 0.10)[3],
            "stale",
        )
        wrong_mapping = lowstate_frame(10.0, left_arm_indices=list(range(14, 21)))
        self.assertEqual(
            bridge.validate_lowstate_frame(wrong_mapping, 10.0, 0.10)[3],
            "mapping_rejected",
        )
        invalid_q = [0.0] * bridge.MOTOR_COUNT
        invalid_q[bridge.ARM_INDICES[-1]] = math.nan
        invalid = lowstate_frame(10.0, full_q=invalid_q)
        self.assertEqual(
            bridge.validate_lowstate_frame(invalid, 10.0, 0.10)[3],
            "invalid_state",
        )
        invalid_crc = lowstate_frame(10.0, crc_valid=False)
        self.assertEqual(
            bridge.validate_lowstate_frame(invalid_crc, 10.0, 0.10)[3],
            "state_contract_rejected",
        )

    def test_source_contains_no_command_or_hand_control_path(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        forbidden = (
            "ChannelPublisher",
            "ChannelFactoryInitialize",
            "ChannelSubscriber",
            "unitree_sdk2py",
            "LowCmd",
            "MotionSwitcher",
            "Enter_Debug_Mode",
            "Enter_Debug_Mode",
            "ctrl_dual_arm",
            "/dev/tty",
            "import socket",
            "import serial",
            "network_interface",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)
        self.assertIn('choices=("dry-run",)', source)
        self.assertIn('"publishing_enabled": False', source)
        self.assertNotIn('"control"', source)
        self.assertNotIn("r1_a7_workspace", source)
        self.assertNotIn("workspace_limited", source)
        self.assertNotIn("raw_position_offset_m", source)
        launcher = (REPO_ROOT / "teleop" / "run_r1_a7_arm_live_bridge.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--mode dry-run", launcher)
        self.assertNotIn("enable-real", launcher)
        self.assertLess(source.index("row, load_error = load_json"), source.index("validation_now = time.monotonic()"))
        self.assertLess(
            source.index("validation_now = time.monotonic()"),
            source.index("evaluate_loaded_frame(\n                        gate,"),
        )
        self.assertIn("signal.SIGTERM", source)
        self.assertLess(
            source.index('status="startup"'),
            source.index("ik = R1A7DryRunIK("),
        )
        shadow_source = PUBLISHER_PATH.read_text(encoding="utf-8")
        launchers = "\n".join(
            (REPO_ROOT / "teleop" / name).read_text(encoding="utf-8")
            for name in (
                "run_r1_a7_arm_live_bridge.sh",
                "run_r1_a7_arm_publisher_shadow.sh",
            )
        )
        shadow_forbidden = tuple(token for token in forbidden if token != "linker_o6")
        for text in (shadow_source, launchers):
            for token in shadow_forbidden:
                with self.subTest(source="shadow_or_launcher", token=token):
                    self.assertNotIn(token, text)

    def test_lowstate_source_is_subscriber_only(self):
        source = LOWSTATE_PATH.read_text(encoding="utf-8")
        self.assertIn("ChannelSubscriber", source)
        self.assertIn('ChannelSubscriber("rt/lowstate", LowState_)', source)
        for token in (
            "ChannelPublisher",
            "LowCmd",
            "MotionSwitcher",
            "Enter_Debug_Mode",
            "robot_arm",
            "ctrl_dual_arm",
            "import serial",
            "/dev/tty",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_solver_exception_clears_candidate_and_latches(self):
        class FailingIK:
            def __init__(self):
                self.last_q = bridge.np.zeros(14)
                self.reset_called = False

            def solve(self, offset):
                raise RuntimeError("synthetic failure")

            def reset(self):
                self.reset_called = True

            def accept(self, solution):
                raise AssertionError("failed solution must not be accepted")

        gate = self.make_gate()
        ik = FailingIK()
        result = bridge.evaluate_ik_candidate(ik, gate, bridge.np.zeros(6), 0.05)
        self.assertEqual(result["status"], "ik_rejected:synthetic failure")
        self.assertIsNone(result["solution"])
        self.assertTrue(gate.latched)
        self.assertTrue(ik.reset_called)

    def test_model_limit_does_not_accept_candidate_or_latch_ik_gate(self):
        class LimitedIK:
            def __init__(self):
                self.last_q = bridge.np.zeros(14)
                self.reset_called = False

            def solve(self, offset):
                raise bridge.ModelLimitError("synthetic model boundary")

            def reset(self):
                self.reset_called = True

            def accept(self, solution):
                raise AssertionError("limited solution must not be accepted")

        gate = self.make_gate()
        ik = LimitedIK()
        result = bridge.evaluate_ik_candidate(ik, gate, bridge.np.zeros(6), 0.05)
        self.assertEqual(result["status"], "model_limited:synthetic model boundary")
        self.assertIsNone(result["solution"])
        self.assertFalse(gate.latched)
        self.assertFalse(ik.reset_called)

    def test_command_intent_decisions_are_fail_closed(self):
        solution = bridge.np.arange(14, dtype=bridge.np.float64) / 100.0
        last_safe = solution + 0.01
        cases = (
            ("dry_run_ok", solution, None, False, False, "new", solution.tolist()),
            ("duplicate", None, last_safe, True, False, "hold", last_safe.tolist()),
            ("duplicate_source_sample", None, last_safe, True, False, "hold", last_safe.tolist()),
            ("duplicate", None, last_safe, False, False, "disarm", None),
            ("waiting_for_second_fresh_frame", None, None, False, False, "disarm", None),
            ("model_limited:boundary", None, last_safe, True, False, "fault", None),
            ("stale", None, last_safe, True, True, "fault", None),
        )
        for status, candidate, previous, ready, latched, expected, expected_q in cases:
            with self.subTest(status=status, ready=ready):
                payload = command_intent(
                    status,
                    solution=candidate,
                    last_safe=previous,
                    command_ready=ready,
                    latched=latched,
                )
                self.assertEqual(payload["decision"], expected)
                self.assertEqual(payload["candidate_arm_q"], expected_q)
                self.assertFalse(payload["physical_publishing_enabled"])
                self.assertFalse(payload["o6_enabled"])
                self.assertEqual(payload["platform_profile"], bridge.PLATFORM_PROFILE)

    def test_only_model_derived_verified_collision_payload_passes_shadow_validator(self):
        solution = bridge.np.zeros(14, dtype=bridge.np.float64)
        payload = command_intent("dry_run_ok", solution=solution)
        normalized, status = publisher.validate_bridge_candidate(payload, 10.0, 0.10)
        self.assertIsNone(normalized)
        self.assertEqual(status, "collision_model_rejected")

        future_verified_payload = command_intent(
            "dry_run_ok",
            solution=solution,
            collision_verified=True,
        )
        normalized, status = publisher.validate_bridge_candidate(
            future_verified_payload, 10.0, 0.10
        )
        self.assertEqual(status, "valid")
        self.assertEqual(normalized["candidate_q"], solution.tolist())

        gate = publisher.ArmPublisherSafetyGate(
            lowstate_timeout=0.10,
            command_timeout=0.10,
            deadman_timeout=0.10,
            sequence_freeze_timeout=0.25,
            max_xr_age=0.25,
            max_following_error_rad=0.05,
            following_error_duration=0.10,
            zero_hold_cycles=2,
        )
        decision = gate.step(
            lowstate_frame(10.0),
            future_verified_payload,
            {
                "schema": publisher.DEADMAN_SCHEMA,
                "sequence": 1,
                "held": True,
                "updated_monotonic_ns": 10_000_000_000,
            },
            10.0,
            arm_request=True,
        )
        self.assertEqual(decision["status"], "zero_hold")
        self.assertEqual(decision["full_q"], lowstate_frame(10.0)["full_q"])

    def test_command_writer_lock_is_single_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            command_path = Path(directory) / "command.json"
            first = bridge.acquire_command_writer_lock(command_path)
            try:
                with self.assertRaises(RuntimeError):
                    bridge.acquire_command_writer_lock(command_path)
            finally:
                first.close()


class ArmBridgeIkIntegrationTest(unittest.TestCase):
    ACTUAL_ARM_Q = bridge.np.asarray(
        [
            -0.071408,
            -0.004434,
            0.005556,
            0.788643,
            0.056465,
            -0.741967,
            -0.094675,
            -0.030470,
            0.104970,
            -0.421895,
            0.741715,
            -0.204363,
            -0.745395,
            0.286183,
        ],
        dtype=bridge.np.float64,
    )
    ACTUAL_FIXED_Q = bridge.np.asarray(
        [1.647274, 0.000515, 2.007127],
        dtype=bridge.np.float64,
    )

    @classmethod
    def setUpClass(cls):
        from teleop.o6_readonly_contract import o6_joint_positions

        cls.ik = bridge.R1A7DryRunIK(
            0.01,
            0.10,
            0.005,
            500.0,
            o6_joint_positions(
                {
                    "left": [254, 255, 254, 254, 254, 254],
                    "right": [254, 255, 254, 254, 254, 254],
                }
            ),
        )

    def setUp(self):
        self.ik.set_live_state(self.ACTUAL_ARM_Q, self.ACTUAL_FIXED_Q)
        self.ik.reset()

    def test_zero_offset_is_exact_actual_hold(self):
        solution, position_error, orientation_error, sigma_min, condition = self.ik.solve(
            bridge.np.zeros(6, dtype=bridge.np.float64)
        )
        self.assertTrue(bridge.np.array_equal(solution, self.ACTUAL_ARM_Q))
        self.assertTrue(bridge.np.array_equal(self.ik.last_q, self.ACTUAL_ARM_Q))
        self.assertEqual(position_error, 0.0)
        self.assertEqual(orientation_error, 0.0)
        self.assertGreaterEqual(sigma_min, 0.005)
        self.assertLessEqual(condition, 500.0)

    def test_small_retraction_produces_finite_candidate(self):
        solution, position_error, orientation_error, sigma_min, condition = self.ik.solve(
            bridge.np.asarray([-0.005, 0.0, 0.0, -0.005, 0.0, 0.0])
        )
        self.assertEqual(solution.shape, (14,))
        self.assertTrue(bridge.np.isfinite(solution).all())
        self.assertLessEqual(position_error, 0.01)
        self.assertLessEqual(orientation_error, 0.10)
        self.assertGreaterEqual(sigma_min, 0.005)
        self.assertLessEqual(condition, 500.0)

    def test_actual_pose_and_small_trajectory_are_collision_free(self):
        self.assertGreater(self.ik.collision_pair_count, 100)
        self.assertEqual(self.ik.collision_geometry_count, 44)
        self.assertEqual(self.ik.o6_collision_geometry_count, 26)
        self.assertGreater(self.ik.o6_collision_pair_count, 0)
        self.assertIsNone(self.ik._first_collision(self.ACTUAL_ARM_Q, self.ACTUAL_FIXED_Q))
        solution, *_ = self.ik.solve(
            bridge.np.asarray([-0.005, 0.0, 0.0, -0.005, 0.0, 0.0])
        )
        self.assertIsNone(
            self.ik._trajectory_collision(
                self.ACTUAL_ARM_Q,
                solution,
                self.ACTUAL_FIXED_Q,
            )
        )

    def test_o6_geometry_keeps_wrist_and_cross_hand_collision_pairs(self):
        pairs = [
            (
                self.ik.collision_model.geometryObjects[pair.first].name,
                self.ik.collision_model.geometryObjects[pair.second].name,
            )
            for pair in self.ik.collision_model.collisionPairs
        ]
        self.assertTrue(
            any(
                (first.startswith(("lh_", "left_o6_")) and second.startswith("left_wrist_pitch_link"))
                or (second.startswith(("lh_", "left_o6_")) and first.startswith("left_wrist_pitch_link"))
                for first, second in pairs
            )
        )
        self.assertTrue(
            any(
                {first, second} == {"lh_index_distal_0", "rh_index_distal_0"}
                for first, second in pairs
            )
        )

    def test_known_self_collision_is_detected(self):
        colliding_q = bridge.np.asarray(
            [
                -2.8887559174816135,
                0.18099035473318786,
                1.7003261471993387,
                1.0718842799272352,
                -0.26126394719156587,
                0.07190403883926266,
                1.1291645309009886,
                -1.308155081008012,
                -0.8996399049794008,
                0.6685743820812708,
                0.17668332756229566,
                0.06951467114810472,
                0.8033812744739894,
                1.2393223062145784,
            ]
        )
        self.assertIsNotNone(
            self.ik._first_collision(colliding_q, bridge.np.zeros(3))
        )

    def test_actual_waist_changes_fk_and_rotates_waist_local_offset(self):
        left_zero, _ = self.ik._forward_frames(self.ACTUAL_ARM_Q, bridge.np.zeros(3))
        left_actual, _ = self.ik._forward_frames(
            self.ACTUAL_ARM_Q,
            self.ACTUAL_FIXED_Q,
        )
        self.assertGreater(
            float(bridge.np.linalg.norm(left_actual[:3, 3] - left_zero[:3, 3])),
            0.30,
        )
        offset = bridge.np.asarray([0.01, 0.0, 0.0, 0.0, 0.01, 0.0])
        root_offset = self.ik._offset_in_root(offset)
        angle = float(self.ACTUAL_FIXED_Q[0])
        expected_left = bridge.np.asarray([math.cos(angle), math.sin(angle), 0.0]) * 0.01
        expected_right = bridge.np.asarray([-math.sin(angle), math.cos(angle), 0.0]) * 0.01
        self.assertTrue(bridge.np.allclose(root_offset[:3], expected_left))
        self.assertTrue(bridge.np.allclose(root_offset[3:], expected_right))

    def test_reset_recaptures_latest_live_state_and_fixed_drift_is_visible(self):
        next_arm_q = self.ACTUAL_ARM_Q.copy()
        next_arm_q[0] += 0.001
        next_fixed_q = self.ACTUAL_FIXED_Q.copy()
        next_fixed_q[0] += 0.01
        self.ik.set_live_state(next_arm_q, next_fixed_q)
        self.assertAlmostEqual(self.ik.fixed_joint_drift(), 0.01)
        self.ik.reset()
        self.assertTrue(bridge.np.array_equal(self.ik.reference_q, next_arm_q))
        self.assertTrue(bridge.np.array_equal(self.ik.reference_fixed_q, next_fixed_q))
        self.assertEqual(self.ik.fixed_joint_drift(), 0.0)

    def test_computation_loads_no_robot_io_modules(self):
        self.assertFalse(any(name.startswith("unitree_sdk2py") for name in sys.modules))
        self.assertNotIn("teleop.robot_control.robot_arm", sys.modules)
        self.assertNotIn("teleop.utils.motion_switcher", sys.modules)


if __name__ == "__main__":
    unittest.main()
