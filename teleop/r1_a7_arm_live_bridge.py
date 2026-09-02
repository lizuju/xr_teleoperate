#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from teleop.o6_readonly_contract import O6OpenPoseGate


ARM_SCHEMA = "r1_a7_arm_ik_target_v1"
ARM_MAPPING = "r1_a7_visionpro_cartesian_v1"
ARM_REFERENCE = "armed_actual_wrist_yaw_pose"
ARM_FRAME = "r1_waist_yaw"
COMMAND_SCHEMA = "r1_a7_arm_command_intent_v1"
PLATFORM_PROFILE = "r1_a7_dual_linker_o6_v1"
VERIFIED_COLLISION_MODEL = "r1_a7_linker_o6_open"
UNVERIFIED_COLLISION_MODEL = "r1_a7_linker_o6_unverified"
COLLISION_MODEL_SHA256 = "5a64e7c2e6d4fa6b11e3243ffb4d7bf398e3537338cc0cf147c46ab1b8ef3bc5"
O6_VERIFIED_POSE = "fresh_verified_open"
O6_UNVERIFIED_POSE = "not_verified"
HAND_ORDER = ["left", "right"]
LOWSTATE_SCHEMA = "r1_a7_lowstate_shadow_v1"
MOTOR_COUNT = 35
WAIST_INDEX = 13
LEFT_ARM_INDICES = list(range(15, 22))
RIGHT_ARM_INDICES = list(range(22, 29))
ARM_INDICES = LEFT_ARM_INDICES + RIGHT_ARM_INDICES
HEAD_INDICES = [29, 30]
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
FIXED_JOINT_NAMES = ["waist_yaw_joint", "head_pitch_joint", "head_yaw_joint"]
O6_JOINT_PREFIXES = ("lh_", "rh_")
O6_COLLISION_MARKERS = ("lh_", "rh_", "left_o6_", "right_o6_")


class ModelLimitError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fail-closed Vision Pro to R1-A7 arm-only validation bridge"
    )
    parser.add_argument("--mode", choices=("dry-run",), default="dry-run")
    parser.add_argument(
        "--arm-live-state",
        default="/run/user/1000/unitree-r1-arm-live/target.json",
    )
    parser.add_argument(
        "--status-file",
        default="/run/user/1000/unitree-r1-arm-bridge/status.json",
    )
    parser.add_argument(
        "--command-intent-file",
        default="/run/user/1000/unitree-r1-arm-bridge/command.json",
    )
    parser.add_argument(
        "--lowstate-shadow",
        default="/run/user/1000/unitree-r1-lowstate/state.json",
    )
    parser.add_argument(
        "--o6-readonly-state",
        default="/run/user/1000/unitree-o6-readonly/state.json",
    )
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--tracking-timeout", type=float, default=0.25)
    parser.add_argument("--lowstate-timeout", type=float, default=0.10)
    parser.add_argument("--o6-state-timeout", type=float, default=0.25)
    parser.add_argument("--max-fixed-joint-drift-rad", type=float, default=0.02)
    parser.add_argument("--max-sequence-gap", type=int, default=8)
    parser.add_argument("--max-target-speed-m-s", type=float, default=0.35)
    parser.add_argument("--max-ik-position-error-m", type=float, default=0.01)
    parser.add_argument("--max-ik-orientation-error-rad", type=float, default=0.10)
    parser.add_argument("--min-jacobian-sigma", type=float, default=0.005)
    parser.add_argument("--max-jacobian-condition", type=float, default=500.0)
    parser.add_argument("--max-candidate-step-rad", type=float, default=0.05)
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


def acquire_command_writer_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.parent / "writer.lock"
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError(f"another arm bridge owns {lock_path}")
    os.fchmod(lock_file.fileno(), 0o600)
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"r1_a7_arm_live_bridge pid={os.getpid()}\n")
    lock_file.flush()
    return lock_file


def stop_on_signal(_signum, _frame):
    raise KeyboardInterrupt


def load_json(path):
    try:
        row = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, json.JSONDecodeError, UnicodeError, ValueError):
        return None, "unavailable"
    if not isinstance(row, dict):
        return None, "invalid_document"
    return row, None


def finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def validate_arm_frame(row, now, tracking_timeout):
    sequence = row.get("sequence")
    if (
        not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence <= 0
        or sequence > 2**63 - 1
    ):
        return None, "invalid_sequence", None, None, None, False
    if row.get("schema") != ARM_SCHEMA:
        return None, "schema_rejected", sequence, None, None, False
    if row.get("mapping") != ARM_MAPPING:
        return None, "mapping_rejected", sequence, None, None, False
    armed = row.get("armed")
    if not isinstance(armed, bool):
        return None, "invalid_armed", sequence, None, None, False

    published_ns = row.get("published_monotonic_ns")
    source_timestamp = row.get("monotonic_timestamp")
    if (
        not isinstance(published_ns, int)
        or isinstance(published_ns, bool)
        or published_ns <= 0
        or published_ns > 2**63 - 1
    ):
        return None, "invalid_published_timestamp", sequence, None, None, False
    if not finite_number(source_timestamp):
        return None, "invalid_source_timestamp", sequence, None, None, False
    source_timestamp = float(source_timestamp)
    transport_age = now - published_ns / 1_000_000_000.0
    tracking_age = now - source_timestamp
    if transport_age < 0.0 or tracking_age < 0.0:
        return None, "future_timestamp", sequence, tracking_age, transport_age, False
    if transport_age >= tracking_timeout or tracking_age >= tracking_timeout:
        return None, "stale", sequence, tracking_age, transport_age, False
    if not armed:
        reason = row.get("reason")
        if reason not in ("disarmed", "stale", "tracking_jump"):
            return None, "disarm_reason_rejected", sequence, tracking_age, transport_age, False
        return None, reason, sequence, tracking_age, transport_age, True

    if row.get("reference") != ARM_REFERENCE:
        return None, "reference_rejected", sequence, tracking_age, transport_age, False
    if row.get("frame") != ARM_FRAME:
        return None, "frame_rejected", sequence, tracking_age, transport_age, False
    if row.get("target_units") != "m":
        return None, "units_rejected", sequence, tracking_age, transport_age, False
    if row.get("target_hand_order") != HAND_ORDER:
        return None, "hand_order_rejected", sequence, tracking_age, transport_age, False
    values = row.get("position_offset_m")
    if not isinstance(values, list) or len(values) != 6 or not all(
        finite_number(value) for value in values
    ):
        return None, "invalid_target", sequence, tracking_age, transport_age, False
    offset = np.asarray(values, dtype=np.float64)
    return offset, "valid", sequence, tracking_age, transport_age, False


def validate_lowstate_frame(row, now, lowstate_timeout):
    if not isinstance(row, dict):
        return None, None, None, "invalid_document", None, None
    if row.get("schema") != LOWSTATE_SCHEMA:
        return None, None, None, "schema_rejected", None, None
    if row.get("motor_count") != MOTOR_COUNT or row.get("crc_valid") is not True:
        return None, None, None, "state_contract_rejected", None, None
    if row.get("arm_indices") != ARM_INDICES or row.get("waist_index") != WAIST_INDEX:
        return None, None, None, "mapping_rejected", None, None
    if row.get("left_arm_indices") != LEFT_ARM_INDICES:
        return None, None, None, "mapping_rejected", None, None
    if row.get("right_arm_indices") != RIGHT_ARM_INDICES:
        return None, None, None, "mapping_rejected", None, None
    if row.get("head_indices") != HEAD_INDICES:
        return None, None, None, "mapping_rejected", None, None

    sequence = row.get("sequence")
    sample_ns = row.get("sample_monotonic_ns")
    published_ns = row.get("published_monotonic_ns")
    mode_machine = row.get("mode_machine")
    if (
        not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence <= 0
        or sequence > 2**63 - 1
    ):
        return None, None, None, "invalid_sequence", None, None
    for value in (sample_ns, published_ns):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > 2**63 - 1
        ):
            return None, None, None, "invalid_timestamp", None, None
    if not isinstance(mode_machine, int) or isinstance(mode_machine, bool):
        return None, None, None, "invalid_mode", None, None

    full_q = row.get("full_q")
    full_dq = row.get("full_dq")
    if (
        not isinstance(full_q, list)
        or len(full_q) != MOTOR_COUNT
        or not all(finite_number(value) for value in full_q)
        or not isinstance(full_dq, list)
        or len(full_dq) != MOTOR_COUNT
        or not all(finite_number(value) for value in full_dq)
    ):
        return None, None, None, "invalid_state", None, None

    sample_age = now - sample_ns / 1_000_000_000.0
    transport_age = now - published_ns / 1_000_000_000.0
    if sample_age < 0.0 or transport_age < 0.0 or published_ns < sample_ns:
        return None, None, None, "future_timestamp", sample_age, transport_age
    if sample_age >= lowstate_timeout or transport_age >= lowstate_timeout:
        return None, None, None, "stale", sample_age, transport_age
    arm_q = [full_q[index] for index in ARM_INDICES]
    arm_dq = [full_dq[index] for index in ARM_INDICES]
    fixed_q = np.asarray(
        [full_q[WAIST_INDEX], full_q[HEAD_INDICES[0]], full_q[HEAD_INDICES[1]]],
        dtype=np.float64,
    )
    return (
        np.asarray(arm_q, dtype=np.float64),
        np.asarray(arm_dq, dtype=np.float64),
        fixed_q,
        "valid",
        sample_age,
        transport_age,
    )


class SafetyGate:
    def __init__(self, tracking_timeout, max_sequence_gap, max_target_speed_m_s):
        self.tracking_timeout = tracking_timeout
        self.max_sequence_gap = max_sequence_gap
        self.max_target_speed_m_s = max_target_speed_m_s
        self.latched = False
        self.source_disarm_seen = False
        self.fresh_frames = 0
        self.last_sequence = None
        self.last_source_timestamp = None
        self.last_offset = None

    def _reset_history(self):
        self.fresh_frames = 0
        self.last_sequence = None
        self.last_source_timestamp = None
        self.last_offset = None

    def latch_fault(self):
        self.latched = True
        self._reset_history()

    def evaluate(self, row, now):
        (
            offset,
            status,
            sequence,
            tracking_age,
            transport_age,
            validated_disarm,
        ) = validate_arm_frame(row, now, self.tracking_timeout)
        if validated_disarm:
            self.latched = False
            self.source_disarm_seen = True
            self._reset_history()
            return None, status, sequence, tracking_age, transport_age
        if status != "valid":
            self.latch_fault()
            return None, status, sequence, tracking_age, transport_age
        if self.latched:
            return None, "fault_latched_waiting_for_source_disarm", sequence, tracking_age, transport_age
        if not self.source_disarm_seen:
            return None, "startup_requires_source_disarm", sequence, tracking_age, transport_age

        source_timestamp = float(row["monotonic_timestamp"])
        if self.last_sequence is not None:
            if sequence == self.last_sequence:
                return None, "duplicate", sequence, tracking_age, transport_age
            if sequence < self.last_sequence:
                self.latch_fault()
                return None, "sequence_regression", sequence, tracking_age, transport_age
            if sequence - self.last_sequence > self.max_sequence_gap:
                self.latch_fault()
                return None, "sequence_gap", sequence, tracking_age, transport_age
            sample_dt = source_timestamp - self.last_source_timestamp
            if sample_dt == 0.0:
                return None, "duplicate_source_sample", sequence, tracking_age, transport_age
            if sample_dt < 0.0:
                self.latch_fault()
                return None, "source_time_nonincreasing", sequence, tracking_age, transport_age
            speed = max(
                float(np.linalg.norm(offset[:3] - self.last_offset[:3])),
                float(np.linalg.norm(offset[3:] - self.last_offset[3:])),
            ) / sample_dt
            if speed > self.max_target_speed_m_s:
                self.latch_fault()
                return None, "target_jump", sequence, tracking_age, transport_age

        self.last_sequence = sequence
        self.last_source_timestamp = source_timestamp
        self.last_offset = offset.copy()
        self.fresh_frames += 1
        if self.fresh_frames < 2:
            return None, "waiting_for_second_fresh_frame", sequence, tracking_age, transport_age
        return offset, "ready", sequence, tracking_age, transport_age


def evaluate_loaded_frame(gate, row, load_error, now):
    if row is None:
        gate.latch_fault()
        return None, load_error, None, None, None
    return gate.evaluate(row, now)


class R1A7DryRunIK:
    def __init__(
        self,
        max_position_error_m,
        max_orientation_error_rad,
        min_jacobian_sigma,
        max_jacobian_condition,
        o6_joint_q,
    ):
        import casadi
        import pinocchio as pin
        from pinocchio import casadi as cpin

        profile_path = REPO_ROOT / "config" / "r1_a7_dual_linker_o6_v1.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile.get("configuration_id") != PLATFORM_PROFILE:
            raise RuntimeError("R1-A7 + O6 platform profile mismatch")
        combined = profile.get("combined_urdf", {})
        urdf_path = (REPO_ROOT / combined["workspace_relative_path"]).resolve()
        package_dir = (
            REPO_ROOT / combined["pinocchio_package_dir_workspace_relative"]
        ).resolve()
        actual_hash = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
        if actual_hash != COLLISION_MODEL_SHA256 or combined.get("sha256") != actual_hash:
            raise RuntimeError("R1-A7 + O6 combined URDF hash mismatch")
        raw_model, raw_collision, raw_visual = pin.buildModelsFromUrdf(
            str(urdf_path),
            package_dirs=[str(package_dir)],
        )
        o6_joint_names = [
            name for name in raw_model.names if name.startswith(O6_JOINT_PREFIXES)
        ]
        if raw_model.nq != 39 or len(o6_joint_names) != 22 or set(o6_joint_names) != set(o6_joint_q):
            raise RuntimeError("Unexpected R1-A7 + O6 joint layout")
        lock_q = pin.neutral(raw_model)
        o6_joint_ids = []
        for name in o6_joint_names:
            joint_id = raw_model.getJointId(name)
            q_index = raw_model.joints[joint_id].idx_q
            value = float(o6_joint_q[name])
            if not math.isfinite(value):
                raise RuntimeError("O6 lock pose contains a non-finite value")
            if not raw_model.lowerPositionLimit[q_index] <= value <= raw_model.upperPositionLimit[q_index]:
                raise RuntimeError(f"O6 lock pose is outside URDF limits: {name}")
            lock_q[q_index] = value
            o6_joint_ids.append(joint_id)
        self.model, geometry_models = pin.buildReducedModel(
            raw_model,
            [raw_collision, raw_visual],
            o6_joint_ids,
            lock_q,
        )
        self.collision_model, self.visual_model = geometry_models
        self.collision_model_sha256 = actual_hash
        self.o6_lock_joint_q = dict(o6_joint_q)
        self.data = self.model.createData()
        self.left_frame_id = self.model.getFrameId("lh_hand_base_link")
        self.right_frame_id = self.model.getFrameId("rh_hand_base_link")
        if self.left_frame_id >= len(self.model.frames) or self.right_frame_id >= len(self.model.frames):
            raise RuntimeError("R1-A7 + O6 hand-base frames are missing from the combined model")
        self.pin = pin
        self.max_position_error_m = max_position_error_m
        self.max_orientation_error_rad = max_orientation_error_rad
        self.min_jacobian_sigma = min_jacobian_sigma
        self.max_jacobian_condition = max_jacobian_condition
        self.arm_q_indices = np.asarray(
            [self.model.joints[self.model.getJointId(name)].idx_q for name in ARM_JOINT_NAMES],
            dtype=np.int64,
        )
        self.fixed_q_indices = np.asarray(
            [self.model.joints[self.model.getJointId(name)].idx_q for name in FIXED_JOINT_NAMES],
            dtype=np.int64,
        )
        if self.model.nq != 17 or sorted(
            self.arm_q_indices.tolist() + self.fixed_q_indices.tolist()
        ) != list(range(17)):
            raise RuntimeError("Unexpected R1-A7 URDF joint layout")

        self.collision_geometry_count = len(self.collision_model.geometryObjects)
        self.o6_collision_geometry_count = sum(
            object_.name.startswith(O6_COLLISION_MARKERS)
            for object_ in self.collision_model.geometryObjects
        )
        if self.collision_geometry_count != 44 or self.o6_collision_geometry_count != 26:
            raise RuntimeError("R1-A7 + O6 collision geometry set is incomplete")
        self.collision_kinematic_model = self.model
        self.collision_kinematic_data = self.collision_kinematic_model.createData()
        self.collision_model.addAllCollisionPairs()
        collision_pairs = []
        for pair in self.collision_model.collisionPairs:
            first = self.collision_model.geometryObjects[pair.first]
            second = self.collision_model.geometryObjects[pair.second]
            first_joint = first.parentJoint
            second_joint = second.parentJoint
            if first_joint == second_joint:
                continue
            adjacent = (
                self.model.parents[first_joint] == second_joint
                or self.model.parents[second_joint] == first_joint
            )
            involves_o6 = first.name.startswith(O6_COLLISION_MARKERS) or second.name.startswith(
                O6_COLLISION_MARKERS
            )
            if adjacent and not involves_o6:
                continue
            collision_pairs.append(pair)
        self.collision_model.removeAllCollisionPairs()
        for pair in collision_pairs:
            self.collision_model.addCollisionPair(pair)
        self.collision_data = self.collision_model.createData()
        self.collision_pair_count = len(self.collision_model.collisionPairs)
        self.o6_collision_pair_count = sum(
            self.collision_model.geometryObjects[pair.first].name.startswith(
                O6_COLLISION_MARKERS
            )
            or self.collision_model.geometryObjects[pair.second].name.startswith(
                O6_COLLISION_MARKERS
            )
            for pair in self.collision_model.collisionPairs
        )
        if self.o6_collision_pair_count == 0:
            raise RuntimeError("R1-A7 + O6 collision pairs are missing")

        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()
        cq = casadi.SX.sym("q", self.model.nq, 1)
        left_target = casadi.SX.sym("left_target", 4, 4)
        right_target = casadi.SX.sym("right_target", 4, 4)
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        translation_error = casadi.Function(
            "r1_bridge_translation_error",
            [cq, left_target, right_target],
            [
                casadi.vertcat(
                    cdata.oMf[self.left_frame_id].translation - left_target[:3, 3],
                    cdata.oMf[self.right_frame_id].translation - right_target[:3, 3],
                )
            ],
        )
        orientation_error = casadi.Function(
            "r1_bridge_orientation_error",
            [cq, left_target, right_target],
            [
                casadi.vertcat(
                    cpin.log3(
                        cdata.oMf[self.left_frame_id].rotation
                        @ left_target[:3, :3].T
                    ),
                    cpin.log3(
                        cdata.oMf[self.right_frame_id].rotation
                        @ right_target[:3, :3].T
                    ),
                )
            ],
        )
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.param_left = self.opti.parameter(4, 4)
        self.param_right = self.opti.parameter(4, 4)
        self.param_last_q = self.opti.parameter(14)
        self.param_reference_q = self.opti.parameter(14)
        self.param_fixed_q = self.opti.parameter(3)
        for q_index in self.arm_q_indices:
            self.opti.subject_to(
                self.opti.bounded(
                    self.model.lowerPositionLimit[int(q_index)],
                    self.var_q[int(q_index)],
                    self.model.upperPositionLimit[int(q_index)],
                )
            )
        for parameter_index, q_index in enumerate(self.fixed_q_indices):
            self.opti.subject_to(self.var_q[int(q_index)] == self.param_fixed_q[parameter_index])
        arm_var_q = casadi.vertcat(
            *[self.var_q[int(index)] for index in self.arm_q_indices]
        )
        self.opti.minimize(
            50.0
            * casadi.sumsqr(
                translation_error(self.var_q, self.param_left, self.param_right)
            )
            + casadi.sumsqr(
                orientation_error(self.var_q, self.param_left, self.param_right)
            )
            + 0.02 * casadi.sumsqr(arm_var_q - self.param_reference_q)
            + 0.10 * casadi.sumsqr(arm_var_q - self.param_last_q)
        )
        self.opti.solver(
            "ipopt",
            {
                "expand": True,
                "detect_simple_bounds": True,
                "calc_lam_p": False,
                "print_time": False,
                "ipopt.sb": "yes",
                "ipopt.print_level": 0,
                "ipopt.max_iter": 30,
                "ipopt.tol": 1.0e-4,
                "ipopt.acceptable_tol": 5.0e-4,
                "ipopt.acceptable_iter": 5,
            },
        )
        self.live_q = None
        self.live_fixed_q = None
        self.reference_q = None
        self.reference_fixed_q = None
        self.reference_left = None
        self.reference_right = None
        self.last_q = None

    def set_live_state(self, arm_q, fixed_q):
        arm_q = np.asarray(arm_q, dtype=np.float64)
        fixed_q = np.asarray(fixed_q, dtype=np.float64)
        if arm_q.shape != (14,) or fixed_q.shape != (3,):
            raise ValueError("R1-A7 live-state shape mismatch")
        if not np.isfinite(arm_q).all() or not np.isfinite(fixed_q).all():
            raise ValueError("R1-A7 live-state contains non-finite values")
        full_q = self._compose_full(arm_q, fixed_q)
        if np.any(full_q[self.arm_q_indices] < self.model.lowerPositionLimit[self.arm_q_indices] - 1.0e-5) or np.any(
            full_q[self.arm_q_indices] > self.model.upperPositionLimit[self.arm_q_indices] + 1.0e-5
        ):
            raise ModelLimitError("R1-A7 arm live-state is outside URDF joint limits")
        if np.any(full_q[self.fixed_q_indices] < self.model.lowerPositionLimit[self.fixed_q_indices] - 1.0e-4) or np.any(
            full_q[self.fixed_q_indices] > self.model.upperPositionLimit[self.fixed_q_indices] + 1.0e-4
        ):
            raise ModelLimitError("R1-A7 waist/head live-state is outside URDF joint limits")
        self.live_q = arm_q.copy()
        self.live_fixed_q = fixed_q.copy()

    def reset(self):
        if self.live_q is None or self.live_fixed_q is None:
            raise RuntimeError("Fresh R1-A7 lowstate is required before reset")
        collision = self._first_collision(self.live_q, self.live_fixed_q)
        if collision is not None:
            raise ModelLimitError(f"R1-A7 live-state model collision: {collision}")
        self.reference_q = self.live_q.copy()
        self.reference_fixed_q = self.live_fixed_q.copy()
        self.reference_left, self.reference_right = self._forward_frames(
            self.reference_q,
            self.reference_fixed_q,
        )
        self.last_q = self.reference_q.copy()

    def accept(self, solution):
        self.last_q = solution.copy()

    def fixed_joint_drift(self):
        if self.reference_fixed_q is None or self.live_fixed_q is None:
            return None
        return float(np.max(np.abs(self.live_fixed_q - self.reference_fixed_q)))

    def _offset_in_root(self, offset):
        waist_yaw = float(self.reference_fixed_q[0])
        cosine = math.cos(waist_yaw)
        sine = math.sin(waist_yaw)
        waist_to_root = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return np.concatenate(
            (waist_to_root @ offset[:3], waist_to_root @ offset[3:])
        )

    def _compose_full(self, arm_q, fixed_q):
        full_q = np.zeros(self.model.nq, dtype=np.float64)
        full_q[self.arm_q_indices] = arm_q
        full_q[self.fixed_q_indices] = fixed_q
        return full_q

    def _forward_frames(self, arm_q, fixed_q):
        full_q = self._compose_full(arm_q, fixed_q)
        self.pin.framesForwardKinematics(self.model, self.data, full_q)
        self.pin.updateFramePlacements(self.model, self.data)
        left = self.data.oMf[self.left_frame_id].homogeneous.copy()
        right = self.data.oMf[self.right_frame_id].homogeneous.copy()
        return left, right

    def _jacobian_metrics(self, arm_q, fixed_q):
        full_q = self._compose_full(arm_q, fixed_q)
        self.pin.computeJointJacobians(self.model, self.data, full_q)
        self.pin.updateFramePlacements(self.model, self.data)
        left = self.pin.getFrameJacobian(
            self.model,
            self.data,
            self.left_frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        right = self.pin.getFrameJacobian(
            self.model,
            self.data,
            self.right_frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        arm_jacobian = np.vstack((left, right))[:, self.arm_q_indices]
        singular_values = np.linalg.svd(arm_jacobian, compute_uv=False)
        sigma_min = float(singular_values[-1])
        condition = float(singular_values[0] / singular_values[-1])
        return sigma_min, condition

    def _first_collision(self, arm_q, fixed_q):
        full_q = self._compose_full(arm_q, fixed_q)
        self.pin.computeCollisions(
            self.collision_kinematic_model,
            self.collision_kinematic_data,
            self.collision_model,
            self.collision_data,
            full_q,
            False,
        )
        for pair, result in zip(
            self.collision_model.collisionPairs,
            self.collision_data.collisionResults,
        ):
            if result.isCollision():
                first = self.collision_model.geometryObjects[pair.first].name
                second = self.collision_model.geometryObjects[pair.second].name
                return f"{first}<->{second}"
        return None

    def _trajectory_collision(self, start_q, target_q, fixed_q):
        steps = max(1, int(np.ceil(np.max(np.abs(target_q - start_q)) / 0.01)))
        for index in range(1, steps + 1):
            q = start_q + (target_q - start_q) * (index / steps)
            pair = self._first_collision(q, fixed_q)
            if pair is not None:
                return pair
        return None

    def solve(self, offset):
        if (
            self.last_q is None
            or self.reference_q is None
            or self.reference_fixed_q is None
            or self.live_fixed_q is None
        ):
            raise RuntimeError("R1-A7 IK has not been anchored to fresh lowstate")
        left = self.reference_left.copy()
        right = self.reference_right.copy()
        root_offset = self._offset_in_root(offset)
        left[:3, 3] += root_offset[:3]
        right[:3, 3] += root_offset[3:]
        fixed_q = self.reference_fixed_q.copy()
        if float(np.max(np.abs(offset))) == 0.0:
            sigma_min, condition = self._jacobian_metrics(self.reference_q, fixed_q)
            if sigma_min < self.min_jacobian_sigma or condition > self.max_jacobian_condition:
                raise ModelLimitError(
                    f"IK Jacobian rejected: sigma={sigma_min:.6f} condition={condition:.1f}"
                )
            return self.reference_q.copy(), 0.0, 0.0, sigma_min, condition

        self.opti.set_initial(self.var_q, self._compose_full(self.last_q, fixed_q))
        self.opti.set_value(self.param_left, left)
        self.opti.set_value(self.param_right, right)
        self.opti.set_value(self.param_last_q, self.last_q)
        self.opti.set_value(self.param_reference_q, self.reference_q)
        self.opti.set_value(self.param_fixed_q, fixed_q)
        try:
            result = self.opti.solve()
        except Exception as error:
            raise ModelLimitError(f"IK target is unreachable: {error}") from error
        full_solution = np.asarray(result.value(self.var_q), dtype=np.float64)
        if full_solution.shape != (17,) or not np.isfinite(full_solution).all():
            raise RuntimeError("IK returned an invalid joint target")
        if np.any(
            full_solution[self.arm_q_indices]
            < self.model.lowerPositionLimit[self.arm_q_indices] - 1.0e-7
        ) or np.any(
            full_solution[self.arm_q_indices]
            > self.model.upperPositionLimit[self.arm_q_indices] + 1.0e-7
        ):
            raise ModelLimitError("IK joint limit rejected")
        solution = full_solution[self.arm_q_indices]
        solved_left, solved_right = self._forward_frames(solution, fixed_q)
        position_error = max(
            float(np.linalg.norm(solved_left[:3, 3] - left[:3, 3])),
            float(np.linalg.norm(solved_right[:3, 3] - right[:3, 3])),
        )
        if position_error > self.max_position_error_m:
            raise ModelLimitError(f"IK residual rejected: {position_error:.6f} m")
        orientation_error = max(
            float(
                np.linalg.norm(
                    self.pin.log3(solved_left[:3, :3] @ left[:3, :3].T)
                )
            ),
            float(
                np.linalg.norm(
                    self.pin.log3(solved_right[:3, :3] @ right[:3, :3].T)
                )
            ),
        )
        if orientation_error > self.max_orientation_error_rad:
            raise ModelLimitError(f"IK orientation rejected: {orientation_error:.6f} rad")
        sigma_min, condition = self._jacobian_metrics(solution, fixed_q)
        if sigma_min < self.min_jacobian_sigma or condition > self.max_jacobian_condition:
            raise ModelLimitError(
                f"IK Jacobian rejected: sigma={sigma_min:.6f} condition={condition:.1f}"
            )
        collision = self._trajectory_collision(self.last_q, solution, fixed_q)
        if collision is not None:
            raise ModelLimitError(f"R1-A7 model collision: {collision}")
        return solution, position_error, orientation_error, sigma_min, condition


def evaluate_ik_candidate(ik, gate, offset, max_candidate_step_rad):
    result = {
        "status": None,
        "solution": None,
        "position_error": None,
        "orientation_error": None,
        "jacobian_sigma_min": None,
        "jacobian_condition": None,
        "max_candidate_step": None,
    }
    try:
        previous_q = ik.last_q.copy()
        (
            solution,
            result["position_error"],
            result["orientation_error"],
            result["jacobian_sigma_min"],
            result["jacobian_condition"],
        ) = ik.solve(offset)
        result["max_candidate_step"] = float(np.max(np.abs(solution - previous_q)))
        if result["max_candidate_step"] > max_candidate_step_rad:
            raise ModelLimitError(
                f"candidate step rejected: {result['max_candidate_step']:.6f} rad"
            )
    except ModelLimitError as error:
        result["status"] = f"model_limited:{error}"
        return result
    except Exception as error:
        gate.latch_fault()
        ik.reset()
        result["status"] = f"ik_rejected:{error}"
        return result
    ik.accept(solution)
    result["status"] = "dry_run_ok"
    result["solution"] = solution
    return result


def build_command_intent(
    *,
    status,
    solution,
    last_safe_arm_q,
    command_ready,
    gate_latched,
    source_row,
    sequence,
    tracking_age,
    transport_age,
    mode_machine,
    reference_arm_q,
    reference_fixed_q,
    fixed_joint_drift,
    collision_model,
    o6_collision_pose,
    o6_state_sequence,
    o6_state_age,
    o6_angles_raw,
    updated_monotonic_ns,
):
    if gate_latched:
        decision = "fault"
        command_arm_q = None
    elif solution is not None:
        decision = "new"
        command_arm_q = solution
    elif (
        command_ready
        and last_safe_arm_q is not None
        and status in ("duplicate", "duplicate_source_sample")
    ):
        decision = "hold"
        command_arm_q = last_safe_arm_q
    elif status in (
        "disarmed",
        "duplicate",
        "duplicate_source_sample",
        "startup_requires_source_disarm",
        "waiting_for_second_fresh_frame",
        "startup",
        "stopped",
    ):
        decision = "disarm"
        command_arm_q = None
    else:
        decision = "fault"
        command_arm_q = None

    return {
        "schema": COMMAND_SCHEMA,
        "decision": decision,
        "platform_profile": PLATFORM_PROFILE,
        "physical_publishing_enabled": False,
        "o6_enabled": False,
        "collision_model": collision_model,
        "collision_model_sha256": COLLISION_MODEL_SHA256,
        "o6_collision_pose": o6_collision_pose,
        "o6_state_sequence": o6_state_sequence,
        "o6_state_age_ms": None
        if o6_state_age is None
        else round(o6_state_age * 1000.0, 3),
        "o6_angles_raw": o6_angles_raw,
        "source_armed": None if source_row is None else source_row.get("armed"),
        "source_sequence": sequence,
        "source_monotonic_timestamp": None
        if source_row is None
        else source_row.get("monotonic_timestamp"),
        "source_tracking_age_ms": None
        if tracking_age is None
        else round(tracking_age * 1000.0, 3),
        "source_transport_age_ms": None
        if transport_age is None
        else round(transport_age * 1000.0, 3),
        "mode_machine": mode_machine,
        "candidate_arm_q": None
        if command_arm_q is None
        else np.asarray(command_arm_q, dtype=np.float64).tolist(),
        "reference_arm_q": None
        if reference_arm_q is None
        else np.asarray(reference_arm_q, dtype=np.float64).tolist(),
        "reference_fixed_q": None
        if reference_fixed_q is None
        else np.asarray(reference_fixed_q, dtype=np.float64).tolist(),
        "fixed_joint_drift_rad": fixed_joint_drift,
        "bridge_status": status,
        "bridge_monotonic_ns": updated_monotonic_ns,
    }


def validate_cli(args):
    positive_finite = (
        "frequency",
        "tracking_timeout",
        "lowstate_timeout",
        "o6_state_timeout",
        "max_fixed_joint_drift_rad",
        "max_target_speed_m_s",
        "max_ik_position_error_m",
        "max_ik_orientation_error_rad",
        "min_jacobian_sigma",
        "max_jacobian_condition",
        "max_candidate_step_rad",
    )
    for name in positive_finite:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if args.max_sequence_gap <= 0:
        raise ValueError("--max-sequence-gap must be positive")
    if not math.isfinite(args.duration) or args.duration < 0.0:
        raise ValueError("--duration must be finite and non-negative")


def main():
    args = parse_args()
    validate_cli(args)
    arm_path = Path(args.arm_live_state)
    lowstate_path = Path(args.lowstate_shadow)
    o6_state_path = Path(args.o6_readonly_state)
    status_path = Path(args.status_file)
    command_path = Path(args.command_intent_file)
    command_writer_lock = acquire_command_writer_lock(command_path)
    startup_ns = time.monotonic_ns()
    atomic_write_json(
        command_path,
        build_command_intent(
            status="startup",
            solution=None,
            last_safe_arm_q=None,
            command_ready=False,
            gate_latched=False,
            source_row=None,
            sequence=None,
            tracking_age=None,
            transport_age=None,
            mode_machine=None,
            reference_arm_q=None,
            reference_fixed_q=None,
            fixed_joint_drift=None,
            collision_model=UNVERIFIED_COLLISION_MODEL,
            o6_collision_pose=O6_UNVERIFIED_POSE,
            o6_state_sequence=None,
            o6_state_age=None,
            o6_angles_raw=None,
            updated_monotonic_ns=startup_ns,
        ),
    )
    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_signal)
    gate = SafetyGate(
        args.tracking_timeout,
        args.max_sequence_gap,
        args.max_target_speed_m_s,
    )
    o6_gate = O6OpenPoseGate()
    ik = None
    locked_o6_angles = None
    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    next_tick = started
    last_printed_status = None
    next_status_print = started
    counters = {
        "frames": 0,
        "ready": 0,
        "ik_ok": 0,
        "rejected": 0,
        "combined_model_builds": 0,
    }
    last_safe_arm_q = None
    command_ready = False
    print("[R1 ARM BRIDGE] DRY RUN ONLY; computation and local files only", flush=True)
    print(
        "[R1 ARM BRIDGE] fresh dual-O6 FC04 state, read-only lowstate, and a source disarm are required before IK",
        flush=True,
    )

    try:
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.01))
                continue
            next_tick = max(next_tick + 1.0 / args.frequency, now)
            row, load_error = load_json(arm_path)
            lowstate_row, lowstate_load_error = load_json(lowstate_path)
            o6_row, o6_load_error = load_json(o6_state_path)
            validation_now = time.monotonic()
            counters["frames"] += 1
            status = load_error
            sequence = None
            tracking_age = None
            transport_age = None
            offset = None
            ik_position_error = None
            ik_orientation_error = None
            jacobian_sigma_min = None
            jacobian_condition = None
            max_candidate_step = None
            solution = None
            lowstate_sample_age = None
            lowstate_transport_age = None
            mode_machine = None
            arm_q = None
            arm_dq = None
            fixed_q = None
            fixed_joint_drift = None
            o6_state_age = None
            o6_sequence = None
            o6_angles_raw = None
            o6_joint_q = None
            if o6_row is None:
                o6_gate.reset()
                o6_status = o6_load_error
            else:
                o6_joint_q, o6_status, o6_state_age, o6_sequence = o6_gate.evaluate(
                    o6_row,
                    validation_now,
                    args.o6_state_timeout,
                )
                if isinstance(o6_row.get("hands"), dict) and all(
                    isinstance(o6_row["hands"].get(side), dict)
                    for side in ("left", "right")
                ):
                    o6_angles_raw = {
                        side: o6_row["hands"][side].get("angles_raw")
                        for side in ("left", "right")
                    }

            if o6_status != O6_VERIFIED_POSE:
                gate.latch_fault()
                ik = None
                locked_o6_angles = None
                last_safe_arm_q = None
                command_ready = False
                status = f"o6_{o6_status}"
            else:
                pose_changed = (
                    locked_o6_angles is not None
                    and any(
                        abs(current - locked) > 1
                        for side in ("left", "right")
                        for current, locked in zip(o6_angles_raw[side], locked_o6_angles[side])
                    )
                )
                if pose_changed:
                    gate.latch_fault()
                    ik = None
                    locked_o6_angles = None
                    last_safe_arm_q = None
                    command_ready = False
                    status = "o6_pose_changed_requires_rebuild"
                if ik is None:
                    try:
                        ik = R1A7DryRunIK(
                            args.max_ik_position_error_m,
                            args.max_ik_orientation_error_rad,
                            args.min_jacobian_sigma,
                            args.max_jacobian_condition,
                            o6_joint_q,
                        )
                    except Exception as error:
                        gate.latch_fault()
                        status = f"o6_combined_model_rejected:{error}"
                    else:
                        locked_o6_angles = {
                            side: list(o6_angles_raw[side]) for side in ("left", "right")
                        }
                        counters["combined_model_builds"] += 1

            if lowstate_row is None:
                lowstate_status = lowstate_load_error
            else:
                (
                    arm_q,
                    arm_dq,
                    fixed_q,
                    lowstate_status,
                    lowstate_sample_age,
                    lowstate_transport_age,
                ) = validate_lowstate_frame(
                    lowstate_row,
                    validation_now,
                    args.lowstate_timeout,
                )
                mode_machine = lowstate_row.get("mode_machine")

            if o6_status != O6_VERIFIED_POSE or ik is None:
                gate.latch_fault()
            elif lowstate_status != "valid":
                gate.latch_fault()
                status = f"lowstate_{lowstate_status}"
            else:
                try:
                    ik.set_live_state(arm_q, fixed_q)
                except (ModelLimitError, ValueError) as error:
                    gate.latch_fault()
                    status = f"lowstate_model_rejected:{error}"
                else:
                    offset, status, sequence, tracking_age, transport_age = evaluate_loaded_frame(
                        gate,
                        row,
                        load_error,
                        validation_now,
                    )
                    if row is not None and row.get("armed") is False and not gate.latched:
                        try:
                            ik.reset()
                            last_safe_arm_q = ik.reference_q.copy()
                            command_ready = False
                        except ModelLimitError as error:
                            gate.latch_fault()
                            status = f"lowstate_model_rejected:{error}"
                    fixed_joint_drift = ik.fixed_joint_drift()
                    if (
                        offset is not None
                        and fixed_joint_drift is not None
                        and fixed_joint_drift > args.max_fixed_joint_drift_rad
                    ):
                        gate.latch_fault()
                        offset = None
                        status = "fixed_joint_drift_requires_rearm"

            if offset is not None:
                counters["ready"] += 1
                candidate = evaluate_ik_candidate(
                    ik,
                    gate,
                    offset,
                    args.max_candidate_step_rad,
                )
                status = candidate["status"]
                solution = candidate["solution"]
                ik_position_error = candidate["position_error"]
                ik_orientation_error = candidate["orientation_error"]
                jacobian_sigma_min = candidate["jacobian_sigma_min"]
                jacobian_condition = candidate["jacobian_condition"]
                max_candidate_step = candidate["max_candidate_step"]
                if solution is not None:
                    counters["ik_ok"] += 1
                    last_safe_arm_q = solution.copy()
                    command_ready = True
            elif status not in ("duplicate", "disarmed", "stale"):
                counters["rejected"] += 1

            reference_alignment = (
                None
                if ik is None or ik.reference_q is None or ik.live_q is None
                else float(np.max(np.abs(ik.reference_q - ik.live_q)))
            )
            collision_verified = (
                ik is not None
                and o6_status == O6_VERIFIED_POSE
                and locked_o6_angles is not None
                and o6_angles_raw is not None
                and all(
                    abs(current - locked) <= 1
                    for side in ("left", "right")
                    for current, locked in zip(o6_angles_raw[side], locked_o6_angles[side])
                )
            )
            collision_model = (
                VERIFIED_COLLISION_MODEL if collision_verified else UNVERIFIED_COLLISION_MODEL
            )
            o6_collision_pose = O6_VERIFIED_POSE if collision_verified else O6_UNVERIFIED_POSE

            updated_monotonic_ns = time.monotonic_ns()
            payload = {
                "schema": "r1_a7_arm_bridge_status_v1",
                "platform_profile": PLATFORM_PROFILE,
                "mode": args.mode,
                "publishing_enabled": False,
                "o6_enabled": False,
                "workspace_policy": "r1_a7_model_constrained",
                "collision_model": collision_model,
                "collision_model_sha256": COLLISION_MODEL_SHA256,
                "o6_collision_pose": o6_collision_pose,
                "o6_state_source": "pc2_fc04_only_via_ssh_stdin",
                "o6_state_sequence": o6_sequence,
                "o6_state_age_ms": None
                if o6_state_age is None
                else round(o6_state_age * 1000.0, 3),
                "o6_angles_raw": o6_angles_raw,
                "collision_pair_count": None if ik is None else ik.collision_pair_count,
                "collision_geometry_count": None if ik is None else ik.collision_geometry_count,
                "o6_collision_geometry_count": None
                if ik is None
                else ik.o6_collision_geometry_count,
                "o6_collision_pair_count": None
                if ik is None
                else ik.o6_collision_pair_count,
                "robot_state_source": "read_only_rt_lowstate_shadow",
                "robot_state_sample_age_ms": None
                if lowstate_sample_age is None
                else round(lowstate_sample_age * 1000.0, 3),
                "robot_state_transport_age_ms": None
                if lowstate_transport_age is None
                else round(lowstate_transport_age * 1000.0, 3),
                "mode_machine": mode_machine,
                "live_arm_q": None if ik is None or ik.live_q is None else ik.live_q.tolist(),
                "reference_arm_q": None
                if ik is None or ik.reference_q is None
                else ik.reference_q.tolist(),
                "live_waist_yaw_rad": None
                if ik is None or ik.live_fixed_q is None
                else float(ik.live_fixed_q[0]),
                "reference_waist_yaw_rad": None
                if ik is None or ik.reference_fixed_q is None
                else float(ik.reference_fixed_q[0]),
                "live_head_q": None
                if ik is None or ik.live_fixed_q is None
                else ik.live_fixed_q[1:].tolist(),
                "reference_head_q": None
                if ik is None or ik.reference_fixed_q is None
                else ik.reference_fixed_q[1:].tolist(),
                "reference_alignment_max_rad": reference_alignment,
                "fixed_joint_drift_rad": fixed_joint_drift,
                "status": status,
                "sequence": sequence,
                "tracking_age_ms": None if tracking_age is None else round(tracking_age * 1000.0, 3),
                "transport_age_ms": None if transport_age is None else round(transport_age * 1000.0, 3),
                "ik_position_error_m": ik_position_error,
                "ik_orientation_error_rad": ik_orientation_error,
                "jacobian_sigma_min": jacobian_sigma_min,
                "jacobian_condition": jacobian_condition,
                "max_candidate_step_rad": max_candidate_step,
                "candidate_q": None if solution is None else solution.tolist(),
                "counters": counters,
                "updated_monotonic_ns": updated_monotonic_ns,
            }
            atomic_write_json(status_path, payload)
            atomic_write_json(
                command_path,
                build_command_intent(
                    status=status,
                    solution=solution,
                    last_safe_arm_q=last_safe_arm_q,
                    command_ready=command_ready,
                    gate_latched=gate.latched,
                    source_row=row,
                    sequence=sequence,
                    tracking_age=tracking_age,
                    transport_age=transport_age,
                    mode_machine=mode_machine,
                    reference_arm_q=None if ik is None else ik.reference_q,
                    reference_fixed_q=None if ik is None else ik.reference_fixed_q,
                    fixed_joint_drift=fixed_joint_drift,
                    collision_model=collision_model,
                    o6_collision_pose=o6_collision_pose,
                    o6_state_sequence=o6_sequence,
                    o6_state_age=o6_state_age,
                    o6_angles_raw=o6_angles_raw,
                    updated_monotonic_ns=updated_monotonic_ns,
                ),
            )
            if status != last_printed_status or now >= next_status_print:
                print(
                    f"[R1 ARM BRIDGE] status={status} sequence={sequence} "
                    f"tracking_age_ms={payload['tracking_age_ms']} "
                    f"transport_age_ms={payload['transport_age_ms']} "
                    f"ik_error_m={ik_position_error}",
                    flush=True,
                )
                last_printed_status = status
                next_status_print = now + 2.0
    except KeyboardInterrupt:
        pass
    finally:
        stopped_ns = time.monotonic_ns()
        atomic_write_json(
            command_path,
            build_command_intent(
                status="stopped",
                solution=None,
                last_safe_arm_q=None,
                command_ready=False,
                gate_latched=False,
                source_row=None,
                sequence=None,
                tracking_age=None,
                transport_age=None,
                mode_machine=None,
                reference_arm_q=None,
                reference_fixed_q=None,
                fixed_joint_drift=None,
                collision_model=UNVERIFIED_COLLISION_MODEL,
                o6_collision_pose=O6_UNVERIFIED_POSE,
                o6_state_sequence=None,
                o6_state_age=None,
                o6_angles_raw=None,
                updated_monotonic_ns=stopped_ns,
            ),
        )
        atomic_write_json(
            status_path,
            {
                "schema": "r1_a7_arm_bridge_status_v1",
                "platform_profile": PLATFORM_PROFILE,
                "mode": args.mode,
                "publishing_enabled": False,
                "o6_enabled": False,
                "status": "stopped",
                "counters": counters,
                "updated_monotonic_ns": stopped_ns,
            },
        )
        print(f"[R1 ARM BRIDGE] stopped counters={counters}", flush=True)
        signal.signal(signal.SIGTERM, previous_sigterm)
        command_writer_lock.close()


if __name__ == "__main__":
    main()
