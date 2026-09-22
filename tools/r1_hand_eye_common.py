#!/usr/bin/env python3
"""Shared helpers for R1 eye-in-hand (wrist / head) calibration.

FK uses pinocchio on ``assets/r1/r1_a7.urdf`` (nq=17). Motor indices match
``R1_A7_JointIndex`` on production. Does not open DDS and does not publish.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_PATTERN = (9, 6)
DEFAULT_SQUARE_M = 0.030
DEFAULT_URDF = REPO / "assets" / "r1" / "r1_a7.urdf"
DEFAULT_OUT_ROOT = Path("~/r1-cam-calib")

# Motor index -> pinocchio q slot (URDF joint order on r1_a7.urdf).
MOTOR_TO_PIN_Q = (
    (13, 0),   # waist_yaw_joint
    (29, 1),   # head_pitch_joint
    (30, 2),   # head_yaw_joint
    (15, 3),   # left_shoulder_pitch_joint
    (16, 4),
    (17, 5),
    (18, 6),
    (19, 7),
    (20, 8),
    (21, 9),   # left_wrist_yaw_joint
    (22, 10),  # right_shoulder_pitch_joint
    (23, 11),
    (24, 12),
    (25, 13),
    (26, 14),
    (27, 15),
    (28, 16),  # right_wrist_yaw_joint
)

TARGETS = {
    "left_wrist": {
        "camera": "left_wrist",
        "frame": "left_wrist_yaw_link",
        "kind": "wrist",
        "side": "left",
        "port": 55556,
        "topic": "left_wrist_camera",
        "service": "r1-camera-forward-55556.service",
        "image_size": (640, 480),
        "distortion_model": "plumb_bob",
    },
    "right_wrist": {
        "camera": "right_wrist",
        "frame": "right_wrist_yaw_link",
        "kind": "wrist",
        "side": "right",
        "port": 55557,
        "topic": "right_wrist_camera",
        "service": "r1-camera-forward-55557.service",
        "image_size": (640, 480),
        "distortion_model": "plumb_bob",
    },
    "head": {
        "camera": "head_left",
        "frame": "head_yaw_link",
        "kind": "head",
        "side": "head",
        "port": 55555,
        "topic": "head_camera",
        "service": "r1-camera-forward-55555.service",
        "image_size": (544, 448),
        "distortion_model": "fisheye",
        "stereo_frame_size": (1088, 448),
        "also_camera": "head_right",
    },
}


class HandEyeError(RuntimeError):
    """Raised when hand-eye capture or fit cannot proceed."""


def target_spec(name):
    try:
        return TARGETS[name]
    except KeyError as error:
        raise HandEyeError(f"target must be one of {sorted(TARGETS)}, got {name!r}") from error


def motor_q_to_pinocchio_q(full_q):
    """Map a length-35 lowstate motor_q vector to the 17-DoF URDF q."""
    full_q = np.asarray(full_q, dtype=float).reshape(-1)
    if full_q.size < 31:
        raise HandEyeError(f"motor_q length {full_q.size} is too short (need >= 31)")
    q = np.zeros(17, dtype=float)
    for motor_index, pin_index in MOTOR_TO_PIN_Q:
        value = float(full_q[motor_index])
        if not np.isfinite(value):
            raise HandEyeError(f"motor_q[{motor_index}] is not finite")
        q[pin_index] = value
    return q


def pinocchio_q_to_named(q):
    q = np.asarray(q, dtype=float).reshape(17)
    return {
        "waist_yaw": float(q[0]),
        "head_pitch": float(q[1]),
        "head_yaw": float(q[2]),
        "left_arm": [float(x) for x in q[3:10]],
        "right_arm": [float(x) for x in q[10:17]],
        "pinocchio_q": [float(x) for x in q],
    }


class R1A7FK:
    """Forward kinematics for wrist / head links in the URDF root frame."""

    def __init__(self, urdf_path=None):
        import pinocchio as pin

        urdf_path = Path(urdf_path or DEFAULT_URDF).expanduser().resolve()
        if not urdf_path.is_file():
            raise HandEyeError(f"URDF not found: {urdf_path}")
        self.pin = pin
        self.urdf_path = urdf_path
        self.robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), str(urdf_path.parent))
        if int(self.robot.model.nq) != 17:
            raise HandEyeError(f"expected r1_a7 nq=17, got {self.robot.model.nq}")
        self.data = self.robot.model.createData()
        self._frame_ids = {}

    def frame_id(self, frame_name):
        if frame_name not in self._frame_ids:
            if not self.robot.model.existFrame(frame_name):
                raise HandEyeError(f"URDF has no frame {frame_name!r}")
            self._frame_ids[frame_name] = self.robot.model.getFrameId(frame_name)
        return self._frame_ids[frame_name]

    def link_pose(self, pinocchio_q, frame_name):
        """Return 4x4 T_link_in_root (gripper2base for OpenCV eye-in-hand)."""
        q = np.asarray(pinocchio_q, dtype=float).reshape(17)
        if not np.all(np.isfinite(q)):
            raise HandEyeError("pinocchio_q must be finite")
        self.pin.framesForwardKinematics(self.robot.model, self.data, q)
        pose = self.data.oMf[self.frame_id(frame_name)]
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = np.asarray(pose.rotation, dtype=float)
        matrix[:3, 3] = np.asarray(pose.translation, dtype=float).reshape(3)
        return matrix


def se3_from_rt(R, t):
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return matrix


def rt_from_se3(matrix):
    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    return matrix[:3, :3].copy(), matrix[:3, 3].copy()


def invert_se3(matrix):
    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rotation_geodesic_deg(Ra, Rb):
    Ra = np.asarray(Ra, dtype=float).reshape(3, 3)
    Rb = np.asarray(Rb, dtype=float).reshape(3, 3)
    R = Ra.T @ Rb
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def object_points(pattern=DEFAULT_PATTERN, square=DEFAULT_SQUARE_M):
    cols, rows = int(pattern[0]), int(pattern[1])
    points = np.zeros((cols * rows, 1, 3), dtype=np.float64)
    points[:, 0, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square)
    return points


def capture_day_dirs(out_root, target, day=None):
    import time

    spec = target_spec(target)
    root = Path(out_root).expanduser()
    day = day or time.strftime("%Y%m%d")
    day_dir = root / day
    sample_dir = day_dir / "hand_eye" / spec["camera"]
    return day_dir, sample_dir
