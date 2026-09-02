import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np


HARDWARE_AXIS_ORDER = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)

FINGER_CHAINS = {
    "index_mcp_pitch": (5, 6, 7, 8, 9),
    "middle_mcp_pitch": (10, 11, 12, 13, 14),
    "ring_mcp_pitch": (15, 16, 17, 18, 19),
    "pinky_mcp_pitch": (20, 21, 22, 23, 24),
}


def is_tracking_fresh(ready, last_update, timeout, now=None):
    if not ready or last_update <= 0.0:
        return False
    current = time.monotonic() if now is None else now
    return 0.0 <= current - last_update <= timeout


def _angle(first, second):
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm < 1e-9 or second_norm < 1e-9:
        return 0.0
    cosine = np.dot(first, second) / (first_norm * second_norm)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _chain_flexion(points, indices, max_bend):
    segments = [points[end] - points[start] for start, end in zip(indices[:-1], indices[1:])]
    bend = sum(_angle(first, second) for first, second in zip(segments[:-1], segments[1:]))
    return float(np.clip(bend / max_bend, 0.0, 1.0))


def _project_to_plane(vector, normal):
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-9:
        return vector
    unit_normal = normal / normal_norm
    return vector - np.dot(vector, unit_normal) * unit_normal


def _thumb_adduction(points):
    palm_longitudinal = points[10] - points[0]
    palm_lateral = points[20] - points[5]
    palm_normal = np.cross(palm_longitudinal, palm_lateral)
    thumb_direction = _project_to_plane(points[4] - points[1], palm_normal)
    palm_direction = _project_to_plane(palm_longitudinal, palm_normal)
    spread_angle = _angle(thumb_direction, palm_direction)
    return float(1.0 - np.clip(spread_angle / (np.pi / 2.0), 0.0, 1.0))


class LinkerO6HandRetargeter:
    def __init__(self, urdf_path, side):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")

        self.urdf_path = Path(urdf_path)
        self.prefix = "lh_" if side == "left" else "rh_"
        root = ET.parse(self.urdf_path).getroot()

        joint_limits = {}
        joint_axes = {}
        self.urdf_joint_order = []
        for joint in root.findall("joint"):
            if joint.get("type") != "revolute" or joint.find("mimic") is not None:
                continue
            name = joint.get("name")
            limit = joint.find("limit")
            if limit is None:
                continue
            self.urdf_joint_order.append(name)
            joint_limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
            joint_axes[name] = tuple(float(value) for value in joint.find("axis").get("xyz").split())

        self.hardware_joint_order = tuple(self.prefix + axis for axis in HARDWARE_AXIS_ORDER)
        if set(self.urdf_joint_order) != set(self.hardware_joint_order):
            raise ValueError(f"Unexpected O6 actuated joints in {self.urdf_path}")

        self.urdf_to_hardware = tuple(
            self.urdf_joint_order.index(name) for name in self.hardware_joint_order
        )
        self.hardware_lower = np.array([joint_limits[name][0] for name in self.hardware_joint_order])
        self.hardware_upper = np.array([joint_limits[name][1] for name in self.hardware_joint_order])
        self.hardware_axes = np.array([joint_axes[name] for name in self.hardware_joint_order])

    def retarget(self, hand_points):
        points = np.asarray(hand_points, dtype=float)
        if points.shape != (25, 3) or not np.isfinite(points).all():
            raise ValueError("hand_points must be a finite 25x3 array")

        normalized = {
            "thumb_cmc_pitch": _chain_flexion(points, (1, 2, 3, 4), np.pi),
            "thumb_cmc_yaw": _thumb_adduction(points),
        }
        for axis, indices in FINGER_CHAINS.items():
            normalized[axis] = _chain_flexion(points, indices, 1.5 * np.pi)

        normalized_by_name = {
            self.prefix + axis: normalized[axis] for axis in HARDWARE_AXIS_ORDER
        }
        limits_by_name = {
            name: (lower, upper)
            for name, lower, upper in zip(
                self.hardware_joint_order, self.hardware_lower, self.hardware_upper
            )
        }
        urdf_radians = np.array([
            limits_by_name[name][0]
            + normalized_by_name[name] * (limits_by_name[name][1] - limits_by_name[name][0])
            for name in self.urdf_joint_order
        ])
        hardware_radians = urdf_radians[list(self.urdf_to_hardware)]
        hardware_normalized = (
            hardware_radians - self.hardware_lower
        ) / (self.hardware_upper - self.hardware_lower)
        return np.clip(hardware_normalized, 0.0, 1.0)


class LinkerO6Calibration:
    def __init__(self, path):
        self.path = Path(path)
        config = json.loads(self.path.read_text(encoding="utf-8"))
        if tuple(config["axis_order"]) != HARDWARE_AXIS_ORDER:
            raise ValueError("Calibration axis_order does not match Linker O6 hardware order")

        self.name = config["name"]
        self.left_min = np.asarray(config["left"]["input_min"], dtype=float)
        self.left_max = np.asarray(config["left"]["input_max"], dtype=float)
        self.right_min = np.asarray(config["right"]["input_min"], dtype=float)
        self.right_max = np.asarray(config["right"]["input_max"], dtype=float)
        for lower, upper in (
            (self.left_min, self.left_max),
            (self.right_min, self.right_max),
        ):
            if lower.shape != (6,) or upper.shape != (6,):
                raise ValueError("Calibration bounds must contain six axes per hand")
            if not np.isfinite(lower).all() or not np.isfinite(upper).all():
                raise ValueError("Calibration bounds must be finite")
            if np.any(upper <= lower):
                raise ValueError("Calibration input_max must be greater than input_min")

    def apply(self, left_target, right_target):
        left = np.asarray(left_target, dtype=float)
        right = np.asarray(right_target, dtype=float)
        if left.shape != (6,) or right.shape != (6,):
            raise ValueError("Calibration targets must contain six axes per hand")
        return (
            np.clip((left - self.left_min) / (self.left_max - self.left_min), 0.0, 1.0),
            np.clip((right - self.right_min) / (self.right_max - self.right_min), 0.0, 1.0),
        )


class DualLinkerO6Retargeter:
    def __init__(self, urdf_root):
        root = Path(urdf_root)
        self.left = LinkerO6HandRetargeter(
            root / "left" / "linkerhand_o6_left.urdf", "left"
        )
        self.right = LinkerO6HandRetargeter(
            root / "right" / "linkerhand_o6_right.urdf", "right"
        )

    def retarget(self, left_hand_points, right_hand_points):
        return self.left.retarget(left_hand_points), self.right.retarget(right_hand_points)
