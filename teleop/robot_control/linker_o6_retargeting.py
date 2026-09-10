from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np


HARDWARE_AXIS_ORDER = (
    "thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch",
    "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch",
)
METHODS = ("vector", "position", "dexpilot")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HUMAN_TIPS = np.array([4, 9, 14, 19, 24])
# TeleVuer wrist-local Unitree hand convention -> O6 hand_base_link.
UNITREE_HAND_TO_O6 = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
# Distal STL +Z apex, in metres; no arm installation transform belongs here.
TIP_OFFSETS = {
    "thumb": (-0.00639294, 0.0, 0.04851740),
    "index": (0.01466039, 0.0, 0.03757212),
    "middle": (0.01466039, 0.0, 0.03757212),
    "ring": (0.01466039, 0.0, 0.03757212),
    "pinky": (0.01466039, 0.0, 0.03757212),
}


def is_tracking_fresh(ready, last_update, timeout, now=None):
    if not ready or last_update <= 0.0:
        return False
    current = time.monotonic() if now is None else now
    return 0.0 <= current - last_update <= timeout


class LinkerO6HandRetargeter:
    def __init__(self, urdf_path, side, method="vector"):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        from dex_retargeting import RetargetingConfig

        self.urdf_path = Path(urdf_path).resolve()
        self.method = method
        self.prefix = "lh_" if side == "left" else "rh_"
        self.hardware_joint_order = tuple(self.prefix + name for name in HARDWARE_AXIS_ORDER)
        self.wrist_link_name = self.prefix + "hand_base_link"
        self.tip_link_names = [self.prefix + finger + "_tip" for finger in FINGERS]
        root = ET.parse(self.urdf_path).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        active_names = {
            name for name, joint in joints.items()
            if joint.get("type") == "revolute" and joint.find("mimic") is None
        }
        if active_names != set(self.hardware_joint_order):
            raise ValueError(f"Unexpected O6 actuated joints in {self.urdf_path}")
        limits = np.array([
            [float(joints[name].find("limit").get(bound)) for bound in ("lower", "upper")]
            for name in self.hardware_joint_order
        ])
        self.hardware_lower, self.hardware_upper = limits[:, 0].copy(), limits[:, 1].copy()

        # Propagate dependent joint bounds without changing the vendor hardware scale.
        for joint in joints.values():
            mimic = joint.find("mimic")
            if mimic is None:
                continue
            source_index = self.hardware_joint_order.index(mimic.get("joint"))
            multiplier = float(mimic.get("multiplier", "1"))
            offset = float(mimic.get("offset", "0"))
            bound = joint.find("limit")
            source_bounds = sorted((float(bound.get(key)) - offset) / multiplier for key in ("lower", "upper"))
            limits[source_index, 0] = max(limits[source_index, 0], source_bounds[0])
            limits[source_index, 1] = min(limits[source_index, 1], source_bounds[1])
        if np.any(limits[:, 1] <= limits[:, 0]):
            raise ValueError("O6 mimic joint bounds have an empty intersection")
        for name, (lower, upper) in zip(self.hardware_joint_order, limits):
            joints[name].find("limit").set("lower", str(lower))
            joints[name].find("limit").set("upper", str(upper))

        links = {link.get("name"): link for link in root.findall("link")}
        for link in links.values():
            for geometry in list(link.findall("visual")) + list(link.findall("collision")):
                link.remove(geometry)
        for finger, tip_name in zip(FINGERS, self.tip_link_names):
            distal_name = self.prefix + finger + "_distal"
            if distal_name not in links or tip_name in links:
                raise ValueError(f"Unexpected O6 distal/tip links: {distal_name}, {tip_name}")
            ET.SubElement(root, "link", name=tip_name)
            joint = ET.SubElement(root, "joint", name=tip_name + "_fixed", type="fixed")
            ET.SubElement(joint, "parent", link=distal_name)
            ET.SubElement(joint, "child", link=tip_name)
            ET.SubElement(joint, "origin", xyz=" ".join(map(str, TIP_OFFSETS[finger])), rpy="0 0 0")

        config = {
            "type": method,
            "target_joint_names": list(self.hardware_joint_order),
            "add_dummy_free_joint": False,
            "ignore_mimic_joint": False,
            "has_joint_limits": True,
            "scaling_factor": 1.0,
            "low_pass_alpha": 1.0,  # Pass through; the O6 controller owns time-based smoothing.
        }
        if method == "vector":
            config.update(
                target_origin_link_names=[self.wrist_link_name] * 5,
                target_task_link_names=self.tip_link_names,
                target_link_human_indices_vector=[np.zeros(5, dtype=int), HUMAN_TIPS],
            )
        elif method == "position":
            config.update(
                target_link_names=self.tip_link_names,
                target_link_human_indices_position=HUMAN_TIPS,
            )
        else:
            logical_tips = np.concatenate(([0], HUMAN_TIPS))
            pairs = [(j, i) for i in range(1, 5) for j in range(i + 1, 6)] + [(0, i) for i in range(1, 6)]
            config.update(
                wrist_link_name=self.wrist_link_name,
                finger_tip_link_names=self.tip_link_names,
                target_link_human_indices_dexpilot=logical_tips[np.array(pairs).T],
            )
        with tempfile.TemporaryDirectory(prefix="linker-o6-retarget-") as directory:
            model_path = Path(directory) / self.urdf_path.name
            ET.ElementTree(root).write(model_path, encoding="utf-8", xml_declaration=True)
            config["urdf_path"] = str(model_path)
            self.retargeting = RetargetingConfig.from_dict(config).build()
        self.retargeting.optimizer.set_joint_limit(self.retargeting.joint_limits, epsilon=0.0)
        self.retargeting.optimizer.opt.set_maxeval(80)
        self.retargeting.optimizer.opt.set_maxtime(0.02)
        self._hardware_indices = [self.retargeting.joint_names.index(name) for name in self.hardware_joint_order]
        self.reset()

    def to_robot_points(self, hand_points):
        points = np.asarray(hand_points, dtype=float)
        if points.shape != (25, 3) or not np.isfinite(points).all():
            raise ValueError("hand_points must be a finite 25x3 array")
        relative = points - points[0]
        if np.max(np.linalg.norm(relative, axis=1)) < 1e-6:
            raise ValueError("hand_points must contain a tracked hand, not an empty skeleton")
        return relative @ UNITREE_HAND_TO_O6.T

    def retarget(self, hand_points):
        points = self.to_robot_points(hand_points)
        indices = self.retargeting.optimizer.target_link_human_indices
        reference = points[indices] if self.method == "position" else points[indices[1]] - points[indices[0]]
        try:
            qpos = self.retargeting.retarget(reference)
        except Exception as error:
            self.reset()
            raise RuntimeError(f"O6 {self.method} optimization failed") from error
        if self.retargeting.optimizer.opt.last_optimize_result() < 0 or not np.isfinite(qpos).all():
            self.reset()
            raise RuntimeError(f"O6 {self.method} optimization failed")
        hardware_radians = qpos[self._hardware_indices]
        return np.clip((hardware_radians - self.hardware_lower) / (self.hardware_upper - self.hardware_lower), 0.0, 1.0)

    def reset(self):
        self.retargeting.reset()
        self.retargeting.last_qpos = np.clip(
            np.zeros(6, dtype=np.float32),
            self.retargeting.joint_limits[:, 0], self.retargeting.joint_limits[:, 1],
        )
        self.retargeting.filter.reset()
        if self.method == "dexpilot":
            self.retargeting.optimizer.projected[:] = False


class DualLinkerO6Retargeter:
    def __init__(self, urdf_root, method="vector"):
        root = Path(urdf_root)
        self.left = LinkerO6HandRetargeter(root / "left" / "linkerhand_o6_left.urdf", "left", method)
        self.right = LinkerO6HandRetargeter(root / "right" / "linkerhand_o6_right.urdf", "right", method)
        self.method = method
        self.mapping_name = f"linker_o6_dex_{method}_v1"

    def retarget(self, left_hand_points, right_hand_points):
        return self.left.retarget(left_hand_points), self.right.retarget(right_hand_points)

    def reset(self):
        self.left.reset()
        self.right.reset()
