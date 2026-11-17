"""The deployed ice-cream robot's observation and action contract."""

import numpy as np

from teleop.utils.camera_color import IMAGE_COLOR_PROFILE

GROUPS = {"left_arm": 7, "right_arm": 7, "left_ee": 6, "right_ee": 1, "body": 3}
GROUP_SLICES = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "left_ee": slice(14, 20),
                "right_ee": slice(20, 21), "body": slice(21, 24)}
ACTION_DIM = 24
CAMERAS = ["head_left", "head_right", "left_wrist", "right_wrist"]
DEVICES = {"left": "linker_o6", "right": "dex1_1"}
UNITS = ["rad"] * 14 + ["normalized_0_to_1"] * 6 + ["rad"] * 4
JOINT_NAMES = [
    *[f"{side}_arm/k{side.title()}{name}" for side in ("left", "right")
      for name in ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "Elbow", "WristRoll", "WristPitch",
                   "Wristyaw" if side == "left" else "WristYaw")],
    *[f"left_ee/lh_{name}" for name in ("thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch",
                                      "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch")],
    "right_ee/dex1_1_gripper", "body/waist_yaw", "body/head_pitch", "body/head_yaw",
]


def validate_contract(metadata):
    for key, expected in (("state_dim", ACTION_DIM), ("action_dim", ACTION_DIM), ("joint_groups", GROUPS),
                          ("end_effectors", DEVICES), ("units", UNITS), ("joint_names", JOINT_NAMES),
                          ("camera_names", CAMERAS)):
        if metadata.get(key) != expected:
            raise ValueError(f"24D left O6 + right Dex1 contract mismatch: {key}")
    limits = (metadata.get("gripper_position_limits") or {}).get("right", {})
    low, high = limits.get("min_q_rad"), limits.get("max_q_rad")
    if (not isinstance(low, (int, float)) or not isinstance(high, (int, float))
            or not np.isfinite([low, high]).all() or not 0 <= low < high <= 5.4):
        raise ValueError("Checkpoint requires the recorded right Dex1 limits in radians")
    if metadata.get("image_transform_kind") != "letterbox_rgb_v1":
        raise ValueError("Checkpoint requires the shared letterbox_rgb_v1 image transform")
    if metadata.get("image_color_profile") != IMAGE_COLOR_PROFILE:
        raise ValueError("Dataset/checkpoint color profile is uncorrected; use a corrected RGB export and retrain")
    controller = metadata.get("controller_config") or {}
    if controller.get("dex1_transport") != "lowcmd" or controller.get("dex1_slot") != 33:
        raise ValueError("This policy runner requires the right Dex1 in lowcmd slot 33")
    for key in ("dex1_kp", "dex1_kd", "dex1_grip_delta"):
        value = controller.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Missing or invalid recorded controller setting: {key}")
    return float(low), float(high)
