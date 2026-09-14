import hashlib
import importlib.metadata
from pathlib import Path
import socket
import sys
import time

import numpy as np


def capture_metadata(args, camera_config, retargeter):
    from teleop.robot_control.robot_arm import R1_A7_JointArmIndex

    root = Path(__file__).resolve().parents[2]
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "opencv-python", "pin", "dex-retargeting", "vuer", "televuer", "teleimager"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    code_paths = [root / "teleop/teleop_hand_and_arm.py", Path(__file__),
                  root / "teleop/utils/episode_writer.py",
                  root / "teleop/teleimager/src/teleimager/image_client.py"]
    code_paths += list((root / "teleop/robot_control").glob("*.py"))
    models = {
        "arm": root / "assets/r1/r1_a7.urdf",
        "left_hand": retargeter.left.urdf_path,
        "right_hand": retargeter.right.urdf_path,
    }
    names = [joint.name for joint in R1_A7_JointArmIndex]
    height, width = camera_config["head_camera"]["image_shape"]
    return {
        "robot": "R1_A7", "end_effector": "linker_o6",
        "retargeting_method": retargeter.method,
        "retargeting_mapping": retargeter.mapping_name,
        "frequency": args.frequency,
        "image": {"width": width // 2, "height": height,
                  "fps": camera_config["head_camera"]["fps"]},
        "joint_names": {
            "left_arm": names[:7], "right_arm": names[7:],
            "left_ee": list(retargeter.left.hardware_joint_order),
            "right_ee": list(retargeter.right.hardware_joint_order),
            "body": ["waist_yaw", "head_pitch", "head_yaw"],
        },
        "units": {"arm_and_body_qpos": "rad", "arm_qvel": "rad/s",
                  "hand_qpos": "normalized_vendor_axis_0_to_1", "hand_points": "m"},
        "hand_axis_normalization": {
            side: {"lower_rad": hand.hardware_lower.tolist(), "upper_rad": hand.hardware_upper.tolist(),
                   "formula": "q_normalized = (q_rad - lower_rad) / (upper_rad - lower_rad)"}
            for side, hand in (("left", retargeter.left), ("right", retargeter.right))
        },
        "action_semantics": {
            "actions": "requested position targets before publisher limiting or hand smoothing",
            "sample.commands": "last successful SDK Write per publisher; not an execution acknowledgement",
            "states": "latest received motor feedback; each source retains its own receive time",
        },
        "clock": {"host": socket.gethostname(), "sample.timestamp_ns": "Unix wall clock",
                  "source_times": "Ubuntu CLOCK_MONOTONIC receive times; not camera exposure times",
                  "image_sequence": "local received JPEG sequence; repeated=true means reused sample"},
        "camera_config": camera_config,
        "launch_arguments": vars(args).copy(),
        "versions": versions,
        "code_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in code_paths},
        "models": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in models.items()},
    }


class R1Capture:
    def __init__(self, arm_controller, hand_controller, hand_loop, tracking_timeout, image_shape):
        self.arm = arm_controller
        self.hand = hand_controller
        self.hand_loop = hand_loop
        self.tracking_timeout = tracking_timeout
        self.image_shape = tuple(image_shape)
        self.last_image_sequence = None

    def reset_episode(self):
        self.last_image_sequence = None

    def frame(self, tele_data, image, mode):
        arm = self.arm.get_recording_snapshot()
        hand_sample = self.hand_loop.get_recording_sample()
        hand = hand_sample["hand"]
        hand_inputs = hand_sample["target_inputs"]
        now = time.monotonic_ns()
        wall = time.time_ns()

        def source(timestamp, timeout, valid=True, sequence=None):
            received = int(timestamp or 0)
            age = None if received <= 0 else (now - received) / 1e6
            item = {"received_monotonic_ns": received, "age_ms": age,
                    "fresh": bool(valid and age is not None and 0 <= age <= timeout * 1000)}
            if sequence is not None:
                item["sequence"] = int(sequence)
            return item

        if image is None or image.bgr is None:
            raise RuntimeError("Capture lost the decoded stereo image; episode is incomplete")
        image_source = source(image.received_monotonic_ns, 0.5, sequence=image.sequence)
        if not image_source["fresh"] or image.sequence <= 0:
            raise RuntimeError("Capture stereo image is stale; episode is incomplete")
        if image.bgr.shape != (*self.image_shape, 3):
            raise RuntimeError(f"Capture stereo dimensions changed: {image.bgr.shape}")
        image_source["repeated"] = image.sequence == self.last_image_sequence
        self.last_image_sequence = image.sequence
        sources = {
            "image": image_source,
            "xr": source(int(tele_data.motion_data_timestamp * 1e9), self.tracking_timeout,
                         tele_data.motion_data_ready),
            "robot": source(arm["state"]["monotonic_ns"], 0.25,
                             sequence=arm["state"]["sequence"]),
        }
        for side in ("left", "right"):
            sources[f"{side}_hand_tracking"] = source(
                int(getattr(tele_data, f"{side}_hand_timestamp") * 1e9), self.tracking_timeout,
                tele_data.motion_data_ready,
            )
            sources[f"{side}_hand_feedback"] = source(
                hand["state"][side]["monotonic_ns"], 0.25,
                sequence=hand["state"][side]["sequence"],
            )
        if not all(sources[name]["fresh"] for name in ("robot", "left_hand_feedback", "right_hand_feedback")):
            raise RuntimeError("Capture motor feedback is stale; episode is incomplete")

        state, requested = arm["state"], arm["requested"]
        states = {
            "left_arm": {"qpos": state["q"][:7], "qvel": state["dq"][:7], "torque": []},
            "right_arm": {"qpos": state["q"][7:], "qvel": state["dq"][7:], "torque": []},
            "left_ee": {"qpos": hand["state"]["left"]["q"], "qvel": [], "torque": []},
            "right_ee": {"qpos": hand["state"]["right"]["q"], "qvel": [], "torque": []},
            "body": {"qpos": [state["waist_q"], *state["head_q"]]},
        }
        actions = {
            "left_arm": {"qpos": requested["arm_q"][:7]},
            "right_arm": {"qpos": requested["arm_q"][7:]},
            "left_ee": {"qpos": hand["requested"]["left_q"]},
            "right_ee": {"qpos": hand["requested"]["right_q"]},
            "body": {"qpos": [state["waist_q"] if requested["waist_q"] is None else requested["waist_q"],
                               *requested["head_q"]]},
        }
        half = self.image_shape[1] // 2
        return {
            "colors": {"color_0": image.bgr[:, :half], "color_1": image.bgr[:, half:]},
            "states": states,
            "actions": actions,
            "sample": {
                "timestamp_ns": wall, "monotonic_ns": now, "mode": mode, "sources": sources,
                "xr": {
                    "hand_points_frame": "televuer_unitree_hand_wrist_local_meters",
                    "left_hand_points": np.asarray(tele_data.left_hand_pos).tolist(),
                    "right_hand_points": np.asarray(tele_data.right_hand_pos).tolist(),
                    "left_wrist_pose": np.asarray(tele_data.left_wrist_pose).tolist(),
                    "right_wrist_pose": np.asarray(tele_data.right_wrist_pose).tolist(),
                    "head_pose": np.asarray(tele_data.head_pose).tolist(),
                },
                "hand_target_inputs": hand_inputs,
                "commands": {"arm": {"requested": requested, "published": arm["published"]},
                             "hands": {"requested": hand["requested"], "published": hand["published"]}},
            },
        }
