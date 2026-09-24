"""Measured R1 upper-body kinematics and calibrated rigid camera/tool frames."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from teleop.utils.camera_calibration import resolve_image_mirror

ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = ROOT / "assets/r1/r1_a7.urdf"
FRAME_CALIBRATION_PATH = Path.home() / ".config/xr_teleoperate/recording_frames.json"
ARM_JOINTS = [f"{side}_{joint}_joint" for side in ("left", "right")
              for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                            "wrist_roll", "wrist_pitch", "wrist_yaw")]
JOINT_NAMES = ["waist_yaw_joint", "head_pitch_joint", "head_yaw_joint", *ARM_JOINTS]


def rigid_matrix(entry, label):
    rotation = np.asarray(entry["rotation"], dtype=float)
    translation = np.asarray(entry["translation_m"], dtype=float)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError(f"{label}: expected a proper 3x3 rotation and translation in metres")
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix


class R1RecordingTF:
    def __init__(self, camera_calibration=None, frame_calibration_path=FRAME_CALIBRATION_PATH,
                 urdf_path=URDF_PATH):
        import pinocchio as pin

        self.pin = pin
        self.urdf_path = Path(urdf_path)
        urdf = self.urdf_path.read_bytes()
        xml = ET.fromstring(urdf)
        self.parents = {joint.find("child").get("link"): joint.find("parent").get("link")
                        for joint in xml.findall("joint")}
        self.links = [link.get("name") for link in xml.findall("link")]
        self.root = "pelvis_link"
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        if set(self.model.names[1:]) != set(JOINT_NAMES) or self.model.nq != len(JOINT_NAMES):
            raise ValueError("Recording TF requires the 17-joint R1_A7 upper-body URDF")
        self.q_indices = [self.model.joints[self.model.getJointId(name)].idx_q for name in JOINT_NAMES]
        self.data = self.model.createData()
        self.frame_ids = {name: self.model.getFrameId(name) for name in self.links}
        self.static = {}
        self.camera_frames = {}
        self.tcp_frames = {}
        self.task_frames = {}
        self.missing = {"world_to_pelvis": "no base localization; root is the current pelvis",
                        "cup": "no object pose tracker",
                        "left_tcp": "not calibrated", "right_tcp": "not calibrated",
                        "table": "not calibrated", "icecream_machine": "not calibrated"}
        for key, entry in (camera_calibration or {}).get("hand_eye", {}).items():
            camera = entry["camera"]
            child = f"{camera}_optical"
            parent = entry["frame"]
            if parent not in self.frame_ids or child in self.static:
                raise ValueError(f"camera {key}: unknown parent or duplicate optical frame")
            self.static[child] = {"parent": parent, "matrix": rigid_matrix(entry, key),
                                  "source": "camera_hand_eye_calibration",
                                  "image_mirror": resolve_image_mirror(
                                      (camera_calibration or {}).get("cameras", {}).get(camera), entry)}
            for metric in ("rotation_rms_deg", "translation_rms_m"):
                if metric in entry:
                    self.static[child][metric] = entry[metric]
            self.camera_frames[camera] = child
        for camera in ("head_left", "head_right", "left_wrist", "right_wrist"):
            if camera not in self.camera_frames:
                self.missing[f"{camera}_optical"] = "no camera hand-eye extrinsics"
        self.frame_calibration = None
        path = None if frame_calibration_path is None else Path(frame_calibration_path)
        if path is not None and path.is_file():
            raw = path.read_bytes()
            document = json.loads(raw)
            if document.get("schema") != "r1_recording_frames_v1" or document.get("root_frame") != self.root:
                raise ValueError(f"{path}: expected r1_recording_frames_v1 in pelvis_link")
            self.frame_calibration = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                      "document": document}
            for side, entry in document.get("tcp", {}).items():
                if side not in ("left", "right"):
                    raise ValueError(f"{path}: unknown TCP side {side}")
                if entry is None:
                    continue
                child = f"{side}_tcp"
                self.static[child] = {"parent": f"{side}_wrist_yaw_link",
                                      "matrix": rigid_matrix(entry, child), "source": "measured_tcp_calibration"}
                self.tcp_frames[side] = child
                del self.missing[child]
            for name, entry in document.get("task_frames", {}).items():
                if name not in ("table", "icecream_machine"):
                    raise ValueError(f"{path}: unsupported stationary task frame {name}")
                if entry is None:
                    continue
                if document.get("stationary_pelvis_confirmed") is not True:
                    raise ValueError(f"{path}: task frames require stationary_pelvis_confirmed=true")
                self.static[name] = {"parent": self.root, "matrix": rigid_matrix(entry, name),
                                     "source": "measured_stationary_task_calibration"}
                self.task_frames[name] = name
                del self.missing[name]
        self.parents.update({child: entry["parent"] for child, entry in self.static.items()})
        self.metadata = {
            "schema": "r1_tf_v1", "root_frame": self.root,
            "matrix_convention": "T_parent_child; column vectors; p_parent = T_parent_child @ p_child",
            "units": {"translation": "m", "joint_position": "rad"},
            "poses_in_root": "4x4 matrices from each named link/optical/TCP/task frame into pelvis_link",
            "source": "measured robot feedback chosen for camera anchor; no commanded angles or IK lock state",
            "urdf": {"path": str(self.urdf_path), "sha256": hashlib.sha256(urdf).hexdigest(),
                     "xml": urdf.decode("utf-8")},
            "joint_names": JOINT_NAMES, "parents": self.parents,
            "static_transforms": {child: {key: value.tolist() if isinstance(value, np.ndarray) else value
                                           for key, value in entry.items()} for child, entry in self.static.items()},
            "camera_frames": self.camera_frames, "tcp_frames": self.tcp_frames,
            "task_frames": self.task_frames, "unavailable": self.missing,
            "frame_calibration": self.frame_calibration,
            "task_frame_assumption": "valid only while pelvis pose relative to the environment matches calibration; fixed feet do not measure sway",
            "optical_convention": "OpenCV x right, y down, z forward; unmirror raw wrist images before using intrinsics",
        }

    def sample(self, robot, target_ns, camera_time_valid):
        result = {"schema": "r1_tf_v1", "root_frame": self.root,
                  "target_monotonic_ns": target_ns, "state_monotonic_ns": None,
                  "state_sequence": None, "offset_ms": None, "valid": False,
                  "aligned_to_camera": False, "poses_in_root": {}, "tcp_in_task": {}}
        if robot is None:
            result["invalid_reason"] = "no robot feedback at camera time"
            return result
        values = np.asarray([robot["waist_q"], *robot["head_q"], *robot["q"]], dtype=float)
        result.update(state_monotonic_ns=robot["monotonic_ns"], state_sequence=robot["sequence"],
                      offset_ms=(robot["monotonic_ns"] - target_ns) / 1e6)
        if values.shape != (17,) or not np.isfinite(values).all():
            result["invalid_reason"] = "robot joint feedback is not a finite 17-joint vector"
            return result
        q = np.empty(self.model.nq)
        q[self.q_indices] = values
        self.pin.framesForwardKinematics(self.model, self.data, q)
        poses = {name: self.data.oMf[index].homogeneous.copy() for name, index in self.frame_ids.items()}
        for child, entry in self.static.items():
            poses[child] = poses[entry["parent"]] @ entry["matrix"]
        result.update(valid=True, aligned_to_camera=bool(camera_time_valid and abs(result["offset_ms"]) <= 25.0),
                      poses_in_root={name: matrix.tolist() for name, matrix in poses.items()})
        result["tcp_in_task"] = {
            task: {side: np.linalg.solve(poses[task], poses[tcp]).tolist()
                   for side, tcp in self.tcp_frames.items()}
            for task in self.task_frames}
        return result
