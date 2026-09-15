import hashlib
import importlib.metadata
from pathlib import Path
import socket
import sys
import time

import numpy as np

from teleop.utils.camera_calibration import (COLOR_KEYS, WRIST_COLOR_KEYS, camera_calibration_metadata,
                                             expected_cameras)
from teleop.utils.frame_sync import CameraSynchronizer


def color_image_specs(camera_config):
    """Describe every colour stream this run records, keyed by colour key.

    `color_0`/`color_1` are the two halves of the head stereo frame and
    `color_2`/`color_3` the palm cameras, matching the long-standing upstream
    mapping. The per-key sizes matter because the streams no longer share one
    resolution: a head half is 544x448 while a wrist frame is 640x480.
    """
    head = camera_config.get("head_camera") or {}
    height, width = head.get("image_shape") or [480, 640]
    specs = {}
    if head.get("binocular"):
        for key, camera, half in (("color_0", COLOR_KEYS["color_0"], "left"),
                                  ("color_1", COLOR_KEYS["color_1"], "right")):
            specs[key] = {"camera": camera, "stream": f"head_stereo_{half}",
                          "width": width // 2, "height": height, "fps": head.get("fps")}
    else:
        specs["color_0"] = {"camera": COLOR_KEYS["color_0"], "stream": "head_mono",
                            "width": width, "height": height, "fps": head.get("fps")}
    for side in ("left", "right"):
        wrist = camera_config.get(f"{side}_wrist_camera") or {}
        if not wrist.get("enable_zmq"):
            continue
        wrist_height, wrist_width = wrist.get("image_shape") or [height, width]
        specs[WRIST_COLOR_KEYS[side]] = {"camera": COLOR_KEYS[WRIST_COLOR_KEYS[side]],
                                         "stream": f"{side}_wrist",
                                         "width": wrist_width, "height": wrist_height,
                                         "fps": wrist.get("fps")}
    return specs


def capture_metadata(args, camera_config, retargeter, calibration=None):
    from teleop.robot_control.robot_arm import R1_A7_JointArmIndex

    root = Path(__file__).resolve().parents[2]
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "opencv-python", "pin", "dex-retargeting", "vuer", "televuer", "teleimager"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    code_paths = [root / "teleop/teleop_hand_and_arm.py", Path(__file__),
                  root / "teleop/utils/camera_calibration.py",
                  root / "teleop/utils/episode_writer.py",
                  root / "teleop/teleimager/src/teleimager/client.py"]
    code_paths += list((root / "teleop/robot_control").glob("*.py"))
    models = {
        "arm": root / "assets/r1/r1_a7.urdf",
        "left_hand": retargeter.left.urdf_path,
        "right_hand": retargeter.right.urdf_path,
    }
    names = [joint.name for joint in R1_A7_JointArmIndex]
    height, width = camera_config["head_camera"]["image_shape"]
    images = color_image_specs(camera_config)
    return {
        "robot": "R1_A7", "end_effector": "linker_o6",
        "retargeting_method": retargeter.method,
        "retargeting_mapping": retargeter.mapping_name,
        "frequency": args.frequency,
        "image": {"width": width // 2, "height": height,
                  "fps": camera_config["head_camera"]["fps"]},
        # `image` above describes the head stereo half only, because the stereo
        # preview tools read it. `images` is the authoritative roster: one entry
        # per colour key actually written to the episode.
        "images": images,
        "camera_sync": {
            "method": "nearest arrival time to the anchor stream's newest frame",
            "anchor": "head_image",
            "tolerance_ms": float(getattr(args, "camera_sync_tolerance_ms",
                                          CameraSynchronizer.DEFAULT_TOLERANCE_MS)),
            "streams": sorted(images),
            "note": ("The head stereo pair is one frame off one sensor, so its two eyes are "
                     "inherently simultaneous. The palm cameras are separate UVC devices on PC2 "
                     "with no shared trigger, so each sample pairs them to the head frame in "
                     "hand rather than to the sample clock."),
        },
        "joint_names": {
            "left_arm": names[:7], "right_arm": names[7:],
            "left_ee": list(retargeter.left.hardware_joint_order),
            "right_ee": list(retargeter.right.hardware_joint_order),
            "body": ["waist_yaw", "head_pitch", "head_yaw"],
        },
        "units": {"arm_and_body_qpos": "rad", "arm_qvel": "rad/s",
                  "arm_torque": "N*m measured at the joint (DDS tau_est)",
                  "arm_torque_command": "N*m requested feed-forward (arm_tau)",
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
            "states.*.torque": ("measured joint torque; an empty list means the source exposes no "
                                "torque, never that the torque was zero"),
            "colors": ("one JPEG per colour key per sample; a missing camera is null, and the "
                       "matching sample.sources entry carries fresh=false"),
        },
        "clock": {"host": socket.gethostname(), "sample.timestamp_ns": "Unix wall clock",
                  "source_times": "Ubuntu CLOCK_MONOTONIC receive times; not camera exposure times",
                  "image_sequence": "local received JPEG sequence; repeated=true means reused sample",
                  "wrist_image_sequence": ("per-side received JPEG sequence; the wrist streams run at "
                                           "~21-22 fps against a 40 Hz sample loop, so repeated=true "
                                           "is expected and is not a stall"),
                  "camera_alignment": ("sources.*.offset_ms is the frame's arrival time minus the "
                                       "anchor (head) frame's arrival time; sample.camera_alignment.skew_ms "
                                       "is the spread across cameras. The three streams are independent "
                                       "devices with no shared trigger and the ZMQ packets carry no "
                                       "capture timestamp, so these are arrival-time differences, not "
                                       "exposure-time differences.")},
        "camera_calibration": camera_calibration_metadata(
            calibration, expected_cameras(camera_config),
            {spec["camera"]: (spec["width"], spec["height"]) for spec in images.values()}),
        "camera_config": camera_config,
        "launch_arguments": vars(args).copy(),
        "versions": versions,
        "code_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in code_paths},
        "models": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in models.items()},
    }


class R1Capture:
    def __init__(self, arm_controller, hand_controller, hand_loop, tracking_timeout, image_shape,
                 wrist_image_shapes=None, wrist_timeout=0.5, sync_capacity=8,
                 sync_tolerance_ms=CameraSynchronizer.DEFAULT_TOLERANCE_MS):
        self.arm = arm_controller
        self.hand = hand_controller
        self.hand_loop = hand_loop
        self.tracking_timeout = tracking_timeout
        self.image_shape = tuple(image_shape)
        if self.image_shape[1] % 2:
            raise ValueError(f"stereo image width must be even to split into two eyes: {self.image_shape}")
        # side -> (height, width) for every palm camera that is recorded. Empty
        # means the episode has only the two head colours.
        self.wrist_image_shapes = {side: tuple(shape) for side, shape in (wrist_image_shapes or {}).items()}
        self.wrist_timeout = wrist_timeout
        # The head is the anchor: it is the slowest stream (15 Hz against the
        # palms' ~21-22 Hz) and the one the episode cannot be recorded without,
        # so anchoring there minimises the worst-case spread across cameras.
        self.sync = CameraSynchronizer(
            ["head"] + [f"{side}_wrist" for side in sorted(self.wrist_image_shapes)],
            anchor="head", capacity=sync_capacity, tolerance_ms=sync_tolerance_ms)
        self.last_image_sequence = None
        self.last_wrist_sequences = {}
        self.wrist_seen = {}

    def reset_episode(self):
        self.last_image_sequence = None
        self.last_wrist_sequences = {}
        self.wrist_seen = {}
        self.sync.clear()

    def observe(self, image, wrist_images=None):
        """Record the newest frame of every stream so pairing has history.

        Called from the control loop on every iteration, including while no
        episode is being recorded, so the first sample of an episode already
        has frames to choose between instead of falling back to "the latest".
        """
        self.sync.offer("head", image)
        for side in self.wrist_image_shapes:
            self.sync.offer(f"{side}_wrist", (wrist_images or {}).get(side))

    def frame(self, tele_data, image, mode, wrist_images=None):
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
        # Measured torque is absent, not zero, when the DDS build exposes no
        # torque field; an empty list keeps that distinction visible.
        torque = state.get("tau")
        states = {
            "left_arm": {"qpos": state["q"][:7], "qvel": state["dq"][:7],
                         "torque": [] if torque is None else [float(value) for value in torque[:7]]},
            "right_arm": {"qpos": state["q"][7:], "qvel": state["dq"][7:],
                          "torque": [] if torque is None else [float(value) for value in torque[7:]]},
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
        colors = {"color_0": image.bgr[:, :half], "color_1": image.bgr[:, half:]}
        # Pair every stream around the head frame in hand instead of taking
        # whatever each palm camera happened to deliver most recently, which
        # measured a median 31.6 ms apart and 12.8% of samples over 50 ms.
        self.observe(image, wrist_images)
        pairing = self.sync.pair(now)
        for side in sorted(self.wrist_image_shapes):
            name = f"{side}_wrist"
            entry = None if pairing is None else pairing.frames.get(name)
            wrist = None if entry is None else entry.payload
            pixels = None
            sample = None
            if wrist is not None and wrist.bgr is not None and int(wrist.sequence or 0) > 0:
                sample = source(wrist.received_monotonic_ns, self.wrist_timeout, sequence=wrist.sequence)
                if sample["fresh"]:
                    if wrist.bgr.shape != (*self.wrist_image_shapes[side], 3):
                        raise RuntimeError(f"Capture {side} wrist dimensions changed: {wrist.bgr.shape}")
                    sample["repeated"] = wrist.sequence == self.last_wrist_sequences.get(side)
                    sample["offset_ms"] = float(pairing.offsets_ms.get(name, 0.0))
                    self.last_wrist_sequences[side] = wrist.sequence
                    self.wrist_seen[side] = True
                    pixels = wrist.bgr
            if pixels is None:
                # A palm camera that never delivered at all is a wiring fault and
                # fails the episode. A camera that drops one frame mid-episode is
                # recorded as an absent image with fresh=false, so the offline
                # checker excludes exactly those frames instead of the operator
                # losing the whole demonstration.
                if not self.wrist_seen.get(side, False):
                    raise RuntimeError(f"Capture {side} wrist image never arrived; episode is incomplete")
                sample = sample or {"received_monotonic_ns": 0, "age_ms": None, "fresh": False}
                if "offset_ms" not in sample and pairing is not None and name in pairing.offsets_ms:
                    sample["offset_ms"] = float(pairing.offsets_ms[name])
            sources[f"{side}_wrist_image"] = sample
            colors[WRIST_COLOR_KEYS[side]] = pixels
        if pairing is not None:
            image_source["offset_ms"] = float(pairing.offsets_ms.get("head", 0.0))
        return {
            "colors": colors,
            "states": states,
            "actions": actions,
            "sample": {
                "timestamp_ns": wall, "monotonic_ns": now, "mode": mode, "sources": sources,
                "camera_alignment": None if pairing is None else self.sync.alignment(pairing),
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
