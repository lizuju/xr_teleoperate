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


def capture_metadata(args, camera_config, retargeter, calibration=None, recording_tf=None):
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
                  root / "teleop/utils/recording_tf.py",
                  root / "teleop/utils/episode_writer.py",
                  root / "teleop/teleimager/src/teleimager/client.py"]
    code_paths += list((root / "teleop/robot_control").glob("*.py"))
    if getattr(args, "tracking_source", "webxr") == "visionpro":
        code_paths += [root / "teleop/utils/visionpro_source.py", root / "tools/visionpro_bridge.py",
                       root / "tools/visionpro_protocol/handtracking.proto"]
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
        "tf": None if recording_tf is None else recording_tf.metadata,
        "retargeting_method": retargeter.method,
        "retargeting_mapping": retargeter.mapping_name,
        "frequency": args.frequency,
        "tracking_source": getattr(args, "tracking_source", "webxr"),
        "tracking_input": ({
            "protocol_version": 1,
            "client_upstream_commit": "4c549905c2a8b214d79f7cd88e535101a1ce32af",
            "mapping": "ARKit first25 -> WebKit per-side joint axes -> existing TeleVuerWrapper",
            "timestamp_method": "minimum receive-minus-device clock offset; estimated pose age excludes best observed transport delay; not hardware synchronization",
            "prediction": "reported per tracking diagnostic as prediction_seconds; set app offset to 0 ms for capture",
        } if getattr(args, "tracking_source", "webxr") == "visionpro" else {"backend": "televuer"}),
        "image": {"width": width // 2, "height": height,
                  "fps": camera_config["head_camera"]["fps"]},
        # `image` above describes the head stereo half only, because the stereo
        # preview tools read it. `images` is the authoritative roster: one entry
        # per colour key actually written to the episode.
        "images": images,
        "camera_sync": {
            "method": "minimax nearest frame across newest stream anchors; no sequence rewind",
            "anchor": "selected per sample",
            "tolerance_ms": float(getattr(args, "camera_sync_tolerance_ms",
                                          CameraSynchronizer.DEFAULT_TOLERANCE_MS)),
            "streams": sorted(images),
            "note": ("PC2 software acquisition times mapped by round trips to Ubuntu monotonic time. "
                     "Head eyes use separate RTP streams; eye timestamps and skew are preserved. "
                     "No shared hardware trigger or exposure-time guarantee."),
        },
        "sensor_sync": {
            "schema": "r1_sensor_sync_v1", "feedback_tolerance_ms": 25.0,
            "clock_uncertainty_limit_ms": 5.0,
            "method": "nearest received feedback to selected camera anchor; no interpolation or extrapolation",
            "imu": {"quaternion_order": "wxyz", "gyroscope_unit": "rad/s",
                    "accelerometer_unit": "m/s^2", "rpy_unit": "rad",
                    "temperature_unit": "SDK raw int16; temperature scale not verified",
                    "frame": "Unitree SDK native IMU frame; no extrinsic rotation applied",
                    "quaternion_reference": "vendor attitude reference; not calibrated to camera/world",
                    "tick": "raw LowState uint32 device counter; not mapped to host time",
                    "valid": "finite values and quaternion norm within [0.9, 1.1]; zeros remain raw but invalid",
                    "packets": "all received DDS packets since previous sample; first sample seeds latest; buffer losses counted"},
            "clock_mapping": "host_ns = PC2_ns + offset_ns; minimum-delay round-trip estimate",
            "uncertainty": "half network round trip plus 100 ppm age allowance; excludes sensor/pipeline latency",
            "xr": "raw received poses only; XR device clock is unavailable and poses are not retimed",
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
                  "hand_qpos": ("normalized_vendor_axis_0_to_1, from a one-byte device "
                                "register: 256 levels over the joint range"),
                  "hand_points": "m",
                  "hand_qvel": "normalized_0_to_1 relative joint speed, not rad/s",
                  "hand_torque": ("normalized_0_to_1 relative joint torque from the O6 "
                                  "telemetry block. Not N*m and not a measured contact "
                                  "force: it is inferred from motor current, so it also "
                                  "carries the finger's own weight and inertia, and no "
                                  "calibration can separate those out. The O6 register is "
                                  "one byte per channel (LinkerHand's native scale), so the "
                                  "value has 256 levels -- 1/255 resolution, saturating at "
                                  "1.0 -- and the device is the quantizer: PC2 forwards it "
                                  "already normalised and leaves q_raw, dq_raw and tau_est_raw "
                                  "at zero, so no more precision can be recovered "
                                  "downstream. Use it for contact and grasp events, not as "
                                  "a force regressor."),
                  "hand_temperature": "raw device register value",
                  "hand_errors": ("per-joint O6 fault code; all zeros means no fault was "
                                  "reported. PC2 only fills reserve[0] once hand_dds_service "
                                  "forwards hand_state.errors")},
        "hand_axis_normalization": {
            side: {"lower_rad": hand.hardware_lower.tolist(), "upper_rad": hand.hardware_upper.tolist(),
                   "formula": "q_normalized = (q_rad - lower_rad) / (upper_rad - lower_rad)"}
            for side, hand in (("left", retargeter.left), ("right", retargeter.right))
        },
        "action_semantics": {
            "actions": "requested position targets before publisher limiting or hand smoothing",
            "sample.commands": "last successful SDK Write per publisher; not an execution acknowledgement",
            "states": "latest received motor feedback; each source retains its own receive time",
            "sample.aligned_states": "separate nearest-feedback observations at camera anchor; raw actions unchanged",
            "sample.imu": "latest SDK IMU with the same receive sequence and tick as robot state",
            "sample.imu_packets": "unresampled IMU packet batch since previous recording sample",
            "states.*.torque": ("measured joint torque; an empty list means the source exposes no "
                                "torque, never that the torque was zero"),
            "colors": ("one JPEG per colour key per sample; a missing camera is null, and the "
                       "matching sample.sources entry carries fresh=false"),
        },
        "clock": {"host": socket.gethostname(), "sample.timestamp_ns": "Unix wall clock",
                  "source_times": "Ubuntu CLOCK_MONOTONIC receive timestamps are retained",
                  "image_sequence": "local received JPEG sequence; repeated=true means reused image",
                  "camera_alignment": "mapped source time when all cameras have clock mappings; otherwise arrival time",
                  "source_timing": "sources.*.timing preserves PC2 timestamp, clock id, offset, uncertainty and stereo eye skew"},
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


def _end_effector_state(entry):
    """One hand's recorded state.

    The O6 reports joint speed, torque and temperature alongside position. Those
    reach us normalised to 0..1 (the device register is one byte), so they are
    logged as the relative measures they are and never as N*m or rad/s. A
    channel the firmware did not send stays an empty list, which the offline
    checker reads as "absent", not as zero.
    """
    def channel(name):
        values = entry.get(name)
        if not values or any(value is None for value in values):
            # A channel the firmware did not send is absent, never a zero reading.
            return []
        return [float(value) for value in values]
    return {"qpos": entry["q"], "qvel": channel("qvel"),
            "torque": channel("torque"), "temperature": channel("temperature"),
            "errors": channel("errors")}


class R1Capture:
    def __init__(self, arm_controller, hand_controller, hand_loop, tracking_timeout, image_shape,
                 wrist_image_shapes=None, wrist_timeout=0.5, sync_capacity=8,
                 sync_tolerance_ms=CameraSynchronizer.DEFAULT_TOLERANCE_MS, recording_tf=None):
        self.recording_tf = recording_tf
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
        self.sync = CameraSynchronizer(
            ["head"] + [f"{side}_wrist" for side in sorted(self.wrist_image_shapes)],
            anchor="head", capacity=sync_capacity, tolerance_ms=sync_tolerance_ms)
        self.last_image_sequence = None
        self.last_wrist_sequences = {}
        self.wrist_seen = {}
        self.last_imu_sequence = None

    def reset_episode(self):
        self.last_image_sequence = None
        self.last_wrist_sequences = {}
        self.wrist_seen = {}
        self.last_imu_sequence = None
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
        self.observe(image, wrist_images)
        pairing = self.sync.pair(now)
        if pairing is None or pairing.frames.get("head") is None:
            raise RuntimeError("Capture stereo image has no valid timestamp; episode is incomplete")
        image = pairing.frames["head"].payload
        image_source = source(image.received_monotonic_ns, 0.5, sequence=image.sequence)
        image_source["timing"] = getattr(image, "timing", None)
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
            "left_ee": _end_effector_state(hand["state"]["left"]),
            "right_ee": _end_effector_state(hand["state"]["right"]),
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
        # The writer stores a colour key once per source frame and references it
        # again on every later sample that reuses it. The 40 Hz loop sees a fresh
        # head frame about a quarter of the time, so without this the same JPEG is
        # encoded and written four times over.
        color_sequences = {"color_0": int(image.sequence), "color_1": int(image.sequence)}
        for side in sorted(self.wrist_image_shapes):
            name = f"{side}_wrist"
            entry = None if pairing is None else pairing.frames.get(name)
            wrist = None if entry is None else entry.payload
            pixels = None
            sample = None
            if wrist is not None and wrist.bgr is not None and int(wrist.sequence or 0) > 0:
                sample = source(wrist.received_monotonic_ns, self.wrist_timeout, sequence=wrist.sequence)
                sample["timing"] = getattr(wrist, "timing", None)
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
            if pixels is not None:
                color_sequences[WRIST_COLOR_KEYS[side]] = int(wrist.sequence)
        if pairing is not None:
            image_source["offset_ms"] = float(pairing.offsets_ms.get("head", 0.0))
        target_ns = pairing.anchor_ns
        robot_samples = self.arm.get_recording_samples(target_ns, self.last_imu_sequence, now)
        if robot_samples["imu_packets"]:
            self.last_imu_sequence = robot_samples["imu_packets"][-1]["sequence"]
        aligned_robot = robot_samples["nearest"]
        aligned_hands = self.hand.get_recording_states_at(target_ns, now)
        aligned_sources = {"robot": aligned_robot,
                           **{f"{side}_hand_feedback": value for side, value in aligned_hands.items()}}
        offsets = {name: None if value is None else (value["monotonic_ns"] - target_ns) / 1e6
                   for name, value in aligned_sources.items()}
        camera_sources = [sources["image"]] + [sources[f"{side}_wrist_image"]
                                               for side in sorted(self.wrist_image_shapes)]
        clock_ok = pairing.timestamp_basis == "mapped_source" and all(
            (entry.get("timing") or {}).get("clock_valid") is True
            and 0 <= now - entry["timing"]["clock_measured_monotonic_ns"] <= 10_000_000_000
            and entry["timing"]["clock_uncertainty_ns"] <= 5_000_000
            for entry in camera_sources)
        stereo_ok = (image_source.get("timing") or {}).get("stereo_skew_ns")
        stereo_ok = stereo_ok is not None and stereo_ok / 1e6 <= self.sync.tolerance_ms
        feedback_ok = all(value is not None and abs(value) <= 25.0 for value in offsets.values())
        imu_valid = bool(state["imu"]["valid"] and aligned_robot is not None
                         and aligned_robot["imu"]["valid"]
                         and all(packet["valid"] for packet in robot_samples["imu_packets"]))
        sensor_alignment = {
            "schema": "r1_sensor_sync_v1", "target_monotonic_ns": target_ns,
            "feedback_offset_ms": offsets, "feedback_tolerance_ms": 25.0,
            "clock_valid": bool(clock_ok), "stereo_aligned": bool(stereo_ok),
            "feedback_aligned": feedback_ok, "imu_valid": imu_valid,
            "imu_dropped_packets": robot_samples["dropped"],
            "usable": bool(clock_ok and stereo_ok and feedback_ok
                           and pairing.skew_ms <= self.sync.tolerance_ms
                           and all(entry["fresh"] for entry in camera_sources)
                           and robot_samples["dropped"] == 0),
        }
        sensor_alignment["usable_with_imu"] = sensor_alignment["usable"] and imu_valid
        return {
            "colors": colors,
            "color_sequences": color_sequences,
            "states": states,
            "actions": actions,
            "sample": {
                "timestamp_ns": wall, "monotonic_ns": now, "mode": mode, "sources": sources,
                "camera_alignment": self.sync.alignment(pairing),
                "imu": {"monotonic_ns": state["monotonic_ns"], "sequence": state["sequence"],
                        "tick": state["tick"], **state["imu"]},
                "imu_packets": robot_samples["imu_packets"],
                "aligned_states": {"robot": aligned_robot, "hands": aligned_hands},
                "sensor_alignment": sensor_alignment,
                "tf": None if self.recording_tf is None else self.recording_tf.sample(
                    aligned_robot, target_ns,
                    sensor_alignment["clock_valid"] and stereo_ok
                    and pairing.skew_ms <= self.sync.tolerance_ms
                    and all(entry["fresh"] for entry in camera_sources)),
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
