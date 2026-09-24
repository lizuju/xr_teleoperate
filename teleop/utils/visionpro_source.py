import json
import math
from pathlib import Path
import subprocess
import threading
import time

import numpy as np


# ARKit joint axes to WebXR joint axes, matching WebKit WKXRTrackingManager.
JOINT_AXES = {
    "left": np.array([[0., 0., -1., 0.], [0., -1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 0., 1.]]),
    "right": np.array([[0., 0., 1., 0.], [0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 0., 1.]]),
}


def rigid_matrices(value, shape):
    matrices = np.asarray(value, dtype=float)
    if matrices.shape != shape or not np.isfinite(matrices).all():
        raise ValueError(f"Expected finite poses with shape {shape}")
    rotations = matrices[..., :3, :3]
    if (not np.allclose(matrices[..., 3, :], [0, 0, 0, 1], atol=1e-4)
            or not np.allclose(np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=2e-3)
            or not np.allclose(np.linalg.det(rotations), 1., atol=2e-3)):
        raise ValueError("Tracking poses must be rigid transforms")
    return matrices


class VisionProMotionSource:
    def __init__(self, host, python, port=12345, timeout=0.25):
        self._lock = threading.RLock()
        self._head = np.eye(4)
        self._head[:3, 3] = [0., 1.5, -0.2]
        self._snapshot = {"motion_data_ready": False, "motion_sample_seq": 0,
                          "motion_data_timestamp": 0., "left_hand_timestamp": 0., "right_hand_timestamp": 0.}
        for side, x in (("left", -0.15), ("right", 0.15)):
            pose = np.eye(4)
            pose[:3, 3] = [x, 1.13, -0.3]
            self._snapshot.update({
                f"{side}_arm_pose": pose,
                f"{side}_hand_positions": np.tile(pose[:3, 3], (25, 1)),
                f"{side}_hand_orientations": np.tile(np.eye(3), (25, 1, 1)),
                f"{side}_hand_pinch": False, f"{side}_hand_pinchValue": 0.,
                f"{side}_hand_squeeze": False, f"{side}_hand_squeezeValue": 0.,
            })
        self.timeout = timeout
        self._anchor_times = {key: 0. for key in ("head", "left", "right")}
        self._timestamps = dict(self._anchor_times)
        self._valid = {key: False for key in self._anchor_times}
        self._received = 0.
        self._packet_time = 0.
        self._clock_offset = None
        self._connected = False
        self._ever_head = False
        self._head_live = False
        self._realign_required = False
        self._error = None
        self._prediction = 0.
        self._stream_id = 0
        self._frames_coalesced = 0
        self._hand_loss_seq = {side: 0 for side in ("left", "right")}
        self._observed_hand_loss = threading.local()
        self._hello = threading.Event()
        self._process = None
        self._reader = None
        if host is not None:
            bridge = Path(__file__).resolve().parents[2] / "tools/visionpro_bridge.py"
            self._process = subprocess.Popen(
                [str(python), "-u", str(bridge), "--host", host, "--port", str(port),
                 "--timeout", str(timeout)],
                stdout=subprocess.PIPE, text=True,
            )
            self._reader = threading.Thread(target=self._read, daemon=True)
            self._reader.start()
            if not self._hello.wait(10.) or self._error:
                reason = self._error or "No Tracking Streamer packet within 10 seconds"
                self.close()
                raise RuntimeError(reason)

    def _read(self):
        try:
            for line in self._process.stdout:
                packet = json.loads(line)
                if "error" in packet:
                    raise ValueError(packet["error"])
                if "disconnected" in packet:
                    with self._lock:
                        self._connected = False
                        self._realign_required |= self._ever_head
                        self._error = packet["disconnected"]
                        self._refresh(time.monotonic())
                    continue
                self.accept_packet(packet)
                self._hello.set()
        except (ValueError, KeyError, TypeError) as exc:
            self._error = f"Vision Pro input rejected: {exc}"
        finally:
            with self._lock:
                self._connected = False
                self._realign_required |= self._ever_head
                if self._error is None:
                    self._error = "Tracking Streamer receiver ended; restart the Vision Pro script"
            self._hello.set()

    def accept_packet(self, packet):
        now = time.monotonic()
        if packet["tracking_protocol_version"] != 1:
            raise ValueError("Tracking Streamer needs the supplied tracking-validity patch (protocol 1)")
        received, sample = float(packet["received_monotonic"]), float(packet["sample_time"])
        prediction = float(packet["prediction_seconds"])
        if (not all(math.isfinite(v) for v in (received, sample, prediction))
                or received <= 0. or received > now + 0.01 or sample <= 0.
                or not 0. <= prediction <= 0.1):
            raise ValueError("Invalid tracking clock or prediction offset")
        with self._lock:
            stream_id = packet.get("stream_id", self._stream_id)
            if stream_id != self._stream_id:
                self._realign_required |= self._ever_head
                self._connected = False
                self._head_live = False
                self._packet_time = 0.
                self._clock_offset = None
                self._anchor_times = {key: 0. for key in self._anchor_times}
                self._timestamps = dict(self._anchor_times)
                self._valid = {key: False for key in self._valid}
                self._stream_id = stream_id
            if sample < self._packet_time:
                raise ValueError("Vision Pro tracking clock restarted; realignment is required")
            self._refresh(now)
            offset = received - sample
            if "clock_offset" in packet:
                bridge_offset = float(packet["clock_offset"])
                if not math.isfinite(bridge_offset) or bridge_offset > offset + 1e-8:
                    raise ValueError("Invalid tracking clock offset")
                offset = min(offset, bridge_offset)
            offset = offset if self._clock_offset is None else min(offset, self._clock_offset)
            anchor_times, timestamps, validity = dict(self._anchor_times), dict(self._timestamps), {}
            updates = {}
            for key in ("head", "left", "right"):
                anchor_time = float(packet[f"{key}_time"])
                valid = packet[f"{key}_valid"]
                if not isinstance(valid, bool) or not math.isfinite(anchor_time):
                    raise ValueError("Invalid tracking validity or anchor timestamp")
                age = sample - anchor_time
                valid = valid and anchor_time > 0. and -prediction - 0.01 <= age <= self.timeout
                if valid and anchor_time < self._anchor_times[key]:
                    raise ValueError(f"{key} tracking clock moved backwards")
                if valid and anchor_time > self._anchor_times[key]:
                    if key == "head":
                        updates[key] = rigid_matrices(packet["head"], (4, 4))
                    else:
                        wrist = rigid_matrices(packet[f"{key}_wrist"], (4, 4))
                        joints = rigid_matrices(packet[f"{key}_joints"], (25, 4, 4))
                        world = wrist @ joints @ JOINT_AXES[key]
                        if np.max(np.linalg.norm(world[:, :3, 3] - world[0, :3, 3], axis=1)) > 0.35:
                            raise ValueError("Hand joints exceed 0.35 m from the wrist")
                        updates[key] = world
                    anchor_times[key] = anchor_time
                    timestamps[key] = min(received, anchor_time + offset)
                validity[key] = valid
            self._anchor_times, self._timestamps, self._valid = anchor_times, timestamps, validity
            self._clock_offset = offset
            if "head" in updates:
                self._head = updates["head"].copy()
            for side in ("left", "right"):
                if side not in updates:
                    continue
                world = updates[side]
                positions = world[:, :3, 3]
                pinch = float(np.linalg.norm(positions[4] - positions[9]))
                self._snapshot.update({
                    f"{side}_arm_pose": world[0].copy(),
                    f"{side}_hand_positions": positions.copy(),
                    f"{side}_hand_orientations": world[:, :3, :3].copy(),
                    f"{side}_hand_pinch": pinch < 0.02,
                    f"{side}_hand_pinchValue": pinch,
                })
            self._prediction = prediction
            lost = packet.get("tracking_lost", ())
            if "head" in lost:
                self._realign_required |= self._ever_head
            for side in ("left", "right"):
                if side in lost:
                    self._hand_loss_seq[side] += 1
            self._frames_coalesced = packet.get("frames_coalesced", self._frames_coalesced)
            self._received, self._packet_time = received, sample
            self._connected = True
            self._error = None
            if updates:
                self._snapshot["motion_sample_seq"] += 1
            self._refresh(now)

    def _refresh(self, now):
        connected = self._connected and 0. <= now - self._received <= self.timeout
        head_live = connected and self._valid["head"] and 0. <= now - self._timestamps["head"] <= self.timeout
        if self._head_live and not head_live:
            self._realign_required = True
        self._head_live = head_live
        self._ever_head |= head_live
        for side in ("left", "right"):
            live = head_live and self._valid[side] and 0. <= now - self._timestamps[side] <= self.timeout
            self._snapshot[f"{side}_hand_timestamp"] = self._timestamps[side] if live else 0.
        stamps = [self._snapshot[f"{side}_hand_timestamp"] for side in ("left", "right")]
        self._snapshot["motion_data_ready"] = any(stamps)
        self._snapshot["motion_data_timestamp"] = min(stamps)

    @property
    def head_pose(self):
        with self._lock:
            return self._head.copy()

    def get_hand_motion_snapshot(self, include_orientations=False):
        with self._lock:
            self._refresh(time.monotonic())
            snapshot = {key: value.copy() if isinstance(value, np.ndarray) else value
                        for key, value in self._snapshot.items()}
            snapshot["head_pose"] = self._head.copy()
            seen = getattr(self._observed_hand_loss, "sequences", {})
            # Arm and hand control threads must each observe losses hidden by coalescing.
            for side, sequence in self._hand_loss_seq.items():
                if sequence > seen.get(side, 0):
                    snapshot[f"{side}_hand_timestamp"] = 0.
            self._observed_hand_loss.sequences = dict(self._hand_loss_seq)
            stamps = [snapshot[f"{side}_hand_timestamp"] for side in ("left", "right")]
            snapshot["motion_data_ready"] = any(stamps)
            snapshot["motion_data_timestamp"] = min(stamps)
            return snapshot

    def pop_hand_motion_sample(self):
        return self.get_hand_motion_snapshot(include_orientations=True)

    @property
    def needs_realign(self):
        with self._lock:
            self._refresh(time.monotonic())
            return self._realign_required

    def consume_realign_required(self):
        with self._lock:
            self._refresh(time.monotonic())
            required = self._realign_required
            self._realign_required = False
            return required

    def get_tracking_diagnostics(self):
        with self._lock:
            now = time.monotonic()
            self._refresh(now)
            result = {"source": "visionpro", "protocol_version": 1,
                      "head_tracking": self._head_live, "error": self._error,
                      "prediction_seconds": self._prediction,
                      "stream_id": self._stream_id, "frames_coalesced": self._frames_coalesced,
                      "motion_sample_seq": self._snapshot["motion_sample_seq"]}
            for side in ("left", "right"):
                stamp = self._snapshot[f"{side}_hand_timestamp"]
                result[f"{side}_age_ms"] = (now - stamp) * 1000. if stamp else None
                result[f"{side}_tracking"] = bool(stamp)
                result[f"{side}_status"] = "tracking" if stamp else "missing"
            stamp = self._snapshot["motion_data_timestamp"]
            result["pair_age_ms"] = (now - stamp) * 1000. if stamp else None
            return result

    def close(self):
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            if self._reader is not None:
                self._reader.join(timeout=1.)
            self._process.stdout.close()
        with self._lock:
            self._connected = False
            self._refresh(time.monotonic())
