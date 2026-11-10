"""Lightly smooth wrist targets during finger motion without freezing small movements.

Translation and SO(3) orientation use a 40 ms exponential time constant while the
fingers move. The configured distances bound tracking lag, not a dead zone. Every
nonzero change responds on the next positive time step, including small intentional
motion. After the short hold/release, settled fingers give exact passthrough.

This reduces fast tracking jitter; slow finger-induced bias cannot be distinguished
from deliberate wrist motion using these poses alone.
"""
import math
from collections import deque

import numpy as np


FINGERTIPS = (4, 9, 14, 19, 24)
DEFAULT_POSITION_M = 0.003
DEFAULT_ROTATION_RAD = math.radians(3.0)
# Fingertip speed in the wrist frame that counts as the fingers moving. With the fingers
# still the capture measured p99 0.02 (left) and 0.04 m/s (right), with peaks near 0.07.
DEFAULT_FINGER_SPEED = 0.08
DEFAULT_HOLD_S = 0.1
SMOOTHING_TIME_S = 0.04


def rotation_vector(matrix):
    """Axis times angle of a rotation matrix."""
    cosine = min(max((np.trace(matrix) - 1.0) / 2.0, -1.0), 1.0)
    angle = math.acos(cosine)
    skew = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    norm = float(np.linalg.norm(skew))
    if norm > 1e-9:
        return skew / norm * angle
    if angle < 1.0:
        return np.zeros(3)
    # Half a turn: the skew part vanishes, so take the axis from the symmetric part.
    axis = np.sqrt(np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0))
    largest = int(np.argmax(axis))
    for index in range(3):
        if index != largest and matrix[largest, index] + matrix[index, largest] < 0.0:
            axis[index] = -axis[index]
    return axis / max(float(np.linalg.norm(axis)), 1e-12) * angle


def rotation_from_vector(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


class WristStabilizer:
    def __init__(self, position_m=DEFAULT_POSITION_M, rotation_rad=DEFAULT_ROTATION_RAD,
                 finger_speed=DEFAULT_FINGER_SPEED, window_s=0.1, attack_s=0.05, hold_s=DEFAULT_HOLD_S,
                 release_s=0.1, max_gap_s=0.25):
        values = {"position_m": position_m, "rotation_rad": rotation_rad, "finger_speed": finger_speed,
                  "window_s": window_s, "attack_s": attack_s, "hold_s": hold_s, "release_s": release_s,
                  "max_gap_s": max_gap_s}
        for name, value in values.items():
            if not (math.isfinite(float(value)) and float(value) >= 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if finger_speed <= 0.0 or window_s <= 0.0 or attack_s <= 0.0 or release_s <= 0.0 or max_gap_s <= 0.0:
            raise ValueError("finger_speed, window_s, attack_s, release_s and max_gap_s must be positive")
        self.position_m, self.rotation_rad = float(position_m), float(rotation_rad)
        self.finger_speed, self.window_s = float(finger_speed), float(window_s)
        self.attack_s, self.hold_s, self.release_s = float(attack_s), float(hold_s), float(release_s)
        self.max_gap_s = float(max_gap_s)
        self.frames = 0
        self.active_frames = 0
        self.activations = 0
        self.max_held_m = 0.0
        self.max_held_rad = 0.0
        self.reset()

    def reset(self):
        """Forget the smoothing state; the next update passes its pose through."""
        self._position = None
        self._rotation = None
        self._time = None
        self._gate = 0.0
        self._active_until = None
        self._tips = deque()
        self.last_finger_speed = 0.0

    @property
    def gate(self):
        """0 for passthrough, up to 1 for full smoothing strength."""
        return self._gate

    def _fingertip_speed(self, hand_points, now):
        if hand_points is None:
            self._tips.clear()
            return 0.0
        points = np.asarray(hand_points, dtype=float)
        if points.ndim != 2 or points.shape[0] <= max(FINGERTIPS) or points.shape[1] != 3:
            self._tips.clear()
            return 0.0
        tips = points[list(FINGERTIPS)]
        if not np.isfinite(tips).all():
            self._tips.clear()
            return 0.0
        self._tips.append((now, tips.copy()))
        while len(self._tips) > 2 and self._tips[1][0] <= now - self.window_s:
            self._tips.popleft()
        then, earlier = self._tips[0]
        elapsed = now - then
        if elapsed < self.window_s / 2.0:
            return 0.0
        return float(np.max(np.linalg.norm(tips - earlier, axis=1)) / elapsed)

    def update(self, pose, hand_points, now):
        """Stabilised copy of a 4x4 wrist pose. ``hand_points`` are in the wrist frame."""
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("pose must be a finite 4x4 matrix")
        position, rotation = pose[:3, 3], pose[:3, :3]
        if self._time is None or not 0.0 <= now - self._time <= self.max_gap_s:
            self.reset()
            self._fingertip_speed(hand_points, now)
            self._position, self._rotation, self._time = position.copy(), rotation.copy(), now
            return pose.copy()
        dt = now - self._time
        self._time = now
        speed = self._fingertip_speed(hand_points, now)
        self.last_finger_speed = speed
        if speed >= self.finger_speed:
            if self._gate == 0.0 and (self._active_until is None or now > self._active_until):
                self.activations += 1
            self._active_until = now + self.hold_s
        if self._active_until is not None and now <= self._active_until:
            self._gate = min(1.0, self._gate + dt / self.attack_s)
        else:
            self._gate = max(0.0, self._gate - dt / self.release_s)
        self.frames += 1
        if self._gate == 0.0:
            self._position, self._rotation = position.copy(), rotation.copy()
            return pose.copy()
        self.active_frames += 1

        retained = self._gate * math.exp(-dt / SMOOTHING_TIME_S)
        offset = position - self._position
        distance = float(np.linalg.norm(offset))
        if distance > 0.0 and self.position_m > 0.0:
            weight = min(retained, self._gate * self.position_m / distance)
            self._position = position - offset * weight
        else:
            self._position = position.copy()
        turn = rotation_vector(rotation @ self._rotation.T)
        angle = float(np.linalg.norm(turn))
        if angle > 0.0 and self.rotation_rad > 0.0:
            weight = min(retained, self._gate * self.rotation_rad / angle)
            self._rotation = rotation_from_vector(turn * (1.0 - weight)) @ self._rotation
        else:
            self._rotation = rotation.copy()
        self.max_held_m = max(self.max_held_m, float(np.linalg.norm(position - self._position)))
        self.max_held_rad = max(self.max_held_rad,
                                float(np.linalg.norm(rotation_vector(rotation @ self._rotation.T))))
        result = pose.copy()
        result[:3, 3] = self._position
        result[:3, :3] = self._rotation
        return result

    def summary(self):
        share = 100.0 * self.active_frames / self.frames if self.frames else 0.0
        return (f"active {share:.1f}% of {self.frames} frames in {self.activations} bursts, "
                f"max smoothing offset {self.max_held_m * 1000:.1f} mm / {math.degrees(self.max_held_rad):.1f} deg")
