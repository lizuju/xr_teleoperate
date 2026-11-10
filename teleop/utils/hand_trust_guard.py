"""Treat a Vision Pro hand as lost while the headset cannot really see it.

ARKit keeps reporting a hand as tracked when the other hand hides it from the
headset cameras, and fills in a guessed pose that can swing, jump or snap onto the
other hand. Downstream that guess drives the robot arm. This guard flags such
samples, so the source reports the hand as lost (its arm and gripper or O6 hold)
until the hand has been clearly visible and self-consistent for a short dwell.

All checks use world-frame ARKit joints (metres, y up):
- occluded: the other hand lies on the line of sight from the headset to this
  hand's wrist, palm or knuckles;
- jump: the wrist moves or turns faster between consecutive ARKit samples than a
  hand does, which is how a guessed pose snaps;
- merged: both wrists within a few centimetres, where ARKit can swap the hands.
Normal motion is never slowed: a sample is either passed unchanged or rejected.
"""
import math

import numpy as np

# WebXR joint order used by visionpro_source: wrist, thumb (4 joints), then index,
# middle, ring and little (metacarpal, proximal, intermediate, distal, tip).
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8, 9), (0, 10, 11, 12, 13, 14),
          (0, 15, 16, 17, 18, 19), (0, 20, 21, 22, 23, 24))
PALM = (0, 5, 10, 15, 20, 6, 11, 16, 21)
KNUCKLES = (6, 11, 16, 21)
_SPANS = tuple((a, b) for chain in CHAINS for a, b in zip(chain, chain[1:])) + (
    (5, 10), (10, 15), (15, 20), (6, 11), (11, 16), (16, 21), (5, 11), (10, 16), (15, 21))
REASONS = {
    "occluded": "hidden from the headset by the other hand",
    "jump": "tracked pose jumped faster than a hand moves",
    "merged": "hands too close together to tell apart",
}


def occluder_points(positions):
    """Joints plus bone and palm midpoints, so a hand blocks the view like a solid shape."""
    positions = np.asarray(positions, dtype=float)
    spans = np.array([(positions[a] + positions[b]) / 2.0 for a, b in _SPANS])
    return np.vstack([positions, spans, positions[list(PALM)].mean(axis=0)[None]])


def target_points(positions):
    """Wrist, the four knuckles and the palm centre: what the headset must see."""
    positions = np.asarray(positions, dtype=float)
    return np.vstack([positions[[0, *KNUCKLES]], positions[list(PALM)].mean(axis=0)[None]])


def blocked_fraction(eye, targets, occluders, radius, near=0.05, far_margin=0.02):
    """Share of targets whose sight line from eye passes within radius of an occluder point.

    Only occluders between the eye and the target count (not behind it, not at the eye).
    """
    eye = np.asarray(eye, dtype=float)
    rays = np.asarray(targets, dtype=float) - eye
    lengths2 = np.einsum("ij,ij->i", rays, rays)
    if np.any(lengths2 <= 1e-8):
        return 0.0
    occluders = np.asarray(occluders, dtype=float)
    along = ((occluders - eye) @ rays.T) / lengths2
    in_front = (along > near) & (along < 1.0 - far_margin / np.sqrt(lengths2))
    closest = eye + along[..., None] * rays[None]
    distance = np.linalg.norm(occluders[:, None, :] - closest, axis=2)
    return float(np.mean(np.any(in_front & (distance < radius), axis=0)))


def rotation_angle(rotation):
    return math.acos(max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0)))


class _Side:
    def __init__(self):
        self.last = None
        self.untrusted = False
        self.reason = None
        self.clear_since = None
        self.events = {reason: 0 for reason in REASONS}


class HandTrustGuard:
    def __init__(self, block_radius=0.03, clear_radius=0.04, occluded_fraction=0.5,
                 clear_fraction=0.2, max_speed=3.0, max_angular_speed=math.radians(1000.0),
                 min_wrist_gap=0.05, dwell=0.25, max_sample_gap=0.2):
        values = (block_radius, clear_radius, occluded_fraction, clear_fraction, max_speed,
                  max_angular_speed, min_wrist_gap, dwell, max_sample_gap)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("hand guard settings must be finite and positive")
        if clear_radius < block_radius or clear_fraction > occluded_fraction or occluded_fraction > 1.0:
            raise ValueError("hand guard clearing must be at least as strict as blocking")
        self.block_radius, self.clear_radius = float(block_radius), float(clear_radius)
        self.occluded_fraction, self.clear_fraction = float(occluded_fraction), float(clear_fraction)
        self.max_speed, self.max_angular_speed = float(max_speed), float(max_angular_speed)
        self.min_wrist_gap, self.dwell = float(min_wrist_gap), float(dwell)
        self.max_sample_gap = float(max_sample_gap)
        self._sides = {side: _Side() for side in ("left", "right")}

    def trusted(self, side):
        return not self._sides[side].untrusted

    def reason(self, side):
        return self._sides[side].reason

    def diagnostics(self):
        return {side: {"trusted": not state.untrusted, "reason": state.reason, "events": dict(state.events)}
                for side, state in self._sides.items()}

    def update(self, side, sample_time, joints, eye, other_positions=None):
        """Return True when this world-frame sample (25 x 4 x 4 joints) may drive the robot."""
        state = self._sides[side]
        joints = np.asarray(joints, dtype=float)
        positions = joints[:, :3, 3]
        wrist_position, wrist_rotation = positions[0].copy(), joints[0, :3, :3].copy()
        sample_time = float(sample_time)
        reason = None
        if state.last is not None and 0.0 < sample_time - state.last[0] <= self.max_sample_gap:
            dt = max(sample_time - state.last[0], 1.0 / 120.0)
            speed = np.linalg.norm(wrist_position - state.last[1]) / dt
            turn = rotation_angle(wrist_rotation @ state.last[2].T) / dt
            if speed > self.max_speed or turn > self.max_angular_speed:
                reason = "jump"
        state.last = (sample_time, wrist_position, wrist_rotation)
        if reason is None and other_positions is not None:
            other = np.asarray(other_positions, dtype=float)
            if np.linalg.norm(wrist_position - other[0]) < self.min_wrist_gap:
                reason = "merged"
            else:
                radius = self.clear_radius if state.untrusted else self.block_radius
                fraction = blocked_fraction(eye, target_points(positions), occluder_points(other), radius)
                if (fraction > self.clear_fraction) if state.untrusted else (fraction >= self.occluded_fraction):
                    reason = "occluded"
        if reason is not None:
            if not state.untrusted:
                state.events[reason] += 1
            state.untrusted, state.reason, state.clear_since = True, reason, None
            return False
        if state.untrusted:
            if state.clear_since is None:
                state.clear_since = sample_time
            if sample_time - state.clear_since < self.dwell:
                return False
            state.untrusted, state.reason, state.clear_since = False, None, None
        return True
