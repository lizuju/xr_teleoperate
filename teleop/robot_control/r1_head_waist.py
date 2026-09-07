import math

import numpy as np


def _yaw_rotation(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _finite_pose(pose, name):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return pose


def compensate_wrist_for_waist(target, waist_actual, waist_reference):
    target = _finite_pose(target, "target")
    waist_actual, waist_reference = float(waist_actual), float(waist_reference)
    if not math.isfinite(waist_actual) or not math.isfinite(waist_reference):
        raise ValueError("waist angles must be finite")
    # The R1-A7 URDF waist axis is +Z at the pelvis origin.
    rotation = _yaw_rotation(waist_reference - waist_actual)
    result = target.copy()
    result[:3, :3] = rotation @ target[:3, :3]
    result[:3, 3] = rotation @ target[:3, 3]
    return result


class R1HeadWaistFollower:
    def __init__(self, waist_reference, now, tracking_timeout=0.25):
        self.waist_reference = float(waist_reference)
        self._last_time = float(now)
        self.tracking_timeout = float(tracking_timeout)
        if not all(math.isfinite(value) for value in (
            self.waist_reference, self._last_time, self.tracking_timeout
        )) or self.tracking_timeout <= 0.0:
            raise ValueError("reference/time must be finite and tracking_timeout must be positive")
        self.reset(self.waist_reference, self._last_time)

    def reset(self, waist_actual, now):
        waist_actual, now = float(waist_actual), float(now)
        if not math.isfinite(waist_actual) or not math.isfinite(now):
            raise ValueError("waist_actual and now must be finite")
        self._last_time = now
        self.waist_target = waist_actual
        self.total_yaw = 0.0
        self.following = False
        self._last_heading = None
        self._heading_was_degenerate = False
        self._trigger_since = None
        self._trigger_sign = 0.0
        self._velocity = 0.0
        self._goal = waist_actual

    def update(self, current_head_pose, reference_head_pose, waist_actual, now):
        current = _finite_pose(current_head_pose, "current_head_pose")
        reference = _finite_pose(reference_head_pose, "reference_head_pose")
        waist_actual, now = float(waist_actual), float(now)
        if not math.isfinite(waist_actual) or not math.isfinite(now):
            raise ValueError("waist_actual and now must be finite")
        dt = now - self._last_time
        if dt < 0.0:
            raise ValueError("now must not move backwards")

        relative_rotation = reference[:3, :3].T @ current[:3, :3]
        heading = math.atan2(relative_rotation[1, 0], relative_rotation[0, 0])
        heading_is_valid = math.hypot(relative_rotation[0, 0], relative_rotation[1, 0]) >= 0.15
        timed_out = dt > self.tracking_timeout
        if timed_out:
            self.reset(waist_actual, now)
        if heading_is_valid:
            if self._last_heading is None:
                if self._heading_was_degenerate:
                    heading += 2.0 * math.pi * round((self.total_yaw - heading) / (2.0 * math.pi))
                self.total_yaw = heading
            else:
                delta = heading - self._last_heading
                self.total_yaw += math.atan2(math.sin(delta), math.cos(delta))
            self._last_heading = heading
            self._heading_was_degenerate = False
        else:
            # A nearly vertical viewing direction does not provide an observable heading.
            self._last_heading = None
            self._heading_was_degenerate = True
            self.following = False
            self._trigger_since = None
            self._trigger_sign = 0.0
        self._last_time = now

        if not timed_out:
            residual = self.total_yaw - (waist_actual - self.waist_reference)
            if self.following:
                if abs(residual) < math.radians(5.0):
                    self.following = False
                    self._trigger_since = None
                    self._trigger_sign = 0.0
            elif heading_is_valid and abs(residual) > math.radians(20.0):
                trigger_sign = math.copysign(1.0, residual)
                if self._trigger_since is None or trigger_sign != self._trigger_sign:
                    self._trigger_since = now
                    self._trigger_sign = trigger_sign
                elif now - self._trigger_since >= 0.4:
                    self.following = True
                    self._trigger_since = None
                    self._trigger_sign = 0.0
            else:
                self._trigger_since = None
                self._trigger_sign = 0.0

            if self.following:
                self._goal = float(np.clip(
                    self.waist_reference + self.total_yaw,
                    -2.618, 2.618,
                ))
                desired_velocity = float(np.clip(
                    (self._goal - self.waist_target) / 0.8, -0.35, 0.35,
                ))
            else:
                desired_velocity = 0.0
            previous_velocity = self._velocity
            self._velocity += float(np.clip(
                desired_velocity - self._velocity, -0.5 * dt, 0.5 * dt,
            ))
            step = self._velocity * dt
            remaining = self._goal - self.waist_target
            if (
                remaining * step > 0.0 and abs(step) >= abs(remaining)
                and abs(previous_velocity) <= 0.5 * dt
            ):
                self.waist_target = self._goal
                self._velocity = remaining / dt
            else:
                # A changed goal can fall inside the stopping distance; brake smoothly.
                self.waist_target += step
            self.waist_target = float(np.clip(
                self.waist_target, -2.618, 2.618,
            ))

        # Pitch precedes yaw in the R1 head chain: Ry(pitch) @ Rz(yaw).
        local_rotation = _yaw_rotation(
            self.waist_reference - waist_actual
        ) @ relative_rotation
        forward = local_rotation[:, 0]
        pitch = math.atan2(-forward[2], forward[0])
        horizontal = math.hypot(forward[0], forward[2])
        yaw = math.atan2(forward[1], horizontal)
        if abs(pitch) > math.pi / 2.0:
            pitch = math.atan2(forward[2], -forward[0])
            yaw = math.atan2(forward[1], -horizontal)
        residual = self.total_yaw - (waist_actual - self.waist_reference)
        yaw += 2.0 * math.pi * round((residual - yaw) / (2.0 * math.pi))
        head_target = np.clip(
            [pitch, yaw], [-0.62832, -2.0071], [0.62832, 2.0071],
        )
        return head_target, self.waist_target
