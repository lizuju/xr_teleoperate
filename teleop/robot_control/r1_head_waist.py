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
    #: Residual yaw needed to start following, and how long it must hold. The residual
    #: is the head yaw the waist has not absorbed yet, not the raw head yaw.
    DEFAULT_ENGAGE_THRESHOLD = math.radians(20.0)
    DEFAULT_ENGAGE_DURATION = 0.2
    #: Residual below which following stops, leaving a hysteresis band around the
    #: engage threshold so the waist cannot chatter on and off at the boundary.
    DEFAULT_RELEASE_THRESHOLD = math.radians(5.0)
    #: Slew limits of the waist target while following. The old 20 deg/s ceiling with a
    #: 0.5 rad/s^2 ramp took 2.2 s to absorb a 45 deg body turn, which the operator read
    #: as the robot lagging behind them.
    DEFAULT_MAX_VELOCITY = math.radians(40.0)
    DEFAULT_MAX_ACCELERATION = math.radians(90.0)
    #: Time constant of the proportional slew law: the target chases the goal with this
    #: lag and saturates at DEFAULT_MAX_VELOCITY once the residual exceeds the product.
    SLEW_TIME_CONSTANT = 0.8
    #: Key-commanded turns move the waist between fixed stops this far apart, placed
    #: around the waist angle at activation, so every turn lands on the same angle.
    TURN_STEP = math.pi / 2.0
    #: A stop closer than this to the current waist target counts as reached, so a
    #: waist resting a few degrees off a stop still needs only one press per stop.
    TURN_STOP_MARGIN = math.radians(10.0)
    #: Slew limits of a key-commanded turn. Gentler than the follower limits because
    #: the operator may be carrying something (a cup) through the turn.
    DEFAULT_TURN_VELOCITY = math.radians(30.0)
    DEFAULT_TURN_ACCELERATION = math.radians(60.0)
    #: Braking plans with this fraction of the acceleration limit, leaving headroom
    #: for loop-period jitter so the ramp lands on the stop without overshooting.
    TURN_BRAKE_FRACTION = 0.9
    #: URDF waist yaw range; stops beyond it do not exist.
    WAIST_LIMIT = 2.618

    def __init__(
        self,
        waist_reference,
        now,
        tracking_timeout=0.25,
        engage_threshold=DEFAULT_ENGAGE_THRESHOLD,
        engage_duration=DEFAULT_ENGAGE_DURATION,
        max_velocity=DEFAULT_MAX_VELOCITY,
        max_acceleration=DEFAULT_MAX_ACCELERATION,
        release_threshold=DEFAULT_RELEASE_THRESHOLD,
        turn_origin=None,
        turn_velocity=DEFAULT_TURN_VELOCITY,
        turn_acceleration=DEFAULT_TURN_ACCELERATION,
        head_follow=True,
    ):
        self.waist_reference = float(waist_reference)
        #: False leaves the waist to key-commanded turns: an operator head or body turn
        #: never engages following, and the robot head alone follows their head yaw.
        self.head_follow = bool(head_follow)
        self.turn_origin = self.waist_reference if turn_origin is None else float(turn_origin)
        self.turn_velocity = float(turn_velocity)
        self.turn_acceleration = float(turn_acceleration)
        if not all(math.isfinite(value) for value in (
            self.turn_origin, self.turn_velocity, self.turn_acceleration,
        )) or min(self.turn_velocity, self.turn_acceleration) <= 0.0:
            raise ValueError("turn origin must be finite and turn limits finite and positive")
        #: Yaw added to the operator's head heading. A key turn moves it together with
        #: the waist target, so the turn reads like the operator having turned their
        #: body: the residual the follower acts on does not change and nothing turns back.
        self.heading_offset = 0.0
        self._turn = None
        self._turn_events = []
        self._last_time = float(now)
        self.tracking_timeout = float(tracking_timeout)
        self.engage_threshold = float(engage_threshold)
        self.engage_duration = float(engage_duration)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.release_threshold = float(release_threshold)
        if not all(math.isfinite(value) for value in (
            self.waist_reference,
            self._last_time,
            self.tracking_timeout,
            self.engage_threshold,
            self.engage_duration,
            self.max_velocity,
            self.max_acceleration,
            self.release_threshold,
        )) or min(
            self.tracking_timeout,
            self.engage_threshold,
            self.engage_duration,
            self.max_velocity,
            self.max_acceleration,
            self.release_threshold,
        ) <= 0.0:
            raise ValueError(
                "reference/time and engagement settings must be finite and positive"
            )
        self.reset(self.waist_reference, self._last_time)

    def reset(self, waist_actual, now):
        waist_actual, now = float(waist_actual), float(now)
        if not math.isfinite(waist_actual) or not math.isfinite(now):
            raise ValueError("waist_actual and now must be finite")
        turn = getattr(self, "_turn", None)
        if turn is not None:
            # An interrupted turn stops where the waist is. Keep the heading offset
            # matched to that angle so the follower does not undo the part done.
            self.heading_offset = turn["offset_start"] + (waist_actual - turn["target_start"])
            self._turn = None
            self._turn_events.append({
                "event": "cancelled", "direction": turn["direction"],
                "goal": turn["goal"], "waist": waist_actual,
            })
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

    @property
    def turning(self):
        return self._turn is not None

    @property
    def turn_goal(self):
        return None if self._turn is None else self._turn["goal"]

    def pop_turn_events(self):
        """Return and clear the turn events (reached, cancelled) since the last call."""
        events, self._turn_events = self._turn_events, []
        return events

    def next_turn_stop(self, direction):
        """The stop a turn in this direction would go to, or None past the waist range.

        direction is +1 for left (+Z yaw, the URDF waist axis) and -1 for right.
        """
        if direction not in (1, -1):
            raise ValueError("direction must be +1 (left) or -1 (right)")
        offset = self.waist_target - self.turn_origin
        if direction > 0:
            index = math.ceil((offset + self.TURN_STOP_MARGIN) / self.TURN_STEP)
        else:
            index = math.floor((offset - self.TURN_STOP_MARGIN) / self.TURN_STEP)
        goal = self.turn_origin + index * self.TURN_STEP
        return goal if abs(goal) <= self.WAIST_LIMIT else None

    def command_turn(self, direction):
        """Turn the waist to the next stop on the left (+1) or right (-1).

        Returns the goal angle, or None when no stop is left in that direction; the
        follower is then unchanged. A press during a turn retargets it from where the
        waist target is, so the opposite key takes the robot back.
        """
        goal = self.next_turn_stop(direction)
        if goal is None:
            return None
        self._turn = {
            "direction": direction, "goal": goal,
            "offset_start": self.heading_offset, "target_start": self.waist_target,
        }
        self._goal = goal
        self.following = False
        self._trigger_since = None
        self._trigger_sign = 0.0
        return goal

    def _advance_turn(self, dt):
        turn = self._turn
        goal = turn["goal"]
        remaining = goal - self.waist_target
        if dt <= 0.0:
            return
        # Fastest speed from which the discrete ramp (one velocity step of brake * dt
        # per tick) still stops within the remaining distance; continuous v^2 = 2ad
        # would leave a final one-tick stop of about twice the acceleration limit.
        brake = self.TURN_BRAKE_FRACTION * self.turn_acceleration
        brake_speed = brake * dt * (
            math.sqrt(0.25 + 2.0 * abs(remaining) / (brake * dt * dt)) - 0.5
        )
        desired_velocity = math.copysign(min(self.turn_velocity, brake_speed), remaining)
        self._velocity += float(np.clip(
            desired_velocity - self._velocity,
            -self.turn_acceleration * dt, self.turn_acceleration * dt,
        ))
        step = self._velocity * dt
        reached = remaining == 0.0 or (step * remaining > 0.0 and abs(step) >= abs(remaining))
        if reached:
            self.waist_target = goal
            self._velocity = 0.0
        else:
            self.waist_target += step
        self.waist_target = float(np.clip(self.waist_target, -self.WAIST_LIMIT, self.WAIST_LIMIT))
        self.heading_offset = turn["offset_start"] + (self.waist_target - turn["target_start"])
        if reached:
            self._turn = None
            self._turn_events.append({
                "event": "reached", "direction": turn["direction"], "goal": goal,
            })

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

        if not timed_out and self._turn is not None:
            self._advance_turn(dt)
        elif not timed_out:
            residual = self.total_yaw + self.heading_offset - (waist_actual - self.waist_reference)
            if self.following:
                if abs(residual) < self.release_threshold:
                    self.following = False
                    self._trigger_since = None
                    self._trigger_sign = 0.0
            elif self.head_follow and heading_is_valid and abs(residual) > self.engage_threshold:
                trigger_sign = math.copysign(1.0, residual)
                if self._trigger_since is None or trigger_sign != self._trigger_sign:
                    self._trigger_since = now
                    self._trigger_sign = trigger_sign
                elif now - self._trigger_since >= self.engage_duration:
                    self.following = True
                    self._trigger_since = None
                    self._trigger_sign = 0.0
            else:
                self._trigger_since = None
                self._trigger_sign = 0.0

            if self.following:
                self._goal = float(np.clip(
                    self.waist_reference + self.total_yaw + self.heading_offset,
                    -2.618, 2.618,
                ))
                desired_velocity = float(np.clip(
                    (self._goal - self.waist_target) / self.SLEW_TIME_CONSTANT,
                    -self.max_velocity, self.max_velocity,
                ))
            else:
                desired_velocity = 0.0
            previous_velocity = self._velocity
            self._velocity += float(np.clip(
                desired_velocity - self._velocity,
                -self.max_acceleration * dt, self.max_acceleration * dt,
            ))
            step = self._velocity * dt
            remaining = self._goal - self.waist_target
            if (
                remaining * step > 0.0 and abs(step) >= abs(remaining)
                and abs(previous_velocity) <= self.max_acceleration * dt
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
            self.waist_reference + self.heading_offset - waist_actual
        ) @ relative_rotation
        forward = local_rotation[:, 0]
        pitch = math.atan2(-forward[2], forward[0])
        horizontal = math.hypot(forward[0], forward[2])
        yaw = math.atan2(forward[1], horizontal)
        if abs(pitch) > math.pi / 2.0:
            pitch = math.atan2(forward[2], -forward[0])
            yaw = math.atan2(forward[1], -horizontal)
        residual = self.total_yaw + self.heading_offset - (waist_actual - self.waist_reference)
        yaw += 2.0 * math.pi * round((residual - yaw) / (2.0 * math.pi))
        head_target = np.clip(
            [pitch, yaw], [-0.62832, -2.0071], [0.62832, 2.0071],
        )
        return head_target, self.waist_target
