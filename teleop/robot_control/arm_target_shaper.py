"""Bounded-rate shaping of the R1_A7 arm joint target.

Why this exists
---------------
The Vision Pro hand stream does not reach the control loop evenly. Measured on the
real robot (2026-09-16, 40 Hz loop, 1068 recorded control ticks, 39.9 s):

* the control loop only saw about 11 fresh hand samples per second;
* the sample it was using was 45 ms old at the median, 251 ms at p95, 446 ms at worst;
* every time a stale sample was replaced, the operator's whole hand motion during
  that gap arrived as one target step: p50 1.2 deg, p95 22.8 deg, max 84.5 deg of
  joint motion inside a single 25 ms tick.

The arm cannot execute that. The joints saturate around 5-7 rad/s (measured
``dq_actual`` p95 5.30, max 7.19 rad/s) while those steps imply up to 46 rad/s, so
the servo runs 0.25-0.45 rad behind the command and catches up in bursts. The
operator sees exactly that: the arm holds still, then lurches.

What this does
--------------
Ramps the reference toward the newest IK target at a bounded speed, so the servo is
never asked for more than it can deliver, and the velocity feed-forward is taken
from a continuous signal instead of a step train. The steady-state joint demand
measured on the robot is ~0.8 rad/s with peaks near 2 rad/s, so the default limit
keeps up with ordinary hand motion and only stretches the catch-up that follows a
stale sample.

The cost is lag: while the reference is rate limited it trails the target. That lag
is bounded because the limit is above the sustained demand, and it clears as soon as
the operator's hand pauses. ``--arm-target-velocity-limit 0`` disables the shaper and
restores the raw target.
"""

import math
import time

import numpy as np

__all__ = ["ArmTargetShaper"]


class ArmTargetShaper:
    """Turn an irregular target stream into a continuous, achievable reference."""

    #: A longer gap than this is treated as a stalled loop rather than real elapsed
    #: time, so a scheduling hiccup cannot turn into an unbounded jump.
    MAX_TICK_S = 0.10

    def __init__(self, velocity_limit=4.0, accel_limit=40.0, dof=14, name="arm"):
        velocity_limit = float(velocity_limit)
        accel_limit = float(accel_limit)
        if not np.isfinite(velocity_limit) or velocity_limit < 0.0:
            raise ValueError("velocity_limit must be a finite, non-negative value")
        if not np.isfinite(accel_limit) or accel_limit < 0.0:
            raise ValueError("accel_limit must be a finite, non-negative value")
        if int(dof) <= 0:
            raise ValueError("dof must be a positive integer")
        self.velocity_limit = velocity_limit
        self.accel_limit = accel_limit
        self.dof = int(dof)
        self.name = str(name)
        self.reference = None
        self.velocity = np.zeros(self.dof)
        self.last_time = None
        self.reset_stats()

    @property
    def enabled(self):
        """False when the shaper is configured as a pass-through."""
        return self.velocity_limit > 0.0

    @property
    def configured_limits(self):
        return self.velocity_limit, self.accel_limit

    def reset_stats(self):
        self.stats = {
            "ticks": 0,
            "limited_ticks": 0,
            "max_residual_deg": 0.0,
            "max_speed_rad_s": 0.0,
        }

    def reset(self, reference_q=None):
        """Drop the reference.

        With ``reference_q`` the shaper restarts exactly there. Without it the shaper
        keeps the pose it had already reached and only drops its velocity, which is
        the safe behaviour at a re-anchor: the next target is followed from where the
        arm actually is instead of being jumped to.
        """
        if reference_q is not None:
            reference = np.asarray(reference_q, dtype=np.float64)
            if reference.shape != (self.dof,) or not np.isfinite(reference).all():
                raise ValueError(
                    "%s reference must contain %d finite values" % (self.name, self.dof)
                )
            self.reference = reference.copy()
        self.velocity = np.zeros(self.dof)
        self.last_time = None

    def shape(self, target_q, now=None):
        """Return the reference to send this tick, stepping toward ``target_q``."""
        target = np.asarray(target_q, dtype=np.float64)
        if target.shape != (self.dof,) or not np.isfinite(target).all():
            raise ValueError(
                "%s target must contain %d finite values" % (self.name, self.dof)
            )
        if not self.enabled:
            return target.copy()
        if now is None:
            now = time.monotonic()

        if self.reference is None:
            # Bootstrap: adopt the first target outright instead of ramping to it.
            self.reference = target.copy()
            self.velocity = np.zeros(self.dof)
            self.last_time = now
            return target.copy()

        dt = now - self.last_time if self.last_time is not None else 0.0
        self.last_time = now
        if not np.isfinite(dt) or dt <= 0.0:
            # Clock went backwards or two calls landed in the same instant: hold.
            return self.reference.copy()
        dt = min(dt, self.MAX_TICK_S)

        delta = target - self.reference
        desired = np.clip(delta / dt, -self.velocity_limit, self.velocity_limit)
        if self.accel_limit > 0.0:
            # Ramp the speed rather than stepping it, so the feed-forward the servo sees
            # has no discontinuity at the start and end of a catch-up. Zero means "no
            # acceleration limit", not "never accelerate".
            self.velocity = self.velocity + np.clip(
                desired - self.velocity, -self.accel_limit * dt, self.accel_limit * dt
            )
        else:
            self.velocity = desired
        step = np.clip(self.velocity * dt, -np.abs(delta), np.abs(delta))
        # Keep speed and displacement consistent: if the step was clipped to avoid
        # overshoot, the velocity must come down with it or the next tick jumps.
        self.velocity = step / dt
        self.reference = self.reference + step

        self.stats["ticks"] += 1
        residual_deg = float(np.max(np.abs(target - self.reference))) * 180.0 / math.pi
        speed = float(np.max(np.abs(self.velocity)))
        if residual_deg > self.stats["max_residual_deg"]:
            self.stats["max_residual_deg"] = residual_deg
        if speed > self.stats["max_speed_rad_s"]:
            self.stats["max_speed_rad_s"] = speed
        if np.any(np.abs(delta) > self.velocity_limit * dt + 1e-9):
            self.stats["limited_ticks"] += 1
        return self.reference.copy()

    def snapshot(self):
        """Diagnostics for the alignment JSONL."""
        current_speed = float(np.max(np.abs(self.velocity)))
        return {
            "velocity_limit_rad_s": self.velocity_limit,
            "accel_limit_rad_s2": self.accel_limit,
            "ticks": self.stats["ticks"],
            "limited_ticks": self.stats["limited_ticks"],
            "max_residual_deg": round(self.stats["max_residual_deg"], 3),
            "max_speed_rad_s": round(self.stats["max_speed_rad_s"], 3),
            "current_speed_rad_s": round(current_speed, 4),
        }
