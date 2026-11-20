"""Operator lock for the R1 head.

h toggles it (H too). While locked, the robot head holds the pitch/yaw target it had when
the lock was taken, whatever the operator's head does; the arms, hands and waist keep
their own behaviour. On unlock the head does not jump to the operator's current head: its
target moves there at no more than UNLOCK_SPEED_RAD_S and then follows directly again.
"""
import math
import threading

import numpy as np

HEAD_LOCK_KEYS = ("h", "H")
UNLOCK_SPEED_RAD_S = 1.0


class R1HeadLock:
    def __init__(self, unlock_speed=UNLOCK_SPEED_RAD_S):
        if not (math.isfinite(unlock_speed) and unlock_speed > 0.0):
            raise ValueError("unlock_speed must be positive")
        self.unlock_speed = float(unlock_speed)
        self._lock = threading.Lock()
        self._locked = False
        self._held = None          # head target held while locked
        self._catching_up = False  # after an unlock, until the target reaches the operator's head
        self._last = None          # last head target sent (or held by a pause)
        self._last_time = None
        self._announce = False     # log the held angle on the first tick after locking
        self._events = []

    def toggle(self):
        """Flip the lock (keyboard thread) and return its new state."""
        with self._lock:
            self._locked = not self._locked
            if self._locked:
                self._held = None if self._last is None else self._last.copy()
                self._catching_up = False
                self._announce = True
            else:
                self._held = None
                self._catching_up = self._last is not None
                self._announce = False
            return self._locked

    def locked(self):
        with self._lock:
            return self._locked

    def observe(self, head_target, now):
        """Record a head target the loop sent without apply(), e.g. while paused."""
        target = np.asarray(head_target, dtype=float).copy()
        with self._lock:
            self._last, self._last_time = target, float(now)

    def apply(self, follow_target, now):
        """Head target to send this tick, given what following would command now."""
        follow = np.asarray(follow_target, dtype=float).copy()
        now = float(now)
        with self._lock:
            if self._locked:
                if self._held is None:
                    self._held = follow.copy()
                if self._announce:
                    self._events.append(self._message("locked; the head holds", self._held))
                    self._announce = False
                out = self._held.copy()
            elif self._catching_up and self._last is not None and self._last_time is not None:
                step = self.unlock_speed * max(0.0, now - self._last_time)
                delta = follow - self._last
                if float(np.max(np.abs(delta))) <= step:
                    out = follow
                    self._catching_up = False
                    self._events.append("[R1 HEAD LOCK] unlocked; the head follows you again.")
                else:
                    out = self._last + np.clip(delta, -step, step)
            else:
                self._catching_up = False
                out = follow
            self._last, self._last_time = out.copy(), now
            return out

    def pop_events(self):
        with self._lock:
            events, self._events = self._events, []
            return events

    @staticmethod
    def _message(text, target):
        return (f"[R1 HEAD LOCK] {text} pitch {math.degrees(target[0]):.1f} deg, "
                f"yaw {math.degrees(target[1]):.1f} deg; h unlocks.")
