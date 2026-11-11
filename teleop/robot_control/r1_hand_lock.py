"""Operator locks for one R1 arm and its hand (O6 or Dex1-1 gripper).

[ toggles the left side, ] the right side (the full-width 【 】 of a Chinese input
method work too). A lock is a deliberate command, not a tracking fault: the locked arm
and its hand keep their last target while the rest of the robot keeps following, the
waist may still turn, and recorded frames stay "following" so they remain usable for
training. The control loop re-anchors an unlocked arm at its held pose, and the hand
loop makes an unlocked gripper wait until the operator's pinch matches its grip.
"""
import threading

SIDES = ("left", "right")
HAND_LOCK_KEYS = {"[": "left", "\u3010": "left", "]": "right", "\u3011": "right"}


class R1HandLocks:
    def __init__(self):
        self._lock = threading.Lock()
        self._locked = {side: False for side in SIDES}

    def toggle(self, side):
        """Flip one side and return its new state."""
        with self._lock:
            self._locked[side] = not self._locked[side]
            return self._locked[side]

    def locked(self, side):
        with self._lock:
            return self._locked[side]

    def snapshot(self):
        """(left, right) lock states read together."""
        with self._lock:
            return tuple(self._locked[side] for side in SIDES)
