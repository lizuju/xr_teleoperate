import threading
import time

from teleop.robot_control.r1_hand_tracking import hand_tracking_freshness


class R1PauseState:
    def __init__(self):
        self.lock = threading.Lock()
        self._paused = False
        self._input_hold = False
        self.generation = 0
        self.resume_after = None
        self.stable_since = None
        self.last_timestamps = None
        self.samples = 0

    @property
    def paused(self):
        with self.lock:
            return self._paused

    @property
    def input_hold(self):
        with self.lock:
            return self._input_hold

    def pause(self, *, input_lost=False, now=None):
        with self.lock:
            auto_resume = input_lost and (not self._paused or self.resume_after is not None)
            self._paused = True
            self._input_hold = auto_resume
            self.generation += 1
            self.resume_after = (time.monotonic() if now is None else now) if auto_resume else None
            self.stable_since = None
            self.last_timestamps = None
            self.samples = 0

    def request_resume(self, now=None):
        with self.lock:
            if not self._paused:
                return
            self.resume_after = time.monotonic() if now is None else now
            self.stable_since = None
            self.last_timestamps = None
            self.samples = 0

    def poll_resume(self, sample, timeout, now):
        with self.lock:
            if not self._paused or self.resume_after is None:
                return None
            timestamps = (sample.left_hand_timestamp, sample.right_hand_timestamp)
            if (
                not all(hand_tracking_freshness(sample, timeout, now))
                or min(timestamps) <= self.resume_after
            ):
                self.stable_since = None
                self.last_timestamps = None
                self.samples = 0
                return None
            if self.last_timestamps is None or all(
                current > previous for current, previous in zip(timestamps, self.last_timestamps)
            ):
                if self.stable_since is None:
                    self.stable_since = now
                self.last_timestamps = timestamps
                self.samples += 1
            if self.samples >= 5 and now - self.stable_since >= 0.35:
                return self.generation
            return None

    def complete_resume(self, generation):
        with self.lock:
            if self.generation != generation or self.resume_after is None:
                return False
            self._paused = False
            self._input_hold = False
            self.resume_after = None
            return True
