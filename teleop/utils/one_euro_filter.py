import numpy as np


class OneEuroFilter:
    # Initial tuning for radian joint targets; algorithm: https://gery.casiez.net/1euro/
    def __init__(self, min_cutoff=3.0, beta=8.0, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.reset()

    def reset(self):
        self._value = None
        self._derivative = None
        self._timestamp = None

    def filter(self, value, timestamp, initial_value=None):
        value = np.asarray(value, dtype=float)
        if self._timestamp is None or timestamp - self._timestamp > 0.25:
            # After a pause, start at the measured pose instead of catching up.
            self._value = np.array(value if initial_value is None else initial_value, dtype=float, copy=True)
            self._derivative = np.zeros_like(value)
            self._timestamp = timestamp
            return self._value.copy()

        dt = timestamp - self._timestamp
        if dt <= 0.0:
            return self._value.copy()
        derivative = (value - self._value) / dt
        derivative_alpha = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * self.derivative_cutoff * dt))
        self._derivative += derivative_alpha * (derivative - self._derivative)
        cutoff = self.min_cutoff + self.beta * np.abs(self._derivative)
        alpha = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))
        self._value += alpha * (value - self._value)
        self._timestamp = timestamp
        return self._value.copy()
