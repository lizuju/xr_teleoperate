import numpy as np


class R1WristWorkspace:
    def __init__(self, reference):
        self.offset = np.zeros(3)
        self.previous_position = reference[:3, 3].copy()
        self.return_residual = np.zeros(3)

    def _return_correction(self, position):
        distance = np.linalg.norm(self.return_residual)
        if self.previous_position is None or distance <= 0.005:
            return np.zeros(3)
        direction = self.return_residual / distance
        inward_step = -np.dot(position - self.previous_position, direction)
        # Pay down a confirmed overreach only while retracting, at most one extra
        # input step per frame. A stationary hand must never move the target.
        return min(max(inward_step, 0.0), distance - 0.005) * direction

    def target(self, raw_target):
        target = raw_target.copy()
        target[:3, 3] += self.offset - self._return_correction(raw_target[:3, 3])
        return target

    def hold(self):
        self.previous_position = None
        self.return_residual[:] = 0.0

    def observe(self, raw_target, solved_pose):
        position = raw_target[:3, 3]
        if self.previous_position is not None:
            correction = self._return_correction(position)
            self.offset -= correction
            self.return_residual -= correction
            residual = position + self.offset - solved_pose[:3, 3]
            distance = np.linalg.norm(residual)
            return_distance = np.linalg.norm(self.return_residual)
            if return_distance > 0.0:
                direction = self.return_residual / return_distance
                remaining = min(return_distance, max(np.dot(residual, direction), 0.0))
                self.return_residual = remaining * direction
            # Ignore the small residual from the IK posture and limit costs.
            if distance > 0.02:
                direction = residual / distance
                outward_step = np.dot(position - self.previous_position, direction)
                # Consume only new outward motion, never drift while the hand is still
                # or erase an inward command because the robot has not caught up yet.
                correction = min(max(outward_step, 0.0), distance - 0.02)
                self.offset -= correction * direction
                if correction > 0.0:
                    self.return_residual = min(distance - correction, 0.02) * direction
        self.previous_position = position.copy()
