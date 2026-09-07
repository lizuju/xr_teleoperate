import math

import numpy as np


def hand_tracking_freshness(tele_data, timeout, now):
    return tuple(
        bool(tele_data.motion_data_ready)
        and math.isfinite(timestamp)
        and timestamp > 0.0
        and 0.0 <= now - timestamp <= timeout
        for timestamp in (tele_data.left_hand_timestamp, tele_data.right_hand_timestamp)
    )


class R1WristHold:
    def __init__(self, target):
        self.target = target.copy()
        self.tracking = True
        self.position_offset = np.zeros(3)
        self.rotation_offset = np.eye(3)

    def hold(self):
        self.tracking = False

    def prepare(self, raw_target, fresh):
        if not fresh:
            self.hold()
            return self.target.copy()
        if not self.tracking:
            # Re-clutch at the held pose, without replaying motion during occlusion.
            self.position_offset = self.target[:3, 3] - raw_target[:3, 3]
            self.rotation_offset = raw_target[:3, :3].T @ self.target[:3, :3]
            self.tracking = True
        target = raw_target.copy()
        target[:3, 3] += self.position_offset
        target[:3, :3] = raw_target[:3, :3] @ self.rotation_offset
        return target

    def commit(self, target):
        self.target = target.copy()
