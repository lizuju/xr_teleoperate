import math


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

    def hold(self):
        self.tracking = False

    def prepare(self, raw_target, fresh):
        if not fresh:
            self.hold()
            return self.target.copy()
        if not self.tracking:
            self.tracking = True
            return self.target.copy()
        return raw_target.copy()

    def commit(self, target):
        self.target = target.copy()
