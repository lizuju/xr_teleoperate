import math


def _hand_timestamps(tele_data):
    return (tele_data.left_hand_timestamp, tele_data.right_hand_timestamp)


def hand_tracking_present(tele_data):
    """True while this hand still has a last sample.

    Vision Pro zeroes the timestamp in the same HAND_MOVE that drops a hand.
    A Wi-Fi hole sends nothing, so the last timestamp stays put and only ages.
    Hold on the zeroed timestamp, not on age.
    """
    return tuple(
        bool(tele_data.motion_data_ready)
        and math.isfinite(timestamp)
        and timestamp > 0.0
        for timestamp in _hand_timestamps(tele_data)
    )


def hand_tracking_freshness(tele_data, timeout, now):
    return tuple(
        present and 0.0 <= now - timestamp <= timeout
        for present, timestamp in zip(hand_tracking_present(tele_data), _hand_timestamps(tele_data))
    )


class R1WristHold:
    def __init__(self, target):
        self.target = target.copy()
        self.tracking = True
        self.resume_streak = 0

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
