"""Pair camera streams by mapped source time when available, otherwise receive time.

Source timestamps describe software acquisition, not sensor exposure. Head eyes
arrive over independent RTP streams; their source-time spread is recorded too.
"""

from collections import namedtuple


# One observed frame. `payload` is whatever the caller needs back (normally the
# decoded TeleImage), kept by reference and never copied.
Frame = namedtuple("Frame", "sequence received_monotonic_ns payload")

# The result of pairing one sample across all streams.
Pairing = namedtuple("Pairing", "anchor anchor_ns frames offsets_ms skew_ms timestamp_basis")


class StreamHistory:
    """Bounded, sequence-deduplicated history of one camera stream."""

    def __init__(self, name, capacity=8):
        if capacity < 2:
            raise ValueError("capacity must hold at least two frames")
        self.name = name
        self.capacity = capacity
        self._frames = []
        self._sequences = set()

    def __len__(self):
        return len(self._frames)

    def clear(self):
        self._frames = []
        self._sequences = set()

    def offer(self, sequence, received_monotonic_ns, payload):
        """Add a frame. Returns False for a repeat or an unusable packet.

        The caller polls the camera client once per control-loop iteration,
        which is faster than every stream, so the same frame is offered several
        times; the sequence check keeps the history one entry per real frame.
        """
        if payload is None:
            return False
        try:
            sequence = int(sequence or 0)
            received_monotonic_ns = int(received_monotonic_ns or 0)
        except (TypeError, ValueError):
            return False
        if sequence <= 0 or received_monotonic_ns <= 0:
            return False
        if sequence in self._sequences:
            return False
        self._frames.append(Frame(sequence, received_monotonic_ns, payload))
        self._sequences.add(sequence)
        while len(self._frames) > self.capacity:
            self._sequences.discard(self._frames.pop(0).sequence)
        return True

    def latest(self, min_sequence=None):
        """Newest frame, ignoring the history entirely if it is older than the floor."""
        if not self._frames:
            return None
        newest = self._frames[-1]
        if min_sequence is not None and newest.sequence < min_sequence:
            return None
        return newest

    def nearest(self, target_ns, min_sequence=None, source_time=False):
        """Frame whose arrival time is closest to ``target_ns``.

        Only frames at or after ``min_sequence`` are eligible, so a stream can
        repeat its previous frame but can never be rewound to an older one. A
        repeat is not hidden: the caller still compares sequences and reports
        ``repeated``.
        """
        candidates = self._frames
        if min_sequence is not None:
            candidates = [frame for frame in candidates if frame.sequence >= min_sequence]
        if source_time:
            candidates = [frame for frame in candidates
                          if (getattr(frame.payload, "timing", None) or {}).get("clock_valid")]
        if not candidates:
            return None if source_time else self.latest()
        return min(candidates, key=lambda frame: abs(frame_time(frame, source_time) - target_ns))


def frame_time(frame, source_time):
    return (frame.payload.timing["mapped_monotonic_ns"] if source_time
            else frame.received_monotonic_ns)


class CameraSynchronizer:
    """Selects the latest complete camera pair within the allowed time spread."""

    #: Cross-camera spread above which a sample is reported as not aligned.
    DEFAULT_TOLERANCE_MS = 25.0

    def __init__(self, streams, anchor="head", capacity=8, tolerance_ms=DEFAULT_TOLERANCE_MS):
        if anchor not in streams:
            raise ValueError(f"anchor {anchor!r} is not one of {list(streams)}")
        self.anchor = anchor
        self.tolerance_ms = float(tolerance_ms)
        self.histories = {name: StreamHistory(name, capacity) for name in streams}
        self.last_used = {}

    def clear(self):
        for history in self.histories.values():
            history.clear()
        self.last_used = {}

    def offer(self, stream, frame):
        """Record the frame a client just returned. Safe to call every iteration."""
        history = self.histories.get(stream)
        if history is None or frame is None:
            return False
        previous = history.latest()
        timing = getattr(frame, "timing", None) or {}
        previous_timing = (getattr(previous.payload, "timing", None) or {}) if previous else {}
        if any(previous_timing.get(key) and timing.get(key)
               and previous_timing[key] != timing[key] for key in ("clock_id", "source_epoch")):
            history.clear()
            self.last_used.pop(stream, None)
        return history.offer(getattr(frame, "sequence", 0),
                             getattr(frame, "received_monotonic_ns", 0), frame)

    def latest(self, stream):
        return self.histories[stream].latest()

    def pair(self, now_ns):
        """Prefer the freshest complete set whose actual time spread is allowed.

        Freshness is the oldest camera time in the set. With no qualifying set,
        record the latest available images and their actual spread. Every image
        recorded advances its sequence floor, including an unaligned sample.
        """
        candidates = {}
        for name, history in self.histories.items():
            floor = self.last_used.get(name, 0)
            candidates[name] = [frame for frame in history._frames
                                if frame.sequence >= floor and frame.received_monotonic_ns <= now_ns]
        if not any(candidates.values()):
            return None
        source_time = all(frames and (getattr(frames[-1].payload, "timing", None) or {}).get("clock_valid") is True
                          for frames in candidates.values())
        if source_time:
            candidates = {name: [frame for frame in frames
                                if (getattr(frame.payload, "timing", None) or {}).get("clock_valid") is True
                                and 0 < frame_time(frame, True) <= now_ns]
                          for name, frames in candidates.items()}
        frames = {name: max(values, key=lambda frame: (frame_time(frame, source_time), frame.sequence))
                  if values else None for name, values in candidates.items()}
        if not any(frame is not None for frame in frames.values()):
            return None
        if all(candidates.values()):
            times = sorted({frame_time(frame, source_time) for values in candidates.values()
                            for frame in values}, reverse=True)
            for oldest_ns in times:
                matched = {}
                for name, values in candidates.items():
                    eligible = [frame for frame in values
                                if oldest_ns <= frame_time(frame, source_time)
                                <= oldest_ns + self.tolerance_ms * 1e6]
                    if not eligible:
                        break
                    matched[name] = max(eligible, key=lambda frame: (frame_time(frame, source_time), frame.sequence))
                if len(matched) == len(self.histories):
                    frames = matched
                    break
        anchor_name = self.anchor if frames[self.anchor] is not None else next(
            name for name, frame in frames.items() if frame is not None)
        anchor_ns = frame_time(frames[anchor_name], source_time)
        offsets = {name: (frame_time(frame, source_time) - anchor_ns) / 1e6
                   for name, frame in frames.items() if frame is not None}
        for name, frame in frames.items():
            if frame is not None:
                self.last_used[name] = frame.sequence
        skew = max(offsets.values()) - min(offsets.values()) if len(offsets) > 1 else 0.0
        return Pairing(anchor=anchor_name, anchor_ns=anchor_ns, frames=frames,
                       offsets_ms=offsets, skew_ms=skew,
                       timestamp_basis="mapped_source" if source_time else "host_receive")

    def alignment(self, pairing):
        """The per-sample alignment block written into the episode."""
        if pairing is None:
            return None
        return {
            "anchor": pairing.anchor,
            "anchor_monotonic_ns": int(pairing.anchor_ns),
            "offset_ms": {name: float(offset) for name, offset in pairing.offsets_ms.items()},
            "skew_ms": float(pairing.skew_ms),
            "tolerance_ms": self.tolerance_ms,
            "aligned": bool(all(frame is not None for frame in pairing.frames.values())
                            and pairing.skew_ms <= self.tolerance_ms),
            "method": "latest complete set within actual camera time spread; no stream is rewound",
            "timestamp_basis": pairing.timestamp_basis,
            "complete": all(frame is not None for frame in pairing.frames.values()),
            "clock": "Ubuntu CLOCK_MONOTONIC; mapped PC2 software acquisition or local receive, not exposure",
        }
