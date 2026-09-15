"""Timestamp-based pairing of the independent camera streams.

The head stereo pair and the two palm cameras are three separate streams from
two devices. The head pair is one 1088-wide frame off the head sensor, so its
two eyes are inherently simultaneous; the palms are UVC modules on PC2 that
nothing triggers together with the head. A sample that simply takes "the latest
frame from each stream" therefore mixes moments that can be most of a frame
period apart.

Measured on the live rig (2026-09-15, 20 s, arrivals observed at 200 Hz and
replayed at the production 40 Hz loop, both policies over the same trace):

    head  15.05 Hz, period p50 66.7 ms / p95 107.0 ms
    left  22.53 Hz, period p50 33.6 ms / p95  74.3 ms
    right 20.87 Hz, period p50 34.5 ms / p95  73.4 ms

    latest-frame pairing   skew p50 35.6 ms  p95 71.1 ms  max 106.1 ms  >25 ms 70.7%  >50 ms 14.3%
    this module            skew p50 11.1 ms  p95 39.2 ms  max  71.6 ms  >25 ms 45.5%  >50 ms  3.4%

So this module keeps a short history per stream and pairs every sample around
the head frame that anchors it: the head is the slowest stream, so anchoring
there minimises the worst-case spread. The residual skew is reported per sample
rather than hidden, because a stall in one stream (head periods reach 107 ms at
p95) cannot be fixed by choosing a different frame -- only by excluding the
sample downstream.

Two limits are worth stating plainly:

* The ZMQ packets carry no capture timestamp, so ``received_monotonic_ns`` (the
  host arrival time) is the only clock available. Pipeline delay is roughly
  constant per stream, so differences in arrival time are a usable proxy for
  differences in exposure time, but the absolute exposure time is not
  recoverable and a constant per-stream pipeline offset is not observable.
* Images cannot be interpolated. Pairing is nearest-neighbour in time; the
  residual is recorded so a consumer can filter or weight it.
"""

from collections import namedtuple


# One observed frame. `payload` is whatever the caller needs back (normally the
# decoded TeleImage), kept by reference and never copied.
Frame = namedtuple("Frame", "sequence received_monotonic_ns payload")

# The result of pairing one sample across all streams.
Pairing = namedtuple("Pairing", "anchor anchor_ns frames offsets_ms skew_ms")


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

    def nearest(self, target_ns, min_sequence=None):
        """Frame whose arrival time is closest to ``target_ns``.

        Only frames at or after ``min_sequence`` are eligible, so a stream can
        repeat its previous frame but can never be rewound to an older one. A
        repeat is not hidden: the caller still compares sequences and reports
        ``repeated``.
        """
        candidates = self._frames
        if min_sequence is not None:
            candidates = [frame for frame in candidates if frame.sequence >= min_sequence]
        if not candidates:
            return self.latest()
        return min(candidates, key=lambda frame: abs(frame.received_monotonic_ns - target_ns))


class CameraSynchronizer:
    """Pairs the frames of several streams around a chosen anchor stream."""

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
        return history.offer(getattr(frame, "sequence", 0),
                             getattr(frame, "received_monotonic_ns", 0), frame)

    def latest(self, stream):
        return self.histories[stream].latest()

    def pair(self, now_ns):
        """Choose one frame per stream, all as close to a common instant as possible.

        The instant is not pinned to a fixed stream. A fixed head anchor only
        works while the head is the slowest stream: it is then stale enough that
        a palm frame has already arrived on either side of it. Once the head
        stereo is made faster than the palms (measured 25 Hz against 20 Hz after
        the head fps raise of 2026-09-15) the head anchor is too fresh, the
        nearest palm frame is always the previous one, and the pairing stops
        helping -- the skew went from 10.1 ms back up to 27.8 ms.

        So the instant is picked by minimax over the newest frame of every
        stream: try each as the anchor, keep the one whose worst per-stream
        distance is smallest. Frames already used by an earlier sample are never
        revisited, so a stream can repeat but cannot be rewound.
        """
        candidates = []
        for name, history in self.histories.items():
            frame = history.latest(self.last_used.get(name))
            if frame is not None:
                candidates.append((name, frame))
        if not candidates:
            return None
        best = None
        for anchor_name, anchor_frame in candidates:
            target = anchor_frame.received_monotonic_ns
            frames, offsets, worst = {}, {}, 0.0
            for name, history in self.histories.items():
                frame = history.nearest(target, self.last_used.get(name))
                frames[name] = frame
                if frame is None:
                    continue
                offset = (frame.received_monotonic_ns - target) / 1e6
                offsets[name] = offset
                worst = max(worst, abs(offset))
            if best is None or worst < best[0]:
                best = (worst, anchor_name, target, frames, offsets)
        _, anchor_name, anchor_ns, frames, offsets = best
        for name, frame in frames.items():
            if frame is not None:
                self.last_used[name] = frame.sequence
        skew = max(offsets.values()) - min(offsets.values()) if len(offsets) > 1 else 0.0
        return Pairing(anchor=anchor_name, anchor_ns=anchor_ns, frames=frames,
                       offsets_ms=offsets, skew_ms=skew)

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
            "aligned": bool(pairing.skew_ms <= self.tolerance_ms),
            "method": ("per-stream frame nearest the instant that minimises the worst "
                       "per-stream distance; no stream is rewound"),
            "clock": ("host arrival times (CLOCK_MONOTONIC); ZMQ packets carry no capture "
                      "timestamp, so exposure times are not directly observable"),
        }
