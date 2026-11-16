"""Keep Python's garbage collector from pausing the teleop main loop.

On 2026-11-16 the loop stalled ~105 ms about every 5 s while recording. A probe showed
each stall was a full (generation 2) collection of ~84 ms in MainThread: every full
collection walked the whole heap loaded at startup (IK models, DDS types, the XR
bridge), while recording's steady allocation triggered one every few seconds.
freeze_startup_heap() collects once and then gc.freeze()s that heap, so later full
collections only walk objects created by the loop. The cost: a reference cycle made
only of startup objects that later becomes garbage is never collected.

SlowGcMonitor records collections of SLOW_GC_MS or longer. Its callback only appends to
a bounded deque: it must not log, because a collection can start inside a logging or
queue call that already holds a non-reentrant lock in the same thread.
"""
from collections import deque
import gc
import threading
import time

SLOW_GC_MS = 20.0


class SlowGcMonitor:
    def __init__(self, threshold_ms=SLOW_GC_MS, keep=50):
        self.threshold_ms = threshold_ms
        self.events = deque(maxlen=keep)
        self.count = 0
        self.max_ms = 0.0
        self._started = None
        self._closed = False
        gc.callbacks.append(self._on_gc)

    def _on_gc(self, phase, info):
        if phase == "start":
            self._started = time.perf_counter()
            return
        started, self._started = self._started, None
        if started is None:
            return
        duration_ms = (time.perf_counter() - started) * 1000.0
        if duration_ms >= self.threshold_ms:
            self.count += 1
            self.max_ms = max(self.max_ms, duration_ms)
            self.events.append((time.monotonic(), info.get("generation"), duration_ms,
                                threading.current_thread().name))

    def summary(self):
        recent = ", ".join(f"gen{generation} {duration:.0f} ms ({thread})"
                           for _, generation, duration, thread in list(self.events)[-5:])
        return (f"[R1 GC] collections >= {self.threshold_ms:.0f} ms: {self.count}"
                + (f", max {self.max_ms:.0f} ms; last: {recent}" if self.count else "")
                + f"; frozen startup objects: {gc.get_freeze_count()}")

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            gc.callbacks.remove(self._on_gc)
        except ValueError:
            pass


def freeze_startup_heap(log, threshold_ms=SLOW_GC_MS):
    """Collect, freeze the startup heap and start a SlowGcMonitor. Call once, before the loop."""
    started = time.perf_counter()
    collected = gc.collect()
    gc.freeze()
    log.info("[R1 GC] froze %d startup objects after collecting %d in %.0f ms; "
             "later full collections skip them",
             gc.get_freeze_count(), collected, (time.perf_counter() - started) * 1000.0)
    return SlowGcMonitor(threshold_ms)
