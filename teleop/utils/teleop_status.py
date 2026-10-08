import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid


logger = logging.getLogger(__name__)


class TeleopStatusPublisher:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else Path(f"/tmp/r1-teleop-status-{os.getuid()}.json")
        self.run_id = str(uuid.uuid4())
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._sequence = 0
        self._sample_ns = 0
        self._state_key = None
        self._worker = threading.Thread(target=self._write_pending, name="teleop-status", daemon=True)
        self._worker.start()

    def submit(self, motion, recorder=None, record_enabled=False, error=None, force=False,
               hold_reason=None):
        recording = recorder.status_snapshot() if recorder is not None else {
            "state": "idle" if record_enabled else "disabled", "episode": None,
            "frames_accepted": 0, "last_saved": None, "quality": None, "error": None,
        }
        if motion != "tracking_hold":
            hold_reason = None
        state_key = (motion, recording["state"], recording["episode"], recording["last_saved"],
                     recording.get("quality"), recording["error"], error, hold_reason)
        sample_ns = time.monotonic_ns()
        with self._condition:
            if self._closed:
                return
            if not force and state_key == self._state_key and sample_ns - self._sample_ns < 100_000_000:
                return
            self._sequence += 1
            self._sample_ns = sample_ns
            self._state_key = state_key
            self._pending = {
                "schema": "r1_teleop_status_v1", "run_id": self.run_id,
                "sequence": self._sequence, "sample_monotonic_ns": sample_ns,
                "motion": motion, "recording": recording, "error": error,
            }
            if hold_reason is not None:
                self._pending["hold_reason"] = hold_reason
            self._condition.notify()

    def _write_pending(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._pending is None:
                    return
                snapshot, self._pending = self._pending, None
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=f".{self.path.name}.", delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(snapshot, stream, ensure_ascii=False, allow_nan=False)
                    stream.write("\n")
                os.replace(temporary, self.path)
            except OSError as error:
                logger.warning("[NATIVE STATUS] Could not publish status: %s", error)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def close(self, recorder=None, record_enabled=False, error=None):
        self.submit("failed" if error else "stopped", recorder, record_enabled, error, force=True)
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._worker.join(timeout=1.0)
        if self._worker.is_alive():
            logger.warning("[NATIVE STATUS] Status writer did not finish within one second")
