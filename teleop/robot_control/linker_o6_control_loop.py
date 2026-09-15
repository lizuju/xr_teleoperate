import threading
import time
from copy import deepcopy

from teleop.robot_control.r1_hand_tracking import hand_tracking_freshness


class LinkerO6ControlLoop:
    def __init__(self, get_tele_data, retargeter, controller, frequency, tracking_timeout,
                 should_stop, on_update, should_pause=lambda: False):
        self.get_tele_data = get_tele_data
        self.retargeters = (retargeter.left, retargeter.right)
        self.controller = controller
        self.period = 1.0 / frequency
        self.tracking_timeout = tracking_timeout
        self.should_stop = should_stop
        self.should_pause = should_pause
        self.was_paused = False
        self.on_update = on_update
        self.stop_event = threading.Event()
        self.error = None
        self.fresh = (False, False)
        self.command_fresh = (False, False)
        self.timestamps = [None, None]
        self.targets = list(controller.get_action())
        self.computed_inputs = [None, None]
        self.target_inputs = {"left": None, "right": None}
        self.target_inputs_lock = threading.Lock()
        self.recording_sample = {
            "hand": controller.get_recording_snapshot(),
            "target_inputs": deepcopy(self.target_inputs),
        }
        self.thread = threading.Thread(target=self._run, name="linker-o6-control", daemon=True)

    def start(self):
        self.thread.start()

    def _step(self):
        if self.stop_event.is_set() or self.should_stop():
            return
        if self.should_pause():
            self._hold_paused_targets()
            return
        if self.was_paused:
            for retargeter in self.retargeters:
                retargeter.reset()
            self.timestamps = [None, None]
            self.was_paused = False
        # Pull the latest XR state directly; there is no queue behind the arm IK.
        sample = self.get_tele_data()
        fresh = hand_tracking_freshness(sample, self.tracking_timeout, time.monotonic())
        for side, (valid, timestamp, points, retargeter) in enumerate(zip(
            fresh, (sample.left_hand_timestamp, sample.right_hand_timestamp),
            (sample.left_hand_pos, sample.right_hand_pos), self.retargeters,
        )):
            if not valid:
                if self.fresh[side]:
                    retargeter.reset()
                self.timestamps[side] = None
            elif timestamp != self.timestamps[side]:
                self.targets[side] = retargeter.retarget(points)
                self.computed_inputs[side] = {
                    "received_monotonic_ns": int(timestamp * 1_000_000_000),
                    "points": points.tolist(),
                }
                self.timestamps[side] = timestamp
        self.fresh = fresh

        # A frame can expire or tracking can be lost while retargeting runs.
        latest = self.get_tele_data()
        now = time.monotonic()
        fresh = tuple(a and b for a, b in zip(
            hand_tracking_freshness(sample, self.tracking_timeout, now),
            hand_tracking_freshness(latest, self.tracking_timeout, now),
        ))
        if self.stop_event.is_set() or self.should_stop():
            return
        if self.should_pause():
            self._hold_paused_targets()
            return
        self.controller.update(*self.targets, tracking_fresh=fresh)
        self.command_fresh = fresh
        for index, side in enumerate(("left", "right")):
            if fresh[index]:
                self.target_inputs[side] = self.computed_inputs[index]
        self._cache_recording_sample()
        self.on_update(self.controller.get_state(), self.controller.get_action())

    def _hold_paused_targets(self):
        self.was_paused = True
        self.targets = list(self.controller.get_action())
        self.controller.update(*self.targets, tracking_fresh=self.command_fresh)
        self._cache_recording_sample()
        self.on_update(self.controller.get_state(), self.controller.get_action())

    def _cache_recording_sample(self):
        snapshot = self.controller.get_recording_snapshot()
        with self.target_inputs_lock:
            self.recording_sample = {"hand": snapshot, "target_inputs": deepcopy(self.target_inputs)}

    def get_recording_sample(self):
        with self.target_inputs_lock:
            return deepcopy(self.recording_sample)

    def get_target_inputs(self):
        with self.target_inputs_lock:
            return deepcopy(self.recording_sample["target_inputs"])

    def _run(self):
        try:
            while not self.stop_event.is_set() and not self.should_stop():
                started = time.monotonic()
                self._step()
                self.stop_event.wait(max(0.0, self.period - (time.monotonic() - started)))
        except Exception as error:
            self.error = error
        finally:
            try:
                self.controller.stop()
            except Exception as error:
                if self.error is None:
                    self.error = error

    def raise_if_failed(self):
        if self.error is not None:
            raise RuntimeError("Linker O6 control loop failed") from self.error

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            raise RuntimeError("Linker O6 control loop did not stop")
        self.raise_if_failed()
