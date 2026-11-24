import threading
import time
from copy import deepcopy

import numpy as np

from teleop.robot_control.r1_hand_tracking import hand_tracking_freshness


SIDES = ("left", "right")


class LinkerO6ControlLoop:
    def __init__(self, get_tele_data, retargeter, controller, frequency, tracking_timeout,
                 should_stop, on_update, should_pause=lambda: False, is_locked=lambda index: False,
                 pickup_tolerance=(None, None), log=lambda message: None, is_external=lambda: False,
                 takeover_tolerance=(None, None)):
        # is_locked(index): the operator locked that side ([ or ]). Its target is held and
        # still sent and recorded as a deliberate command. A side with a pickup tolerance
        # (the Dex1-1 gripper) keeps holding after the unlock until the live target comes
        # within that tolerance of the held one or crosses it, so a gripped cup is not
        # dropped because the operator's hand relaxed while it was locked.
        self.is_locked = is_locked
        self.pickup_tolerance = tuple(pickup_tolerance)
        # is_external(): a policy drives the hands (teleop takeover, g), so this loop sends
        # nothing. When the operator takes over, a side with a takeover tolerance keeps the
        # policy's target, as after an unlock, until the operator's hand is that close to it or
        # crosses it, so a gripped cup or a pulled lever is not let go at the handover.
        self.is_external = is_external
        self.takeover_tolerance = tuple(takeover_tolerance)
        self.handover = False
        self.log = log
        self.locked = [False, False]
        self.pickup = [None, None]
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
            "tracking_fresh": {"left": False, "right": False},
            "paused": False,
        }
        # Achieved rate, reported once at exit: on 2026-11-16 the 90 Hz loop really ran at
        # 22-47 Hz, which aged every finger command by tens of milliseconds.
        self.step_count = 0
        self.late_steps = 0
        self.max_step_interval_ms = 0.0
        self.run_started = None
        self.run_stopped = None
        self.thread = threading.Thread(target=self._run, name="linker-o6-control", daemon=True)

    def start(self):
        self.thread.start()

    def _step(self):
        if self.stop_event.is_set() or self.should_stop():
            return
        if self.is_external():
            self._yield_to_external()
            return
        if self.should_pause():
            self._hold_paused_targets()
            return
        if self.was_paused:
            for retargeter in self.retargeters:
                retargeter.reset()
            self.timestamps = [None, None]
            self.was_paused = False
        if self.handover:
            self._arm_takeover_pickup()
        # Pull the latest XR state directly; there is no queue behind the arm IK.
        sample = self.get_tele_data()
        fresh = hand_tracking_freshness(sample, self.tracking_timeout, time.monotonic())
        locked = tuple(bool(self.is_locked(index)) for index in range(2))
        for side, (valid, timestamp, points, retargeter) in enumerate(zip(
            fresh, (sample.left_hand_timestamp, sample.right_hand_timestamp),
            (sample.left_hand_pos, sample.right_hand_pos), self.retargeters,
        )):
            if locked[side]:
                if not self.locked[side]:
                    retargeter.reset()
                    self.timestamps[side] = None
                self.pickup[side] = None
                continue
            if self.locked[side] and self.pickup_tolerance[side] is not None:
                self.pickup[side] = {"sign": None}
                self.log(f"[R1 HAND LOCK] {SIDES[side]} gripper keeps its grip until your pinch matches it.")
            if not valid:
                if self.fresh[side]:
                    retargeter.reset()
                self.timestamps[side] = None
            elif timestamp != self.timestamps[side]:
                target = retargeter.retarget(points)
                self.timestamps[side] = timestamp
                if self.pickup[side] is not None and not self._picked_up(side, target):
                    continue
                self.targets[side] = target
                self.computed_inputs[side] = {
                    "received_monotonic_ns": int(timestamp * 1_000_000_000),
                    "points": points.tolist(),
                }
        self.fresh = fresh
        self.locked = list(locked)

        # A frame can expire or tracking can be lost while retargeting runs.
        latest = self.get_tele_data()
        now = time.monotonic()
        fresh = tuple(a and b for a, b in zip(
            hand_tracking_freshness(sample, self.tracking_timeout, now),
            hand_tracking_freshness(latest, self.tracking_timeout, now),
        ))
        if self.stop_event.is_set() or self.should_stop():
            return
        if self.is_external():
            self._yield_to_external()
            return
        if self.should_pause():
            self._hold_paused_targets()
            return
        # A locked side, or a gripper waiting for pickup, is commanded on purpose: its held
        # target counts as fresh even while the operator's real hand is out of view.
        held = tuple(self.locked[index] or self.pickup[index] is not None for index in range(2))
        command = tuple(a or b for a, b in zip(fresh, held))
        self.controller.update(*self.targets, tracking_fresh=command)
        self.command_fresh = command
        for index, side in enumerate(("left", "right")):
            if fresh[index] and not held[index]:
                self.target_inputs[side] = self.computed_inputs[index]
        self._cache_recording_sample()
        self.on_update(self.controller.get_state(), self.controller.get_action())

    def _picked_up(self, side, live):
        held = np.asarray(self.targets[side], dtype=float).reshape(-1)
        difference = np.asarray(live, dtype=float).reshape(-1) - held
        sign = np.sign(difference)
        previous = self.pickup[side]["sign"]
        crossed = previous is not None and bool(np.any(sign * previous < 0))
        tolerance = self.pickup[side].get("tolerance", self.pickup_tolerance[side])
        if float(np.max(np.abs(difference))) <= tolerance or crossed:
            message = self.pickup[side].get(
                "released", f"[R1 HAND LOCK] {SIDES[side]} gripper follows your pinch again.")
            self.pickup[side] = None
            self.log(message)
            return True
        self.pickup[side]["sign"] = sign
        return False

    def _yield_to_external(self):
        self.was_paused = True
        self.handover = True
        self.targets = list(self.controller.get_action())
        self._cache_recording_sample()
        self.on_update(self.controller.get_state(), self.controller.get_action())

    def _arm_takeover_pickup(self):
        self.handover = False
        for side, tolerance in enumerate(self.takeover_tolerance):
            if tolerance is None:
                continue
            self.pickup[side] = {"sign": None, "tolerance": tolerance,
                                 "released": f"[R1 TAKEOVER] {SIDES[side]} hand follows you again."}
            self.log(f"[R1 TAKEOVER] {SIDES[side]} hand keeps the policy's target until yours matches it.")

    def _hold_paused_targets(self):
        self.was_paused = True
        self.targets = list(self.controller.get_action())
        self.controller.update(*self.targets, tracking_fresh=self.command_fresh)
        self._cache_recording_sample()
        self.on_update(self.controller.get_state(), self.controller.get_action())

    def _cache_recording_sample(self):
        snapshot = self.controller.get_recording_snapshot()
        with self.target_inputs_lock:
            self.recording_sample = {
                "hand": snapshot, "target_inputs": deepcopy(self.target_inputs),
                "tracking_fresh": dict(zip(("left", "right"), self.command_fresh)),
                "paused": self.was_paused,
                # Held on purpose: locked, or a gripper still waiting for pickup after unlock.
                "held": {side: bool(self.locked[index] or self.pickup[index] is not None)
                         for index, side in enumerate(SIDES)},
            }

    def get_recording_sample(self):
        with self.target_inputs_lock:
            return deepcopy(self.recording_sample)

    def get_target_inputs(self):
        with self.target_inputs_lock:
            return deepcopy(self.recording_sample["target_inputs"])

    def _run(self):
        previous = None
        try:
            while not self.stop_event.is_set() and not self.should_stop():
                started = time.monotonic()
                if previous is None:
                    self.run_started = started
                else:
                    interval = started - previous
                    self.max_step_interval_ms = max(self.max_step_interval_ms, interval * 1000.0)
                    self.late_steps += interval > 2.0 * self.period
                previous = started
                self._step()
                self.step_count += 1
                self.stop_event.wait(max(0.0, self.period - (time.monotonic() - started)))
        except Exception as error:
            self.error = error
        finally:
            self.run_stopped = time.monotonic()
            try:
                self.controller.stop()
            except Exception as error:
                if self.error is None:
                    self.error = error

    def timing_summary(self):
        end = self.run_stopped if self.run_stopped is not None else time.monotonic()
        elapsed = end - self.run_started if self.run_started is not None else 0.0
        rate = self.step_count / elapsed if elapsed > 0.0 else 0.0
        return (f"[R1 HAND LOOP] {rate:.1f} Hz achieved of {1.0 / self.period:.0f} Hz over {elapsed:.0f} s; "
                f"steps more than 2 periods apart: {self.late_steps}; "
                f"longest gap {self.max_step_interval_ms:.0f} ms")

    def raise_if_failed(self):
        if self.error is not None:
            raise RuntimeError("Linker O6 control loop failed") from self.error

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            raise RuntimeError("Linker O6 control loop did not stop")
        self.raise_if_failed()
