import threading
import time

from teleop.robot_control.r1_hand_tracking import hand_tracking_freshness


class LinkerO6ControlLoop:
    def __init__(self, get_tele_data, retargeter, controller, frequency, tracking_timeout,
                 should_stop, on_update):
        self.get_tele_data = get_tele_data
        self.retargeters = (retargeter.left, retargeter.right)
        self.controller = controller
        self.period = 1.0 / frequency
        self.tracking_timeout = tracking_timeout
        self.should_stop = should_stop
        self.on_update = on_update
        self.stop_event = threading.Event()
        self.error = None
        self.fresh = (False, False)
        self.timestamps = [None, None]
        self.targets = list(controller.get_action())
        self.thread = threading.Thread(target=self._run, name="linker-o6-control", daemon=True)

    def start(self):
        self.thread.start()

    def _step(self):
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
        self.controller.update(*self.targets, tracking_fresh=fresh)
        self.on_update(self.controller.get_state(), self.controller.get_action())

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
