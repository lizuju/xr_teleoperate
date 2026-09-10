import ast
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from teleop.robot_control.linker_o6_control_loop import LinkerO6ControlLoop


class FakeRetargeter:
    def __init__(self):
        self.calls = 0
        self.resets = 0

    def retarget(self, points):
        self.calls += 1
        return points[0, :].repeat(2)

    def reset(self):
        self.resets += 1


class FakeController:
    def __init__(self):
        self.actions = [np.zeros(6), np.zeros(6)]
        self.updates = []
        self.stopped = False
        self.updated = threading.Event()

    def get_action(self):
        return tuple(a.copy() for a in self.actions)

    def get_state(self):
        return np.zeros(6), np.zeros(6)

    def update(self, left, right, tracking_fresh):
        if self.stopped:
            raise AssertionError("publish after stop")
        self.actions = [a.copy() for a in (left, right)]
        self.updates.append((time.monotonic(), tracking_fresh, self.get_action()))
        self.updated.set()

    def stop(self):
        self.stopped = True


def sample(timestamp=None, fresh=(True, True), value=0.2):
    timestamp = time.monotonic() if timestamp is None else timestamp
    return SimpleNamespace(
        motion_data_ready=True,
        left_hand_timestamp=timestamp if fresh[0] else 0.0,
        right_hand_timestamp=timestamp if fresh[1] else 0.0,
        left_hand_pos=np.full((25, 3), value),
        right_hand_pos=np.full((25, 3), value),
    )


class LinkerO6ControlLoopTest(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        self.retargeter = SimpleNamespace(left=FakeRetargeter(), right=FakeRetargeter())
        self.outputs = []
        self.loop = LinkerO6ControlLoop(
            sample, self.retargeter, self.controller, 30.0, 0.25,
            lambda: False, lambda state, action: self.outputs.append((state, action)),
        )

    def test_same_xr_frame_is_not_reoptimized_but_commands_keep_ticking(self):
        snapshot = sample()
        self.loop.get_tele_data = lambda: snapshot
        for _ in range(3):
            self.loop._step()
        self.assertEqual(self.retargeter.left.calls, 1)
        self.assertEqual(self.retargeter.right.calls, 1)
        self.assertEqual(len(self.controller.updates), 3)

    def test_one_hand_can_hold_while_the_other_continues(self):
        self.loop._step()
        self.loop.get_tele_data = lambda: sample(fresh=(False, True))
        self.loop._step()
        self.assertEqual(self.controller.updates[-1][1], (False, True))
        self.assertEqual(self.retargeter.left.calls, 1)
        self.assertEqual(self.retargeter.left.resets, 1)
        self.assertEqual(self.retargeter.right.calls, 2)

    def test_source_disconnect_is_not_refreshed_by_worker_heartbeat(self):
        self.loop.get_tele_data = lambda: sample(timestamp=time.monotonic() - 1)
        self.loop._step()
        self.assertEqual(self.controller.updates[-1][1], (False, False))
        self.assertEqual(self.retargeter.left.calls, 0)

    def test_tracking_loss_during_retarget_is_checked_before_publication(self):
        snapshots = iter([sample(), sample(fresh=(False, True))])
        self.loop.get_tele_data = lambda: next(snapshots)
        self.loop._step()
        self.assertEqual(self.controller.updates[-1][1], (False, True))

    def test_original_frame_expiry_is_checked_even_if_new_tracking_arrives(self):
        snapshots = iter([sample(timestamp=100), sample(timestamp=101)])
        self.loop.get_tele_data = lambda: next(snapshots)
        with mock.patch("teleop.robot_control.linker_o6_control_loop.time.monotonic", side_effect=[100.01, 101, 101]):
            self.loop._step()
        self.assertEqual(self.controller.updates[-1][1], (False, False))

    def test_stop_during_retarget_suppresses_publication(self):
        original = self.retargeter.left.retarget

        def request_stop(points):
            self.loop.stop_event.set()
            return original(points)

        self.retargeter.left.retarget = request_stop
        self.loop._step()
        self.assertEqual(self.controller.updates, [])

    def test_external_quit_stops_hands_even_while_main_loop_is_blocked(self):
        quit_event = threading.Event()
        self.loop.should_stop = quit_event.is_set
        self.loop.start()
        try:
            self.assertTrue(self.controller.updated.wait(1.0))
            quit_event.set()
            self.loop.thread.join(0.5)
            self.assertFalse(self.loop.thread.is_alive())
            self.assertTrue(self.controller.stopped)
        finally:
            self.loop.stop()

    def test_hands_read_latest_source_while_arm_work_is_blocked(self):
        self.loop.start()
        try:
            self.assertTrue(self.controller.updated.wait(1.0))
            before = len(self.controller.updates)
            # Model an arm solver waiting for 200 ms; hand work has no dependency on it.
            threading.Event().wait(0.2)
            self.assertGreaterEqual(len(self.controller.updates) - before, 4)
            self.assertGreaterEqual(self.retargeter.left.calls, 5)
        finally:
            self.loop.stop()
        count = len(self.controller.updates)
        time.sleep(0.04)
        self.assertEqual(len(self.controller.updates), count)
        self.assertTrue(self.controller.stopped)

    def test_failure_stops_controller_and_reaches_main_thread(self):
        error = TimeoutError("feedback stale")
        self.controller.update = mock.Mock(side_effect=error)
        self.loop.start()
        self.loop.thread.join(1.0)
        self.assertTrue(self.controller.stopped)
        with self.assertRaises(RuntimeError) as raised:
            self.loop.stop()
        self.assertIs(raised.exception.__cause__, error)

    def test_failed_solver_is_not_published(self):
        self.retargeter.left.retarget = mock.Mock(side_effect=ValueError("invalid skeleton"))
        self.loop.start()
        self.loop.thread.join(1.0)
        self.assertEqual(self.controller.updates, [])
        self.assertTrue(self.controller.stopped)
        with self.assertRaises(RuntimeError):
            self.loop.stop()

    def test_shutdown_error_is_not_hidden(self):
        self.loop.should_stop = lambda: True
        self.controller.stop = mock.Mock(side_effect=OSError("DDS close failed"))
        self.loop.start()
        self.loop.thread.join(1.0)
        with self.assertRaises(RuntimeError) as raised:
            self.loop.stop()
        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_wrapper_serializes_mutable_snapshot_access(self):
        path = Path(__file__).resolve().parents[1] / "teleop/televuer/src/televuer/tv_wrapper.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TeleVuerWrapper")
        get_data = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_tele_data")
        namespace = {}
        exec(compile(ast.Module(body=[get_data], type_ignores=[]), str(path), "exec"), namespace)
        active = []
        peak = []

        def read_cache():
            active.append(1)
            peak.append(len(active))
            time.sleep(0.005)
            active.pop()

        wrapper = SimpleNamespace(_tele_data_lock=threading.Lock(), _get_tele_data=read_cache)
        threads = [threading.Thread(target=namespace["get_tele_data"], args=(wrapper,)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(max(peak), 1)

    def test_main_no_longer_owns_o6_retarget_or_publish(self):
        path = Path(__file__).resolve().parents[1] / "teleop/teleop_hand_and_arm.py"
        source = path.read_text()
        loop = source.split("# main loop. robot start to follow VR user's motion", 1)[1].split("except KeyboardInterrupt:", 1)[0]
        self.assertNotIn("hand_ctrl.update(", loop)
        self.assertNotIn("hand_ctrl.hold(", loop)
        self.assertNotIn("linker_o6_retargeter.", loop)
        self.assertIn("linker_o6_loop.raise_if_failed()", loop)


if __name__ == "__main__":
    unittest.main()
