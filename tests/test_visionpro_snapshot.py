from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils.visionpro_source import VisionProMotionSource
from test_visionpro_bridge import SERVER


SNAPSHOT_SERVER = SERVER.replace(
    "import sys, time", "import sys, time\nfrom pathlib import Path"
).replace(
    "        yield message",
    """        command = Path(sys.argv[2]).read_text()
        if command == 'fatal':
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'snapshot fixture fatal')
        if command == 'head':
            message.head_valid = False
        if command == 'left':
            message.left_valid = False
        yield message""",
)


@unittest.skipUnless(os.environ.get("VISIONPRO_PYTHON"), "Set VISIONPRO_PYTHON to the isolated receiver Python")
class VisionProSnapshotTest(unittest.TestCase):
    def connect(self, timeout=1.):
        python = os.environ["VISIONPRO_PYTHON"]
        fixture = tempfile.TemporaryDirectory(prefix="visionpro-snapshot-test-")
        self.addCleanup(fixture.cleanup)
        self.command = Path(fixture.name) / "command"
        self.command.write_text("valid")
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "tools"))
        server = subprocess.Popen(
            [python, "-u", "-c", SNAPSHOT_SERVER, "valid", str(self.command)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        def stop():
            server.terminate()
            server.wait(timeout=3.)
            server.stdout.close()
            server.stderr.close()
        self.addCleanup(stop)
        line = server.stdout.readline()
        self.assertTrue(line, server.stderr.read() if server.poll() is not None else "No port")
        source = VisionProMotionSource("127.0.0.1", python, int(line), timeout=timeout)
        self.addCleanup(source.close)
        return source

    def wait_for(self, predicate, timeout=3.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.005)
        self.fail("snapshot receiver did not reach the expected state")

    def snapshot(self, source):
        return json.loads((Path(source._snapshot_dir.name) / "snapshot.json").read_text())

    @contextmanager
    def blocked_reader(self, source):
        entered = threading.Event()
        release = threading.Event()
        resumed = threading.Event()
        consumed = []
        original = source.accept_packet

        def accept(packet):
            first = not entered.is_set()
            if first:
                entered.set()
                if not release.wait(5.):
                    raise ValueError("test consumer gate timed out")
            original(packet)
            consumed.append(packet)
            if not first:
                resumed.set()

        with patch.object(source, "accept_packet", side_effect=accept):
            try:
                self.assertTrue(entered.wait(2.), "reader never entered the gate")
                yield release, resumed, consumed
            finally:
                release.set()
                self.assertTrue(resumed.wait(2.), "reader did not consume a new snapshot after release")

    def test_blocked_consumer_does_not_backpressure_bridge_or_replay_old_queue(self):
        source = self.connect()
        with self.blocked_reader(source) as (release, resumed, consumed):
            initial = self.snapshot(source)
            self.wait_for(lambda: self.snapshot(source)["bridge_publish_seq"] >= initial["bridge_publish_seq"] + 25)
            latest = self.snapshot(source)
            self.assertGreaterEqual(latest["transport"]["packets_received"],
                                    initial["transport"]["packets_received"] + 25)
            self.assertEqual(consumed, [])
            self.assertIsNone(source._process.poll())
            release.set()
            self.assertTrue(resumed.wait(2.))
            self.assertGreaterEqual(consumed[1]["bridge_publish_seq"], latest["bridge_publish_seq"])
            self.assertGreaterEqual(consumed[1]["sample_time"], latest["sample_time"])
            self.assertGreater(consumed[1]["bridge_publish_seq"], consumed[0]["bridge_publish_seq"] + 10)
        self.assertTrue(source.get_tracking_diagnostics()["head_tracking"])
        self.assertTrue(source.get_hand_motion_snapshot()["motion_data_ready"])

    def test_head_loss_and_recovery_overwritten_during_stall_still_require_realign(self):
        source = self.connect(timeout=2.)
        self.assertFalse(source.needs_realign)
        with self.blocked_reader(source) as (release, resumed, consumed):
            baseline = self.snapshot(source)["tracking_loss_seq"]["head"]
            self.command.write_text("head")
            self.wait_for(lambda: not self.snapshot(source).get("head_valid", True))
            self.command.write_text("valid")
            self.wait_for(lambda: self.snapshot(source).get("head_valid", False))
            latest = self.snapshot(source)
            self.assertGreater(latest["tracking_loss_seq"]["head"], baseline)
            release.set()
            self.assertTrue(resumed.wait(2.))
            self.assertGreaterEqual(consumed[1]["bridge_publish_seq"], latest["bridge_publish_seq"])
        self.assertTrue(source.get_tracking_diagnostics()["head_tracking"])
        self.assertTrue(source.needs_realign)
        self.assertEqual(source.get_hold_reason(), "tracking_interrupted")

    def test_hidden_left_loss_is_observed_by_each_consumer_without_losing_right(self):
        source = self.connect(timeout=2.)
        with self.blocked_reader(source) as (release, resumed, consumed):
            baseline = self.snapshot(source)["tracking_loss_seq"]["left"]
            self.command.write_text("left")
            self.wait_for(lambda: not self.snapshot(source).get("left_valid", True))
            self.command.write_text("valid")
            self.wait_for(lambda: self.snapshot(source).get("left_valid", False))
            latest = self.snapshot(source)
            self.assertGreater(latest["tracking_loss_seq"]["left"], baseline)
            release.set()
            self.assertTrue(resumed.wait(2.))
            self.assertGreaterEqual(consumed[1]["bridge_publish_seq"], latest["bridge_publish_seq"])
        self.assertFalse(source.needs_realign)
        observations = []

        def observe():
            observations.append((source.get_hand_motion_snapshot(), source.get_hand_motion_snapshot()))
        observe()
        worker = threading.Thread(target=observe)
        worker.start()
        worker.join(timeout=2.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(observations), 2)
        for first, second in observations:
            self.assertEqual(first["left_hand_timestamp"], 0.)
            self.assertGreater(first["right_hand_timestamp"], 0.)
            self.assertGreater(second["left_hand_timestamp"], 0.)
            self.assertGreater(second["right_hand_timestamp"], 0.)

    def test_fatal_rpc_exit_cannot_leave_the_last_snapshot_valid(self):
        source = self.connect()
        self.assertTrue(source.get_hand_motion_snapshot()["motion_data_ready"])
        self.command.write_text("fatal")
        self.wait_for(lambda: source._process.poll() is not None)
        self.wait_for(lambda: source.get_tracking_diagnostics()["error"] is not None)
        self.assertFalse(source.get_hand_motion_snapshot()["motion_data_ready"])
        self.assertTrue(source.needs_realign)

    def test_abrupt_bridge_exit_invalidates_cached_pose_and_close_removes_slot(self):
        source = self.connect()
        directory = Path(source._snapshot_dir.name)
        self.assertTrue((directory / "snapshot.json").is_file())
        source._process.kill()
        source._process.wait(timeout=2.)
        self.wait_for(lambda: source.get_tracking_diagnostics()["error"] is not None)
        self.assertFalse(source.get_hand_motion_snapshot()["motion_data_ready"])
        self.assertTrue(source.needs_realign)
        source.close()
        self.assertFalse(directory.exists())
        self.assertFalse(source._reader.is_alive())
        self.assertTrue(source._reader_stop.is_set())


if __name__ == "__main__":
    unittest.main()
