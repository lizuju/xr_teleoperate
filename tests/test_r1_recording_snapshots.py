import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import test_linker_o6_controller as hand_fixtures
from test_r1_a7_official_activation import FakeCRC, FakeLowCmd, load_r1_controller_namespace
from check_teleop_episode import SOURCE_NAMES, validate_episode
from teleop.utils import r1_capture


class R1RecordingSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hand_fixtures.LinkerO6ControllerTest.setUpClass()
        cls.hand_module = hand_fixtures.LinkerO6ControllerTest.module

    @classmethod
    def tearDownClass(cls):
        hand_fixtures.LinkerO6ControllerTest.tearDownClass()

    def setUp(self):
        self.now = 50.0
        self.namespace = load_r1_controller_namespace()
        self.arm = self.namespace["R1_A7_ArmController"](deferred_activation=True)
        self.arm.lowstate_subscriber.Close()
        self.namespace["time"] = SimpleNamespace(monotonic=lambda: self.now, sleep=lambda _: None)
        self.arm_state = self.arm.lowstate_buffer.GetData()
        self.arm_state.monotonic_timestamp = self.now
        self.arm_state.sequence = 17
        for motor in self.arm_state.motor_state:
            motor.q = 0.1
            motor.dq = -0.02
        self.arm.msg = FakeLowCmd()
        self.arm.crc = FakeCRC()
        self.arm_writes = []

        def write_arm(message):
            self.arm_writes.append(copy.deepcopy(message))
            self.arm.publish_running = False
            return True

        self.arm.lowcmd_publisher = Mock(Write=Mock(side_effect=write_arm))
        # Snapshot tests assert requested == the ctrl argument. Leave shaping to
        # test_arm_target_shaper; a same-timestamp 0.8 -> 0.9 would otherwise hold.
        self.arm.target_shaper.velocity_limit = 0.0
        hand_fixtures.FakePublisher.instances.clear()
        hand_fixtures.FakeSubscriber.instances.clear()
        hand_fixtures.FakeSubscriber.queued_messages = {}
        self.hand_time = patch.object(self.hand_module, "time", SimpleNamespace(monotonic=lambda: self.now))
        self.hand_time.start()
        self.addCleanup(self.hand_time.stop)
        self.hand = self.hand_module.LinkerO6Controller(apply_grip_cap=False)
        self.hand._on_left_state(hand_fixtures.state_message([0.1] * 6, mode=2))
        self.hand._on_right_state(hand_fixtures.state_message([0.2] * 6, mode=2))
        self.hand.ready = self.hand.active = True
        self.hand.left_action = np.full(6, 0.1)
        self.hand.right_action = np.full(6, 0.2)
        self.hand.requested_targets = (self.hand.left_action.copy(), self.hand.right_action.copy())
        self.hand.action_time = self.now - 0.02

    def tearDown(self):
        self.hand.active = False
        self.hand.stop()
        self.arm.stop()

    def publish_arm(self, target=0.8):
        self.arm.ctrl_dual_arm_and_head(np.full(14, target), np.full(14, 0.3), [0.2, -0.3])
        self.arm.publish_running = True
        self.arm._ctrl_motor_state()
        self.arm.raise_if_failed()

    def test_arm_snapshot_distinguishes_requested_limited_published_and_actual_feedback(self):
        self.publish_arm()
        snapshot = self.arm.get_recording_snapshot()
        np.testing.assert_allclose(snapshot["requested"]["arm_q"], 0.8)
        np.testing.assert_allclose(snapshot["state"]["q"], 0.1)
        self.assertTrue(np.all(np.asarray(snapshot["published"]["arm_q"]) > 0.1))
        self.assertTrue(np.all(np.asarray(snapshot["published"]["arm_q"]) < 0.8))
        indices = self.namespace["R1_A7_JointArmIndex"]
        self.assertEqual(snapshot["published"]["arm_q"], [self.arm_writes[-1].motor_cmd[i].q for i in indices])
        self.assertEqual(snapshot["published"]["sequence"], 1)
        self.assertEqual(snapshot["state"]["sequence"], 17)
        self.assertEqual(snapshot["published"]["monotonic_ns"], 50_000_000_000)
        self.assertEqual(snapshot["state"]["head_q"], [0.1, 0.1])
        self.assertEqual(snapshot["published"]["head_q"], [0.2, -0.3])

    def test_arm_snapshot_read_has_no_write_or_target_refresh_and_copies_all_layers(self):
        self.publish_arm()
        before = self.arm.get_recording_snapshot()
        requested_at = self.arm.target_updated_at
        writes = self.arm.lowcmd_publisher.Write.call_count
        for _ in range(3):
            self.assertEqual(self.arm.get_recording_snapshot(), before)
        self.assertEqual(self.arm.lowcmd_publisher.Write.call_count, writes)
        self.assertEqual(self.arm.target_updated_at, requested_at)
        changed = self.arm.get_recording_snapshot()
        changed["requested"]["arm_q"][0] = -900
        changed["published"]["arm_q"][1] = -900
        changed["state"]["q"][2] = -900
        self.assertEqual(self.arm.get_recording_snapshot(), before)
        self.arm.q_target[0] = 0.6
        self.arm.published_command["arm_q"][1] = 0.7
        self.arm_state.motor_state[next(iter(self.namespace["R1_A7_JointArmIndex"]))].q = 0.9
        self.assertEqual(before["requested"]["arm_q"][0], 0.8)
        self.assertLess(before["published"]["arm_q"][1], 0.8)
        self.assertEqual(before["state"]["q"][0], 0.1)

    def test_arm_failed_write_keeps_last_successful_publication(self):
        self.publish_arm()
        before = self.arm.get_recording_snapshot()["published"]
        self.arm.ctrl_dual_arm_and_head(np.full(14, 0.9), np.zeros(14), [0.1, 0.2])
        self.arm.lowcmd_publisher.Write = Mock(return_value=False)
        with self.assertRaisesRegex(RuntimeError, "Write failed"):
            self.arm._write_command(target_updated_at=self.arm.target_updated_at)
        after = self.arm.get_recording_snapshot()
        self.assertEqual(after["published"], before)
        self.assertEqual(after["requested"]["arm_q"], [0.9] * 14)
        self.assertEqual(self.arm.published_sequence, 1)
        self.assertEqual(after["state"]["q"], [0.1] * 14)

    def test_unpublished_arm_snapshot_is_explicitly_null_and_does_not_create_a_write(self):
        snapshot = self.arm.get_recording_snapshot()
        self.assertIsNone(snapshot["published"])
        self.assertIsNone(snapshot["requested"]["monotonic_ns"])
        self.arm.lowcmd_publisher.Write.assert_not_called()

    def test_arm_command_history_queries_accepted_requests_without_future_or_boundary_crossing(self):
        self.assertIsNone(self.arm.get_recording_command_at(50_000_000_000, 0))
        self.arm.ctrl_dual_arm(np.full(14, 0.4), np.full(14, 0.2))
        self.now += 0.025
        self.arm.ctrl_dual_arm_and_head(np.full(14, 0.6), np.full(14, 0.3), [0.1, -0.2], 0.7)
        old = self.arm.get_recording_command_at(50_010_000_000, 0)
        latest = self.arm.get_recording_command_at(50_025_000_000, 50_010_000_000)
        self.assertEqual(old["requested"]["sequence"], 1)
        self.assertEqual(old["requested"]["arm_q"], [0.4] * 14)
        self.assertEqual(old["requested"]["monotonic_ns"], 50_000_000_000)
        self.assertIsNone(old["published"])
        self.assertEqual(latest["requested"]["sequence"], 2)
        self.assertEqual(latest["requested"]["head_q"], [0.1, -0.2])
        self.assertEqual(latest["requested"]["waist_q"], 0.7)
        self.assertIsNone(self.arm.get_recording_command_at(49_999_999_999, 0))
        self.assertIsNone(self.arm.get_recording_command_at(50_025_000_000, 50_025_000_001))
        self.arm.lowcmd_publisher.Write.assert_not_called()

    def test_arm_command_history_is_bounded_and_copies_request_and_publication(self):
        self.publish_arm()
        before = self.arm.get_recording_command_at(50_000_000_000, 0)
        changed = self.arm.get_recording_command_at(50_000_000_000, 0)
        changed["requested"]["arm_q"][0] = -900
        changed["published"]["arm_q"][0] = -900
        self.arm.q_target[0] = 0.7
        self.arm.published_command["arm_q"][0] = 0.6
        self.assertEqual(self.arm.get_recording_command_at(50_000_000_000, 0), before)
        for _ in range(64):
            self.arm.ctrl_dual_arm(np.full(14, 0.5), np.zeros(14))
        self.assertEqual(len(self.arm.command_history), 64)
        self.assertEqual(self.arm.command_history[0]["sequence"], 2)
        self.assertEqual(self.arm.command_history[-1]["sequence"], 65)
        self.assertEqual(self.arm.command_publications, {})

    def test_arm_publication_keeps_request_sequence_captured_before_concurrent_update(self):
        self.arm.ctrl_dual_arm_and_head(np.full(14, 0.8), np.zeros(14), [0.2, -0.3])

        def update_during_write(message):
            self.arm_writes.append(copy.deepcopy(message))
            self.now += 0.01
            self.arm.ctrl_dual_arm_and_head(np.full(14, 0.9), np.zeros(14), [0.1, 0.2])
            self.arm.publish_running = False
            return True

        self.arm.lowcmd_publisher.Write = Mock(side_effect=update_during_write)
        self.arm.publish_running = True
        self.arm._ctrl_motor_state()
        self.arm.raise_if_failed()
        old = self.arm.get_recording_command_at(50_000_000_000, 0)
        latest = self.arm.get_recording_command_at(50_010_000_000, 0)
        self.assertEqual(old["requested"]["sequence"], 1)
        self.assertEqual(old["published"]["request_sequence"], 1)
        self.assertEqual(old["published"]["head_q"], [0.2, -0.3])
        self.assertEqual(latest["requested"]["sequence"], 2)
        self.assertIsNone(latest["published"])
        self.assertEqual(self.arm.get_recording_snapshot()["published"]["request_sequence"], 1)

    def test_failed_arm_publication_preserves_per_request_last_success(self):
        self.publish_arm()
        before = self.arm.get_recording_command_at(50_000_000_000, 0)
        self.arm.lowcmd_publisher.Write = Mock(return_value=False)
        with self.assertRaisesRegex(RuntimeError, "Write failed"):
            self.arm._write_command(target_updated_at=self.arm.target_updated_at, request_sequence=1)
        self.assertEqual(self.arm.get_recording_command_at(50_000_000_000, 0), before)
        self.now += 0.025
        self.arm.ctrl_dual_arm(np.full(14, 0.9), np.zeros(14))
        with self.assertRaisesRegex(RuntimeError, "Write failed"):
            self.arm._write_command(target_updated_at=self.arm.target_updated_at, request_sequence=2)
        self.assertIsNone(self.arm.get_recording_command_at(50_025_000_000, 0)["published"])
        self.assertEqual(self.arm.get_recording_command_at(50_000_000_000, 0), before)

    def test_hold_clears_history_without_fabricating_an_accepted_request(self):
        self.publish_arm()
        self.now += 0.025
        self.arm.hold_targets()
        self.assertEqual(self.arm.requested_sequence, 1)
        self.assertEqual(self.arm.target_updated_at, self.now)
        self.assertIsNone(self.arm.get_recording_command_at(50_025_000_000, 0))
        self.assertEqual(self.arm.command_publications, {})
        self.arm.ctrl_dual_arm(np.full(14, 0.5), np.zeros(14))
        resumed = self.arm.get_recording_command_at(50_025_000_000, 50_025_000_000)
        self.assertEqual(resumed["requested"]["sequence"], 2)
        self.assertIsNone(resumed["published"])
        self.assertIsNone(self.arm.get_recording_command_at(50_000_000_000, 0))

    def test_hold_during_inflight_publication_cannot_repopulate_history(self):
        self.arm.ctrl_dual_arm(np.full(14, 0.8), np.zeros(14))

        def hold_during_write(message):
            self.now += 0.01
            self.arm.hold_targets()
            self.arm.ctrl_dual_arm(np.full(14, 0.5), np.zeros(14))
            self.arm.publish_running = False
            return True

        self.arm.lowcmd_publisher.Write = Mock(side_effect=hold_during_write)
        self.arm.publish_running = True
        self.arm._ctrl_motor_state()
        self.arm.raise_if_failed()
        self.assertIsNone(self.arm.get_recording_command_at(50_000_000_000, 0))
        self.assertEqual(self.arm.get_recording_command_at(50_010_000_000, 0)["requested"]["sequence"], 2)
        self.assertIsNone(self.arm.get_recording_command_at(50_010_000_000, 0)["published"])
        self.assertEqual(self.arm.command_publications, {})

    def test_arm_history_keeps_last_successful_publication_of_the_same_request(self):
        self.publish_arm()
        self.now += 0.01
        self.arm.publish_running = True
        self.arm._ctrl_motor_state()
        result = self.arm.get_recording_command_at(50_000_000_000, 0)
        self.assertEqual(result["requested"]["sequence"], 1)
        self.assertEqual(result["published"]["request_sequence"], 1)
        self.assertEqual(result["published"]["sequence"], 2)
        self.assertEqual(result["published"]["monotonic_ns"], 50_010_000_000)

    def test_hand_snapshot_distinguishes_requested_smoothed_published_and_actual(self):
        self.hand.update([0.9] * 6, [0.8] * 6)
        snapshot = self.hand.get_recording_snapshot()
        self.assertEqual(snapshot["requested"]["left_q"], [0.9] * 6)
        self.assertEqual(snapshot["requested"]["monotonic_ns"], 50_000_000_000)
        self.assertEqual(snapshot["requested"]["sequence"], 1)
        self.assertEqual(snapshot["state"]["left"]["q"], [0.1] * 6)
        self.assertTrue(0.1 < snapshot["published"]["left"]["q"][0] < 0.9)
        self.assertTrue(0.2 < snapshot["published"]["right"]["q"][0] < 0.8)
        for side in ("left", "right"):
            message = getattr(self.hand, side + "_publisher").writes[-1]
            self.assertEqual(snapshot["published"][side]["q"], [command.q for command in message.cmds])
            self.assertEqual(snapshot["published"][side]["mode"], message.cmds[0].mode)
            self.assertEqual(snapshot["published"][side]["sequence"], 1)
            self.assertEqual(snapshot["published"][side]["request_sequence"], 1)

    def test_hand_snapshot_read_does_not_publish_refresh_or_share_mutable_values(self):
        self.hand.update([0.9] * 6, [0.8] * 6)
        before = self.hand.get_recording_snapshot()
        timestamps = (self.hand.action_time, self.hand.left_state_time, self.hand.right_state_time)
        counts = [len(publisher.writes) for publisher in hand_fixtures.FakePublisher.instances]
        self.now += 0.01
        for _ in range(3):
            self.assertEqual(self.hand.get_recording_snapshot(), before)
        self.assertEqual([len(publisher.writes) for publisher in hand_fixtures.FakePublisher.instances], counts)
        self.assertEqual((self.hand.action_time, self.hand.left_state_time, self.hand.right_state_time), timestamps)
        changed = self.hand.get_recording_snapshot()
        changed["state"]["left"]["q"][0] = -900
        changed["requested"]["left_q"][1] = -900
        changed["published"]["right"]["q"][2] = -900
        self.assertEqual(self.hand.get_recording_snapshot(), before)
        self.hand.left_state[0] = 0.6
        self.hand.requested_targets[0][1] = 0.7
        self.hand.published_commands["right"]["q"][2] = 0.9
        self.assertEqual(before["state"]["left"]["q"][0], 0.1)
        self.assertEqual(before["requested"]["left_q"][1], 0.9)
        self.assertLess(before["published"]["right"]["q"][2], 0.8)

    def test_failed_hand_write_only_advances_the_successful_side(self):
        self.hand.update([0.5] * 6, [0.6] * 6)
        for failing_side in ("left", "right"):
            with self.subTest(failing_side=failing_side):
                self.now += 0.02
                before = self.hand.get_recording_snapshot()
                publisher = getattr(self.hand, failing_side + "_publisher")
                publisher.write_result = False
                try:
                    with self.assertRaisesRegex(RuntimeError, "Failed to publish"):
                        self.hand.update([0.8] * 6, [0.9] * 6)
                finally:
                    publisher.write_result = True
                after = self.hand.get_recording_snapshot()
                successful_side = "right" if failing_side == "left" else "left"
                self.assertEqual(after["published"][failing_side], before["published"][failing_side])
                self.assertEqual(after["published"][successful_side]["sequence"],
                                 before["published"][successful_side]["sequence"] + 1)
                self.assertGreater(after["published"][successful_side]["monotonic_ns"],
                                   before["published"][successful_side]["monotonic_ns"])
                self.assertEqual(after["state"], before["state"])
                self.assertEqual(after["requested"], {
                    "left_q": [0.8] * 6, "right_q": [0.9] * 6,
                    "monotonic_ns": int(self.now * 1e9),
                    "sequence": before["requested"]["sequence"] + 1})
                self.assertEqual(after["published"][successful_side]["request_sequence"],
                                 after["requested"]["sequence"])
                self.assertNotEqual(after["published"][failing_side]["request_sequence"],
                                    after["requested"]["sequence"])

    def test_hand_transport_exception_preserves_that_side_snapshot(self):
        self.hand.update([0.5] * 6, [0.6] * 6)
        before = self.hand.get_recording_snapshot()
        with patch.object(self.hand.left_publisher, "Write", side_effect=OSError("fake transport failure")):
            with self.assertRaisesRegex(OSError, "fake transport"):
                self.hand.update([0.8] * 6, [0.9] * 6)
        after = self.hand.get_recording_snapshot()
        self.assertEqual(after["published"]["left"], before["published"]["left"])
        self.assertEqual(after["published"]["right"]["sequence"], 2)

    def test_current_r1_capture_output_is_accepted_by_offline_checker(self):
        hand_loop = SimpleNamespace(get_recording_sample=lambda: {
            "hand": self.hand.get_recording_snapshot(), "target_inputs": None,
            "paused": False, "tracking_fresh": {"left": True, "right": True},
        })
        capture = r1_capture.R1Capture(self.arm, self.hand, hand_loop,
                                       tracking_timeout=0.25, image_shape=(48, 128))
        tele_data = SimpleNamespace(
            motion_data_timestamp=self.now, motion_data_ready=True,
            left_hand_timestamp=self.now, right_hand_timestamp=self.now,
            left_hand_tracking_valid=True, right_hand_tracking_valid=True,
            left_hand_pos=np.zeros((25, 3)), right_hand_pos=np.zeros((25, 3)),
            left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4), head_pose=np.eye(4),
        )
        image = SimpleNamespace(bgr=np.zeros((48, 128, 3), dtype=np.uint8),
                                received_monotonic_ns=int(self.now * 1e9), sequence=1)
        with patch.object(r1_capture, "time", SimpleNamespace(monotonic_ns=lambda: 50_001_000_000,
                                                            time_ns=lambda: 1_800_000_000_000_000_000)):
            frame = capture.frame(tele_data, image, "following")
        self.arm.lowcmd_publisher.Write.assert_not_called()
        self.assertEqual([len(publisher.writes) for publisher in hand_fixtures.FakePublisher.instances], [0, 0])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "colors").mkdir()
            for key, pixels in frame["colors"].items():
                relative = f"colors/{key}.png"
                self.assertTrue(cv2.imwrite(str(directory / relative), pixels))
                frame["colors"][key] = relative
            frame["idx"] = 0
            manifest = {"schema": "xr_teleop_episode_v2", "status": "complete", "outcome": "unspecified",
                        "episode_id": 1, "frames": "frames.jsonl", "frame_count": 1, "text": {},
                        "info": {"frequency": 30, "image": {"width": 64, "height": 48, "fps": 15},
                                 "joint_names": {group: [str(i) for i in range(len(state["qpos"]))]
                                                 for group, state in frame["states"].items()}}}
            (directory / "episode.json").write_text(json.dumps(manifest))
            (directory / "frames.jsonl").write_text(json.dumps(frame, allow_nan=False) + "\n")
            report = validate_episode(directory)
            self.assertTrue(report["valid"], report)
            self.assertEqual(set(report["sources"]), set(SOURCE_NAMES))
            self.assertEqual(report["images_checked"], 2)
            self.assertEqual(report["tracking_invalid_frames"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
