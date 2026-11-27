import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from teleop.utils.act_runtime import (ActionEnsemble, ChunkExecutor, GravityModel, ensemble_window_ticks, ObservationBuilder, PolicyClient,
                                      PolicyGuard, decode_guide, decode_request, decode_response, encode_request, encode_response,
                                      group_errors, join_action, odd_ticks, prepare_training_image, request_size, response_size,
                                      rtc_weights, smooth_chunk, split_action)
from teleop.utils.frame_sync import CameraSynchronizer

HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None
HAS_PINOCCHIO = importlib.util.find_spec("pinocchio") is not None
HAS_SDK = importlib.util.find_spec("unitree_sdk2py") is not None and importlib.util.find_spec("sshkeyboard") is not None
NORMALIZATION = {"action_min": [-1.0] * 14 + [0.2] * 6 + [4.3] + [-0.5] * 3, "action_max": [1.0] * 14 + [0.6] * 6 + [5.4] + [0.5] * 3,
                 "qpos_min": [-1.0] * 24, "qpos_max": [1.0] * 24}


class EnsembleTests(unittest.TestCase):
    def test_older_chunks_weigh_more_and_ticks_stay_aligned(self):
        ensemble = ActionEnsemble(4, k=1.0)
        ensemble.add(0, np.arange(4.0)[:, None])
        ensemble.add(2, np.array([[10.0], [20.0], [30.0], [40.0]]))
        self.assertEqual(ensemble.get(1)[0], 1.0)
        weights = np.exp([0.0, -1.0])
        expected = (2.0 * weights[0] + 10.0 * weights[1]) / weights.sum()
        self.assertAlmostEqual(ensemble.get(2)[0], expected)

    def test_a_chunk_stops_voting_after_its_horizon_and_cannot_invent_a_target(self):
        ensemble = ActionEnsemble(2)
        self.assertIsNone(ensemble.get(0))
        ensemble.add(0, np.array([[1.0], [2.0]]))
        self.assertEqual(ensemble.get(1)[0], 2.0)
        self.assertIsNone(ensemble.get(2))

    def test_future_chunks_do_not_vote_for_an_earlier_tick(self):
        ensemble = ActionEnsemble(2)
        ensemble.add(5, np.array([[9.0], [9.0]]))
        self.assertIsNone(ensemble.get(4))

    def test_a_window_lets_only_recent_chunks_vote(self):
        chunks = ((0, 1.0), (2, 2.0), (4, 3.0), (5, 4.0))
        everyone, recent = ActionEnsemble(8, k=0.0), ActionEnsemble(8, k=0.0, window=3)
        for start, value in chunks:
            everyone.add(start, np.full((8, 1), value))
            recent.add(start, np.full((8, 1), value))
        self.assertAlmostEqual(everyone.get(6)[0], 2.5)
        self.assertAlmostEqual(recent.get(6)[0], 3.5)  # started at ticks 4 and 5, within 3 ticks of 6

    def test_the_newest_chunk_votes_when_none_is_inside_the_window(self):
        ensemble = ActionEnsemble(8, k=0.0, window=2)
        ensemble.add(0, np.full((8, 1), 1.0))
        ensemble.add(1, np.full((8, 1), 2.0))
        self.assertEqual(ensemble.get(6)[0], 2.0)
        self.assertIsNone(ensemble.get(9))  # the window never stretches a chunk past its horizon

    def test_a_window_is_whole_ticks_from_seconds(self):
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                ActionEnsemble(4, window=bad)
        self.assertEqual([ensemble_window_ticks(s, 40.0) for s in (0.0, 0.001, 0.25, 0.5)], [0, 1, 10, 20])


class ActionLayoutTests(unittest.TestCase):
    def test_split_and_join_match_the_dataset_order(self):
        values = np.arange(24.0)
        parts = split_action(values)
        self.assertEqual(parts["arm_q"].tolist(), list(range(14)))
        self.assertEqual(parts["left_q"].tolist(), list(range(14, 20)))
        self.assertEqual(parts["right_q"].tolist(), list(range(20, 21)))
        self.assertEqual(parts["waist_q"], 21.0)
        self.assertEqual(parts["head_q"].tolist(), [22.0, 23.0])
        np.testing.assert_array_equal(join_action(*(parts[k] for k in ("arm_q", "left_q", "right_q", "waist_q", "head_q"))), values)

    def test_wrong_length_is_refused(self):
        with self.assertRaises(ValueError):
            split_action(np.zeros(28))

    def test_group_errors_report_the_worst_joint_of_each_group(self):
        difference = np.zeros(24)
        difference[3], difference[16], difference[23] = 0.5, -0.25, 0.1
        errors = group_errors(difference)
        self.assertEqual((errors["left_arm"], errors["left_ee"], errors["body"], errors["right_arm"]), (0.5, 0.25, 0.1, 0.0))


class ProtocolTests(unittest.TestCase):
    def test_request_and_response_round_trip_at_their_declared_sizes(self):
        qpos = np.arange(24, dtype=np.float32)
        images = np.random.randint(0, 255, (4, 6, 8, 3), dtype=np.uint8)
        payload = encode_request(qpos, images)
        self.assertEqual(len(payload), request_size(24, 4, (6, 8)))
        decoded_q, decoded_images, context = decode_request(payload, 24, 4, (6, 8))
        np.testing.assert_array_equal(decoded_q, qpos)
        np.testing.assert_array_equal(decoded_images, images)
        chunk = np.arange(5 * 24, dtype=np.float32).reshape(5, 24)
        payload = encode_response(12.5, chunk)
        self.assertEqual(len(payload), response_size(5, 24))
        inference_ms, decoded, complete = decode_response(payload, 5, 24)
        self.assertEqual(inference_ms, 12.5)
        np.testing.assert_array_equal(decoded, chunk)

    FAKE_WORKER = r"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from teleop.utils.act_runtime import encode_response, read_exact, request_size
out = sys.stdout.buffer
out.write((json.dumps({"chunk_size": 3, "action_dim": 24}) + "\n").encode()); out.flush()
if sys.argv[2] == "exit":
    sys.exit(3)
if sys.argv[2] == "hang":
    time.sleep(30)
while read_exact(sys.stdin.buffer, request_size(24, 4, (6, 8))) is not None:
    out.write(encode_response(1.5, np.ones((3, 24)))); out.flush()
"""

    def worker(self, behaviour):
        client = PolicyClient([sys.executable, "-c", self.FAKE_WORKER, str(ROOT), behaviour])
        self.addCleanup(client.close)
        return client

    def request(self, client):
        client.send(7, 0.0, np.zeros(24), np.zeros((4, 6, 8, 3), dtype=np.uint8))

    def test_client_is_idle_only_when_nothing_is_outstanding(self):
        client = self.worker("ok")
        self.assertTrue(client.idle)
        self.request(client)
        self.assertFalse(client.idle)
        deadline = time.monotonic() + 5
        response = None
        while response is None and time.monotonic() < deadline:
            response = client.poll()
            time.sleep(0.005)
        (tick, _), inference_ms, chunk, complete = response
        self.assertEqual((tick, inference_ms, chunk.shape), (7, 1.5, (3, 24)))
        self.assertTrue(client.idle)

    def test_a_dead_worker_is_reported_instead_of_blocking(self):
        client = self.worker("exit")
        deadline = time.monotonic() + 5
        while client.failed is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIn("exited", client.failed)

    def test_a_hung_worker_cannot_block_the_control_loop(self):
        client = self.worker("hang")
        began = time.monotonic()
        self.request(client)
        self.assertIsNone(client.poll())
        self.assertLess(time.monotonic() - began, 0.2)
        self.assertFalse(client.idle)

    def test_shutdown_unblocks_a_hung_worker_with_a_full_image_pipe(self):
        client = self.worker("hang")
        client.send(0,time.monotonic(),np.zeros(24),np.zeros((4,320,384,3),dtype=np.uint8))
        time.sleep(.02)
        started = time.monotonic()
        client.close()
        self.assertLess(time.monotonic()-started,4.0)
        self.assertIsNotNone(client.process.poll())


class GuardTests(unittest.TestCase):
    def guard(self, **options):
        return PolicyGuard(NORMALIZATION, range_margin=0.1, max_tracking_error=0.5, tracking_hold_s=0.5,
                           max_chunk_age_s=0.5, **options)

    def test_targets_are_held_inside_the_trained_range_and_hands_inside_0_to_1(self):
        guard = self.guard()
        clamped = guard.clamp(np.array([5.0] * 14 + [9.0] * 7 + [5.0] * 3))
        self.assertAlmostEqual(clamped[0], 1.0 + 0.2 + 0.02)
        self.assertLessEqual(clamped[14], 1.0)
        self.assertEqual(guard.clamped_ticks, 1)

    def test_an_in_range_target_passes_untouched(self):
        guard = self.guard()
        target = np.array([0.5] * 14 + [0.4] * 6 + [4.8] + [0.1] * 3)
        np.testing.assert_array_equal(guard.clamp(target), target)
        self.assertEqual(guard.clamped_ticks, 0)

    def test_silence_trips_and_stays_tripped(self):
        guard = self.guard()
        self.assertIsNone(guard.check(10.0, 9.8, 0.0))
        self.assertIn("no fresh policy output", guard.check(10.6, 10.0, 0.0))
        self.assertIsNotNone(guard.check(10.7, 10.7, 0.0))

    def test_tracking_error_must_persist_before_it_trips(self):
        guard = self.guard()
        self.assertIsNone(guard.check(10.0, 10.0, 0.6))
        self.assertIsNone(guard.check(10.2, 10.2, 0.0))
        self.assertIsNone(guard.check(10.3, 10.3, 0.6))
        self.assertIsNone(guard.check(10.7, 10.7, 0.6))
        self.assertIn("tracking error", guard.check(10.8, 10.8, 0.6))

    def test_worker_failure_trips_immediately(self):
        self.assertIn("worker failed", self.guard().check(10.0, 10.0, 0.0, "exited"))

    def test_live_pose_outside_the_training_range_is_named(self):
        qpos = np.zeros(24)
        qpos[5] = 3.0
        self.assertEqual(self.guard().out_of_training_range(qpos), [5])


class FakeFrame:
    def __init__(self, sequence, received, mapped, shape, value=10, clock_valid=True, stereo=1_000_000, now=None):
        self.sequence, self.received_monotonic_ns = sequence, received
        self.bgr = np.full((*shape, 3), value, dtype=np.uint8)
        self.bgr[..., 0] = value + 1
        self.timing = {"clock_valid": clock_valid, "mapped_monotonic_ns": mapped, "stereo_skew_ns": stereo,
                       "clock_measured_monotonic_ns": received - 1000, "clock_uncertainty_ns": 1000}


class ObservationTests(unittest.TestCase):
    NOW = 10_000_000_000

    def build(self, feedback_offset_ms=0.0, head_age_ms=40.0, wrist=True, clock_valid=True):
        anchor = self.NOW - int(head_age_ms * 1e6)
        feedback = anchor + int(feedback_offset_ms * 1e6)
        q = list(np.arange(14) / 100)
        robot = {"q": q, "waist_q": 0.7, "head_q": [0.8, 0.9], "monotonic_ns": feedback}
        arm = SimpleNamespace(get_recording_samples=lambda target, since, end: {"nearest": robot if feedback <= end else None})
        hands = {"left": {"q": [0.2] * 6, "monotonic_ns": feedback}, "right": {"q": [0.3], "monotonic_ns": feedback}}
        hand = SimpleNamespace(get_recording_states_at=lambda target, end: hands)
        builder = ObservationBuilder(arm, hand, CameraSynchronizer(["head", "left_wrist", "right_wrist"]), image_hw=(6, 8))
        received = anchor + 30_000_000
        head = FakeFrame(1, received, anchor, (12, 32), value=100, clock_valid=clock_valid)
        builder.observe(head, {"left": FakeFrame(1, received, anchor, (24, 32), value=50) if wrist else None,
                               "right": FakeFrame(1, received, anchor, (24, 32), value=200)})
        return builder

    def test_usable_row_is_in_training_order_and_rgb(self):
        qpos, images, reason = self.build().build(self.NOW)
        self.assertIsNone(reason)
        self.assertEqual(images.shape, (4, 6, 8, 3))
        self.assertEqual(images[2, 0, 0].tolist(), [50, 50, 51])
        self.assertEqual(images[3, 0, 0].tolist(), [200, 200, 201])
        np.testing.assert_allclose(qpos[:2], [0.0, 0.01], atol=1e-6)
        np.testing.assert_allclose(qpos[14:], [0.2] * 6 + [0.3] + [0.7, 0.8, 0.9])

    def test_it_matches_the_row_the_exporter_trains_on(self):
        from check_teleop_episode import training_observation
        builder = self.build()
        qpos = builder.build(self.NOW)[0]
        robot = builder.arm.get_recording_samples(0, None, self.NOW)["nearest"]
        hands = builder.hand.get_recording_states_at(0, self.NOW)
        row = training_observation({"sample": {"aligned_states": {"robot": robot, "hands": hands}}})
        flat = [value for group in ("left_arm", "right_arm", "left_ee", "right_ee", "body") for value in row[group]]
        np.testing.assert_allclose(qpos, flat, atol=1e-6)

    def test_every_training_rule_refuses_a_bad_observation(self):
        self.assertEqual(self.build(head_age_ms=200).build(self.NOW)[2], "observation_age")
        self.assertEqual(self.build(feedback_offset_ms=30).build(self.NOW)[2], "observation_skew")
        self.assertEqual(self.build(wrist=False).build(self.NOW)[2], "cameras_incomplete")
        self.assertEqual(self.build(clock_valid=False).build(self.NOW)[2], "camera_clock_invalid")

    def test_feedback_is_read_mid_images_so_it_cannot_widen_the_spread(self):
        # The head sits at the early edge of the set and hand feedback is sparse:
        # read at the head time it would land 4 ms outside the images, over 25 ms.
        anchor = self.NOW - 60_000_000
        wrist = anchor + 24_000_000
        targets = []
        def samples(target, since, end):
            targets.append(target)
            return {"nearest": {"q": [0.0] * 14, "waist_q": 0.0, "head_q": [0.0, 0.0], "monotonic_ns": target}}
        stamps = [anchor + offset * 1_000_000 for offset in (-4, 6, 16)]
        def states(target, end):
            stamp = min(stamps, key=lambda value: abs(value - target))
            return {side: {"q": [0.2] * (6 if side == "left" else 1), "monotonic_ns": stamp} for side in ("left", "right")}
        builder = ObservationBuilder(SimpleNamespace(get_recording_samples=samples),
                                     SimpleNamespace(get_recording_states_at=states),
                                     CameraSynchronizer(["head", "left_wrist", "right_wrist"]), image_hw=(6, 8))
        received = anchor + 30_000_000
        builder.observe(FakeFrame(1, received, anchor, (12, 32), value=100),
                        {side: FakeFrame(1, received, wrist, (24, 32)) for side in ("left", "right")})
        self.assertIsNone(builder.build(self.NOW)[2])
        self.assertEqual(targets, [anchor + 12_000_000])

    def test_exporter_and_runtime_share_one_image_transform(self):
        import export_r1_training_dataset as exporter
        self.assertIs(exporter.prepare_training_image, prepare_training_image)
        self.assertEqual(exporter.IMAGE_HW, (544, 640))


@unittest.skipUnless(HAS_PINOCCHIO, "needs pinocchio")
class GravityTests(unittest.TestCase):
    def test_gravity_feed_forward_is_finite_and_for_all_fourteen_joints(self):
        tau = GravityModel()(np.zeros(14))
        self.assertEqual(tau.shape, (14,))
        self.assertTrue(np.isfinite(tau).all())


@unittest.skipUnless(HAS_SDK, "needs the robot SDK environment")
class RunnerSafetyTests(unittest.TestCase):
    def load(self):
        import run_r1_act_policy
        return run_r1_act_policy

    def fake_run(self, runner, log):
        arm = Mock()
        arm.get_recording_snapshot.return_value = {"state": {"q": [0.0] * 14, "head_q": [0.0, 0.0], "waist_q": 0.0}}
        hand = Mock()
        hand.get_state.return_value = ([0.2] * 6, [4.8])
        client = Mock(idle=False, failed=None)
        client.poll.return_value = None
        builder = Mock()
        ensemble = ActionEnsemble(2)
        ensemble.add(0, np.zeros((2, 24)))
        run = runner.PolicyRun(arm, hand, lambda: (None, {}), client, builder, ensemble, PolicyGuard(NORMALIZATION), log)
        self.addCleanup(run.log.close)
        return run, arm, hand

    def test_arm_shaping_defaults_match_teleop(self):
        args = self.load().parse_args(["policy.pt"])
        self.assertEqual((args.arm_target_velocity_limit, args.arm_target_accel_limit, args.arm_dq_limit), (6.0, 40.0, 6.0))
        self.assertFalse(args.execute)

    def test_the_ensemble_window_is_off_by_default_and_never_negative(self):
        runner = self.load()
        self.assertEqual(runner.parse_args(["policy.pt"]).ensemble_window, 0.0)
        self.assertEqual(runner.parse_args(["policy.pt", "--ensemble-window", "0.5"]).ensemble_window, 0.5)
        with self.assertRaises(SystemExit):
            runner.parse_args(["policy.pt", "--ensemble-window", "-0.1"])

    def test_policy_trained_on_other_head_images_is_refused(self):
        runner = self.load()
        config = {"head_camera": {"image_shape": [544, 1280]},
                  "left_wrist_camera": {"image_shape": [480, 640]}, "right_wrist_camera": {"image_shape": [480, 640]}}
        sizes = {"color_0": [640, 544], "color_1": [640, 544], "color_2": [640, 480], "color_3": [640, 480]}
        runner.check_policy_cameras({"source_image_sizes": sizes}, config)
        with self.assertRaises(RuntimeError):
            runner.check_policy_cameras({"source_image_sizes": dict(sizes, color_3=[480, 640])}, config)
        for dataset in ({}, {"source_image_sizes": {"color_0": [544, 448]}}):
            with self.assertRaisesRegex(RuntimeError, "Export and train again"):
                runner.check_policy_cameras(dataset, config)

    def test_shadow_mode_reads_but_never_commands(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run, arm, hand = self.fake_run(runner, Path(directory) / "ticks.jsonl")
            runner.run_shadow(run, 0.001, [False], seconds=0.02)
            for method in ("activate", "start_publishing", "ctrl_dual_arm_and_head", "hold_targets", "defer_publishing"):
                getattr(arm, method).assert_not_called()
            for method in ("activate", "update", "hold"):
                getattr(hand, method).assert_not_called()
            self.assertGreater(run.tick, 0)

    def test_phase_reentry_does_not_reset_the_whole_run_deadline(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run, arm, hand = self.fake_run(runner,Path(directory)/"ticks.jsonl")
            expired = run.execution_deadline = time.monotonic()-1
            args = SimpleNamespace(max_seconds=60)
            runner.execute_loop(run,arm,hand,run.guard,lambda q: np.zeros(14),.001,[False],args)
            self.assertTrue(run.finished)
            self.assertEqual(run.execution_deadline,expired)
            self.assertEqual(run.sent,0)

    def test_execute_refuses_a_far_first_target_without_commanding_it(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run, arm, hand = self.fake_run(runner, Path(directory) / "ticks.jsonl")
            run.ignore_before_tick = 0
            chunk = np.zeros((2, 24))
            chunk[:, 0] = 1.0
            run.client.poll.side_effect = [((run.tick, time.monotonic()), 1.0, chunk, None)] + [None] * 10
            args = SimpleNamespace(max_start_delta=0.35, blend_seconds=1.0, max_seconds=1.0)
            pose = runner.execute_loop(run, arm, hand, PolicyGuard(NORMALIZATION), lambda q: np.zeros(14), 0.001, [False], args)
            self.assertEqual(pose.arm_q.tolist(), [0.0] * 14)
            for call in arm.ctrl_dual_arm_and_head.call_args_list:
                self.assertEqual(call.args[0].tolist(), [0.0] * 14)

    def test_execute_blends_from_the_live_pose_and_stops_on_silence(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run, arm, hand = self.fake_run(runner, Path(directory) / "ticks.jsonl")
            arm.get_recording_snapshot.return_value = {
                "state": {"q": [0.0] * 14, "head_q": [0.0, 0.0], "waist_q": 0.0},
                "requested": {"arm_q": [0.0] * 14}}
            run.ensemble = ActionEnsemble(400)
            chunk = np.zeros((400, 24))
            chunk[:, 0] = 0.3
            run.client.poll.side_effect = [((run.tick, time.monotonic()), 1.0, chunk, None)] + [None] * 1000
            args = SimpleNamespace(max_start_delta=0.35, blend_seconds=0.2, max_seconds=5.0)
            guard = PolicyGuard(NORMALIZATION, max_chunk_age_s=0.1)
            runner.execute_loop(run, arm, hand, guard, lambda q: np.zeros(14), 0.002, [False], args)
            self.assertIn("no fresh policy output", guard.tripped)
            commanded = [call.args[0][0] for call in arm.ctrl_dual_arm_and_head.call_args_list]
            self.assertEqual(commanded[0], 0.0)
            self.assertTrue(all(0.0 <= value <= 0.3 + 1e-9 for value in commanded))
            self.assertTrue(all(a <= b + 1e-9 for a, b in zip(commanded, commanded[1:])))

    def test_chunk_requested_too_long_ago_is_dropped(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run, _, _ = self.fake_run(runner, Path(directory) / "ticks.jsonl")
            run.ensemble = ActionEnsemble(400)
            late = np.full((400, 24), 0.3)
            run.client.poll.return_value = ((run.tick, time.monotonic() - 0.6), 900.0, late, None)
            target, _ = run.step()
            self.assertIsNone(target)
            self.assertIsNone(run.last_chunk_at)
            self.assertEqual(run.summary()["stale_chunks"], 1)

    def test_execute_never_commands_an_over_age_first_chunk(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run, arm, hand = self.fake_run(runner, Path(directory) / "ticks.jsonl")
            arm.get_recording_snapshot.return_value = {
                "state": {"q": [0.0] * 14, "head_q": [0.0, 0.0], "waist_q": 0.0},
                "requested": {"arm_q": [0.0] * 14}}
            run.ensemble = ActionEnsemble(400)
            late, fresh = np.zeros((400, 24)), np.zeros((400, 24))
            late[:, 0], fresh[:, 0] = 0.3, 0.1
            polls = [0]
            def poll():
                polls[0] += 1
                if polls[0] == 1:
                    return (run.tick, time.monotonic() - 0.6), 900.0, late, None
                # The next request comes back on time, after the over-age one was dropped.
                return ((run.tick, time.monotonic()), 1.0, fresh, None) if polls[0] == 2 else None
            run.client.poll.side_effect = poll
            args = SimpleNamespace(max_start_delta=0.35, blend_seconds=0.01, max_seconds=5.0)
            guard = PolicyGuard(NORMALIZATION, max_chunk_age_s=0.1)
            runner.execute_loop(run, arm, hand, guard, lambda q: np.zeros(14), 0.002, [False], args)
            self.assertEqual(run.stale_chunks, 1)
            commanded = [call.args[0][0] for call in arm.ctrl_dual_arm_and_head.call_args_list]
            self.assertAlmostEqual(max(commanded), 0.1)

    def test_execute_stops_on_the_tick_a_stalled_loop_resumes(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run, arm, hand = self.fake_run(runner, Path(directory) / "ticks.jsonl")
            arm.get_recording_snapshot.return_value = {
                "state": {"q": [0.0] * 14, "head_q": [0.0, 0.0], "waist_q": 0.0},
                "requested": {"arm_q": [0.0] * 14}}
            run.ensemble = ActionEnsemble(400)
            chunk = np.zeros((400, 24))
            polls, commands_at_stall = [0], []
            def poll():
                polls[0] += 1
                if polls[0] == 1:
                    return (run.tick, time.monotonic()), 1.0, chunk, None
                if polls[0] == 5:
                    commands_at_stall.append(arm.ctrl_dual_arm_and_head.call_count)
                    sent_at = time.monotonic()
                    time.sleep(0.15)
                    return (run.tick, sent_at), 1.0, chunk, None
                return None
            run.client.poll.side_effect = poll
            args = SimpleNamespace(max_start_delta=0.35, blend_seconds=0.2, max_seconds=5.0)
            guard = run.guard = PolicyGuard(NORMALIZATION, max_chunk_age_s=0.1)
            runner.execute_loop(run, arm, hand, guard, lambda q: np.zeros(14), 0.002, [False], args)
            self.assertIn("no fresh policy output", guard.tripped)
            self.assertEqual(run.stale_chunks, 1)
            self.assertEqual(arm.ctrl_dual_arm_and_head.call_count, commands_at_stall[0])


@unittest.skipUnless(HAS_TORCHVISION, "needs .venv-train")
class ModelTests(unittest.TestCase):
    def test_checkpoint_records_the_camera_images_it_trained_on(self):
        import train_r1_act
        sizes = {"color_0": [640, 544]}
        metadata = {"schema": "r1_act_hdf5_v4", "fps": 40.0, "joint_names": [], "camera_names": [],
                    "image_size_hw": [544, 640], "source_image_sizes": sizes}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "dataset.json").write_text("{}")
            info = train_r1_act.dataset_info(directory, metadata, SimpleNamespace(dataset=[]), None)
        self.assertEqual(info["source_image_sizes"], sizes)

    def setUp(self):
        import torch
        from teleop.utils.act_policy import ACTPolicy
        self.torch = torch
        torch.manual_seed(0)
        self.model = ACTPolicy(24, 24, 4, chunk_size=5, hidden_dim=32, nheads=2, dim_feedforward=32, encoder_layers=1,
                               vae_encoder_layers=1, dropout=0.0, pretrained=False)
        self.images = torch.rand(2, 4, 3, 48, 64)
        self.qpos = torch.rand(2, 24)
        self.actions = torch.rand(2, 5, 24)
        self.is_pad = torch.tensor([[False] * 3 + [True] * 2, [False] * 5])

    def test_training_and_deployed_passes_have_the_expected_shapes(self):
        pred, mu, logvar, complete = self.model(self.images, self.qpos, self.actions, self.is_pad)
        self.assertEqual((tuple(pred.shape), tuple(mu.shape), tuple(logvar.shape)), ((2, 5, 24), (2, 32), (2, 32)))
        pred, mu, logvar, complete = self.model(self.images, self.qpos)
        self.assertEqual(tuple(pred.shape), (2, 5, 24))
        self.assertIsNone(mu)

    def test_deployed_pass_is_deterministic(self):
        self.model.eval()
        self.torch.testing.assert_close(self.model(self.images, self.qpos)[0], self.model(self.images, self.qpos)[0])

    def test_style_encoder_ignores_padded_actions(self):
        self.model.eval()
        changed = self.actions.clone()
        changed[0, 3:] += 100
        first = self.model.encode_style(self.qpos, self.actions, self.is_pad)[0]
        second = self.model.encode_style(self.qpos, changed, self.is_pad)[0]
        self.torch.testing.assert_close(first[0], second[0], atol=1e-5, rtol=1e-5)

    def test_padded_steps_do_not_count_toward_the_loss(self):
        from teleop.utils.act_policy import masked_l1
        pred = self.torch.zeros(1, 3, 2)
        target = self.torch.tensor([[[1.0, 1.0], [1.0, 1.0], [99.0, 99.0]]])
        self.assertEqual(float(masked_l1(pred, target, self.torch.tensor([[False, False, True]]))), 1.0)

    def test_cameras_are_not_interchangeable(self):
        self.model.eval()
        swapped = self.images[:, [1, 0, 2, 3]]
        self.assertFalse(self.torch.allclose(self.model(self.images, self.qpos)[0], self.model(swapped, self.qpos)[0]))

    def test_augmentation_keeps_shape_and_range(self):
        from teleop.utils.act_policy import augment_images
        out = augment_images(self.images)
        self.assertEqual(out.shape, self.images.shape)
        self.assertTrue(float(out.min()) >= 0 and float(out.max()) <= 1)

    def test_checkpoint_round_trip_predicts_real_units(self):
        from teleop.utils.act_policy import ChunkPredictor, load_policy, save_checkpoint
        normalization = {"qpos_mean": [0.0] * 24, "qpos_std": [1.0] * 24, "action_mean": [5.0] * 24, "action_std": [2.0] * 24}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.pt"
            save_checkpoint(path, self.model, normalization, {"fps": 40}, 3, metrics={"best": 1.0})
            model, data = load_policy(path)
        self.assertEqual((data["step"], data["dataset"]["fps"]), (3, {"fps": 40}["fps"]))
        predict = ChunkPredictor(model, normalization, "cpu")
        chunk, complete = predict(np.zeros((4, 48, 64, 3), dtype=np.uint8), np.zeros(24, dtype=np.float32))
        self.assertEqual(chunk.shape, (5, 24))
        self.model.eval()
        self.torch.manual_seed(0)
        raw = self.model(self.torch.zeros(1, 4, 3, 48, 64), self.torch.zeros(1, 24))[0][0].detach().numpy()
        np.testing.assert_allclose(chunk, raw * 2 + 5, atol=1e-4)

    def test_a_few_steps_on_one_batch_reduce_the_loss(self):
        from teleop.utils.act_policy import act_loss
        self.model.train()
        optimizer = self.torch.optim.AdamW(self.model.parameter_groups(1e-3, 1e-4))
        losses = []
        for _ in range(25):
            pred, mu, logvar, complete = self.model(self.images, self.qpos, self.actions, self.is_pad)
            loss = act_loss(pred, self.actions, self.is_pad, mu, logvar, 0.1)[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], losses[0] * 0.8)

    def test_an_act_config_does_not_name_a_head(self):
        """ACT checkpoints keep their exact config, so code that predates the flow head still loads them."""
        self.assertNotIn("head", self.model.config)
        self.assertEqual(self.model.head, "act")


@unittest.skipUnless(HAS_TORCHVISION, "needs .venv-train")
class FlowModelTests(unittest.TestCase):
    """The flow-matching head (run 7, 2026-11-25)."""

    def setUp(self):
        import torch
        from teleop.utils.act_policy import ACTPolicy
        self.torch = torch
        torch.manual_seed(0)
        self.model = ACTPolicy(24, 24, 4, chunk_size=5, hidden_dim=32, nheads=2, dim_feedforward=32, encoder_layers=1,
                               decoder_layers=2, dropout=0.0, pretrained=False, head="flow", flow_steps=4)
        self.images = torch.rand(2, 4, 3, 48, 64)
        self.qpos = torch.rand(2, 24)
        self.actions = torch.rand(2, 5, 24)
        self.is_pad = torch.tensor([[False] * 3 + [True] * 2, [False] * 5])

    def test_config_names_the_head_and_there_is_no_style_encoder(self):
        self.assertEqual((self.model.config["head"], self.model.config["flow_steps"]), ("flow", 4))
        names = [name for name, _ in self.model.named_parameters()]
        self.assertFalse(any(name.startswith(("vae_", "latent_proj")) for name in names))
        self.assertNotIn("flow_noise", self.model.state_dict())

    def test_training_and_sampling_passes_have_the_expected_shapes(self):
        velocity, target, complete = self.model.flow_targets(self.images, self.qpos, self.actions, self.is_pad)
        self.assertEqual((tuple(velocity.shape), tuple(target.shape)), ((2, 5, 24), (2, 5, 24)))
        self.assertIsNone(complete)
        pred, mu, logvar, complete = self.model(self.images, self.qpos)
        self.assertEqual(tuple(pred.shape), (2, 5, 24))
        self.assertIsNone(mu)
        with self.assertRaises(ValueError):
            self.model(self.images, self.qpos, self.actions, self.is_pad)

    def test_sampling_starts_from_fixed_noise_so_it_is_deterministic(self):
        self.model.eval()
        first = self.model(self.images, self.qpos)[0]
        self.torch.manual_seed(123)
        self.torch.testing.assert_close(first, self.model(self.images, self.qpos)[0])
        other = self.model(self.images, self.qpos, noise=self.torch.randn(2, 5, 24))[0]
        self.assertFalse(self.torch.allclose(first, other))

    def test_euler_integration_reaches_the_end_of_a_straight_flow(self):
        """With the exact straight-line velocity (chunk - noise), the Euler steps land on the chunk."""
        noise = self.torch.randn(2, 5, 24)
        self.model.velocity = lambda memory, x, t: self.actions - noise
        memory = self.torch.zeros(2, 3, 32)
        self.torch.testing.assert_close(self.model.sample(memory, noise), self.actions)

    def test_padded_steps_do_not_count_toward_the_flow_loss(self):
        from teleop.utils.act_policy import masked_mse
        pred = self.torch.zeros(1, 3, 2)
        target = self.torch.tensor([[[1.0, 1.0], [2.0, 2.0], [99.0, 99.0]]])
        self.assertEqual(float(masked_mse(pred, target, self.torch.tensor([[False, False, True]]))), 2.5)

    def test_checkpoint_round_trip_rebuilds_the_flow_head_and_its_noise(self):
        from teleop.utils.act_policy import ChunkPredictor, load_policy, save_checkpoint
        normalization = {"qpos_mean": [0.0] * 24, "qpos_std": [1.0] * 24, "action_mean": [5.0] * 24, "action_std": [2.0] * 24}
        self.model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "f.pt"
            save_checkpoint(path, self.model, normalization, {"fps": 40}, 3, metrics={"best": 1.0})
            model, data = load_policy(path)
        self.assertEqual((model.head, model.flow_steps), ("flow", 4))
        self.torch.testing.assert_close(model.flow_noise, self.model.flow_noise)
        chunk, complete = ChunkPredictor(model, normalization, "cpu")(
            np.zeros((4, 48, 64, 3), dtype=np.uint8), np.zeros(24, dtype=np.float32))
        raw = self.model(self.torch.zeros(1, 4, 3, 48, 64), self.torch.zeros(1, 24))[0][0].detach().numpy()
        np.testing.assert_allclose(chunk, raw * 2 + 5, atol=1e-4)

    def test_training_on_one_batch_brings_the_sampled_chunk_to_it(self):
        from teleop.utils.act_policy import masked_l1, masked_mse
        self.model.eval()
        before = float(masked_l1(self.model(self.images, self.qpos)[0], self.actions, self.is_pad))
        self.model.train()
        optimizer = self.torch.optim.AdamW(self.model.parameter_groups(3e-3, 1e-4))
        for _ in range(300):
            velocity, target, _ = self.model.flow_targets(self.images, self.qpos, self.actions, self.is_pad)
            loss = masked_mse(velocity, target, self.is_pad)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.model.eval()
        after = float(masked_l1(self.model(self.images, self.qpos)[0], self.actions, self.is_pad))
        self.assertLess(after, before * 0.5)

    def test_train_options_default_to_act_and_flow_gets_more_decoder_layers(self):
        import train_r1_act
        act = train_r1_act.parse_args(["data", "--output", "run"])
        flow = train_r1_act.parse_args(["data", "--output", "run", "--head", "flow"])
        self.assertEqual((act.head, act.decoder_layers), ("act", 1))
        self.assertEqual((flow.head, flow.decoder_layers, flow.flow_steps), ("flow", 4, 10))
        self.assertNotIn("head", train_r1_act.training_settings(act))
        self.assertEqual(train_r1_act.training_settings(flow)["head"], "flow")
        with self.assertRaises(SystemExit):
            train_r1_act.parse_args(["data", "--output", "run", "--head", "flow", "--flow-steps", "0"])


class RealTimeChunkingTests(unittest.TestCase):
    """--executor chunk / rtc (2026-11-26): follow the newest chunk, request every interval, smooth, crossfade,
    and for rtc guide with the current chunk."""

    def test_weights_freeze_the_delay_then_fall_to_zero_at_the_end_of_the_overlap(self):
        weights = rtc_weights(10, 2, 7)
        np.testing.assert_array_equal(weights[:2], [1.0, 1.0])
        self.assertTrue(np.all(np.diff(weights[1:7]) < 0))
        self.assertTrue(np.all((weights[2:7] > 0) & (weights[2:7] < 1)))
        np.testing.assert_array_equal(weights[7:], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(rtc_weights(10, 4, 0), np.zeros(10))
        np.testing.assert_array_equal(rtc_weights(4, 9, 3), [1.0, 1.0, 1.0, 0.0])  # the freeze never outlives the chunk

    def test_the_executor_follows_the_newest_chunk_alone(self):
        executor = ChunkExecutor(4, 2, action_dim=1)
        self.assertIsNone(executor.get(0))
        executor.add(0, np.arange(4.0)[:, None])
        executor.add(2, np.array([[10.0], [20.0], [30.0], [40.0]]))
        executor.add(1, np.full((4, 1), 99.0))  # an older request answered late never replaces a newer chunk
        self.assertEqual(executor.get(3)[0], 20.0)
        self.assertEqual(executor.source, 2)
        self.assertIsNone(executor.get(6))
        executor.reset()
        self.assertIsNone(executor.get(3))

    def test_requests_go_out_at_once_then_every_interval(self):
        executor = ChunkExecutor(80, 10)
        self.assertTrue(executor.due(0))
        executor.sent(0)
        executor.add(0, np.zeros((80, 24)))
        self.assertFalse(executor.due(9))
        self.assertTrue(executor.due(10))

    def test_the_guide_is_the_rest_of_the_current_chunk(self):
        executor = ChunkExecutor(8, 3, action_dim=2)
        chunk = np.arange(16.0).reshape(8, 2)
        executor.add(5, chunk)
        prefix, weights = executor.guide(8, 2)
        np.testing.assert_array_equal(prefix[:5], chunk[3:])
        np.testing.assert_array_equal(prefix[5:], np.zeros((3, 2)))
        np.testing.assert_array_equal(weights, rtc_weights(8, 2, 5))

    def test_the_guide_span_ends_guide_steps_after_the_freeze(self):
        executor = ChunkExecutor(8, 3, action_dim=2, guide_steps=2)
        executor.add(5, np.arange(16.0).reshape(8, 2))
        np.testing.assert_array_equal(executor.guide(8, 1)[1], rtc_weights(8, 1, 3))
        self.assertTrue(executor.guided)
        self.assertFalse(ChunkExecutor(8, 3).guided)

    def test_smoothing_follows_the_chunk_without_lag(self):
        ramp = np.arange(10.0)[:, None] * np.ones((1, 3))
        np.testing.assert_allclose(smooth_chunk(ramp, 5)[2:8], ramp[2:8])  # a straight line is left in place
        jitter = np.zeros((10, 1))
        jitter[4] = 5.0
        np.testing.assert_allclose(smooth_chunk(jitter, 5)[2:7, 0], [1.0] * 5)
        np.testing.assert_array_equal(smooth_chunk(jitter, 1), jitter)
        self.assertEqual([odd_ticks(s, 40.0) for s in (0.0, 0.2, 0.25, 0.375)], [1, 9, 11, 15])
        with self.assertRaises(ValueError):
            ChunkExecutor(8, 3, smooth=4)

    def test_a_switch_crossfades_from_the_old_chunk(self):
        executor = ChunkExecutor(10, 3, action_dim=1, fade=3)
        executor.add(0, np.zeros((10, 1)))
        self.assertEqual(executor.get(0)[0], 0.0)
        executor.add(2, np.full((10, 1), 4.0))
        self.assertEqual([executor.get(t)[0] for t in (3, 4, 5, 6)], [1.0, 2.0, 3.0, 4.0])

    def test_without_a_chunk_the_guide_asks_for_nothing(self):
        prefix, weights = ChunkExecutor(8, 3).guide(0, 2)
        self.assertEqual(prefix.shape, (8, 24))
        self.assertFalse(prefix.any() or weights.any())

    def test_the_interval_is_whole_ticks_inside_the_chunk(self):
        for bad in (0, 8, 1.5, True):
            with self.assertRaises(ValueError):
                ChunkExecutor(8, bad)
        for bad in (dict(fade=8), dict(fade=-1), dict(smooth=0), dict(guide_steps=-1)):
            with self.assertRaises(ValueError):
                ChunkExecutor(8, 3, **bad)

    def test_a_guided_request_round_trips_at_its_declared_size(self):
        qpos = np.arange(24, dtype=np.float32)
        images = np.random.randint(0, 255, (4, 6, 8, 3), dtype=np.uint8)
        prefix = np.random.rand(5, 24).astype(np.float32)
        weights = rtc_weights(5, 1, 4)
        payload = encode_request(qpos, images, None, (prefix, weights))
        self.assertEqual(len(payload), request_size(24, 4, (6, 8), guide_shape=(5, 24)))
        decoded_q, decoded_images, _ = decode_request(payload, 24, 4, (6, 8))
        np.testing.assert_array_equal(decoded_q, qpos)
        np.testing.assert_array_equal(decoded_images, images)
        decoded_prefix, decoded_weights = decode_guide(payload, 5, 24)
        np.testing.assert_array_equal(decoded_prefix, prefix)
        np.testing.assert_array_equal(decoded_weights, weights)


@unittest.skipUnless(HAS_TORCHVISION, "needs torchvision")
class GuidedFlowTests(unittest.TestCase):
    def setUp(self):
        import torch
        from teleop.utils.act_policy import ACTPolicy
        self.torch = torch
        torch.manual_seed(0)
        self.model = ACTPolicy(24, 24, 4, chunk_size=5, hidden_dim=32, nheads=2, dim_feedforward=32, encoder_layers=1,
                               decoder_layers=2, dropout=0.0, pretrained=False, head="flow", flow_steps=10).eval()
        self.memory = torch.rand(1, 7, 32)

    def test_the_guidance_weight_starts_at_beta_and_never_exceeds_it(self):
        from teleop.utils.act_policy import rtc_guidance_weight
        self.assertEqual(rtc_guidance_weight(0.0, 5.0), 5.0)
        self.assertAlmostEqual(rtc_guidance_weight(0.5, 5.0), 2.0)
        self.assertTrue(all(0 < rtc_guidance_weight(k / 10, 5.0) <= 5.0 for k in range(10)))

    def test_zero_weights_sample_exactly_as_without_guidance(self):
        prefix = self.torch.rand(1, 5, 24)
        plain = self.model.sample(self.memory)
        self.torch.testing.assert_close(self.model.sample_guided(self.memory, prefix, self.torch.zeros(5)), plain,
                                        rtol=0, atol=0)

    def test_guidance_pulls_the_weighted_steps_onto_the_prefix(self):
        prefix = self.torch.randn(1, 5, 24)
        weights = self.torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
        plain = self.model.sample(self.memory)
        guided = self.model.sample_guided(self.memory, prefix, weights)
        before = float((plain[0, :3] - prefix[0, :3]).abs().mean())
        after = float((guided[0, :3] - prefix[0, :3]).abs().mean())
        self.assertLess(after, 0.2 * before)
        self.assertFalse(any(p.grad is not None for p in self.model.parameters()))

    def test_the_predictor_continues_a_prefix_in_real_units(self):
        from teleop.utils.act_policy import ChunkPredictor
        normalization = {"qpos_mean": [0.0] * 24, "qpos_std": [1.0] * 24, "action_mean": [5.0] * 24, "action_std": [2.0] * 24}
        predict = ChunkPredictor(self.model.requires_grad_(False), normalization, "cpu")
        images, qpos = np.zeros((4, 48, 64, 3), dtype=np.uint8), np.zeros(24, dtype=np.float32)
        plain, _ = predict(images, qpos)
        same, _ = predict.guided(images, qpos, None, np.zeros((5, 24)), np.zeros(5))
        np.testing.assert_array_equal(same, plain)
        prefix = 5 + 2 * np.random.default_rng(1).standard_normal((5, 24)).astype(np.float32)
        guided, _ = predict.guided(images, qpos, None, prefix, np.array([1, 1, 0, 0, 0], dtype=np.float32))
        self.assertLess(np.abs(guided[:2] - prefix[:2]).mean(), 0.2 * np.abs(plain[:2] - prefix[:2]).mean())


class RealTimeChunkingRunnerTests(unittest.TestCase):
    def load(self):
        import run_r1_act_policy
        return run_r1_act_policy

    def test_chunk_execution_is_off_by_default_and_its_settings_are_checked(self):
        runner = self.load()
        args = runner.parse_args(["policy.pt"])
        self.assertEqual((args.executor, args.chunk_interval, args.chunk_smooth, args.chunk_fade, args.rtc_beta, args.rtc_guide),
                         ("ensemble", 0.25, 0.375, 0.25, 5.0, 0.25))
        args = runner.parse_args(["policy.pt", "--executor", "chunk", "--chunk-interval", "0.5", "--chunk-fade", "0"])
        self.assertEqual((args.executor, args.chunk_interval, args.chunk_fade), ("chunk", 0.5, 0.0))
        for bad in (["--chunk-interval", "0"], ["--rtc-beta", "-1"], ["--chunk-smooth", "-0.1"], ["--executor", "other"]):
            with self.assertRaises(SystemExit):
                runner.parse_args(["policy.pt", *bad])

    def rtc_run(self, runner, log, executor=None):
        arm = Mock()
        arm.get_recording_snapshot.return_value = {"state": {"q": [0.0] * 14, "head_q": [0.0, 0.0], "waist_q": 0.0}}
        hand = Mock()
        hand.get_state.return_value = ([0.2] * 6, [4.8])
        client = Mock(idle=True, failed=None)
        client.poll.return_value = None
        builder = Mock()
        builder.build.return_value = (np.zeros(24, dtype=np.float32), np.zeros((4, 6, 8, 3), dtype=np.uint8), None)
        run = runner.PolicyRun(arm, hand, lambda: (None, {}), client, builder,
                               executor or ChunkExecutor(80, 10, guide_steps=10), PolicyGuard(NORMALIZATION), log)
        run.fps = 1e-6  # ticks advance one per step, whatever the wall clock does
        self.addCleanup(run.log.close)
        return run

    def test_requests_go_out_every_interval_with_the_current_chunk_as_their_guide(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run = self.rtc_run(runner, Path(directory) / "ticks.jsonl")
            chunk = np.linspace(0.0, 0.5, 80)[:, None] * np.ones((1, 24))
            run.ensemble.add(0, chunk)
            run.ensemble.sent(0)
            targets = [run.step()[0] for _ in range(25)]
            sent = run.client.send.call_args_list
            self.assertEqual([call.args[0] for call in sent], [10, 20])
            prefix, weights = sent[0].args[5]
            np.testing.assert_allclose(prefix[:70], chunk[10:])
            np.testing.assert_array_equal(weights, rtc_weights(80, 1, 11))  # the delay plus 10 guided steps
            np.testing.assert_allclose(targets[12], chunk[12])
            summary = run.summary()
            self.assertEqual((summary["executor"], summary["interval_ticks"], summary["guided_requests"]), ("rtc", 10, 2))

    def test_chunk_execution_paces_requests_but_sends_no_guide(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run = self.rtc_run(runner, Path(directory) / "ticks.jsonl", ChunkExecutor(80, 10, smooth=15, fade=10))
            run.ensemble.add(0, np.zeros((80, 24)))
            run.ensemble.sent(0)
            for _ in range(25):
                run.step()
            sent = run.client.send.call_args_list
            self.assertEqual([(call.args[0], call.args[5]) for call in sent], [(10, None), (20, None)])
            self.assertEqual((run.summary()["executor"], run.summary()["smooth_ticks"]), ("chunk", 15))

    def test_the_freeze_covers_the_longest_recent_delay(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run = self.rtc_run(runner, Path(directory) / "ticks.jsonl")
            self.assertEqual(run.rtc_delay(), 1)
            run.client.poll.side_effect = [((0, time.monotonic()), 30.0, np.zeros((80, 24)), None)] + [None] * 10
            run.tick = 3
            run.step()
            self.assertEqual(run.rtc_delay(), 3)
            self.assertEqual(run.ensemble.source, 0)


if __name__ == "__main__":
    unittest.main()
