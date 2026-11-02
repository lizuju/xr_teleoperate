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

from teleop.utils.act_runtime import (ActionEnsemble, GravityModel, ObservationBuilder, PolicyClient, PolicyGuard,
                                      decode_request, decode_response, encode_request, encode_response, group_errors,
                                      join_action, prepare_training_image, request_size, response_size, split_action)
from teleop.utils.frame_sync import CameraSynchronizer

HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None
HAS_PINOCCHIO = importlib.util.find_spec("pinocchio") is not None
HAS_SDK = importlib.util.find_spec("unitree_sdk2py") is not None and importlib.util.find_spec("sshkeyboard") is not None
NORMALIZATION = {"action_min": [-1.0] * 14 + [0.2] * 12 + [-0.5] * 3, "action_max": [1.0] * 14 + [0.6] * 12 + [0.5] * 3,
                 "qpos_min": [-1.0] * 29, "qpos_max": [1.0] * 29}


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


class ActionLayoutTests(unittest.TestCase):
    def test_split_and_join_match_the_dataset_order(self):
        values = np.arange(29.0)
        parts = split_action(values)
        self.assertEqual(parts["arm_q"].tolist(), list(range(14)))
        self.assertEqual(parts["left_q"].tolist(), list(range(14, 20)))
        self.assertEqual(parts["right_q"].tolist(), list(range(20, 26)))
        self.assertEqual(parts["waist_q"], 26.0)
        self.assertEqual(parts["head_q"].tolist(), [27.0, 28.0])
        np.testing.assert_array_equal(join_action(*(parts[k] for k in ("arm_q", "left_q", "right_q", "waist_q", "head_q"))), values)

    def test_wrong_length_is_refused(self):
        with self.assertRaises(ValueError):
            split_action(np.zeros(28))

    def test_group_errors_report_the_worst_joint_of_each_group(self):
        difference = np.zeros(29)
        difference[3], difference[16], difference[28] = 0.5, -0.25, 0.1
        errors = group_errors(difference)
        self.assertEqual((errors["left_arm"], errors["left_ee"], errors["body"], errors["right_arm"]), (0.5, 0.25, 0.1, 0.0))


class ProtocolTests(unittest.TestCase):
    def test_request_and_response_round_trip_at_their_declared_sizes(self):
        qpos = np.arange(29, dtype=np.float32)
        images = np.random.randint(0, 255, (4, 6, 8, 3), dtype=np.uint8)
        payload = encode_request(qpos, images)
        self.assertEqual(len(payload), request_size(29, 4, (6, 8)))
        decoded_q, decoded_images = decode_request(payload, 29, 4, (6, 8))
        np.testing.assert_array_equal(decoded_q, qpos)
        np.testing.assert_array_equal(decoded_images, images)
        chunk = np.arange(5 * 29, dtype=np.float32).reshape(5, 29)
        payload = encode_response(12.5, chunk)
        self.assertEqual(len(payload), response_size(5, 29))
        inference_ms, decoded = decode_response(payload, 5, 29)
        self.assertEqual(inference_ms, 12.5)
        np.testing.assert_array_equal(decoded, chunk)

    FAKE_WORKER = r"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from teleop.utils.act_runtime import encode_response, read_exact, request_size
out = sys.stdout.buffer
out.write((json.dumps({"chunk_size": 3, "action_dim": 29}) + "\n").encode()); out.flush()
if sys.argv[2] == "exit":
    sys.exit(3)
if sys.argv[2] == "hang":
    time.sleep(30)
while read_exact(sys.stdin.buffer, request_size(29, 4, (6, 8))) is not None:
    out.write(encode_response(1.5, np.ones((3, 29)))); out.flush()
"""

    def worker(self, behaviour):
        client = PolicyClient([sys.executable, "-c", self.FAKE_WORKER, str(ROOT), behaviour])
        self.addCleanup(client.close)
        return client

    def request(self, client):
        client.send(7, 0.0, np.zeros(29), np.zeros((4, 6, 8, 3), dtype=np.uint8))

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
        (tick, _), inference_ms, chunk = response
        self.assertEqual((tick, inference_ms, chunk.shape), (7, 1.5, (3, 29)))
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


class GuardTests(unittest.TestCase):
    def guard(self, **options):
        return PolicyGuard(NORMALIZATION, range_margin=0.1, max_tracking_error=0.5, tracking_hold_s=0.5,
                           max_chunk_age_s=0.5, **options)

    def test_targets_are_held_inside_the_trained_range_and_hands_inside_0_to_1(self):
        guard = self.guard()
        clamped = guard.clamp(np.array([5.0] * 14 + [9.0] * 12 + [5.0] * 3))
        self.assertAlmostEqual(clamped[0], 1.0 + 0.2 + 0.02)
        self.assertLessEqual(clamped[14], 1.0)
        self.assertEqual(guard.clamped_ticks, 1)

    def test_an_in_range_target_passes_untouched(self):
        guard = self.guard()
        target = np.array([0.5] * 14 + [0.4] * 12 + [0.1] * 3)
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
        qpos = np.zeros(29)
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
        hands = {"left": {"q": [0.2] * 6, "monotonic_ns": feedback}, "right": {"q": [0.3] * 6, "monotonic_ns": feedback}}
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
        np.testing.assert_allclose(qpos[14:], [0.2] * 6 + [0.3] * 6 + [0.7, 0.8, 0.9])

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

    def test_exporter_and_runtime_share_one_image_transform(self):
        import export_r1_training_dataset as exporter
        self.assertIs(exporter.prepare_training_image, prepare_training_image)
        self.assertEqual(exporter.IMAGE_HW, (240, 320))


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
        hand.get_state.return_value = ([0.2] * 6, [0.2] * 6)
        client = Mock(idle=False, failed=None)
        client.poll.return_value = None
        builder = Mock()
        ensemble = ActionEnsemble(2)
        ensemble.add(0, np.zeros((2, 29)))
        run = runner.PolicyRun(arm, hand, lambda: (None, {}), client, builder, ensemble, PolicyGuard(NORMALIZATION), log)
        self.addCleanup(run.log.close)
        return run, arm, hand

    def test_arm_shaping_defaults_match_teleop(self):
        args = self.load().parse_args(["policy.pt"])
        self.assertEqual((args.arm_target_velocity_limit, args.arm_target_accel_limit, args.arm_dq_limit), (6.0, 40.0, 6.0))
        self.assertFalse(args.execute)

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

    def test_execute_refuses_a_far_first_target_without_commanding_it(self):
        runner = self.load()
        with tempfile.TemporaryDirectory() as directory:
            run, arm, hand = self.fake_run(runner, Path(directory) / "ticks.jsonl")
            run.ignore_before_tick = 0
            chunk = np.zeros((2, 29))
            chunk[:, 0] = 1.0
            run.client.poll.side_effect = [((run.tick, 0.0), 1.0, chunk)] + [None] * 10
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
            chunk = np.zeros((400, 29))
            chunk[:, 0] = 0.3
            run.client.poll.side_effect = [((run.tick, 0.0), 1.0, chunk)] + [None] * 1000
            args = SimpleNamespace(max_start_delta=0.35, blend_seconds=0.2, max_seconds=5.0)
            guard = PolicyGuard(NORMALIZATION, max_chunk_age_s=0.1)
            runner.execute_loop(run, arm, hand, guard, lambda q: np.zeros(14), 0.002, [False], args)
            self.assertIn("no fresh policy output", guard.tripped)
            commanded = [call.args[0][0] for call in arm.ctrl_dual_arm_and_head.call_args_list]
            self.assertEqual(commanded[0], 0.0)
            self.assertTrue(all(0.0 <= value <= 0.3 + 1e-9 for value in commanded))
            self.assertTrue(all(a <= b + 1e-9 for a, b in zip(commanded, commanded[1:])))


@unittest.skipUnless(HAS_TORCHVISION, "needs .venv-train")
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        from teleop.utils.act_policy import ACTPolicy
        self.torch = torch
        torch.manual_seed(0)
        self.model = ACTPolicy(29, 29, 4, chunk_size=5, hidden_dim=32, nheads=2, dim_feedforward=32, encoder_layers=1,
                               vae_encoder_layers=1, dropout=0.0, pretrained=False)
        self.images = torch.rand(2, 4, 3, 48, 64)
        self.qpos = torch.rand(2, 29)
        self.actions = torch.rand(2, 5, 29)
        self.is_pad = torch.tensor([[False] * 3 + [True] * 2, [False] * 5])

    def test_training_and_deployed_passes_have_the_expected_shapes(self):
        pred, mu, logvar = self.model(self.images, self.qpos, self.actions, self.is_pad)
        self.assertEqual((tuple(pred.shape), tuple(mu.shape), tuple(logvar.shape)), ((2, 5, 29), (2, 32), (2, 32)))
        pred, mu, logvar = self.model(self.images, self.qpos)
        self.assertEqual(tuple(pred.shape), (2, 5, 29))
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
        normalization = {"qpos_mean": [0.0] * 29, "qpos_std": [1.0] * 29, "action_mean": [5.0] * 29, "action_std": [2.0] * 29}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.pt"
            save_checkpoint(path, self.model, normalization, {"fps": 40}, 3, metrics={"best": 1.0})
            model, data = load_policy(path)
        self.assertEqual((data["step"], data["dataset"]["fps"]), (3, {"fps": 40}["fps"]))
        predict = ChunkPredictor(model, normalization, "cpu")
        chunk = predict(np.zeros((4, 48, 64, 3), dtype=np.uint8), np.zeros(29, dtype=np.float32))
        self.assertEqual(chunk.shape, (5, 29))
        self.model.eval()
        self.torch.manual_seed(0)
        raw = self.model(self.torch.zeros(1, 4, 3, 48, 64), self.torch.zeros(1, 29))[0][0].detach().numpy()
        np.testing.assert_allclose(chunk, raw * 2 + 5, atol=1e-4)

    def test_a_few_steps_on_one_batch_reduce_the_loss(self):
        from teleop.utils.act_policy import act_loss
        self.model.train()
        optimizer = self.torch.optim.AdamW(self.model.parameter_groups(1e-3, 1e-4))
        losses = []
        for _ in range(25):
            pred, mu, logvar = self.model(self.images, self.qpos, self.actions, self.is_pad)
            loss = act_loss(pred, self.actions, self.is_pad, mu, logvar, 0.1)[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], losses[0] * 0.8)


if __name__ == "__main__":
    unittest.main()
