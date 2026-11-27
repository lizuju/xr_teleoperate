"""Run 12 training options (2026-11-26): EMA weights, warmup + cosine learning rate, image shift, snapshots."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(HAS_TORCH, "needs torch and torchvision")
class TrainingTrickOptionTests(unittest.TestCase):
    def parse(self, *extra):
        import train_r1_act
        return train_r1_act.parse_args(["data", "--output", "run", "--head", "flow", *extra])

    def test_everything_is_off_by_default_so_earlier_runs_still_resume(self):
        import train_r1_act
        args = self.parse()
        self.assertEqual((args.ema_decay, args.lr_schedule, args.warmup_steps, args.image_shift, args.snapshot_every),
                         (None, "constant", 0, 0, 0))
        self.assertEqual(set(train_r1_act.training_settings(args)),
                         {"batch_size", "accumulate", "chunk_size", "conditioned", "seed", "no_augment", "kl_weight", "head"})

    def test_used_options_join_the_resume_settings_but_snapshots_do_not(self):
        import train_r1_act
        args = self.parse("--steps", "40000", "--ema-decay", "0.9995", "--lr-schedule", "cosine", "--warmup-steps",
                          "1000", "--image-shift", "12", "--snapshot-every", "5000")
        settings = train_r1_act.training_settings(args)
        self.assertEqual(settings["ema_decay"], 0.9995)
        self.assertEqual(settings["lr_schedule"], {"schedule": "cosine", "warmup_steps": 1000, "min_lr_ratio": 0.1,
                                                   "steps": 40000})
        self.assertEqual(settings["image_shift"], 12)
        self.assertNotIn("snapshot_every", settings)

    def test_bad_values_are_refused(self):
        for bad in (["--ema-decay", "1"], ["--ema-decay", "0"], ["--warmup-steps", "50000"], ["--warmup-steps", "-1"],
                    ["--min-lr-ratio", "1.5"], ["--image-shift", "-1"], ["--snapshot-every", "-5"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                self.parse(*bad)


@unittest.skipUnless(HAS_TORCH, "needs torch and torchvision")
class ScheduleTests(unittest.TestCase):
    def test_constant_is_one_and_warmup_ramps_linearly(self):
        from train_r1_act import lr_factor
        self.assertEqual(lr_factor(0, 100), 1.0)
        self.assertEqual([lr_factor(s, 100, "constant", 10) for s in (0, 4, 9, 10, 99)], [0.1, 0.5, 1.0, 1.0, 1.0])

    def test_cosine_falls_from_one_to_the_floor_at_the_last_step(self):
        from train_r1_act import lr_factor
        values = [lr_factor(s, 110, "cosine", 10, 0.1) for s in range(10, 111)]
        self.assertAlmostEqual(values[0], 1.0)
        self.assertAlmostEqual(lr_factor(60, 110, "cosine", 10, 0.1), 0.55)
        self.assertAlmostEqual(values[-1], 0.1)
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(lr_factor(500, 110, "cosine", 10, 0.1), 0.1)  # past the end it stays at the floor


@unittest.skipUnless(HAS_TORCH, "needs torch and torchvision")
class EmaTests(unittest.TestCase):
    def test_the_average_moves_toward_the_model_with_a_ramped_decay(self):
        import torch
        from train_r1_act import ema_update
        model = torch.nn.Linear(2, 2)
        ema = copy.deepcopy(model)
        with torch.no_grad():
            model.weight.fill_(1.0)
            ema.weight.fill_(0.0)
        d = ema_update(ema, model, 0.999, 1)
        self.assertAlmostEqual(d, 2 / 11)
        torch.testing.assert_close(ema.weight, torch.full((2, 2), 1 - 2 / 11))
        self.assertEqual(ema_update(ema, model, 0.999, 100000), 0.999)

    def test_buffers_are_copied_not_averaged(self):
        import torch
        from train_r1_act import ema_update
        model = torch.nn.BatchNorm1d(3)
        ema = copy.deepcopy(model)
        model.running_mean.fill_(5.0)
        ema_update(ema, model, 0.9, 1)
        torch.testing.assert_close(ema.running_mean, torch.full((3,), 5.0))



class PhaseLabelTests(unittest.TestCase):
    """Run 13: corrections that start mid-task have empty leading phases."""

    def test_skipped_phases_are_allowed_but_order_and_ends_are_not_relaxed(self):
        import tempfile
        from teleop.utils.icecream_task import source_digest, validate_annotation
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "frames.jsonl").write_text("{}\n")
            label = {"reviewed": True, "frames_sha256": source_digest(root / "frames.jsonl")}
            for ok in ([0, 2, 4, 6, 8, 10], [0, 0, 0, 3, 3, 10], [0, 0, 0, 0, 0, 10]):
                self.assertEqual(validate_annotation(dict(label, boundaries=ok), root, 10), ok)
            for bad in ([0, 4, 2, 6, 8, 10], [1, 2, 4, 6, 8, 10], [0, 2, 4, 6, 8, 9], [0, 2, 4, 6, 10]):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    validate_annotation(dict(label, boundaries=bad), root, 10)

if __name__ == "__main__":
    unittest.main()
