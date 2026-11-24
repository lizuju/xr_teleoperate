import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from teleop.utils.act_contract import GROUPS, GROUP_SLICES, UNITS
from teleop.utils.act_dataset import add_qpos_noise, qpos_noise_scale

STD = np.linspace(0.2, 1.4, 24).astype(np.float32)
JOINTS = np.r_[0:14, 21:24]  # both arms, waist, head pitch, head yaw
EXACT = np.r_[14:21]         # left O6 fingers and the right Dex1


class ScaleTests(unittest.TestCase):
    def test_joint_inputs_get_sigma_in_normalized_units(self):
        scale = qpos_noise_scale(GROUPS, UNITS, STD, 0.05)
        self.assertEqual(scale.shape, (24,))
        np.testing.assert_allclose(scale[JOINTS], 0.05 / STD[JOINTS], rtol=1e-6)

    def test_hand_and_gripper_inputs_stay_exact(self):
        scale = qpos_noise_scale(GROUPS, UNITS, STD, 0.05)
        self.assertTrue(np.all(scale[EXACT] == 0))
        self.assertEqual(list(EXACT), list(range(GROUP_SLICES["left_ee"].start, GROUP_SLICES["right_ee"].stop)))

    def test_bad_sigma_or_mismatched_contract_is_refused(self):
        for sigma in (0.0, -0.05, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                qpos_noise_scale(GROUPS, UNITS, STD, sigma)
        with self.assertRaises(ValueError):
            qpos_noise_scale(GROUPS, UNITS, STD[:23], 0.05)
        with self.assertRaises(ValueError):
            qpos_noise_scale(GROUPS, ["deg"] * 14 + UNITS[14:], STD, 0.05)
        with self.assertRaises(ValueError):
            qpos_noise_scale({}, UNITS, STD, 0.05)


class NoiseTests(unittest.TestCase):
    def test_noise_has_the_requested_spread_per_input(self):
        torch.manual_seed(0)
        scale = torch.as_tensor(qpos_noise_scale(GROUPS, UNITS, STD, 0.05))
        qpos = torch.full((20000, 24), 0.3)
        noisy = add_qpos_noise(qpos, scale)
        delta = (noisy - qpos).numpy()
        np.testing.assert_allclose(delta[:, JOINTS].std(axis=0) * STD[JOINTS], 0.05, rtol=0.05)
        self.assertLess(np.abs(delta[:, JOINTS].mean(axis=0) * STD[JOINTS]).max(), 0.003)  # rad; 8 sigma
        self.assertTrue(np.all(delta[:, EXACT] == 0))

    def test_input_is_not_changed_in_place_and_dtype_is_kept(self):
        scale = torch.as_tensor(qpos_noise_scale(GROUPS, UNITS, STD, 0.05))
        for dtype in (torch.float32, torch.float64):
            qpos = torch.zeros(4, 24, dtype=dtype)
            noisy = add_qpos_noise(qpos, scale)
            self.assertTrue(torch.all(qpos == 0))
            self.assertEqual(noisy.dtype, dtype)


@unittest.skipUnless(importlib.util.find_spec("torchvision"), "training needs torchvision (.venv-train)")
class TrainingArgumentTests(unittest.TestCase):
    def test_flag_is_optional_and_joins_resume_settings_only_when_used(self):
        import train_r1_act
        off = train_r1_act.parse_args(["data", "--output", "run"])
        on = train_r1_act.parse_args(["data", "--output", "run", "--qpos-noise", "0.05"])
        self.assertIsNone(off.qpos_noise)
        self.assertNotIn("qpos_noise", train_r1_act.training_settings(off))
        self.assertEqual(train_r1_act.training_settings(on)["qpos_noise"], 0.05)
        for bad in ("0", "-0.05", "nan"):
            with self.assertRaises(SystemExit):
                train_r1_act.parse_args(["data", "--output", "run", "--qpos-noise", bad])


if __name__ == "__main__":
    unittest.main()
