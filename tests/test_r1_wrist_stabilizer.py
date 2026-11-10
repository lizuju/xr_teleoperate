import math
import unittest

import numpy as np

from teleop.robot_control.r1_wrist_stabilizer import (
    WristStabilizer, rotation_from_vector, rotation_vector,
)

DT = 0.025


def pose(position=(0.0, 0.0, 0.0), turn=(0.0, 0.0, 0.0)):
    matrix = np.eye(4)
    matrix[:3, 3] = position
    matrix[:3, :3] = rotation_from_vector(np.asarray(turn, dtype=float))
    return matrix


def hand(curl=0.0):
    """25 wrist-frame points; ``curl`` slides every fingertip along x."""
    points = np.zeros((25, 3))
    for tip in (4, 9, 14, 19, 24):
        points[tip] = [0.10 - curl, 0.01 * tip, 0.0]
    return points


def angle_between(a, b):
    return float(np.linalg.norm(rotation_vector(a[:3, :3] @ b[:3, :3].T)))


class RotationHelpersTest(unittest.TestCase):
    def test_round_trip_including_half_turn(self):
        for vector in ([0.0, 0.0, 0.0], [0.02, -0.01, 0.03], [1.0, 0.5, -0.2], [math.pi, 0.0, 0.0],
                       [0.0, -math.pi / math.sqrt(2), math.pi / math.sqrt(2)]):
            matrix = rotation_from_vector(np.array(vector))
            np.testing.assert_allclose(rotation_from_vector(rotation_vector(matrix)), matrix, atol=1e-9)


class WristStabilizerTest(unittest.TestCase):
    def run_sequence(self, stabilizer, poses, curls, start=0.0):
        outputs = []
        for index, (target, curl) in enumerate(zip(poses, curls)):
            outputs.append(stabilizer.update(target, hand(curl), start + index * DT))
        return outputs

    def test_still_fingers_pass_the_pose_through_exactly(self):
        stabilizer = WristStabilizer()
        poses = [pose((0.001 * i, 0.0, 0.0), (0.0, 0.002 * i, 0.0)) for i in range(60)]
        for target, output in zip(poses, self.run_sequence(stabilizer, poses, [0.0] * 60)):
            np.testing.assert_array_equal(output, target)
        self.assertEqual(stabilizer.gate, 0.0)
        self.assertEqual(stabilizer.activations, 0)

    def test_tiny_motion_responds_immediately_and_converges_with_moving_fingers(self):
        stabilizer = WristStabilizer()
        target = pose((0.0002, 0.0, 0.0), (0.0, math.radians(0.2), 0.0))
        outputs = self.run_sequence(stabilizer, [pose()] * 20 + [target] * 20,
                                    [0.015 * math.sin(i * 0.9) for i in range(40)])
        self.assertGreater(outputs[20][0, 3], 0.00008)
        self.assertLess(outputs[20][0, 3], target[0, 3])
        self.assertGreater(angle_between(outputs[20], pose()), math.radians(0.08))
        self.assertLess(angle_between(outputs[20], pose()), math.radians(0.2))
        self.assertGreater(outputs[23][0, 3], 0.9 * target[0, 3])
        np.testing.assert_allclose(outputs[-1], target, atol=1e-7)
        self.assertEqual(stabilizer.gate, 1.0)

    def test_slow_fine_motion_is_preserved_while_fast_jitter_is_reduced(self):
        stabilizer = WristStabilizer()
        times = np.arange(480) * DT
        slow = np.sin(2 * math.pi * 0.5 * times)
        fast = np.sin(2 * math.pi * 8 * times)
        signal = slow + 0.25 * fast
        poses = [pose((0.001 * v, 0.0, 0.0), (0.0, math.radians(v), 0.0)) for v in signal]
        outputs = self.run_sequence(stabilizer, poses, [0.015 * math.sin(i * 0.9) for i in range(len(times))])
        # Fit both frequency components after settling; compare amplitude, not phase.
        t = times[80:]
        basis = np.column_stack([np.sin(2 * math.pi * f * t) for f in (0.5, 8)] +
                                [np.cos(2 * math.pi * f * t) for f in (0.5, 8)] + [np.ones(len(t))])
        for values in ([o[0, 3] / 0.001 for o in outputs[80:]],
                       [math.degrees(rotation_vector(o[:3, :3])[1]) for o in outputs[80:]]):
            coefficients = np.linalg.lstsq(basis, values, rcond=None)[0]
            self.assertGreater(math.hypot(coefficients[0], coefficients[2]), 0.97)
            self.assertLess(math.hypot(coefficients[1], coefficients[3]) / 0.25, 0.55)

    def test_output_never_strays_beyond_the_radii(self):
        stabilizer = WristStabilizer(position_m=0.008, rotation_rad=math.radians(4))
        poses = [pose((0.004 * i, 0.0, 0.0), (0.0, 0.0, math.radians(2) * i)) for i in range(80)]
        curls = [0.01 * math.sin(i * 0.9) for i in range(80)]
        for target, output in zip(poses, self.run_sequence(stabilizer, poses, curls)):
            self.assertLessEqual(np.linalg.norm(target[:3, 3] - output[:3, 3]), 0.008 + 1e-9)
            self.assertLessEqual(angle_between(target, output), math.radians(4) + 1e-9)

    def test_fast_motion_and_reversal_keep_up_within_the_offset_limit(self):
        stabilizer = WristStabilizer(position_m=0.008, rotation_rad=math.radians(4))
        poses = [pose((0.01 * i, 0.0, 0.0)) for i in list(range(20)) + list(range(20, 0, -1))]
        curls = [0.015 * math.sin(i * 0.9) for i in range(40)]
        outputs = self.run_sequence(stabilizer, poses, curls)
        lag = poses[-1][0, 3] - outputs[-1][0, 3]
        self.assertAlmostEqual(lag, -0.008, places=6)
        self.assertLess(outputs[21][0, 3], outputs[20][0, 3])
        self.assertLess(outputs[-1][0, 3] - outputs[-2][0, 3], -0.0099)

    def test_fingers_stopping_restores_exact_passthrough_promptly(self):
        stabilizer = WristStabilizer()
        shifted = pose((0.001, 0.0, 0.0), (0.0, math.radians(1), 0.0))
        poses = [pose()] * 20 + [shifted] * 20
        curls = [0.004 * i for i in range(20)] + [0.076] * 20
        outputs = self.run_sequence(stabilizer, poses, curls)
        self.assertGreater(outputs[20][0, 3], 0.0)
        values = [o[0, 3] for o in outputs[20:]]
        self.assertTrue(np.all(np.diff(values) >= -1e-12))
        self.assertLessEqual(max(values), shifted[0, 3])
        # Include the 100 ms detection window as well as hold/release time.
        np.testing.assert_array_equal(outputs[33], shifted)
        self.assertEqual(stabilizer.gate, 0.0)

    def test_zero_offset_limit_disables_only_that_component(self):
        target = pose((0.001, 0.0, 0.0), (0.0, math.radians(1), 0.0))
        curls = [0.015 * math.sin(i * 0.9) for i in range(21)]
        for kwargs, straight, smoothed in (({'position_m': 0.0}, (slice(0, 3), 3), (slice(0, 3), slice(0, 3))),
                                           ({'rotation_rad': 0.0}, (slice(0, 3), slice(0, 3)), (slice(0, 3), 3))):
            output = self.run_sequence(WristStabilizer(**kwargs), [pose()] * 20 + [target], curls)[-1]
            np.testing.assert_array_equal(output[straight], target[straight])
            self.assertFalse(np.array_equal(output[smoothed], target[smoothed]))

    def test_rotation_crosses_pi_on_shortest_arc_and_stays_orthogonal(self):
        stabilizer = WristStabilizer()
        start = pose(turn=(0.0, 0.0, math.radians(179)))
        end = pose(turn=(0.0, 0.0, math.radians(-179)))
        outputs = self.run_sequence(stabilizer, [start] * 20 + [end] * 20,
                                    [0.015 * math.sin(i * 0.9) for i in range(40)])
        self.assertLess(angle_between(outputs[20], start), math.radians(2))
        self.assertGreater(angle_between(outputs[20], start), 0.0)
        self.assertLess(angle_between(outputs[-1], end), 1e-6)
        for output in outputs:
            r = output[:3, :3]
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r), 1.0)

    def test_gap_or_reset_starts_from_the_tracked_pose(self):
        stabilizer = WristStabilizer()
        self.run_sequence(stabilizer, [pose()] * 5 + [pose((0.004, 0.0, 0.0))] * 5,
                          [0.0] * 5 + [0.01 * i for i in range(5)])
        far = pose((0.3, 0.1, 0.0))
        np.testing.assert_array_equal(stabilizer.update(far, hand(0.05), 10.0), far)
        self.assertEqual(stabilizer.gate, 0.0)
        stabilizer.reset()
        np.testing.assert_array_equal(stabilizer.update(pose(), hand(0.0), 10.025), pose())

    def test_missing_or_bad_hand_points_do_not_open_the_gate(self):
        stabilizer = WristStabilizer()
        for index in range(20):
            for points in (None, np.zeros((5, 3)), np.full((25, 3), np.nan)):
                target = pose((0.001 * index, 0.0, 0.0))
                np.testing.assert_array_equal(stabilizer.update(target, points, index * DT), target)
        self.assertEqual(stabilizer.gate, 0.0)

    def test_tracking_noise_does_not_open_the_gate(self):
        stabilizer = WristStabilizer()
        rng = np.random.default_rng(3)
        for index in range(200):
            points = hand(0.0) + rng.normal(0.0, 0.0004, (25, 3))
            stabilizer.update(pose(), points, index * DT)
        self.assertEqual(stabilizer.activations, 0)

    def test_invalid_settings_and_poses_are_rejected(self):
        for kwargs in ({"position_m": -1.0}, {"finger_speed": 0.0}, {"release_s": 0.0}, {"rotation_rad": float("nan")}):
            with self.assertRaises(ValueError):
                WristStabilizer(**kwargs)
        with self.assertRaises(ValueError):
            WristStabilizer().update(np.eye(3), hand(), 0.0)

    def test_summary_reports_smoothing_offset(self):
        stabilizer = WristStabilizer()
        self.run_sequence(stabilizer, [pose()] * 10 + [pose((0.004, 0.0, 0.0))] * 10,
                          [0.0] * 5 + [0.01 * i for i in range(15)])
        self.assertIn("max smoothing offset", stabilizer.summary())
        self.assertGreater(stabilizer.max_held_m, 0.0)
        self.assertLessEqual(stabilizer.max_held_m, stabilizer.position_m)
        self.assertIn("1 bursts", stabilizer.summary())


if __name__ == "__main__":
    unittest.main()
