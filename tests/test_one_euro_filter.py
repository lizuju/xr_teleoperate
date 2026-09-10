import unittest

import numpy as np

from teleop.utils.one_euro_filter import OneEuroFilter


class OneEuroFilterTest(unittest.TestCase):
    def test_first_sample_uses_measured_pose_and_does_not_alias_it(self):
        measured = np.array([0.2, -0.3])
        smoothing = OneEuroFilter()
        output = smoothing.filter([1.0, -1.0], 0.0, measured)
        np.testing.assert_array_equal(output, measured)
        output[:] = 9.0
        measured[:] = 8.0
        np.testing.assert_array_equal(smoothing.filter([1.0, -1.0], 0.0), [0.2, -0.3])

    def test_repeated_values_converge_to_the_stopped_target(self):
        smoothing = OneEuroFilter()
        smoothing.filter([0.0], 0.0)
        outputs = [smoothing.filter([0.1], i / 30.0)[0] for i in range(1, 91)]
        self.assertLess(outputs[0], 0.1)
        self.assertTrue(np.all(np.diff(outputs) >= 0.0))
        self.assertAlmostEqual(outputs[-1], 0.1, places=10)

    def test_actual_time_interval_sets_the_low_pass_response(self):
        for dt in (1 / 60, 1 / 30, 1 / 15):
            with self.subTest(dt=dt):
                smoothing = OneEuroFilter(min_cutoff=2.0, beta=0.0)
                smoothing.filter([0.0], 0.0)
                output = smoothing.filter([1.0], dt)[0]
                expected = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * 2.0 * dt))
                self.assertAlmostEqual(output, expected)

    def test_duplicate_and_out_of_order_timestamps_do_not_advance_state(self):
        smoothing = OneEuroFilter()
        reference = OneEuroFilter()
        for item in (smoothing, reference):
            item.filter([0.0], 1.0)
        np.testing.assert_array_equal(smoothing.filter([3.0], 1.0), [0.0])
        np.testing.assert_array_equal(smoothing.filter([3.0], 0.5), [0.0])
        np.testing.assert_allclose(smoothing.filter([0.1], 1.03), reference.filter([0.1], 1.03))

    def test_long_gap_and_explicit_reset_start_from_measured_pose(self):
        smoothing = OneEuroFilter()
        smoothing.filter([0.0], 0.0)
        smoothing.filter([1.0], 0.03)
        np.testing.assert_array_equal(smoothing.filter([2.0], 1.0, [0.2]), [0.2])
        smoothing.reset()
        np.testing.assert_array_equal(smoothing.filter([2.0], 1.03, [0.3]), [0.3])

    def test_fast_joint_does_not_weaken_filtering_of_other_joints(self):
        joint_filter = OneEuroFilter()
        combined_filter = OneEuroFilter()
        for i in range(150):
            t = i / 30.0
            stationary = 0.005 * np.sin(2.0 * np.pi * 8.0 * t)
            expected = joint_filter.filter([stationary], t)[0]
            observed = combined_filter.filter([stationary, 1.2 * t], t)[0]
            self.assertAlmostEqual(observed, expected)

    def test_static_noise_is_lower_than_the_existing_four_frame_filter(self):
        raw = np.random.default_rng(7).normal(0.0, 0.005, 1800)
        smoothing = OneEuroFilter()
        filtered = np.array([smoothing.filter([q], i / 30.0)[0] for i, q in enumerate(raw)])
        baseline = np.convolve(raw, [0.4, 0.3, 0.2, 0.1], mode="full")[:len(raw)]
        self.assertLess(np.std(filtered[60:]), np.std(baseline[60:]))

    def test_fast_ramp_lag_is_lower_than_the_existing_four_frame_filter(self):
        times = np.arange(120) / 30.0
        raw = 1.2 * times
        smoothing = OneEuroFilter()
        filtered = np.array([smoothing.filter([q], t)[0] for q, t in zip(raw, times)])
        baseline = np.convolve(raw, [0.4, 0.3, 0.2, 0.1], mode="full")[:len(raw)]
        self.assertLess(np.mean(raw[60:] - filtered[60:]), np.mean(raw[60:] - baseline[60:]))

    def test_slow_ramp_lag_is_lower_than_the_existing_four_frame_filter(self):
        times = np.arange(120) / 30.0
        raw = 0.15 * times
        smoothing = OneEuroFilter()
        filtered = np.array([smoothing.filter([q], t)[0] for q, t in zip(raw, times)])
        baseline = np.convolve(raw, [0.4, 0.3, 0.2, 0.1], mode="full")[:len(raw)]
        self.assertLess(np.mean(raw[60:] - filtered[60:]), np.mean(raw[60:] - baseline[60:]))

    def test_reversal_remains_within_previous_output_and_current_target(self):
        smoothing = OneEuroFilter()
        previous = smoothing.filter([0.0], 0.0)[0]
        for i, target in enumerate(np.r_[np.ones(30), -np.ones(30), np.zeros(90)], 1):
            output = smoothing.filter([target], i / 30.0)[0]
            self.assertGreaterEqual(output, min(previous, target))
            self.assertLessEqual(output, max(previous, target))
            previous = output
        self.assertAlmostEqual(previous, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
