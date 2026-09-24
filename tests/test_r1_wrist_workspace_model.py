import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np

from teleop.robot_control.robot_arm_ik import R1_A7_ArmIK
from teleop.robot_control.r1_wrist_workspace import R1WristWorkspace
from teleop.robot_control.arm_target_shaper import ArmTargetShaper


class WristWorkspaceModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = Path.cwd()
        try:
            os.chdir(Path(__file__).resolve().parents[1] / 'teleop')
            cls.ik = R1_A7_ArmIK(waist_yaw=0.0)
        finally:
            os.chdir(previous)
        # Recorded R1 arm poses: near extension and the episode's initial posture.
        cls.extended_q = np.array([
            -1.62795, 0.43125, -1.20677, 1.37448, 1.23206, 0.51801, 0.05804,
            -1.51076, -0.31575, 1.23144, 1.36653, -1.24990, 0.45541, -0.05949,
        ])
        cls.nominal_q = np.array([
            -0.98767, 0.19855, -0.38387, 0.03032, -0.95331, -0.65694, 0.02840,
            -0.72487, -0.26713, 0.22808, -0.15822, 1.15874, -0.42224, 0.29091,
        ])

    def setUp(self):
        self.ik.set_redundancy_weights(
            posture_weight=0.01, limit_weight=0.1, nominal_arm_q=self.nominal_q,
        )
        self.ik.reset_smoothing()
        self.now = 0.0
        self.durations = []

    def step(self, targets, q, alpha=0.3, shaper=None):
        self.now += 1 / 40
        started = time.perf_counter()
        with patch('teleop.robot_control.robot_arm_ik.time.monotonic', return_value=self.now):
            command, _ = self.ik.solve_ik(*targets, q, np.zeros(14), raise_on_failure=True)
        self.durations.append(time.perf_counter() - started)
        raw_poses = self.ik.forward_wrist_poses(self.ik.last_raw_q)
        self.assertTrue(np.isfinite(command).all())
        lower = self.ik.reduced_robot.model.lowerPositionLimit
        upper = self.ik.reduced_robot.model.upperPositionLimit
        self.assertTrue(np.all(command >= lower - 1e-6))
        self.assertTrue(np.all(command <= upper + 1e-6))
        if shaper is not None:
            command = shaper.shape(command, self.now)
        return q + np.clip(alpha * (command - q), -6 / 40, 6 / 40), raw_poses

    def test_overreach_reversal_onset_with_filter_shaper_and_servo_lag(self):
        refs = self.ik.forward_wrist_poses(self.extended_q)
        for speed, onset_limit in ((0.05, [0.3, 0.225]), (0.10, [0.225, 0.175]),
                                   (0.20, [0.15, 0.125])):
            with self.subTest(speed=speed):
                self.ik.reset_smoothing()
                q = self.extended_q.copy()
                workspaces = [R1WristWorkspace(p) for p in refs]
                shaper = ArmTargetShaper()
                shaper.reset(q)
                onset = [None, None]
                previous_targets = None
                for phase, offsets in (
                    ('extend', np.linspace(0, 0.2, 81)),
                    ('hold', np.full(80, 0.2)),
                    ('return', np.arange(0.2 - speed / 40, 0.1 - 1e-9, -speed / 40)),
                ):
                    for index, delta in enumerate(offsets):
                        raw = [p.copy() for p in refs]
                        for p in raw:
                            p[0, 3] += delta
                        targets = [w.target(p) for w, p in zip(workspaces, raw)]
                        if phase == 'return':
                            for target, previous in zip(targets, previous_targets):
                                self.assertLessEqual(np.linalg.norm(target[:3, 3] - previous[:3, 3]),
                                                     2 * speed / 40 + 1e-9)
                        q, solved = self.step(targets, q, shaper=shaper)
                        for w, p, actual in zip(workspaces, raw, solved):
                            w.observe(p, actual)
                        actual = self.ik.forward_wrist_poses(q)
                        if phase == 'hold':
                            held = [p[0, 3] for p in actual]
                        if phase == 'return':
                            for side, p in enumerate(actual):
                                if onset[side] is None and held[side] - p[0, 3] > 0.002:
                                    onset[side] = (index + 1) / 40
                        previous_targets = targets
                for observed, limit in zip(onset, onset_limit):
                    self.assertIsNotNone(observed)
                    self.assertLessEqual(observed, limit)
                print('\nReverse speed (cm/s), onset >2mm (ms):', speed * 100,
                      [round(value * 1000) for value in onset])

    def test_overreach_then_retract_with_real_ik_filter_and_feedback_lag(self):
        refs = self.ik.forward_wrist_poses(self.extended_q)
        results = {}
        for corrected in (False, True):
            self.ik.reset_smoothing()
            q = self.extended_q.copy()
            workspaces = [R1WristWorkspace(p) for p in refs]
            for phase, offsets in (
                ('extend', np.linspace(0, 0.2, 81)),
                ('hold', np.full(40, 0.2)),
                ('return', np.linspace(0.2, 0.1, 41)),
            ):
                offset_before_hold = np.array([w.offset.copy() for w in workspaces])
                for delta in offsets:
                    raw = [p.copy() for p in refs]
                    for p in raw:
                        p[0, 3] += delta
                    targets = [w.target(p) if corrected else p for w, p in zip(workspaces, raw)]
                    q, solved = self.step(targets, q)
                    if corrected:
                        for w, p, actual in zip(workspaces, raw, solved):
                            w.observe(p, actual)
                if phase == 'hold':
                    np.testing.assert_array_equal([w.offset for w in workspaces], offset_before_hold)
                    held_positions = np.array([p[:3, 3] for p in self.ik.forward_wrist_poses(q)])
                    held_elbows = q[[3, 10]].copy()
            returned = np.array([p[:3, 3] for p in self.ik.forward_wrist_poses(q)])
            results[corrected] = held_positions[:, 0] - returned[:, 0]
            if corrected:
                self.assertTrue(np.all(held_elbows - q[[3, 10]] > np.deg2rad(30)))
        self.assertTrue(np.all(results[False] < 0.01), results)
        self.assertTrue(np.all(results[True] > 0.05), results)
        print('\nR1 model hand retraction after 10 cm reverse input (cm):',
              {str(k): (v * 100).round(3).tolist() for k, v in results.items()})
        print('IK/filter step p95 (ms):', round(np.percentile(self.durations, 95) * 1000, 3))

    def test_known_reachable_motion_with_rotation_and_feedback_lag_keeps_anchor(self):
        q0 = self.extended_q.copy()
        targets = list(self.ik.forward_wrist_poses(q0))
        for target in targets:
            target[0, 3] -= 0.1
        for _ in range(100):
            q0, _ = self.step(targets, q0)
        q1 = q0.copy()
        q1[[3, 10]] -= 0.6
        for duration, alpha in ((1.0, 1.0), (1.0, 0.3), (3.0, 0.3)):
            with self.subTest(duration=duration, alpha=alpha):
                self.ik.reset_smoothing()
                q = q0.copy()
                workspaces = [R1WristWorkspace(p) for p in self.ik.forward_wrist_poses(q)]
                for fraction in np.r_[np.linspace(0, 1, int(40 * duration)),
                                      np.linspace(1, 0, int(40 * duration))]:
                    raw = self.ik.forward_wrist_poses(q0 + fraction * (q1 - q0))
                    q, solved = self.step([w.target(p) for w, p in zip(workspaces, raw)], q, alpha)
                    for w, p, actual in zip(workspaces, raw, solved):
                        w.observe(p, actual)
                np.testing.assert_array_equal([w.offset for w in workspaces], np.zeros((2, 3)))


if __name__ == '__main__':
    unittest.main()
