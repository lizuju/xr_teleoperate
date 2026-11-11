"""Hand trust guard: occlusion by the other hand, pose jumps, merged hands, and the
Vision Pro source reporting such a hand as lost until it is clearly visible again."""
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils import visionpro_source
from teleop.utils.hand_trust_guard import HandTrustGuard, blocked_fraction, occluder_points, target_points

EYE = np.array([0.0, 1.6, 0.0])
DT = 1.0 / 90.0  # ARKit hand updates


def hand_offsets(spread=1.0):
    """25 WebXR joints of an open hand in its own frame, fingers along +y, palm in x-y."""
    points = [np.zeros(3)]
    for x, count in ((-0.045, 4), (-0.025, 5), (0.0, 5), (0.02, 5), (0.038, 5)):
        for j in range(count):
            points.append(np.array([x * spread, 0.03 + 0.022 * j, 0.0]))
    return np.array(points)


def hand(center, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    joints = np.tile(np.eye(4), (25, 1, 1))
    joints[:, :3, :3] = rotation
    joints[:, :3, 3] = np.asarray(center) + hand_offsets() @ rotation.T
    return joints


def yaw(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


RIGHT = np.array([0.08, 1.2, -0.55])
LEFT_APART = np.array([-0.25, 1.2, -0.5])
# On the sight line from the eye to the right hand, 20 cm in front of it.
LEFT_IN_FRONT = EYE + (RIGHT - EYE) * (1.0 - 0.2 / np.linalg.norm(RIGHT - EYE))


class GeometryTest(unittest.TestCase):
    def test_hand_in_front_blocks_and_hand_behind_does_not(self):
        right, front = hand(RIGHT), hand(LEFT_IN_FRONT)
        blocked = blocked_fraction(EYE, target_points(right[:, :3, 3]), occluder_points(front[:, :3, 3]), 0.03)
        self.assertGreaterEqual(blocked, 0.8)
        behind = blocked_fraction(EYE, target_points(front[:, :3, 3]), occluder_points(right[:, :3, 3]), 0.03)
        self.assertEqual(behind, 0.0)

    def test_hands_side_by_side_do_not_block(self):
        self.assertEqual(blocked_fraction(
            EYE, target_points(hand(RIGHT)[:, :3, 3]), occluder_points(hand(LEFT_APART)[:, :3, 3]), 0.03), 0.0)

    def test_one_fingertip_across_the_wrist_is_not_occlusion(self):
        right = hand(RIGHT)[:, :3, 3]
        tip = EYE + (right[0] - EYE) * 0.7
        others = np.tile(np.array([-0.3, 1.0, -0.4]), (25, 1))
        others[9] = tip
        fraction = blocked_fraction(EYE, target_points(right), occluder_points(others), 0.03)
        self.assertLess(fraction, 0.5)


class GuardTest(unittest.TestCase):
    def run_side(self, guard, side, poses, start=0.0, other=None, step=DT):
        results = []
        for index, joints in enumerate(poses):
            other_positions = None if other is None else other(index)[:, :3, 3]
            results.append(guard.update(side, start + index * step, joints, EYE, other_positions))
        return results

    def test_occluded_right_hand_is_held_until_clearly_visible_for_the_dwell(self):
        guard = HandTrustGuard()
        self.assertTrue(all(self.run_side(guard, "right", [hand(RIGHT)] * 5, other=lambda i: hand(LEFT_APART))))
        blocked = self.run_side(guard, "right", [hand(RIGHT)] * 20, start=1.0, other=lambda i: hand(LEFT_IN_FRONT))
        self.assertFalse(any(blocked))
        self.assertEqual(guard.reason("right"), "occluded")
        self.assertEqual(guard.diagnostics()["right"]["events"]["occluded"], 1)
        # The left hand moves away: still held for the dwell, then trusted again.
        clear = self.run_side(guard, "right", [hand(RIGHT)] * 40, start=2.0, other=lambda i: hand(LEFT_APART))
        first = clear.index(True)
        self.assertFalse(any(clear[:first]))
        self.assertTrue(all(clear[first:]))
        self.assertAlmostEqual(first * DT, guard.dwell, delta=DT * 1.01)
        self.assertIsNone(guard.reason("right"))

    def test_the_hand_in_front_keeps_following(self):
        guard = HandTrustGuard()
        results = self.run_side(guard, "left", [hand(LEFT_IN_FRONT)] * 20, other=lambda i: hand(RIGHT))
        self.assertTrue(all(results))

    def test_fast_real_motion_passes_and_a_snap_is_rejected(self):
        guard = HandTrustGuard()
        # 2 m/s sweep with the wrist turning at 400 deg/s: a quick but real move.
        poses = [hand(RIGHT + [2.0 * i * DT, 0.0, 0.0], yaw(math.radians(400.0) * i * DT)) for i in range(45)]
        self.assertTrue(all(self.run_side(guard, "right", poses)))
        snapped = self.run_side(guard, "right", [hand(RIGHT + [0.12, 0.0, 0.0])], start=45 * DT)
        self.assertEqual(snapped, [False])
        self.assertEqual(guard.reason("right"), "jump")

    def test_palm_flip_in_one_sample_is_a_jump(self):
        guard = HandTrustGuard()
        self.run_side(guard, "right", [hand(RIGHT)] * 3)
        self.assertEqual(self.run_side(guard, "right", [hand(RIGHT, yaw(math.pi))], start=3 * DT), [False])

    def test_samples_after_a_tracking_gap_are_not_called_jumps(self):
        guard = HandTrustGuard()
        self.run_side(guard, "right", [hand(RIGHT)] * 3)
        self.assertEqual(self.run_side(guard, "right", [hand(RIGHT + [0.3, 0.0, 0.0])], start=1.0), [True])

    def test_merged_wrists_are_not_trusted(self):
        guard = HandTrustGuard()
        results = self.run_side(guard, "right", [hand(RIGHT)] * 3, other=lambda i: hand(RIGHT + [0.03, 0.0, 0.0]))
        self.assertFalse(any(results))
        self.assertEqual(guard.reason("right"), "merged")

    def test_partial_overlap_is_not_held_and_release_is_quick(self):
        # Defaults since 2026-11-11: held at 4 of 6 points within 2 cm, released at
        # 2 of 6 within 3 cm after 0.15 s.
        guard = HandTrustGuard()
        self.assertEqual((guard.block_radius, guard.clear_radius, guard.dwell), (0.02, 0.03, 0.15))
        fractions = [3 / 6, 4 / 6, 3 / 6] + [2 / 6] * 20
        radii = []

        def fake_blocked_fraction(eye, targets, occluders, radius):
            radii.append(radius)
            return fractions[len(radii) - 1]

        with mock.patch("teleop.utils.hand_trust_guard.blocked_fraction", side_effect=fake_blocked_fraction):
            results = [guard.update("right", i * DT, hand(RIGHT), EYE, hand(LEFT_APART)[:, :3, 3])
                       for i in range(len(fractions))]
        self.assertTrue(results[0])   # 3 of 6 blocked: still follows
        self.assertFalse(results[1])  # 4 of 6: held
        self.assertFalse(results[2])  # 3 of 6 while held: stays held
        self.assertEqual(radii[:3], [0.02, 0.02, 0.03])
        first = results.index(True, 3)
        self.assertFalse(any(results[3:first]))
        self.assertAlmostEqual((first - 3) * DT, 0.15, delta=DT * 1.01)
        self.assertEqual(guard.diagnostics()["right"]["events"]["occluded"], 1)

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            HandTrustGuard(dwell=0.0)
        with self.assertRaises(ValueError):
            HandTrustGuard(block_radius=0.05, clear_radius=0.03)


def packet(sample, left, right, received_offset=50.0):
    """A protocol-1 packet whose wrists are identity rotations and joints carry the offsets."""
    result = {"tracking_protocol_version": 1, "received_monotonic": sample + received_offset,
              "sample_time": sample, "prediction_seconds": 0.0}
    head = np.eye(4)
    head[:3, 3] = EYE
    result["head"] = head.tolist()
    for key in ("head", "left", "right"):
        result[f"{key}_time"] = sample
        result[f"{key}_valid"] = True
    for side, center in (("left", left), ("right", right)):
        wrist = np.eye(4)
        wrist[:3, 3] = center
        joints = np.tile(np.eye(4), (25, 1, 1))
        joints[:, :3, 3] = hand_offsets()
        result[f"{side}_wrist"] = wrist.tolist()
        result[f"{side}_joints"] = joints.tolist()
    return result


class SourceGuardTest(unittest.TestCase):
    def setUp(self):
        self.now = 150.0
        mock.patch.object(visionpro_source.time, "monotonic", side_effect=lambda: self.now).start()
        self.addCleanup(mock.patch.stopall)

    def feed(self, source, start, count, left):
        snapshots = []
        for index in range(count):
            sample = start + index * DT
            self.now = sample + 50.0
            source.accept_packet(packet(sample, left, RIGHT))
            snapshots.append(source.get_hand_motion_snapshot())
        return snapshots

    def test_occluded_right_hand_reads_as_lost_and_keeps_its_last_pose(self):
        source = visionpro_source.VisionProMotionSource(host=None, python=None, timeout=0.25)
        before = self.feed(source, 100.0, 5, LEFT_APART)[-1]
        self.assertGreater(before["right_hand_timestamp"], 0.0)
        # Gaps longer than the jump check's window: the test teleports the left hand,
        # which a real hand cannot do and the guard would rightly call a jump.
        during = self.feed(source, 100.3, 20, LEFT_IN_FRONT)
        for snapshot in during:
            self.assertEqual(snapshot["right_hand_timestamp"], 0.0)
            self.assertGreater(snapshot["left_hand_timestamp"], 0.0)
            np.testing.assert_allclose(snapshot["right_arm_pose"], before["right_arm_pose"])
        self.assertEqual(source.get_hand_guard_reason("right"), "hidden from the headset by the other hand")
        self.assertEqual(source.get_hand_guard_reason("left"), None)
        self.assertFalse(source.get_tracking_diagnostics()["hand_guard"]["right"]["trusted"])
        after = self.feed(source, 100.8, 40, LEFT_APART)
        recovered = [snapshot["right_hand_timestamp"] > 0.0 for snapshot in after]
        first = recovered.index(True)
        self.assertFalse(any(recovered[:first]))
        self.assertTrue(all(recovered[first:]))
        self.assertGreaterEqual(first * DT, 0.14)  # the 0.15 s dwell

    def test_guard_off_passes_the_occluded_hand(self):
        source = visionpro_source.VisionProMotionSource(host=None, python=None, timeout=0.25, hand_guard=False)
        self.feed(source, 100.0, 5, LEFT_APART)
        during = self.feed(source, 100.3, 10, LEFT_IN_FRONT)
        self.assertTrue(all(snapshot["right_hand_timestamp"] > 0.0 for snapshot in during))
        self.assertNotIn("hand_guard", source.get_tracking_diagnostics())
        self.assertIsNone(source.get_hand_guard_reason("right"))


if __name__ == "__main__":
    unittest.main()
