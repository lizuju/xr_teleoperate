from pathlib import Path
import sys
import unittest

from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop.utils import hand_torque_hud
from teleop.utils.hand_torque_hud import (
    format_torque_line,
    overlay_stereo_hand_torque,
    publish_torque_hud,
    read_hand_torques,
    render_head_to_xr,
    torque_strip_image,
)


class FakeTv:
    def __init__(self):
        self.frames = []
        self.hud = []

    def render_to_xr(self, img, sequence=None):
        self.frames.append(img)

    def render_torque_hud_to_xr(self, side, image):
        self.hud.append((side, image.copy()))


class FakeHand:
    def __init__(self, snapshot=None, error=None):
        self.snapshot = snapshot
        self.error = error

    def get_recording_snapshot(self):
        if self.error is not None:
            raise self.error
        return self.snapshot


class FormatTorqueLineTests(unittest.TestCase):
    def test_single_digits_zero_to_nine(self):
        self.assertEqual(
            format_torque_line([0.0, 0.02, 0.05, 0.08, 0.12, 0.15]),
            "0 1 3 5 7 9",
        )

    def test_light_grasp_is_not_zero(self):
        # Recorded cup/icecream contact is ~0.05, which used to round to 0
        # when the HUD treated motor full-scale 1.0 as digit 9.
        self.assertEqual(format_torque_line([0.05, 0.03, 0.0, 0.0, 0.0, 0.0]), "3 2 0 0 0 0")

    def test_moving_joint_reads_zero(self):
        self.assertEqual(
            format_torque_line([0.12, 0.08, 0.05], [0.20, 0.0, 0.04]),
            "0 5 3",
        )

    def test_missing_is_dashes(self):
        self.assertEqual(format_torque_line(None), "- - - - - -")
        self.assertEqual(format_torque_line([]), "- - - - - -")

    def test_non_finite_becomes_dash(self):
        self.assertEqual(format_torque_line([0.05, float("nan")]), "3 -")


class ReadHandTorquesTests(unittest.TestCase):
    def test_reads_left_and_right(self):
        hand = FakeHand({
            "state": {
                "left": {"torque": [0.1, 0.2, 0.0, 0.0, 0.0, 0.0]},
                "right": {"torque": [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]},
            }
        })
        left, right = read_hand_torques(hand)
        self.assertEqual(left[0], 0.1)
        self.assertEqual(right[0], 0.3)

    def test_empty_torque_is_absent(self):
        hand = FakeHand({"state": {"left": {"q": [0.1], "torque": []}, "right": {"torque": None}}})
        self.assertEqual(read_hand_torques(hand), (None, None))

    def test_not_ready_is_absent(self):
        self.assertEqual(read_hand_torques(FakeHand(error=RuntimeError("not ready"))), (None, None))

    def test_missing_controller_is_absent(self):
        self.assertEqual(read_hand_torques(None), (None, None))
        self.assertEqual(read_hand_torques(object()), (None, None))


class OverlayTests(unittest.TestCase):
    def test_copy_leaves_source_clean(self):
        source = np.full((80, 160, 3), 40, dtype=np.uint8)
        painted = overlay_stereo_hand_torque(source, [0.9] * 6, [0.1] * 6)
        self.assertIsNot(painted, source)
        self.assertTrue(np.array_equal(source, np.full((80, 160, 3), 40, dtype=np.uint8)))
        self.assertFalse(np.array_equal(painted, source))

    def test_left_eye_differs_from_right_eye(self):
        source = np.full((96, 192, 3), 18, dtype=np.uint8)
        painted = overlay_stereo_hand_torque(source, [0.87] * 6, [0.13] * 6)
        mid = 96
        left_top = painted[:40, :mid]
        right_top = painted[:40, mid:]
        left_bottom = painted[56:, :mid]
        right_bottom = painted[56:, mid:]
        self.assertFalse(np.array_equal(left_top, right_top))
        self.assertGreater(int(np.count_nonzero(left_top != 18)), 0)
        self.assertGreater(int(np.count_nonzero(right_top != 18)), 0)
        self.assertEqual(int(np.count_nonzero(left_bottom != 18)), 0)
        self.assertEqual(int(np.count_nonzero(right_bottom != 18)), 0)

    def test_left_numbers_stay_in_left_half(self):
        source = np.full((96, 192, 3), 18, dtype=np.uint8)
        painted = overlay_stereo_hand_torque(source, [0.99] * 6, None)
        changed = np.any(painted != 18, axis=2)
        left_changed = int(np.count_nonzero(changed[:, :96]))
        right_changed = int(np.count_nonzero(changed[:, 96:]))
        self.assertGreater(left_changed, 0)
        self.assertGreater(right_changed, 0)
        self.assertFalse(np.any(changed[56:, :]))


class RenderToXrTests(unittest.TestCase):
    def test_head_path_does_not_burn_numbers_into_the_frame(self):
        tv = FakeTv()
        frame = np.full((40, 80, 3), 12, dtype=np.uint8)
        sent = render_head_to_xr(tv, frame, enabled=True)
        self.assertIs(sent, frame)
        self.assertIs(tv.frames[0], frame)


class PublishTorqueHudTests(unittest.TestCase):
    def test_publishes_left_and_right_strips(self):
        tv = FakeTv()
        hand = FakeHand({
            "state": {
                "left": {"torque": [0.4] * 6},
                "right": {"torque": [0.8] * 6},
            }
        })
        cache = publish_torque_hud(tv, hand, enabled=True)
        self.assertEqual([side for side, _ in tv.hud], ["left", "right"])
        self.assertEqual(cache["left"], "9 9 9 9 9 9")
        self.assertEqual(cache["right"], "9 9 9 9 9 9")
        self.assertEqual(tv.hud[0][1].shape, (48, 280, 3))

    def test_moving_fingers_publish_zeros(self):
        tv = FakeTv()
        hand = FakeHand({
            "state": {
                "left": {"torque": [0.12] * 6, "qvel": [0.2] * 6},
                "right": {"torque": [0.08] * 6, "qvel": [0.0] * 6},
            }
        })
        cache = publish_torque_hud(tv, hand, enabled=True)
        self.assertEqual(cache["left"], "0 0 0 0 0 0")
        self.assertEqual(cache["right"], "5 5 5 5 5 5")

    def test_terminal_prints_the_same_digits_and_does_not_flood(self):
        tv = FakeTv()
        first = FakeHand({"state": {"left": {"torque": [0.05] * 6}, "right": {"torque": [0.0] * 6}}})
        second = FakeHand({"state": {"left": {"torque": [0.08] * 6}, "right": {"torque": [0.0] * 6}}})
        with mock.patch.object(hand_torque_hud.logger_mp, "info") as info:
            cache = publish_torque_hud(tv, first, enabled=True)
            publish_torque_hud(tv, second, enabled=True, cache=cache)
        self.assertEqual(info.call_count, 1)
        self.assertIn("L 3 3 3 3 3 3", info.call_args.args[1])
        self.assertIn("R 0 0 0 0 0 0", info.call_args.args[1])

    def test_skips_unchanged_text(self):
        tv = FakeTv()
        hand = FakeHand({"state": {"left": {"torque": [0.05] * 6}, "right": {"torque": [0.08] * 6}}})
        cache = publish_torque_hud(tv, hand, enabled=True)
        publish_torque_hud(tv, hand, enabled=True, cache=cache)
        self.assertEqual(len(tv.hud), 2)

    def test_disabled_does_not_publish(self):
        tv = FakeTv()
        publish_torque_hud(tv, enabled=False)
        self.assertEqual(tv.hud, [])

    def test_missing_publisher_is_a_no_op(self):
        class Bare:
            pass
        cache = publish_torque_hud(Bare(), enabled=True)
        self.assertEqual(cache, {})

    def test_strip_contains_drawn_pixels(self):
        image = torque_strip_image("9 0 0 0 0 0")
        self.assertGreater(int(np.count_nonzero(image != 16)), 0)
        mid = image.shape[1] // 2
        left_ink = int(np.count_nonzero(np.any(image[:, :40] != 16, axis=2)))
        center_ink = int(np.count_nonzero(np.any(image[:, mid - 40:mid + 40] != 16, axis=2)))
        self.assertGreater(center_ink, left_ink)


if __name__ == "__main__":
    unittest.main()
