"""The wrist target frame while the waist follows the head.

The waist is driven by head yaw, which the operator can change without moving
their hands at all. Holding the hand still in world space then forces the arm to
fold to absorb the torso rotation; measured on the 2026-09-16 waist run with the
operator hand still to within 2 mm, the elbow moved 0.10 deg per tick while the
waist was still and 3.51 deg per tick once the waist turned more than 2 deg.
These tests pin the frame that keeps the arm posture relative to the torso, and
the diagnostic ordering that made the counter-rotation look harmless to fix.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "teleop" / "teleop_hand_and_arm.py").read_text(encoding="utf-8")
WRAPPERS = ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh")
#: The guard both counter-rotation sites must sit behind. It reads the flag
#: tolerantly because the integration tests build their own args namespaces, and
#: the fallback is the same value the argument defaults to -- unlike a tolerated
#: zero, it cannot hide a misconfiguration.
WORLD_GUARD = 'getattr(args, "waist_follow_compensation", "torso") == \'world\''


class WaistCompensationDefaultTest(unittest.TestCase):
    def test_the_arm_keeps_its_posture_relative_to_the_torso_by_default(self):
        self.assertIn(
            "--waist-follow-compensation', choices=['torso', 'world'], default='torso'",
            SOURCE,
        )

    def test_world_stays_available_for_comparison(self):
        self.assertIn("'world'", SOURCE)

    def test_both_wrappers_pass_the_knob_through(self):
        for name in WRAPPERS:
            script = (ROOT / "teleop" / name).read_text(encoding="utf-8")
            self.assertIn(
                '--waist-follow-compensation "${WAIST_FOLLOW_COMPENSATION:-torso}"',
                script,
                name,
            )

    def test_the_effective_mode_is_logged(self):
        self.assertIn("[R1 WAIST] follow=on compensation=%s", SOURCE)

    def test_the_tolerant_read_falls_back_to_the_argument_default(self):
        self.assertIn(WORLD_GUARD, SOURCE)
        self.assertIn("default='torso'", SOURCE)


class WaistCompensationWiringTest(unittest.TestCase):
    def test_torso_mode_takes_the_wrapper_pose_without_re_referencing(self):
        """The wrapper's pose is already relative to the operator's own torso.

        televuer expresses the wrist in the current head-yaw frame with the origin
        moved from the head to the waist, so a body turn leaves it unchanged.
        Re-referencing it to the activation yaw turns it into a world pose, and a
        world pose fed to a fixed-waist solver while the real waist turns sends the
        hand through twice the body rotation.
        """
        torso = SOURCE.index("left_wrist_pose = tele_data.left_wrist_pose")
        right = SOURCE.index("right_wrist_pose = tele_data.right_wrist_pose")
        guard = SOURCE.rindex(") == 'torso':", 0, torso)
        world = SOURCE.index("left_wrist_pose = wrist_in_reference_head_yaw_frame(", right)
        self.assertLess(guard, torso)
        self.assertLess(torso, right)
        self.assertIn("else:", SOURCE[right:world])

    def test_the_world_path_still_re_references(self):
        self.assertGreater(
            SOURCE.count("left_wrist_pose = wrist_in_reference_head_yaw_frame("), 0
        )

    def test_the_tick_target_is_only_counter_rotated_in_world_mode(self):
        call = SOURCE.index("left_ik_target = compensate_wrist_for_waist(")
        guard = SOURCE.rindex(WORLD_GUARD, 0, call)
        self.assertLess(guard, call)
        # The guard must belong to the compensation branch, not to something earlier
        # in the loop: everything between it and the call is the call's own block.
        between = SOURCE[guard:call]
        self.assertNotIn("if args.waist_follow:", between)

    def test_the_resume_anchor_is_only_counter_rotated_in_world_mode(self):
        call = SOURCE.index(
            "compensate_wrist_for_waist(pose, r1_waist_yaw_reference, actual_waist_yaw)"
        )
        guard = SOURCE.rindex(WORLD_GUARD, 0, call)
        self.assertLess(guard, call)


class WorkspaceSaturationFrameTest(unittest.TestCase):
    """The FK and the IK target must be compared in the same frame.

    The saturation check used to run after the FK had been rotated from the
    fixed-waist IK frame into the actual root frame, while the IK target stayed
    in the IK frame. That charged the waist angle itself as solver error: 100% of
    samples read as outside the workspace beyond 10 deg of waist when about 36%
    actually were.
    """

    def test_saturation_is_measured_before_the_root_frame_conversion(self):
        saturation = SOURCE.index("workspace_saturation = r1_workspace_saturation(")
        conversion = SOURCE.index("# Fixed-waist IK FK must be returned to the actual robot root frame")
        self.assertLess(
            saturation, conversion,
            "the workspace check must run while the FK is still in the IK frame",
        )

    def test_the_conversion_still_happens_for_the_record(self):
        self.assertIn("compensate_wrist_for_waist(pose, r1_waist_yaw_reference, waist_yaw_actual)", SOURCE)
        self.assertIn('"solved_left_pose": solved_left_pose.tolist()', SOURCE)

    def test_both_sides_of_the_comparison_are_the_ik_frame_values(self):
        saturation = SOURCE.index("workspace_saturation = r1_workspace_saturation(")
        block = SOURCE[saturation:saturation + 400]
        self.assertIn("solved_left_pose, solved_right_pose, left_ik_target, right_ik_target", block)


if __name__ == "__main__":
    unittest.main()
