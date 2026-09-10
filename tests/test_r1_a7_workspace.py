from pathlib import Path
import sys
import unittest


STAGE_ROOT = Path(__file__).resolve().parents[1]
SIM_TOOLS = STAGE_ROOT / "sim_tools"
if not SIM_TOOLS.is_dir():
    SIM_TOOLS = STAGE_ROOT.parent / "unitree_sim_isaaclab" / "tools"
sys.path.insert(0, str(SIM_TOOLS))

from r1_a7_workspace import (
    constrain_dual_offset,
    is_dual_offset_in_workspace,
    is_offset_in_workspace,
)


class R1A7WorkspaceTest(unittest.TestCase):
    def test_old_rectangular_corner_is_rejected(self):
        self.assertFalse(is_dual_offset_in_workspace([0.12, 0.10, 0.0, 0.0, 0.0, 0.0]))

    def test_scanned_small_offsets_are_accepted(self):
        self.assertTrue(is_dual_offset_in_workspace([0.005, 0.0, 0.0, 0.005, 0.0, 0.0]))
        self.assertTrue(is_dual_offset_in_workspace([-0.01, 0.0, 0.0, -0.01, 0.0, 0.0]))

    def test_constraint_projects_each_wrist_to_the_continuous_boundary(self):
        requested = [0.12, 0.10, 0.0, 0.12, -0.10, 0.0]
        constrained, limited = constrain_dual_offset(requested)
        self.assertTrue(limited)
        self.assertTrue(is_dual_offset_in_workspace(constrained))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertTrue(
                is_offset_in_workspace(
                    [fraction * value for value in constrained[:3]], "left"
                )
            )
            self.assertTrue(
                is_offset_in_workspace(
                    [fraction * value for value in constrained[3:]], "right"
                )
            )

    def test_visionpro_adapter_is_simulation_isolated(self):
        source = (STAGE_ROOT / "teleop" / "visionpro_r1_a7_o6_sim.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('ARM_MAPPING = "r1_a7_visionpro_cartesian_v1"', source)
        self.assertIn('ARM_REFERENCE = "armed_actual_wrist_yaw_pose"', source)
        self.assertIn('ARM_FRAME = "r1_waist_yaw"', source)
        self.assertIn('arm_reference_mode="head_yaw"', source)
        self.assertIn("baseline = wrist_position.copy()", source)
        rearm_block = source[source.index('if key == "r":') : source.index("if not running:")]
        self.assertIn("baseline = None", rearm_block)
        self.assertIn("fresh_frames = 0", rearm_block)
        self.assertNotIn("r1_a7_workspace", source)
        self.assertNotIn("constrain_dual_offset", source)
        self.assertNotIn("workspace_limited", source)
        self.assertNotIn("raw_position_offset_m", source)
        self.assertIn('sequence, retargeter, "tracking_jump"', source)
        self.assertIn("tracking jump; disarmed, press r again", source)
        self.assertNotIn("ChannelFactoryInitialize", source)
        self.assertNotIn("MotionSwitcher", source)
        self.assertNotIn("/dev/tty", source)


if __name__ == "__main__":
    unittest.main()
