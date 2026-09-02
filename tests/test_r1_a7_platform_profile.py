import hashlib
import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "config" / "r1_a7_dual_linker_o6_v1.json"


class R1A7DualO6PlatformProfileTest(unittest.TestCase):
    def test_profile_binds_combined_model_and_mechanical_acceptance(self):
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(profile["configuration_id"], "r1_a7_dual_linker_o6_v1")

        urdf = (REPO_ROOT / profile["combined_urdf"]["workspace_relative_path"]).resolve()
        self.assertTrue(urdf.is_file())
        self.assertEqual(
            hashlib.sha256(urdf.read_bytes()).hexdigest(),
            profile["combined_urdf"]["sha256"],
        )

        acceptance_path = (
            REPO_ROOT / profile["mechanical_acceptance"]["workspace_relative_path"]
        ).resolve()
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        self.assertEqual(acceptance["result"], "PASS")
        self.assertEqual(acceptance["configuration_id"], profile["configuration_id"])

        safety = profile["current_safety_state"]
        self.assertFalse(safety["physical_publishing_enabled"])
        self.assertFalse(safety["o6_actuation_enabled"])
        self.assertTrue(safety["combined_collision_bridge_integrated"])
        self.assertEqual(safety["mode"], "shadow-only")
        readonly = profile["o6_readonly_state"]
        self.assertEqual(readonly["schema"], "linker_o6_readonly_state_v1")
        self.assertEqual(readonly["left_device"], "/dev/ttyHAND0")
        self.assertEqual(readonly["left_slave_id"], 40)
        self.assertEqual(readonly["right_device"], "/dev/ttyHAND1")
        self.assertEqual(readonly["right_slave_id"], 39)
        self.assertFalse(readonly["actuation_enabled"])


if __name__ == "__main__":
    unittest.main()
