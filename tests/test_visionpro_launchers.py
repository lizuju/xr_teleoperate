import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class VisionProLaunchersTest(unittest.TestCase):
    def launch(self, name, arguments):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / name
            target.write_text((ROOT / "teleop" / name).read_text())
            for old in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
                (root / old).write_text('#!/bin/bash\nprintf "%s\\0" "$@"\n')
            result = subprocess.run(["bash", str(target), *arguments], capture_output=True)
            return result.returncode, result.stdout.decode().split("\0")[:-1]

    def test_vector_uses_old_parameter_defaults_and_forces_native_input(self):
        status, args = self.launch("run_r1_a7_visionpro.sh", ["192.0.2.7", "--arm-translation-scale", "0.8"])
        self.assertEqual(status, 0)
        self.assertEqual(args, ["--arm-translation-scale", "0.8", "--tracking-source", "visionpro", "--visionpro-ip", "192.0.2.7", "--display-mode", "pass-through", "--wrist-display", "off", "--hand-torque-hud", "off"])

    def test_capture_preserves_spaces_and_chinese_task_arguments(self):
        status, args = self.launch("run_r1_a7_visionpro_capture.sh", ["192.0.2.7", "这次测试名", "左手接雪糕 再放回", "--frequency", "30"])
        self.assertEqual(status, 0)
        self.assertEqual(args[:4], ["这次测试名", "左手接雪糕 再放回", "--frequency", "30"])
        self.assertIn("visionpro", args)
        self.assertEqual(args[-6:], ["--display-mode", "pass-through", "--wrist-display", "off", "--hand-torque-hud", "off"])

    def test_check_only_does_not_pass_teleop_arguments_to_old_launcher(self):
        status, args = self.launch("run_r1_a7_visionpro.sh", ["192.0.2.7", "--check-only"])
        self.assertEqual(status, 0)
        self.assertEqual(args, ["--check-only"])

    def test_missing_required_arguments_does_not_call_old_launcher(self):
        for name in ("run_r1_a7_visionpro.sh", "run_r1_a7_visionpro_capture.sh"):
            status, args = self.launch(name, [])
            self.assertEqual(status, 2)
            self.assertEqual(args, [])


if __name__ == "__main__":
    unittest.main()
