import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE_PYTHON = """#!/bin/bash
printf 'R1_RIGHT_EE=%s\\0' "${R1_RIGHT_EE:-unset}"
printf '%s\\0' "$@"
printf 'END\\0'
"""


class RightWristLauncherTest(unittest.TestCase):
    def launch(self, right_ee_file=None, environment=None, arguments=(), min_q_file=None,
               launcher_name="run_r1_a7_vector.sh"):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "xr" / "teleop").mkdir(parents=True)
            (base / "xr" / "tools").mkdir()
            (base / "venv" / "bin").mkdir(parents=True)
            for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
                (base / "xr" / "teleop" / name).write_text((ROOT / "teleop" / name).read_text())
            launcher = base / "xr" / "teleop" / launcher_name
            python = base / ".venv-xr" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text(FAKE_PYTHON)
            python.chmod(0o755)
            home = base / "home"
            (home / ".config" / "xr_teleoperate").mkdir(parents=True)
            if right_ee_file is not None:
                (home / ".config" / "xr_teleoperate" / "right_ee").write_text(right_ee_file)
            if min_q_file is not None:
                (home / ".config" / "xr_teleoperate" / "dex1_min_q").write_text(min_q_file)
            env = {"PATH": os.environ["PATH"], "HOME": str(home)}
            env.update(environment or {})
            task = ["cup-test", "hold a cup"] if launcher_name == "run_r1_a7_capture.sh" else []
            result = subprocess.run(["bash", str(launcher), *task, *arguments], capture_output=True, env=env)
            calls, current = [], []
            for item in result.stdout.decode().split("\0")[:-1]:
                if item == "END":
                    calls.append(current)
                    current = []
                else:
                    current.append(item)
            return result.returncode, calls, result.stderr.decode()

    def test_config_file_selects_the_gripper_for_preflight_and_teleop(self):
        status, calls, _ = self.launch(right_ee_file="dex1\n")
        self.assertEqual(status, 0)
        preflight, teleop = calls
        self.assertEqual(preflight[0], "R1_RIGHT_EE=dex1")
        self.assertTrue(preflight[2].endswith("check_r1_teleop.py"))
        self.assertEqual(teleop[0], "R1_RIGHT_EE=dex1")
        index = teleop.index("--right-ee")
        self.assertEqual(teleop[index + 1], "dex1")
        self.assertEqual(teleop[teleop.index("--ee") + 1], "linker_o6")
        self.assertEqual(teleop[teleop.index("--dex1-transport") + 1], "lowcmd")
        self.assertEqual(teleop[teleop.index("--dex1-slot") + 1], "33")

    def test_default_is_two_linker_o6_hands(self):
        status, calls, _ = self.launch()
        self.assertEqual(status, 0)
        teleop = calls[1]
        self.assertEqual(teleop[teleop.index("--right-ee") + 1], "linker_o6")

    def test_environment_overrides_the_file(self):
        status, calls, _ = self.launch(right_ee_file="dex1", environment={"R1_RIGHT_EE": "linker_o6"})
        self.assertEqual(status, 0)
        self.assertEqual(calls[0][0], "R1_RIGHT_EE=linker_o6")

    def test_external_board_wiring_is_passed_through(self):
        status, calls, _ = self.launch(right_ee_file="dex1", environment={"R1_DEX1_TRANSPORT": "service"})
        self.assertEqual(status, 0)
        teleop = calls[1]
        self.assertEqual(teleop[teleop.index("--dex1-transport") + 1], "service")

    def test_check_only_runs_preflight_with_the_selection(self):
        status, calls, _ = self.launch(right_ee_file="dex1", arguments=["--check-only"])
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "R1_RIGHT_EE=dex1")

    def test_both_launchers_load_the_saved_cup_minimum(self):
        for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
            with self.subTest(launcher=name):
                status, calls, stderr = self.launch(right_ee_file="dex1", min_q_file="2.15\n", launcher_name=name)
                self.assertEqual(status, 0, stderr)
                teleop = calls[-1]
                self.assertEqual(teleop[teleop.index("--dex1-min-q") + 1], "2.15")

    def test_environment_overrides_saved_minimum_in_both_launchers(self):
        for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
            for minimum in ("0", "1.8"):
                with self.subTest(launcher=name, minimum=minimum):
                    status, calls, stderr = self.launch(right_ee_file="dex1", min_q_file="2.15\n",
                        environment={"R1_DEX1_MIN_Q": minimum}, launcher_name=name)
                    self.assertEqual(status, 0, stderr)
                    teleop = calls[-1]
                    self.assertEqual(teleop[teleop.index("--dex1-min-q") + 1], minimum)

    def test_both_launchers_default_to_the_full_closing_stroke(self):
        for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
            with self.subTest(launcher=name):
                status, calls, stderr = self.launch(launcher_name=name)
                self.assertEqual(status, 0, stderr)
                teleop = calls[-1]
                self.assertEqual(teleop[teleop.index("--dex1-min-q") + 1], "0")

    def test_cli_minimum_has_precedence_over_the_saved_value(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--dex1-min-q", type=float)
        for name in ("run_r1_a7_vector.sh", "run_r1_a7_capture.sh"):
            with self.subTest(launcher=name):
                status, calls, stderr = self.launch(right_ee_file="dex1", min_q_file="2.15\n",
                    arguments=["--dex1-min-q", "1.75"], launcher_name=name)
                self.assertEqual(status, 0, stderr)
                teleop = calls[-1]
                selected, _ = parser.parse_known_args(teleop[1:])
                self.assertEqual(selected.dex1_min_q, 1.75)
                self.assertEqual(teleop[teleop.index("--dex1-min-q") + 1], "2.15")

    def test_unknown_value_stops_before_anything_runs(self):
        status, calls, stderr = self.launch(right_ee_file="gripper")
        self.assertEqual(status, 2)
        self.assertEqual(calls, [])
        self.assertIn("R1_RIGHT_EE", stderr)


if __name__ == "__main__":
    unittest.main()
