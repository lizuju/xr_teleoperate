import ast
import builtins
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = STAGE_ROOT / "teleop" / "teleop_hand_and_arm.py"
RETARGET_PATH = STAGE_ROOT / "teleop" / "robot_control" / "linker_o6_retargeting.py"


def attach_parents(tree):
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent


def ancestor_if_tests(node):
    tests = []
    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, ast.If):
            tests.append(ast.unparse(parent.test))
        parent = getattr(parent, "parent", None)
    return tests


class HandOnlyDryRunIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_source = MAIN_PATH.read_text(encoding="utf-8")
        cls.main_tree = ast.parse(cls.main_source)
        attach_parents(cls.main_tree)

    def test_no_hardware_or_dds_import_runs_at_module_import_time(self):
        forbidden = ("unitree_sdk2py", "robot_arm", "robot_arm_ik")
        module_imports = [
            node
            for node in self.main_tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        rendered = "\n".join(ast.unparse(node) for node in module_imports)
        for name in forbidden:
            self.assertNotIn(name, rendered)

    def test_dds_factory_is_guarded_and_publishers_are_after_dry_run_exit(self):
        factory_calls = [
            node
            for node in ast.walk(self.main_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ChannelFactoryInitialize"
        ]
        self.assertTrue(factory_calls)
        for call in factory_calls:
            self.assertIn("not args.dry_run", ancestor_if_tests(call))

        dry_exit = self.main_source.index("raise SystemExit(0)")
        self.assertLess(dry_exit, self.main_source.index("from teleop.robot_control.robot_arm import"))
        self.assertLess(dry_exit, self.main_source.index("from teleop.robot_control.robot_arm_ik import"))
        self.assertLess(dry_exit, self.main_source.index("from unitree_sdk2py.core.channel import ChannelPublisher"))
        self.assertLess(dry_exit, self.main_source.index("from teleop.utils.episode_writer import EpisodeWriter"))

    def test_retargeter_has_no_dds_or_motor_command_dependency(self):
        source = RETARGET_PATH.read_text(encoding="utf-8")
        self.assertNotIn("unitree_sdk2py", source)
        self.assertNotIn("ChannelPublisher", source)
        self.assertNotIn("MotorCmds", source)
        self.assertNotIn("linker_hand_server", source)

    def test_first_stage_cli_requires_the_isolated_combination(self):
        self.assertIn("--hand-only", self.main_source)
        self.assertIn("--dry-run", self.main_source)
        self.assertIn('args.ee == "linker_o6" and not args.dry_run', self.main_source)
        self.assertIn('args.dry_run and not (args.hand_only and args.ee == "linker_o6"', self.main_source)
        self.assertIn("--linker-o6-calibration", self.main_source)
        self.assertIn("args.linker_o6_calibration and not args.dry_run", self.main_source)
        self.assertIn("--linker-o6-live-state", self.main_source)
        self.assertIn("args.linker_o6_live_state and not args.dry_run", self.main_source)
        self.assertIn("temp_path.replace(path)", self.main_source)
        self.assertIn('"published_monotonic_ns": time.monotonic_ns()', self.main_source)
        self.assertIn('mapping_name = linker_o6_calibration.name', self.main_source)
        self.assertIn('"reason": "tracking_stale"', self.main_source)
        self.assertIn('"tracking_age_s": tracking_age_s', self.main_source)
        self.assertIn('"raw_target_12":', self.main_source)
        self.assertIn('"visionpro_input_calibrated":', self.main_source)
        self.assertIn('parser.error("--frequency must be positive.")', self.main_source)
        self.assertIn("math.isfinite(args.frequency)", self.main_source)
        self.assertIn("math.isfinite(args.tracking_timeout)", self.main_source)
        self.assertIn("listen_keyboard_thread.join(timeout=1.0)", self.main_source)
        self.assertIn("elif key == 's' and (START == True or (DRY_RUN_MODE and READY)):", self.main_source)

    def test_dynamic_dry_run_never_imports_or_constructs_control_dependencies(self):
        import_names = []
        publisher_calls = []
        image_client_calls = []
        log_messages = []
        keyboard_ready = threading.Event()
        keyboard_callback = [None]
        original_import = builtins.__import__

        class FakeLogger:
            def debug(self, message):
                log_messages.append(str(message))

            def info(self, message):
                log_messages.append(str(message))

            def warning(self, message):
                log_messages.append(str(message))

            def error(self, message):
                log_messages.append(str(message))

        logging_module = types.ModuleType("logging_mp")
        logging_module.INFO = 20
        logging_module.basicConfig = lambda **kwargs: None
        logging_module.getLogger = lambda name: FakeLogger()

        class FakeImageClient:
            def __init__(self, **kwargs):
                image_client_calls.append(kwargs)

            def get_cam_config(self):
                self.assert_keyboard_ready = keyboard_ready.wait(1.0)
                return {
                    "head_camera": {
                        "binocular": False,
                        "image_shape": [480, 640],
                        "enable_zmq": False,
                        "enable_webrtc": True,
                        "webrtc_port": 60001,
                    },
                    "left_wrist_camera": {"enable_zmq": False},
                    "right_wrist_camera": {"enable_zmq": False},
                }

            def close(self):
                pass

        image_module = types.ModuleType("teleimager.image_client")
        image_module.ImageClient = FakeImageClient
        teleimager_module = types.ModuleType("teleimager")
        teleimager_module.__path__ = []

        ipc_module = types.ModuleType("teleop.utils.ipc")
        ipc_module.IPC_Server = object

        def fake_listen_keyboard(on_press, **kwargs):
            keyboard_callback[0] = on_press
            keyboard_ready.set()

        keyboard_module = types.ModuleType("sshkeyboard")
        keyboard_module.listen_keyboard = fake_listen_keyboard
        keyboard_module.stop_listening = lambda: None

        class FakeTeleVuerWrapper:
            calls = 0

            def __init__(self, **kwargs):
                pass

            def get_tele_data(self):
                type(self).calls += 1
                if type(self).calls == 8:
                    raise KeyboardInterrupt
                timestamp = time.monotonic()
                if type(self).calls == 2:
                    keyboard_callback[0]("r")
                elif type(self).calls == 3:
                    keyboard_callback[0]("s")
                elif type(self).calls == 5:
                    keyboard_callback[0]("r")
                    timestamp -= 1.0
                elif type(self).calls == 7:
                    keyboard_callback[0]("r")
                return types.SimpleNamespace(
                    motion_data_ready=True,
                    motion_data_timestamp=timestamp,
                    left_hand_pos=np.zeros((25, 3)),
                    right_hand_pos=np.zeros((25, 3)),
                )

            def close(self):
                pass

        televuer_module = types.ModuleType("televuer")
        televuer_module.TeleVuerWrapper = FakeTeleVuerWrapper

        class FakePublisher:
            def __init__(self, *args, **kwargs):
                publisher_calls.append((args, kwargs))

        channel_module = types.ModuleType("unitree_sdk2py.core.channel")
        channel_module.ChannelPublisher = FakePublisher
        channel_module.ChannelFactoryInitialize = lambda *args, **kwargs: publisher_calls.append(
            (args, kwargs)
        )
        fake_modules = {
            "logging_mp": logging_module,
            "teleimager": teleimager_module,
            "teleimager.image_client": image_module,
            "teleop.utils.ipc": ipc_module,
            "sshkeyboard": keyboard_module,
            "televuer": televuer_module,
            "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
            "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
            "unitree_sdk2py.core.channel": channel_module,
            "teleop.robot_control.robot_arm": types.ModuleType("teleop.robot_control.robot_arm"),
            "teleop.robot_control.robot_arm_ik": types.ModuleType("teleop.robot_control.robot_arm_ik"),
            "teleop.utils.motion_switcher": types.ModuleType("teleop.utils.motion_switcher"),
            "teleop.utils.episode_writer": types.ModuleType("teleop.utils.episode_writer"),
        }

        def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
            import_names.append(name)
            return original_import(name, globals, locals, fromlist, level)

        local_urdf_root = STAGE_ROOT / "urdf" / "O6"
        urdf_root = (
            local_urdf_root
            if local_urdf_root.exists()
            else Path("/home/hnh/unitree_r1_dev/linkerhand-urdf/O6")
        )
        with tempfile.TemporaryDirectory() as task_dir:
            live_state_path = Path(task_dir) / "live" / "target.json"
            argv = [
                str(MAIN_PATH),
                "--hand-only",
                "--dry-run",
                "--ee",
                "linker_o6",
                "--record",
                "--frequency",
                "1000",
                "--task-dir",
                task_dir,
                "--task-name",
                "dynamic-isolation",
                "--linker-o6-urdf-root",
                str(urdf_root),
                "--linker-o6-live-state",
                str(live_state_path),
            ]
            with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
                sys, "argv", argv
            ), mock.patch.object(builtins, "__import__", side_effect=recording_import):
                with self.assertRaises(SystemExit) as normal_exit:
                    runpy.run_path(str(MAIN_PATH), run_name="__main__")
                self.assertEqual(normal_exit.exception.code, 0)

            record_paths = list((Path(task_dir) / "dynamic-isolation").glob("*.jsonl"))
            self.assertEqual(len(record_paths), 1)
            events = [json.loads(line) for line in record_paths[0].read_text().splitlines()]
            live_event = json.loads(live_state_path.read_text())
            self.assertFalse(live_state_path.with_name(".target.json.tmp").exists())

        forbidden_imports = (
            "unitree_sdk2py",
            "teleop.robot_control.robot_arm",
            "teleop.robot_control.robot_arm_ik",
            "teleop.utils.motion_switcher",
            "teleop.utils.episode_writer",
        )
        for forbidden in forbidden_imports:
            self.assertFalse(any(name.startswith(forbidden) for name in import_names))
        self.assertEqual(publisher_calls, [])
        self.assertEqual(image_client_calls, [])
        self.assertTrue(any('"target_12"' in message for message in log_messages))
        self.assertEqual(
            [event.get("reason") for event in events],
            [None, "tracking_stale", "awaiting_r_key", None],
        )
        self.assertTrue(all("tracking_age_s" in event for event in events))
        self.assertEqual(len(events[0]["left_target"]), 6)
        self.assertEqual(len(events[0]["right_target"]), 6)
        self.assertEqual(len(events[0]["target_12"]), 12)
        self.assertEqual(len(events[0]["raw_target_12"]), 12)
        np.testing.assert_allclose(events[0]["raw_target_12"], events[0]["target_12"])
        self.assertFalse(events[0]["visionpro_input_calibrated"])
        self.assertNotIn("target_12", events[1])
        self.assertNotIn("target_12", events[2])
        self.assertEqual(len(events[3]["target_12"]), 12)
        self.assertTrue(live_event["armed"])
        self.assertEqual(live_event["schema"], "linker_o6_target_v1")
        self.assertEqual(live_event["target_hand_order"], ["left", "right"])
        self.assertEqual(len(live_event["target_12"]), 12)
        self.assertGreater(live_event["sequence"], 0)
        self.assertGreater(live_event["published_monotonic_ns"], 0)

        for option in ("--frequency", "--tracking-timeout"):
            for invalid_value in ("0", "inf", "-inf", "nan"):
                with self.subTest(option=option, invalid_value=invalid_value):
                    invalid_argv = [str(MAIN_PATH), option, invalid_value]
                    with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
                        sys, "argv", invalid_argv
                    ), mock.patch.object(sys, "stderr", io.StringIO()), mock.patch.object(
                        builtins, "__import__", side_effect=recording_import
                    ):
                        with self.assertRaises(SystemExit) as invalid_exit:
                            runpy.run_path(str(MAIN_PATH), run_name="__main__")
                    self.assertEqual(invalid_exit.exception.code, 2)

        FakeTeleVuerWrapper.calls = 0
        keyboard_ready.clear()
        bad_urdf_argv = [
            str(MAIN_PATH),
            "--hand-only",
            "--dry-run",
            "--ee",
            "linker_o6",
            "--linker-o6-urdf-root",
            "/definitely/missing/linker-o6",
        ]
        with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
            sys, "argv", bad_urdf_argv
        ), mock.patch.object(builtins, "__import__", side_effect=recording_import):
            with self.assertRaises(SystemExit) as error_exit:
                runpy.run_path(str(MAIN_PATH), run_name="__main__")
        self.assertEqual(error_exit.exception.code, 1)
        for forbidden in forbidden_imports:
            self.assertFalse(any(name.startswith(forbidden) for name in import_names))
        self.assertEqual(publisher_calls, [])


if __name__ == "__main__":
    unittest.main()
