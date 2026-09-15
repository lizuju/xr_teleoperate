import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from teleop.utils.episode_writer import EpisodeWriter


MAIN_PATH = Path(__file__).resolve().parents[1] / "teleop/teleop_hand_and_arm.py"


class RecordingShutdownTest(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(MAIN_PATH.read_text())
        main_try = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
                        and any(isinstance(child, ast.Try) and any(
                            isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                            and call.func.attr == "abort" for call in ast.walk(child))
                            for child in node.finalbody))
        self.cleanup = next(node for node in main_try.finalbody if isinstance(node, ast.Try)
                            and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                                    and call.func.attr == "abort" for call in ast.walk(node)))
        self.interrupt_handler = next(handler for handler in main_try.handlers
                                      if isinstance(handler.type, ast.Name)
                                      and handler.type.id == "KeyboardInterrupt")
        on_press = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "on_press")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.writer = EpisodeWriter(self.temporary.name, rerun_log=False)
        self.addCleanup(self.close_writer)
        self.assertTrue(self.writer.create_episode())
        self.ns = {
            "R1_PAUSE": None, "STOP": False, "START": True, "READY": True,
            "RECORD_RUNNING": True, "RECORD_TOGGLE": False, "RECORD_OUTCOME": "unspecified",
            "DRY_RUN_MODE": False, "ARM_REQUEST_GENERATION": 1,
            "R1_A7_DEFERRED_REAL_MODE": True, "logger_mp": Mock(),
            "args": SimpleNamespace(record=True), "dry_run_record_file": None,
            "recorder": self.writer, "exit_code": 0, "failure_reason": None,
        }
        exec(compile(ast.Module(body=[on_press], type_ignores=[]), str(MAIN_PATH), "exec"), self.ns)

    def close_writer(self):
        try:
            self.writer.close()
        except RuntimeError:
            pass

    def finish(self, interrupt=False):
        nodes = [self.cleanup]
        if interrupt:
            nodes = [ast.Try(
                body=[ast.Raise(exc=ast.Call(func=ast.Name(id="KeyboardInterrupt", ctx=ast.Load()),
                                           args=[], keywords=[]), cause=None)],
                handlers=[self.interrupt_handler], orelse=[], finalbody=nodes,
            )]
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(module, str(MAIN_PATH), "exec"), self.ns)
        manifests = list(Path(self.temporary.name).glob("episode_*/episode.json"))
        self.assertEqual(len(manifests), 1)
        return json.loads(manifests[0].read_text())

    def check_pending_outcome(self, key, outcome, interrupt):
        self.ns["on_press"](key)
        self.assertTrue(self.ns["RECORD_TOGGLE"])
        if not interrupt:
            self.ns["on_press"]("q")
        manifest = self.finish(interrupt=interrupt)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["outcome"], outcome)
        self.assertEqual(self.ns["exit_code"], 0)

    def test_success_then_immediate_q(self):
        self.check_pending_outcome("y", "success", False)

    def test_failure_then_immediate_q(self):
        self.check_pending_outcome("n", "failure", False)

    def test_discard_then_immediate_q(self):
        self.check_pending_outcome("x", "discarded", False)

    def test_success_then_immediate_keyboard_interrupt(self):
        self.check_pending_outcome("y", "success", True)

    def test_failure_then_immediate_keyboard_interrupt(self):
        self.check_pending_outcome("n", "failure", True)

    def test_discard_then_immediate_keyboard_interrupt(self):
        self.check_pending_outcome("x", "discarded", True)

    def test_ordinary_q_keeps_unspecified_outcome(self):
        self.ns["on_press"]("q")
        manifest = self.finish()
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["outcome"], "unspecified")

    def test_already_saving_outcome_is_not_overwritten_by_exit(self):
        self.writer.save_episode(outcome="success")
        self.ns["RECORD_RUNNING"] = False
        manifest = self.finish()
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["outcome"], "success")

    def test_fault_aborts_instead_of_committing_pending_outcome(self):
        self.ns["on_press"]("y")
        self.ns.update(exit_code=1, failure_reason="robot feedback stale")
        manifest = self.finish()
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(manifest["error"], "robot feedback stale")
        self.assertEqual(self.ns["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
