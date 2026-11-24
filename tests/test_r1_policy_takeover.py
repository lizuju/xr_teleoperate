import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import Mock

import numpy as np

from teleop.robot_control.linker_o6_control_loop import LinkerO6ControlLoop
from teleop.robot_control.r1_hand_lock import HAND_LOCK_KEYS, R1HandLocks
from teleop.robot_control.r1_head_lock import HEAD_LOCK_KEYS, R1HeadLock
from teleop.robot_control.r1_pause import R1PauseState
from teleop.utils.episode_writer import EpisodeWriter
import teleop.utils.policy_takeover as takeover_module
from teleop.utils.policy_takeover import O6_DEX1, POLICY_KEYS, R1PolicyTakeover
import test_linker_o6_control_loop as hand_fixture
import test_r1_pause as pause_fixture

ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "teleop/teleop_hand_and_arm.py"
HAS_SDK = importlib.util.find_spec("unitree_sdk2py") is not None and importlib.util.find_spec("sshkeyboard") is not None
NORMALIZATION = {"action_min": [-1.0] * 14 + [0.2] * 6 + [4.3] + [-1.7, -0.5, -0.5],
                 "action_max": [1.0] * 14 + [0.6] * 6 + [5.4] + [0.2, 0.5, 0.5],
                 "qpos_min": [-1.0] * 24, "qpos_max": [1.0] * 24}


def main_function(name):
    tree = ast.parse(MAIN_PATH.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def chunk(arm=0.1):
    values = np.zeros((400, 24))
    values[:, :14], values[:, 14:20], values[:, 20] = arm, 0.2, 4.8
    return values


class FakeClient:
    """Answers each poll with the next queued chunk, stamped with the run's current tick."""

    def __init__(self):
        self.meta = {"dataset": {"end_effectors": dict(O6_DEX1), "image_size_hw": [320, 384], "fps": 40.0},
                     "normalization": NORMALIZATION, "chunk_size": 400}
        self.chunk_size, self.idle, self.failed, self.closed = 400, False, None, False
        self.queued, self.tick = [], lambda: 0

    def poll(self):
        if not self.queued:
            return None
        return (self.tick(), time.monotonic()), 1.0, self.queued.pop(0), None

    def close(self):
        self.closed = True


@unittest.skipUnless(HAS_SDK, "needs the robot SDK environment")
class DriverTest(unittest.TestCase):
    def setUp(self):
        runner, replay = takeover_module.policy_tools()
        names = ("PolicyRun", "ReplayPose", "WAIST_INDEX", "blended", "pose_from_action")
        self.tools = (SimpleNamespace(**{name: getattr(runner, name) for name in names},
                                      check_policy_metadata=lambda meta: None,
                                      check_policy_cameras=lambda dataset, config: None,
                                      validate_contract=lambda dataset: (4.28, 5.4)), replay)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / "best.pt"
        self.checkpoint.write_bytes(b"weights")
        self.state = {"q": [0.0] * 14, "head_q": [0.0, 0.0], "waist_q": 0.0}
        self.requested = {"arm_q": [0.0] * 14, "arm_tau": [0.0] * 14, "head_q": [0.0, 0.0], "waist_q": 0.0}
        self.arm = Mock()
        self.arm.get_recording_snapshot.side_effect = lambda: {"state": dict(self.state),
                                                               "requested": dict(self.requested)}

        def send(q, tau, head, waist_yaw_target=None):
            self.requested.update(arm_q=list(q), arm_tau=list(tau), head_q=list(head), waist_q=waist_yaw_target)
            self.state.update(q=list(q))

        self.arm.ctrl_dual_arm_and_head.side_effect = send
        self.hand = Mock()
        self.hand.get_state.return_value = ([0.2] * 6, [4.8])
        self.hand.get_action.return_value = (np.full(6, 0.2), np.array([4.8]))
        self.client = FakeClient()
        self.logger = Mock()
        self.driver = self.make()

    def make(self, **kwargs):
        options = dict(checkpoint=self.checkpoint, blend_seconds=0.0, max_seconds=5.0, tools=self.tools)
        options.update(kwargs)
        driver = R1PolicyTakeover(self.client, self.arm, self.hand, lambda q: np.zeros(14), {},
                                  Path(self.directory.name) / "log", self.logger, **options)
        self.client.tick = lambda: driver.run.tick
        return driver

    def ticks(self):
        with open(Path(self.directory.name) / "log/ticks.jsonl") as log:
            return [json.loads(line) for line in log]

    def start_commanding(self, arm=0.1):
        self.assertTrue(self.driver.press_go())
        self.assertTrue(self.driver.driving())
        self.assertTrue(self.driver.step(time.monotonic(), 1))  # holds while the first output is awaited
        self.client.queued.append(chunk(arm))
        return self.driver.step(time.monotonic(), 1)

    def test_it_holds_then_commands_the_policy_and_a_takeover_marks_the_next_episode(self):
        self.assertTrue(self.start_commanding(0.1))
        np.testing.assert_allclose(self.requested["arm_q"], 0.1)
        self.hand.update.assert_called()
        self.assertTrue(self.driver.step(time.monotonic(), 1))
        self.assertTrue(self.driver.takeover())
        self.assertFalse(self.driver.step(time.monotonic(), 1))
        self.assertFalse(self.driver.driving())
        info = self.driver.take_intervention_info()
        self.assertEqual(info["end_reason"], "operator took over")
        self.assertEqual(info["policy_ticks"], 2)
        self.assertEqual(info["policy_checkpoint"], str(self.checkpoint))
        self.assertIn("recording_started_after_s", info)
        self.assertIsNone(self.driver.take_intervention_info())
        modes = [tick["mode"] for tick in self.ticks()]
        self.assertEqual(modes[0], "takeover_start")
        self.assertEqual(modes.count("execute"), 2)
        self.assertEqual(modes[-1], "takeover_stop")

    def test_a_first_target_too_far_from_the_held_pose_is_refused(self):
        self.assertFalse(self.start_commanding(0.5))
        self.assertFalse(self.driver.driving())
        self.assertIn("first arm target", self.driver.take_intervention_info()["end_reason"])
        np.testing.assert_allclose(self.requested["arm_q"], 0.0)
        self.assertIn("ended", self.logger.warning.call_args.args[0])

    def test_no_output_after_the_start_trips(self):
        with mock.patch.object(takeover_module, "FIRST_OUTPUT_SECONDS", 0.0):
            self.assertTrue(self.driver.press_go())
            now = time.monotonic()
            self.assertTrue(self.driver.step(now, 1))
            self.assertFalse(self.driver.step(now + 0.01, 1))
        self.assertIn("no policy output", self.driver.take_intervention_info()["end_reason"])

    def test_a_new_pause_stale_output_or_the_time_limit_stops_it(self):
        for generation, delay, reason in ((2, 0.0, "a new pause"), (1, 1.0, "no fresh policy output"),
                                          (1, 0.2, "time limit")):
            self.driver = self.make(max_seconds=0.1)
            self.assertTrue(self.start_commanding(0.1))
            self.assertFalse(self.driver.step(time.monotonic() + delay, generation))
            self.assertFalse(self.driver.driving())
            self.assertIn(reason, self.driver.take_intervention_info()["end_reason"])

    def test_keys_start_once_and_stop_only_a_requested_or_running_policy(self):
        self.assertFalse(self.driver.takeover())
        self.assertFalse(self.driver.stop())
        self.assertTrue(self.driver.press_go())
        self.assertFalse(self.driver.press_go())
        self.assertTrue(self.driver.stop())
        self.assertFalse(self.driver.driving())
        self.assertFalse(self.driver.step(time.monotonic(), 1))  # the go was cancelled before it started
        self.assertIsNone(self.driver.take_intervention_info())

    def test_p_then_g_restarts_and_stays_driving(self):
        self.assertTrue(self.start_commanding(0.1))
        self.assertTrue(self.driver.stop())
        self.assertTrue(self.driver.press_go())
        self.assertTrue(self.driver.step(time.monotonic(), 2))  # old run ended by the new pause, new one holding
        self.assertTrue(self.driver.driving())
        self.assertEqual(self.driver.segments, 2)

    def test_o_forgets_the_policy_segment(self):
        self.assertTrue(self.start_commanding(0.1))
        self.driver.stop()
        self.driver.step(time.monotonic(), 1)
        self.driver.forget()
        self.assertIsNone(self.driver.take_intervention_info())

    def test_close_ends_the_run_writes_a_summary_and_stops_the_worker(self):
        self.assertTrue(self.start_commanding(0.1))
        self.driver.close()
        self.assertTrue(self.client.closed)
        summary = json.loads((Path(self.directory.name) / "log/summary.json").read_text())
        self.assertEqual(summary["segments"], 1)
        self.assertEqual(self.ticks()[-1]["reason"], "teleop exited")
        run = json.loads((Path(self.directory.name) / "log/run.json").read_text())
        self.assertEqual(run["mode"], "takeover")


class HandoverTest(unittest.TestCase):
    def setUp(self):
        self.controller = hand_fixture.FakeController()
        self.retargeter = SimpleNamespace(left=hand_fixture.FakeRetargeter(), right=hand_fixture.FakeRetargeter())
        self.external, self.paused, self.log = [False], [False], []
        self.loop = LinkerO6ControlLoop(
            hand_fixture.sample, self.retargeter, self.controller, 30.0, 0.25, lambda: False,
            lambda state, action: None, should_pause=lambda: self.paused[0],
            is_external=lambda: self.external[0], takeover_tolerance=(0.2, 0.1), log=self.log.append)

    def drive(self):
        self.external[0] = True
        self.controller.update(np.full(6, 0.9), np.full(6, 0.8), tracking_fresh=(True, True))  # the policy
        self.loop.get_tele_data = Mock(side_effect=AssertionError("read XR while the policy drives"))
        sent = len(self.controller.updates)
        for _ in range(3):
            self.loop._step()
        self.assertEqual(len(self.controller.updates), sent)

    def operator(self, value):
        self.loop.get_tele_data = lambda: hand_fixture.sample(value=value)
        self.loop._step()
        return [action[0] for action in self.controller.get_action()]

    def test_the_loop_sends_nothing_while_the_policy_drives(self):
        self.drive()

    def test_after_a_takeover_each_hand_keeps_the_policy_target_until_the_operator_matches_it(self):
        self.drive()
        self.external[0], self.paused[0] = False, True  # takeover: paused until the realign
        self.loop._step()
        self.paused[0] = False
        np.testing.assert_allclose(self.operator(0.2), [0.9, 0.8])
        self.assertTrue(any("keeps the policy's target" in line for line in self.log))
        self.assertTrue(self.loop.get_recording_sample()["held"]["left"])
        np.testing.assert_allclose(self.operator(0.85), [0.85, 0.85])
        self.assertTrue(any("follows you again" in line for line in self.log))

    def test_a_plain_pause_and_resume_follows_at_once(self):
        self.paused[0] = True
        self.loop._step()
        self.paused[0] = False
        np.testing.assert_allclose(self.operator(0.2), [0.2, 0.2])
        self.assertFalse(any("TAKEOVER" in line for line in self.log))


class EpisodeInfoTest(unittest.TestCase):
    def test_extra_info_goes_into_that_episode_only(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeWriter(directory, rerun_log=False)
            try:
                self.assertTrue(writer.create_episode(extra_info={"intervention": {"end_reason": "operator took over"}}))
                deadline = time.monotonic() + 5.0
                while getattr(writer, "episode_dir", None) is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(writer.info["intervention"], {"end_reason": "operator took over"})
                self.assertNotIn("intervention", writer._base_info)
            finally:
                try:
                    writer.close()
                except RuntimeError:
                    pass

    def test_policy_episode_info_marks_only_a_recording_after_a_policy_run(self):
        ns = {}
        exec(compile(ast.Module(body=[main_function("policy_episode_info")], type_ignores=[]), str(MAIN_PATH), "exec"), ns)
        self.assertIsNone(ns["policy_episode_info"]())
        ns["POLICY"] = Mock(take_intervention_info=Mock(return_value=None))
        self.assertIsNone(ns["policy_episode_info"]())
        ns["POLICY"].take_intervention_info.return_value = {"end_reason": "x"}
        self.assertEqual(ns["policy_episode_info"](), {"intervention": {"end_reason": "x"}})
        self.assertIn("recorder.create_episode(extra_info=globals().get(\"policy_episode_info\"", MAIN_PATH.read_text())


class FakePolicy:
    def __init__(self):
        self.busy, self.calls = False, []

    def press_go(self):
        self.calls.append("go")
        if self.busy:
            return False
        self.busy = True
        return True

    def takeover(self):
        self.calls.append("takeover")
        was, self.busy = self.busy, False
        return was

    def stop(self):
        self.calls.append("stop")
        was, self.busy = self.busy, False
        return was

    def forget(self):
        self.calls.append("forget")


class KeyTest(unittest.TestCase):
    def setUp(self):
        self.pause, self.policy = R1PauseState(), FakePolicy()
        self.standard = Mock()
        self.standard.moving.return_value = False
        self.standard.press.return_value = "go"
        self.ns = {
            "R1_PAUSE": self.pause, "START": True, "STOP": False, "READY": True, "RECORD_ENABLED": True,
            "RECORD_RUNNING": False, "RECORD_TOGGLE": False, "RECORD_OUTCOME": "unspecified", "DRY_RUN_MODE": False,
            "ARM_REQUEST_GENERATION": 1, "R1_A7_DEFERRED_REAL_MODE": True, "logger_mp": Mock(),
            "HAND_LOCK_KEYS": HAND_LOCK_KEYS, "HAND_LOCKS": R1HandLocks(), "HEAD_LOCK_KEYS": HEAD_LOCK_KEYS,
            "HEAD_LOCK": R1HeadLock(), "POLICY": self.policy, "POLICY_KEYS": POLICY_KEYS,
            "STANDARD_POSE": self.standard, "STANDARD_POSE_KEYS": ("o", "O"),
        }
        exec(compile(ast.Module(body=[main_function("on_press")], type_ignores=[]), str(MAIN_PATH), "exec"), self.ns)

    def press(self, key):
        self.ns["on_press"](key)

    def test_g_pauses_teleop_and_starts_the_policy(self):
        self.ns["RECORD_TOGGLE"] = True
        self.press("g")
        self.assertTrue(self.pause.paused)
        self.assertTrue(self.policy.busy)
        self.assertFalse(self.ns["RECORD_TOGGLE"])

    def test_g_is_refused_when_it_cannot_run(self):
        cases = (("START", False), ("RECORD_RUNNING", True), ("POLICY", None))
        for name, value in cases:
            self.setUp()
            self.ns[name] = value
            self.press("g")
            self.assertFalse(self.pause.paused, name)
            self.assertFalse(self.policy.busy, name)
        for lock in ("moving", "hand", "head"):
            self.setUp()
            if lock == "moving":
                self.standard.moving.return_value = True
            elif lock == "hand":
                self.ns["HAND_LOCKS"].toggle("right")
            else:
                self.ns["HEAD_LOCK"].toggle()
            self.press("g")
            self.assertFalse(self.policy.busy, lock)

    def test_s_takes_over_resumes_and_records(self):
        self.press("g")
        self.press("s")
        self.assertIn("takeover", self.policy.calls)
        self.assertFalse(self.policy.busy)
        self.assertIsNotNone(self.pause.resume_after)
        self.assertTrue(self.ns["RECORD_TOGGLE"])

    def test_s_after_a_policy_stop_still_resumes_and_records(self):
        self.press("g")
        self.policy.busy = False  # a guard ended the run; the robot holds
        self.press("s")
        self.assertIsNotNone(self.pause.resume_after)
        self.assertTrue(self.ns["RECORD_TOGGLE"])

    def test_r_takes_over_without_recording(self):
        self.press("g")
        self.press("r")
        self.assertIn("takeover", self.policy.calls)
        self.assertIsNotNone(self.pause.resume_after)
        self.assertFalse(self.ns["RECORD_TOGGLE"])

    def test_p_stops_the_policy_and_holds(self):
        self.press("g")
        generation = self.pause.generation
        self.press("p")
        self.assertIn("stop", self.policy.calls)
        self.assertFalse(self.policy.busy)
        self.assertGreater(self.pause.generation, generation)

    def test_o_stops_the_policy_forgets_its_segment_and_moves_to_the_start_pose(self):
        self.press("g")
        self.press("o")
        self.assertEqual(self.policy.calls[-2:], ["stop", "forget"])
        self.standard.press.assert_called_once()


class LoopTest(unittest.TestCase):
    """The real paused branch of the control loop, through the pause test fixture."""

    def setUp(self):
        fixture = pause_fixture.R1PauseControlIntegrationTest()
        fixture.setUp()
        self.fixture, self.ns, self.state = fixture, fixture.ns, fixture.state
        self.remaining = 5
        self.calls = []

        def step(now, generation):
            self.calls.append(generation)
            if self.remaining > 0:
                self.remaining -= 1
                return True
            return False

        self.standard = Mock()
        self.standard.step.return_value = False
        self.ns.update(POLICY=SimpleNamespace(step=step), STANDARD_POSE=self.standard)

    def test_while_the_policy_drives_the_loop_neither_holds_nor_resumes_then_realigns_where_it_left(self):
        hold = self.ns["arm_ctrl"].hold_targets
        self.state.pause()
        now = 1.0
        self.fixture.tick(now)
        self.state.request_resume(now)  # s pressed while the policy drives
        before = hold.call_count
        for _ in range(4):
            now += 0.1
            self.fixture.tick(now)
        self.assertEqual(hold.call_count, before)
        self.assertTrue(self.state.paused)
        self.standard.step.assert_not_called()
        for _ in range(10):
            now += 0.1
            self.fixture.tick(now)
            if not self.state.paused:
                break
        self.assertFalse(self.state.paused)
        np.testing.assert_allclose(self.ns["arm_ik"].reset_smoothing.call_args.kwargs["reference_q"],
                                   self.fixture.requested["arm_q"])
        self.assertEqual(set(self.calls), {self.state.generation})


if __name__ == "__main__":
    unittest.main()
