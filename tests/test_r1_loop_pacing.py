"""The teleop control loop keeps an absolute 40 Hz schedule."""
import ast
from pathlib import Path
import unittest

MAIN_PATH = Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py"
PERIOD = 0.025


def load():
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "next_loop_tick")
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(MAIN_PATH), "exec"), ns)
    return ns["next_loop_tick"]


def run(work, oversleep=0.0004):
    """Simulate iterations with the given work times; return the start times."""
    next_loop_tick = load()
    tick, now, starts = 0.0, 0.0, []
    for duration in work:
        starts.append(now)
        now += duration
        tick = next_loop_tick(tick, PERIOD, now)
        if tick > now:
            now = tick + oversleep
    return starts


class LoopPacingTest(unittest.TestCase):
    def test_sleep_overshoot_does_not_slow_the_loop(self):
        starts = run([0.020] * 400)
        rate = (len(starts) - 1) / (starts[-1] - starts[0])
        self.assertAlmostEqual(rate, 40.0, delta=0.05)

    def test_a_long_tick_is_caught_up_without_shifting_the_schedule(self):
        starts = run([0.020] * 10 + [0.030] + [0.020] * 10, oversleep=0.0)
        intervals = [b - a for a, b in zip(starts, starts[1:])]
        self.assertAlmostEqual(intervals[10], 0.030)
        self.assertAlmostEqual(intervals[11], 0.020)
        self.assertAlmostEqual(starts[-1], 20 * PERIOD)

    def test_a_stall_restarts_the_schedule_instead_of_bursting(self):
        starts = run([0.020] * 5 + [0.200] + [0.020] * 5, oversleep=0.0)
        intervals = [b - a for a, b in zip(starts, starts[1:])]
        self.assertAlmostEqual(intervals[5], 0.200)
        for interval in intervals[6:]:
            self.assertAlmostEqual(interval, PERIOD)

    def test_the_main_loop_uses_the_absolute_schedule(self):
        text = MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn("loop_next_tick = next_loop_tick(loop_next_tick, 1.0 / args.frequency, time.monotonic())", text)
        self.assertNotIn("sleep_time = max(0, (1 / args.frequency) - time_elapsed)", text)


if __name__ == "__main__":
    unittest.main()
