import ast
import gc
from pathlib import Path
import time
import unittest
from unittest.mock import MagicMock

from teleop.utils.gc_tuning import SlowGcMonitor, freeze_startup_heap

MAIN_PATH = Path(__file__).resolve().parents[1] / "teleop" / "teleop_hand_and_arm.py"


class GcTuningTests(unittest.TestCase):
    def tearDown(self):
        gc.unfreeze()

    def test_freeze_moves_the_startup_heap_out_of_later_collections(self):
        gc.unfreeze()
        startup = [[i] for i in range(50_000)]  # stands in for IK/DDS objects kept alive
        log = MagicMock()
        monitor = freeze_startup_heap(log)
        self.addCleanup(monitor.close)
        self.assertGreaterEqual(gc.get_freeze_count(), len(startup))
        self.assertIn("[R1 GC] froze", log.info.call_args[0][0])
        tracked = len(gc.get_objects())
        self.assertLess(tracked, len(startup))  # frozen objects are no longer in any generation
        started = time.perf_counter()
        gc.collect()
        frozen_ms = (time.perf_counter() - started) * 1000.0
        gc.unfreeze()
        started = time.perf_counter()
        gc.collect()
        full_ms = (time.perf_counter() - started) * 1000.0
        self.assertLess(frozen_ms, full_ms)
        del startup

    def test_slow_collections_are_recorded_without_logging_inside_the_collector(self):
        log = MagicMock()
        monitor = SlowGcMonitor(threshold_ms=0.0)
        self.addCleanup(monitor.close)
        gc.collect()
        self.assertGreaterEqual(monitor.count, 1)
        self.assertEqual(monitor.events[-1][1], 2)
        self.assertEqual(monitor.events[-1][3], "MainThread")
        log.assert_not_called()
        text = monitor.summary()
        self.assertIn("collections >= 0 ms:", text)
        self.assertIn("gen2", text)

    def test_quiet_summary_and_close(self):
        monitor = SlowGcMonitor(threshold_ms=10_000.0)
        gc.collect()
        self.assertEqual(monitor.count, 0)
        self.assertNotIn("max", monitor.summary())
        monitor.close()
        monitor.close()
        self.assertNotIn(monitor._on_gc, gc.callbacks)

    def test_main_loop_freezes_before_it_starts_and_reports_through_globals(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        freeze = source.index("GC_MONITOR = freeze_startup_heap(logger_mp)")
        loop = source.index("        # main loop. robot start to follow VR user's motion\n        while not STOP:")
        self.assertLess(freeze, loop)
        self.assertLess(loop - freeze, 600)
        # Test harnesses execute slices of the main script in their own namespaces.
        loads = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Name) and node.id == "GC_MONITOR" and isinstance(node.ctx, ast.Load)]
        self.assertEqual(loads, [])
        self.assertIn('globals().get("GC_MONITOR")', source)


if __name__ == "__main__":
    unittest.main()
