import unittest

from teleop.robot_control.arm_bus_monitor import (
    ArmBusMonitor,
    DEFAULT_HEAD_VOLTAGE_INDICES,
    DEFAULT_PACK_VOLTAGE_INDICES,
    bus_voltage_from,
    coerce_sensor,
    format_power_summary,
    hottest_from,
    weakest_voltage_from,
)


class CoerceSensorTest(unittest.TestCase):
    def test_scalars_and_sequences(self):
        self.assertEqual(coerce_sensor(48.2), 48.2)
        self.assertEqual(coerce_sensor([41.0, 47.5]), 47.5)
        self.assertIsNone(coerce_sensor(None))
        self.assertIsNone(coerce_sensor(float("nan")))
        self.assertIsNone(coerce_sensor(True))

    def test_bus_picks_the_weakest_positive_driver(self):
        self.assertEqual(weakest_voltage_from([48.0, 0.0, None, 45.5]), 45.5)
        self.assertEqual(bus_voltage_from([48.0, 0.0, None, 45.5]), 48.0)
        self.assertIsNone(bus_voltage_from([0.0, None, -1.0]))
        self.assertEqual(hottest_from([31.0, None, 44.0]), 44.0)

    def test_pack_and_head_rails_are_sampled_separately(self):
        sample = [0.0] * 35
        for index in DEFAULT_PACK_VOLTAGE_INDICES:
            sample[index] = 38.0
        sample[17] = 37.5
        sample[29] = 37.6
        sample[30] = 24.0
        self.assertEqual(bus_voltage_from(sample, DEFAULT_PACK_VOLTAGE_INDICES), 38.0)
        self.assertEqual(weakest_voltage_from(sample, DEFAULT_PACK_VOLTAGE_INDICES), 37.5)
        self.assertEqual(weakest_voltage_from(sample, DEFAULT_HEAD_VOLTAGE_INDICES), 24.0)


class ArmBusMonitorTest(unittest.TestCase):
    def monitor(self, **kwargs):
        settings = dict(sag_ratio=0.10, hold_s=0.10, release_s=0.20, rest_window_s=0.0)
        settings.update(kwargs)
        return ArmBusMonitor(**settings)

    def test_rest_is_the_highest_bus_seen(self):
        monitor = self.monitor()
        monitor.ingest([48.0] * 35, [30.0] * 35, now=0.0)
        monitor.ingest([50.0] * 35, [31.0] * 35, now=0.01)
        monitor.ingest([46.0] * 35, [40.0] * 35, now=0.02)
        snap = monitor.snapshot()
        self.assertEqual(snap["rest_v"], 50.0)
        self.assertEqual(snap["min_v"], 46.0)
        self.assertEqual(snap["head_v"], 46.0)
        self.assertEqual(snap["max_temp"], 40.0)
        self.assertEqual(snap["samples"], 3)
        self.assertFalse(snap["holding"])

    def test_a_single_blip_does_not_hold(self):
        monitor = self.monitor()
        monitor.ingest([50.0] * 35, [30.0] * 35, now=0.0)
        event = monitor.ingest([44.0] * 35, [30.0] * 35, now=0.05)
        self.assertIsNone(event)
        self.assertFalse(monitor.is_hold_active())
        event = monitor.ingest([50.0] * 35, [30.0] * 35, now=0.06)
        self.assertIsNone(event)
        self.assertFalse(monitor.is_hold_active())

    def test_sustained_sag_holds_and_hysteresis_releases(self):
        monitor = self.monitor()
        monitor.ingest([50.0] * 35, [30.0] * 35, now=0.0)
        self.assertIsNone(monitor.ingest([44.0] * 35, [30.0] * 35, now=0.05))
        self.assertEqual(monitor.ingest([44.0] * 35, [30.0] * 35, now=0.20), "enter")
        self.assertTrue(monitor.is_hold_active())
        self.assertEqual(monitor.drain_events(), ["enter"])
        # 5% hysteresis line is 47.5; 46 is still sagged for release.
        self.assertIsNone(monitor.ingest([46.0] * 35, [30.0] * 35, now=0.20))
        self.assertTrue(monitor.is_hold_active())
        self.assertIsNone(monitor.ingest([48.0] * 35, [30.0] * 35, now=0.30))
        self.assertEqual(monitor.ingest([48.0] * 35, [30.0] * 35, now=0.50), "release")
        self.assertFalse(monitor.is_hold_active())
        self.assertEqual(monitor.snapshot(now=0.50)["hold_events"], 1)

    def test_zero_sag_ratio_is_telemetry_only(self):
        monitor = self.monitor(sag_ratio=0.0)
        monitor.ingest([50.0] * 35, [30.0] * 35, now=0.0)
        self.assertIsNone(monitor.ingest([1.0] * 35, [90.0] * 35, now=1.0))
        self.assertFalse(monitor.is_hold_active())
        self.assertEqual(monitor.snapshot()["min_v"], 1.0)

    def test_rest_window_blocks_hold_until_it_elapses(self):
        monitor = self.monitor(rest_window_s=2.0)
        monitor.ingest([50.0] * 35, [30.0] * 35, now=0.0)
        self.assertIsNone(monitor.ingest([40.0] * 35, [30.0] * 35, now=1.9))
        self.assertFalse(monitor.is_hold_active())
        self.assertIsNone(monitor.ingest([40.0] * 35, [30.0] * 35, now=2.1))
        self.assertEqual(monitor.ingest([40.0] * 35, [30.0] * 35, now=2.25), "enter")

    def test_head_rail_does_not_define_pack_rest_or_hold(self):
        sample = [0.0] * 35
        for index in DEFAULT_PACK_VOLTAGE_INDICES:
            sample[index] = 38.0
        sample[29] = 37.6
        sample[30] = 24.0
        monitor = self.monitor()
        monitor.ingest(sample, [30.0] * 35, now=0.0)
        snap = monitor.snapshot()
        self.assertEqual(snap["rest_v"], 38.0)
        self.assertEqual(snap["min_v"], 38.0)
        self.assertEqual(snap["current_v"], 38.0)
        self.assertEqual(snap["head_v"], 24.0)
        self.assertEqual(snap["head_rest_v"], 24.0)
        self.assertFalse(monitor.is_hold_active())
        one_weak = list(sample)
        one_weak[17] = 20.0
        self.assertIsNone(monitor.ingest(one_weak, [30.0] * 35, now=0.05))
        self.assertIsNone(monitor.ingest(one_weak, [30.0] * 35, now=0.20))
        self.assertFalse(monitor.is_hold_active())
        self.assertEqual(monitor.snapshot()["rest_v"], 38.0)
        self.assertEqual(monitor.snapshot()["min_v"], 20.0)
        self.assertEqual(monitor.snapshot()["current_v"], 38.0)
        head_dip = list(sample)
        head_dip[30] = 20.0
        self.assertIsNone(monitor.ingest(head_dip, [30.0] * 35, now=0.21))
        self.assertIsNone(monitor.ingest(head_dip, [30.0] * 35, now=0.22))
        self.assertFalse(monitor.is_hold_active())
        self.assertEqual(monitor.snapshot()["head_v"], 20.0)
        self.assertEqual(monitor.snapshot()["rest_v"], 38.0)
        pack_sag = list(sample)
        for index in DEFAULT_PACK_VOLTAGE_INDICES:
            pack_sag[index] = 33.0
        self.assertIsNone(monitor.ingest(pack_sag, [30.0] * 35, now=0.25))
        self.assertEqual(monitor.ingest(pack_sag, [30.0] * 35, now=0.40), "enter")
        self.assertTrue(monitor.is_hold_active())

    def test_missing_vol_is_unavailable(self):
        monitor = self.monitor()
        monitor.ingest([None] * 35, [None] * 35, now=0.0)
        line = format_power_summary(monitor.snapshot())
        self.assertIn("unavailable", line)
        self.assertIn("[R1 POWER]", format_power_summary({
            "rest_v": 50.0, "min_v": 44.0, "drop_frac": 0.12, "max_temp": 41.0,
            "samples": 10, "hold_events": 1, "hold_s": 0.8, "holding": False,
        }))

    def test_default_ratio_allows_ordinary_load(self):
        self.assertEqual(ArmBusMonitor.default_sag_ratio, 0.50)
        monitor = ArmBusMonitor(hold_s=0.10, release_s=0.20, rest_window_s=0.0)
        monitor.ingest([38.0] * 35, [30.0] * 35, now=0.0)
        self.assertIsNone(monitor.ingest([33.0] * 35, [30.0] * 35, now=0.05))
        self.assertIsNone(monitor.ingest([33.0] * 35, [30.0] * 35, now=0.20))
        self.assertFalse(monitor.is_hold_active())
        self.assertIsNone(monitor.ingest([18.0] * 35, [30.0] * 35, now=0.25))
        self.assertEqual(monitor.ingest([18.0] * 35, [30.0] * 35, now=0.40), "enter")

    def test_rejects_bad_limits(self):
        with self.assertRaises(ValueError):
            ArmBusMonitor(sag_ratio=-0.1)
        with self.assertRaises(ValueError):
            ArmBusMonitor(hold_s=float("nan"))


if __name__ == "__main__":
    unittest.main()
