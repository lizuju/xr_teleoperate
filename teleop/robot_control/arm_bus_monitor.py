"""Track R1_A7 motor bus voltage and decide when to hold pose.

The DDS MotorState already carries ``vol`` and ``temperature``. The controller
used to drop both, so a brownout only showed up 250 ms later as stale
feedback.

The standing R1_A7 has two rails. Waist and both arms sit near 38 V; the
head (especially yaw) sits near 24 V; unused legs report 0. Mixing them
with a global min pins rest at 24 V and freezes the arms on a small head
sag. Hold therefore watches only waist + arms. Head voltage is logged on
its own rail and never arms the hold.

Pack rest is the highest arm/waist reading seen — the 38 V rail, not the
weakest driver. A single wrist sitting at 37.5 V must not pin rest there;
under load that motor dropping ~4 V would freeze both arms every burst.
Hold watches the rail (max of the pack). The weakest pack driver is logged
as min_v. Head voltage is sampled separately (min of pitch/yaw, ~24 V) and
never arms the hold.
"""

import math
import threading

# Waist yaw + left/right 7-DoF arms. Matches R1_A7_JointIndex 13 and 15..28.
DEFAULT_PACK_VOLTAGE_INDICES = (13,) + tuple(range(15, 29))
# Head pitch + yaw. Matches R1_A7_JointIndex 29 and 30. Telemetry only.
DEFAULT_HEAD_VOLTAGE_INDICES = (29, 30)


def coerce_sensor(value):
    """Return a finite float from a DDS scalar or max of a short sequence."""
    if value is None or isinstance(value, (bool, bytes, str)):
        return None
    if isinstance(value, (list, tuple)):
        numbers = [coerce_sensor(item) for item in value]
        numbers = [item for item in numbers if item is not None]
        return max(numbers) if numbers else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        try:
            numbers = [coerce_sensor(item) for item in value]
        except TypeError:
            return None
        numbers = [item for item in numbers if item is not None]
        return max(numbers) if numbers else None
    if not math.isfinite(number):
        return None
    return number


def _positive(values):
    return [value for value in values if value is not None and value > 0.0]


def _voltages_at(voltages, indices=None):
    voltages = list(voltages)
    if not indices:
        return voltages
    return [voltages[index] for index in indices if 0 <= index < len(voltages)]


def group_voltage(voltages, indices=None, pick=min):
    """Reduce the valid voltages in this group; skip missing/zero readings."""
    valid = _positive(_voltages_at(voltages, indices))
    if not valid:
        return None
    return pick(valid)


def bus_voltage_from(voltages, indices=None):
    """38 V rail: highest valid driver in this group."""
    return group_voltage(voltages, indices, pick=max)


def weakest_voltage_from(voltages, indices=None):
    """Weakest valid driver in this group."""
    return group_voltage(voltages, indices, pick=min)


def hottest_from(temperatures):
    valid = [value for value in temperatures if value is not None]
    if not valid:
        return None
    return max(valid)


def format_power_summary(snapshot):
    if not snapshot or snapshot.get("samples", 0) <= 0 or snapshot.get("rest_v") is None:
        return "[R1 POWER] unavailable (no motor vol in DDS); hold disabled."
    rest = snapshot["rest_v"]
    minimum = snapshot["min_v"]
    current = snapshot.get("current_v")
    head = snapshot.get("head_v")
    drop = snapshot.get("drop_frac")
    drop_text = "n/a" if drop is None else "%.1f%%" % (100.0 * drop)
    temp = snapshot.get("max_temp")
    temp_text = "n/a" if temp is None else "%.1f" % temp
    return (
        "[R1 POWER] arm_v=%s rest_v=%.2f min_v=%s head_v=%s drop=%s max_temp=%s "
        "samples=%d hold_events=%d hold_s=%.2f holding=%s"
        % (
            "n/a" if current is None else "%.2f" % current,
            rest,
            "n/a" if minimum is None else "%.2f" % minimum,
            "n/a" if head is None else "%.2f" % head,
            drop_text,
            temp_text,
            snapshot.get("samples", 0),
            snapshot.get("hold_events", 0),
            snapshot.get("hold_s", 0.0),
            "yes" if snapshot.get("holding") else "no",
        )
    )


class ArmBusMonitor:
    default_sag_ratio = 0.50
    default_hold_s = 0.10
    default_release_s = 0.20
    default_rest_window_s = 2.0

    def __init__(
        self,
        sag_ratio=None,
        hold_s=None,
        release_s=None,
        rest_window_s=None,
        pack_indices=None,
        head_indices=None,
    ):
        sag_ratio = self.default_sag_ratio if sag_ratio is None else float(sag_ratio)
        hold_s = self.default_hold_s if hold_s is None else float(hold_s)
        release_s = self.default_release_s if release_s is None else float(release_s)
        rest_window_s = (
            self.default_rest_window_s if rest_window_s is None else float(rest_window_s)
        )
        if not math.isfinite(sag_ratio) or sag_ratio < 0.0:
            raise ValueError("sag_ratio must be finite and >= 0")
        if not math.isfinite(hold_s) or hold_s < 0.0:
            raise ValueError("hold_s must be finite and >= 0")
        if not math.isfinite(release_s) or release_s < 0.0:
            raise ValueError("release_s must be finite and >= 0")
        if not math.isfinite(rest_window_s) or rest_window_s < 0.0:
            raise ValueError("rest_window_s must be finite and >= 0")
        self.sag_ratio = sag_ratio
        self.hold_s = hold_s
        self.release_s = release_s
        self.rest_window_s = rest_window_s
        if pack_indices is None:
            self.pack_indices = DEFAULT_PACK_VOLTAGE_INDICES
        else:
            self.pack_indices = tuple(int(index) for index in pack_indices)
        if head_indices is None:
            self.head_indices = DEFAULT_HEAD_VOLTAGE_INDICES
        else:
            self.head_indices = tuple(int(index) for index in head_indices)
        self._lock = threading.Lock()
        self.reset()

    @property
    def hold_enabled(self):
        return self.sag_ratio > 0.0

    def reset(self):
        with self._lock:
            self._reset_locked()

    def _reset_locked(self):
        self.started_at = None
        self.rest_voltage = None
        self.min_voltage = None
        self.max_temperature = None
        self.current_voltage = None
        self.current_temperature = None
        self.head_voltage = None
        self.head_rest_voltage = None
        self.head_min_voltage = None
        self.samples = 0
        self.hold_active = False
        self.hold_events = 0
        self.hold_started_at = None
        self.hold_s_total = 0.0
        self._sag_since = None
        self._recover_since = None
        self._pending_events = []

    def ingest(self, voltages, temperatures, now):
        """Update telemetry. Return 'enter', 'release', or None this sample."""
        voltages = list(voltages)
        rail = bus_voltage_from(voltages, self.pack_indices)
        pack_min = weakest_voltage_from(voltages, self.pack_indices)
        head = weakest_voltage_from(voltages, self.head_indices)
        hot = hottest_from(list(temperatures))
        with self._lock:
            if self.started_at is None:
                self.started_at = now
            self.samples += 1
            self.current_voltage = rail
            self.head_voltage = head
            if head is not None:
                if self.head_min_voltage is None or head < self.head_min_voltage:
                    self.head_min_voltage = head
                if self.head_rest_voltage is None or head > self.head_rest_voltage:
                    self.head_rest_voltage = head
            if hot is not None:
                self.current_temperature = hot
                if self.max_temperature is None or hot > self.max_temperature:
                    self.max_temperature = hot
            if pack_min is not None and (
                self.min_voltage is None or pack_min < self.min_voltage
            ):
                self.min_voltage = pack_min
            if rail is None:
                return None
            if self.rest_voltage is None or rail > self.rest_voltage:
                self.rest_voltage = rail
            event = self._evaluate_hold_locked(rail, now)
            return event

    def _evaluate_hold_locked(self, bus, now):
        if not self.hold_enabled or self.rest_voltage is None:
            return None
        if (now - self.started_at) < self.rest_window_s:
            return None
        threshold = self.rest_voltage * (1.0 - self.sag_ratio)
        recover_line = self.rest_voltage * (1.0 - 0.5 * self.sag_ratio)
        if bus < threshold:
            self._recover_since = None
            if self._sag_since is None:
                self._sag_since = now
            if (not self.hold_active) and (now - self._sag_since) >= self.hold_s - 1e-9:
                self.hold_active = True
                self.hold_events += 1
                self.hold_started_at = now
                self._pending_events.append("enter")
                return "enter"
            return None
        self._sag_since = None
        if not self.hold_active:
            return None
        if bus < recover_line:
            self._recover_since = None
            return None
        if self._recover_since is None:
            self._recover_since = now
        if (now - self._recover_since) >= self.release_s - 1e-9:
            if self.hold_started_at is not None:
                self.hold_s_total += now - self.hold_started_at
            self.hold_active = False
            self.hold_started_at = None
            self._recover_since = None
            self._pending_events.append("release")
            return "release"
        return None

    def is_hold_active(self):
        with self._lock:
            return self.hold_active

    def drain_events(self):
        with self._lock:
            events = list(self._pending_events)
            self._pending_events = []
            return events

    def snapshot(self, now=None):
        with self._lock:
            hold_s = self.hold_s_total
            if self.hold_active and self.hold_started_at is not None and now is not None:
                hold_s = hold_s + max(0.0, now - self.hold_started_at)
            rest = self.rest_voltage
            minimum = self.min_voltage
            drop = None
            if rest is not None and minimum is not None and rest > 0.0:
                drop = max(0.0, (rest - minimum) / rest)
            return {
                "rest_v": rest,
                "min_v": minimum,
                "current_v": self.current_voltage,
                "head_v": self.head_voltage,
                "head_rest_v": self.head_rest_voltage,
                "head_min_v": self.head_min_voltage,
                "drop_frac": drop,
                "max_temp": self.max_temperature,
                "current_temp": self.current_temperature,
                "samples": self.samples,
                "hold_events": self.hold_events,
                "hold_s": hold_s,
                "holding": self.hold_active,
                "sag_ratio": self.sag_ratio,
                "hold_enabled": self.hold_enabled,
            }
