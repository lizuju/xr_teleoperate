"""Refuse to start an episode when the Vision Pro tracking uplink is stale.

Measured on 2026-09-18: the same record command had left_age_ms p50 = 24 ms in
the morning and 106 ms (7% of frames >500 ms) in the evening. That was the Wi-Fi
uplink, not the recorder. Episodes started on a bad link are not usable as
follow-the-operator data, so [s] waits until a hand sample is fresh again.

limit_ms <= 0 disables the gate (RECORD_MAX_TRACKING_AGE_MS=0).
"""

DEFAULT_RECORD_MAX_TRACKING_AGE_MS = 100.0


def tracking_diagnostics(tv_wrapper):
    """Best-effort read of TeleVuer.get_tracking_diagnostics(); None if unavailable."""
    tvuer = getattr(tv_wrapper, "tvuer", None)
    getter = getattr(tvuer, "get_tracking_diagnostics", None)
    if not callable(getter):
        return None
    try:
        data = getter()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def recording_blocked_by_tracking(diagnostics, limit_ms=DEFAULT_RECORD_MAX_TRACKING_AGE_MS):
    """Return a reason string if recording should not start, else None.

    `diagnostics` is TeleVuer.get_tracking_diagnostics(). A missing hand
    (age None) is not a reason to block: one-handed collection is valid.
    Both ages missing, or any present age above the limit, blocks.
    """
    if diagnostics is None or limit_ms is None:
        return None
    try:
        ceiling = float(limit_ms)
    except (TypeError, ValueError):
        return None
    if ceiling <= 0:
        return None

    ages = []
    for side in ("left", "right"):
        age = diagnostics.get(f"{side}_age_ms")
        if age is None:
            continue
        try:
            ages.append((side, float(age)))
        except (TypeError, ValueError):
            continue
    if not ages:
        return (
            "no XR hand sample yet (left_age_ms and right_age_ms are both missing); "
            "wait until [XR TRACKING] shows a fresh hand"
        )
    stale = [(side, age) for side, age in ages if age > ceiling]
    if not stale:
        return None
    detail = " ".join(f"{side}_age_ms={age:.1f}" for side, age in ages)
    return (
        f"XR tracking age {detail} exceeds {ceiling:.0f} ms; "
        "wait for a fresh [XR TRACKING] line (or RECORD_MAX_TRACKING_AGE_MS=0 to disable)"
    )
