"""Draw Linker O6 joint-torque numbers on the Vision Pro head view.

The lowest-CPU path is to paint text onto a copy of the stereo JPEG that
televuer already encodes. That is one memcpy plus a few ``cv2.putText`` calls
and no extra WebXR panel / extra JPEG. Wrist-style ``ImageBackground`` HUDs
would encode another image every frame.

The copy is the point: ``render_to_xr`` keeps a reference, and recording reads
``head_img.bgr``. Drawing in place would burn the numbers into the episode.
"""
import math
import time

import cv2
import logging_mp
import numpy as np


logger_mp = logging_mp.getLogger(__name__)
LOG_INTERVAL_S = 0.25


FONT = cv2.FONT_HERSHEY_SIMPLEX
BASE_SCALE = 0.9
MIN_SCALE = 0.5
FILL = (0, 255, 255)
OUTLINE = (0, 0, 0)
MARGIN_X = 8
BASELINE_Y = 30
MISSING = "- - - - - -"
# O6 tau_est is 0-1 of motor max. A cup / ice-cream grasp sits around 0.02-0.08
# (median nonzero ~0.05 on the 2026-09-21 icecream episodes). Mapping that
# full 0-1 range onto 0-9 made almost every grasp read as 0. 0.15 is about
# the 90th percentile of finger torque; 9 is a firm press, not motor stall.
DISPLAY_FULL_SCALE = 0.15
# O6 qvel is normalized 0-1. On the 2026-09-21 icecream episode, joints faster
# than this were moving (mean torque 0.14) rather than holding (mean 0.04).
# Blank those digits so a wave does not look like a grasp. A held object keeps
# the finger nearly still, so the contact reading stays.
MOTION_BLANK = 0.05


def format_torque_digit(value, speed=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    if speed is not None:
        try:
            motion = abs(float(speed))
        except (TypeError, ValueError):
            motion = None
        else:
            if math.isfinite(motion) and motion >= MOTION_BLANK:
                return "0"
    scaled = max(0.0, min(1.0, number / DISPLAY_FULL_SCALE))
    digit = int(round(scaled * 9.0))
    return str(min(9, max(0, digit)))


def format_torque_line(values, speeds=None):
    if not values:
        return MISSING
    speeds = list(speeds) if speeds else []
    return " ".join(
        format_torque_digit(value, speeds[index] if index < len(speeds) else None)
        for index, value in enumerate(values)
    ) or MISSING


def _side_series(entry, name):
    if not isinstance(entry, dict):
        return None
    values = entry.get(name)
    if not values:
        return None
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return out or None


def _snapshot_state(hand_ctrl):
    getter = getattr(hand_ctrl, "get_recording_snapshot", None)
    if getter is None:
        return None
    try:
        snapshot = getter()
    except (RuntimeError, TypeError, AttributeError):
        return None
    if not isinstance(snapshot, dict):
        return None
    state = snapshot.get("state")
    return state if isinstance(state, dict) else None


def read_hand_hud(hand_ctrl):
    """Return ((left torque, left speed), (right torque, right speed)).

    Read-only. Missing channels are None. Does not publish hand commands.
    """
    state = _snapshot_state(hand_ctrl)
    if state is None:
        return (None, None), (None, None)
    left = state.get("left")
    right = state.get("right")
    return (
        (_side_series(left, "torque"), _side_series(left, "qvel")),
        (_side_series(right, "torque"), _side_series(right, "qvel")),
    )


def read_hand_torques(hand_ctrl):
    """Return (left, right) torque lists from a recording snapshot, or Nones."""
    (left_torque, _), (right_torque, _) = read_hand_hud(hand_ctrl)
    return left_torque, right_torque


def _draw_line(frame, text, x0, x1, center=False):
    max_width = max(1, x1 - x0 - 2 * MARGIN_X)
    scale = BASE_SCALE
    (text_width, _), _ = cv2.getTextSize(text, FONT, scale, 1)
    while text_width > max_width and scale > MIN_SCALE:
        scale = max(MIN_SCALE, scale - 0.05)
        (text_width, _), _ = cv2.getTextSize(text, FONT, scale, 1)
    if center:
        x = x0 + max(MARGIN_X, (x1 - x0 - text_width) // 2)
    else:
        x = x0 + MARGIN_X
    origin = (int(x), min(frame.shape[0] - 4, BASELINE_Y))
    cv2.putText(frame, text, origin, FONT, scale, OUTLINE, 2, cv2.LINE_8)
    cv2.putText(frame, text, origin, FONT, scale, FILL, 1, cv2.LINE_8)


def overlay_stereo_hand_torque(bgr, left_values, right_values):
    """Return a copy of ``bgr`` with left-eye / right-eye torque numbers.

    Left half is the left eye (left hand). Right half is the right eye (right hand).
    Kept for tests; the headset watches WebRTC, so live display uses
    ``publish_torque_hud`` ImageBackground strips instead.
    """
    if bgr is None:
        return None
    frame = np.ascontiguousarray(bgr.copy())
    if frame.ndim != 3 or frame.shape[1] < 2:
        return frame
    mid = frame.shape[1] // 2
    _draw_line(frame, format_torque_line(left_values), 0, mid)
    _draw_line(frame, format_torque_line(right_values), mid, frame.shape[1])
    return frame


STRIP_HEIGHT = 48
STRIP_WIDTH = 280


def torque_strip_image(text, width=STRIP_WIDTH, height=STRIP_HEIGHT):
    """Small BGR strip for an ImageBackground overlay on top of WebRTC.

    The headset places this plane by its centre. Digits are drawn in the
    middle so they stay inside the video window.
    """
    frame = np.full((height, width, 3), 16, dtype=np.uint8)
    _draw_line(frame, text, 0, width, center=True)
    return frame


def publish_torque_hud(tv_wrapper, hand_ctrl=None, enabled=True, cache=None):
    """Push left/right torque strips to Vision Pro. Skip unchanged text."""
    if cache is None:
        cache = {}
    publisher = getattr(tv_wrapper, "render_torque_hud_to_xr", None)
    if not enabled or tv_wrapper is None or publisher is None:
        return cache
    (left_values, left_speeds), (right_values, right_speeds) = read_hand_hud(hand_ctrl)
    for side, values, speeds in (
        ("left", left_values, left_speeds),
        ("right", right_values, right_speeds),
    ):
        line = format_torque_line(values, speeds)
        if cache.get(side) == line:
            continue
        publisher(side, torque_strip_image(line))
        cache[side] = line
    shown = "L {0}   R {1}".format(cache.get("left", MISSING), cache.get("right", MISSING))
    now = time.monotonic()
    if shown != cache.get("_log_line") and now - cache.get("_log_at", 0.0) >= LOG_INTERVAL_S:
        logger_mp.info("[XR TORQUE] %s", shown)
        cache["_log_line"] = shown
        cache["_log_at"] = now
    return cache


def render_head_to_xr(tv_wrapper, bgr, hand_ctrl=None, enabled=True):
    """Push the head ZMQ frame. Torque HUD is published separately for WebRTC."""
    tv_wrapper.render_to_xr(bgr)
    return bgr
