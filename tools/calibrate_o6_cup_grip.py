#!/usr/bin/env python3
"""Hand-only cup-grip calibration for Linker O6 (per-axis).

Operator places a hand around a cup, arms the hand, adjusts each of the six
normalised close axes independently, then presses ``s`` to persist the current
6-vector as the teleop max. Does not move the arms. ``q`` / Ctrl+C are not an
e-stop — use the robot button.

Hardware axis order (teleop): thumb_pitch, thumb_yaw, index, middle, ring, pinky.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Allow `python tools/calibrate_o6_cup_grip.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop.utils.o6_grip_cap import AXIS_NAMES, AXIS_COUNT


START_CLOSE = 0.05
STEP = 0.01
FINE_STEP = 0.001
MAX_CLOSE = 0.95
LOOP_HZ = 50.0
MOVE_EPSILON = 0.02

# Hand side keeps 1/2/b; axes use the next number-row keys so selection is unambiguous.
AXIS_KEYS = {
    "3": 0,  # thumb_pitch
    "4": 1,  # thumb_yaw
    "5": 2,  # index
    "6": 3,  # middle
    "7": 4,  # ring
    "8": 5,  # pinky
}


def start_vector(level=START_CLOSE):
    level = float(np.clip(level, 0.0, 1.0))
    return np.full(AXIS_COUNT, level, dtype=float)


def can_record(*, armed, ever_moved, close_q, start_level=START_CLOSE):
    if not armed:
        return False, "hand is not armed; press a first"
    if not ever_moved:
        return False, "hand never left the start pose; adjust an axis first"
    values = np.asarray(close_q, dtype=float).reshape(AXIS_COUNT)
    if float(np.max(values)) <= float(start_level) + MOVE_EPSILON:
        return False, "all axes still near the open start; ramp up before saving"
    return True, "ok"


def format_close_status(close_q, selected_axis, apply_to):
    parts = []
    for index, (name, value) in enumerate(zip(AXIS_NAMES, close_q)):
        marker = ">" if index == selected_axis else " "
        parts.append(f"{marker}{index + 3}:{name}={float(value):.3f}")
    selected = AXIS_NAMES[selected_axis]
    return (
        f"side={apply_to} sel={selected_axis + 3}:{selected} | "
        + " ".join(parts)
    )


class CupGripCalibrator:
    def __init__(
        self,
        controller,
        *,
        apply_to="both",
        save_path=None,
        cup_mouth_diameter_cm=9.0,
        start_close=START_CLOSE,
        step=STEP,
        fine_step=FINE_STEP,
    ):
        if apply_to not in ("both", "left", "right"):
            raise ValueError("apply_to must be both|left|right")
        self.controller = controller
        self.apply_to = apply_to
        self.save_path = save_path
        self.cup_mouth_diameter_cm = float(cup_mouth_diameter_cm)
        self.start_close = float(start_close)
        self.step = float(step)
        self.fine_step = float(fine_step)
        self.close_q = start_vector(self.start_close)
        self.selected_axis = 2  # index — usually first contact on a cup
        self.armed = False
        self.ever_moved = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.last_status = (
            "ready: press a to arm (starts open/low); "
            "3-8 select axis; =/- adjust selected"
        )
        self.saved_path = None
        self.error = None

    def _fresh(self):
        if not self.armed:
            return (False, False)
        if self.apply_to == "left":
            return (True, False)
        if self.apply_to == "right":
            return (False, True)
        return (True, True)

    def _status_locked(self):
        return format_close_status(self.close_q, self.selected_axis, self.apply_to)

    def arm(self):
        with self.lock:
            self.armed = True
            self.close_q = start_vector(self.start_close)
            self.last_status = "armed; " + self._status_locked()

    def load_saved(self):
        """Jump the active hand to the cap already stored on disk.

        One close vector drives every armed side, so a shared ``both`` session
        only loads when the two saved vectors match. Otherwise pick ``1`` or ``2``.
        """
        from teleop.utils.o6_grip_cap import GripCapError, load_grip_cap, max_close_for_side

        try:
            document = load_grip_cap(self.save_path)
        except GripCapError as error:
            with self.lock:
                self.last_status = f"saved cap unreadable: {error}"
            return
        if document is None:
            with self.lock:
                self.last_status = f"no saved grip cap at {self.save_path}"
            return
        with self.lock:
            apply_to = self.apply_to
        if apply_to == "both":
            left = max_close_for_side(document, "left")
            right = max_close_for_side(document, "right")
            if left is None or right is None or not np.allclose(left, right):
                with self.lock:
                    self.last_status = "left and right caps differ; press 1 or 2, then g"
                return
            loaded = left
        else:
            loaded = max_close_for_side(document, apply_to)
            if loaded is None:
                with self.lock:
                    self.last_status = f"no saved cap for {apply_to}"
                return
        loaded = np.clip(np.asarray(loaded, dtype=float), 0.0, MAX_CLOSE)
        with self.lock:
            self.armed = True
            self.close_q = loaded
            self.ever_moved = True
            self.last_status = "loaded saved cap; " + self._status_locked()

    def open_hand(self):
        with self.lock:
            self.armed = False
            self.close_q = start_vector(self.start_close)
            self.last_status = "released (mode 0); press a to arm again"

    def select_axis(self, axis_index):
        axis_index = int(axis_index)
        if axis_index < 0 or axis_index >= AXIS_COUNT:
            raise ValueError(f"axis_index must be 0..{AXIS_COUNT - 1}")
        with self.lock:
            self.selected_axis = axis_index
            self.last_status = "axis selected; " + self._status_locked()

    def bump(self, delta):
        with self.lock:
            if not self.armed:
                self.last_status = "ignored ramp: press a to arm first"
                return
            axis = self.selected_axis
            before = float(self.close_q[axis])
            self.close_q[axis] = float(
                np.clip(before + float(delta), self.start_close, MAX_CLOSE)
            )
            if abs(float(self.close_q[axis]) - self.start_close) > MOVE_EPSILON:
                self.ever_moved = True
            self.last_status = (
                f"axis {axis + 3}:{AXIS_NAMES[axis]} "
                f"{before:.3f}->{self.close_q[axis]:.3f}; "
                + self._status_locked()
            )

    def set_side(self, apply_to):
        if apply_to not in ("both", "left", "right"):
            raise ValueError("apply_to must be both|left|right")
        with self.lock:
            self.apply_to = apply_to
            self.last_status = f"active side={apply_to}; " + self._status_locked()

    def record(self):
        from teleop.utils.o6_grip_cap import (
            GripCapError,
            build_document,
            load_grip_cap,
            merge_grip_cap_document,
            save_grip_cap,
        )

        with self.lock:
            ok, reason = can_record(
                armed=self.armed,
                ever_moved=self.ever_moved,
                close_q=self.close_q,
                start_level=self.start_close,
            )
            if not ok:
                self.last_status = f"save refused: {reason}"
                return None
            levels = self.close_q.copy()
            apply_to = self.apply_to
        left_state, right_state = self.controller.get_state()
        snapshot = self.controller.get_recording_snapshot()
        left_tau = (snapshot.get("state") or {}).get("left", {}).get("torque")
        right_tau = (snapshot.get("state") or {}).get("right", {}).get("torque")
        incoming = build_document(
            max_close_q=levels.tolist(),
            apply_to=apply_to,
            cup_mouth_diameter_cm=self.cup_mouth_diameter_cm,
            command_torque=1.0,
            left_q=left_state if apply_to in ("both", "left") else None,
            right_q=right_state if apply_to in ("both", "right") else None,
            left_tau_est=left_tau if apply_to in ("both", "left") else None,
            right_tau_est=right_tau if apply_to in ("both", "right") else None,
        )
        # One-side save must merge into any existing opposite-hand vector.
        existing = None
        try:
            existing = load_grip_cap(self.save_path)
        except GripCapError:
            existing = None
        document = merge_grip_cap_document(existing, incoming)
        path = save_grip_cap(document, self.save_path)
        with self.lock:
            self.saved_path = str(path)
            self.last_status = (
                f"saved max_close_q={np.round(levels, 3).tolist()} "
                f"apply_to={document.get('apply_to', apply_to)} -> {path}"
            )
        return path

    def request_stop(self):
        self.stop_event.set()

    def step_once(self):
        with self.lock:
            armed = self.armed
            close_q = self.close_q.copy()
            apply_to = self.apply_to
        left_target = close_q if apply_to in ("both", "left") else start_vector(0.0)
        right_target = close_q if apply_to in ("both", "right") else start_vector(0.0)
        if apply_to == "left":
            _, right_state = self.controller.get_state()
            right_target = right_state
        elif apply_to == "right":
            left_state, _ = self.controller.get_state()
            left_target = left_state
        fresh = self._fresh()
        if not armed:
            # Hold measured pose in release so the hand stays limp for placement.
            self.controller.update(*self.controller.get_action(), tracking_fresh=(False, False))
        else:
            self.controller.update(left_target, right_target, tracking_fresh=fresh)

    def run_loop(self):
        period = 1.0 / LOOP_HZ
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                self.step_once()
                remaining = period - (time.monotonic() - started)
                if remaining > 0.0:
                    self.stop_event.wait(remaining)
        except Exception as error:
            self.error = error
            self.stop_event.set()


def _print_help(path):
    axis_lines = [
        f"  {key}     select axis {index}: {AXIS_NAMES[index]}"
        for key, index in sorted(AXIS_KEYS.items(), key=lambda item: item[1])
    ]
    print(
        "\n".join(
            [
                "=== O6 cup grip calibration (hand only, per-axis) ===",
                "Why: previous tool broadcast one scalar close to all 6 joints — too coarse.",
                f"Cup mouth diameter metadata: 9 cm (override with --cup-diameter-cm).",
                "Quantity saved: normalised close q[6] in [0,1] per axis (NOT Newtons).",
                "Command torque stays 1.0 (teleop default). tau_est is metadata only.",
                "Axis order: " + ", ".join(AXIS_NAMES),
                f"Save path: {path}",
                "",
                "Keys:",
                "  a     arm hands (all axes start at low close 0.05)",
                "  g     jump the active side to the saved max (press 1 or 2 first if the",
                "        two hands differ; does not write the file)",
                *axis_lines,
                "  =/.   increase SELECTED axis only (+0.01)",
                "  -/,   decrease SELECTED axis only (-0.01)",
                "  ]     fine +0.001 on selected axis",
                "  [     fine -0.001 on selected axis",
                "  o     open / release immediately (mode 0)",
                "  s     SAVE current 6-vector as teleop max (merges into existing file)",
                "  1/2/b left only / right only / both (default both; both writes same vector;",
                "        left/right keep the other hand's prior vector if present)",
                "  h     reprint this help",
                "  q     quit (releases; NOT an e-stop — use the robot button)",
                "",
                "Place the cup, press a, select each contacting joint (3-8), ramp until",
                "the squeeze feels right, then s. Live status prints all six values.",
                "Next teleop session loads the file and clamps each axis before the 15 ms smoother.",
            ]
        ),
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-interface",
        default="eno1",
        help="DDS network interface (default eno1)",
    )
    parser.add_argument(
        "--cup-diameter-cm",
        type=float,
        default=9.0,
        help="Cup mouth diameter recorded as metadata (default 9)",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="JSON path (default ~/.config/xr_teleoperate/o6_grip_cap.json)",
    )
    parser.add_argument(
        "--side",
        choices=["both", "left", "right"],
        default="both",
        help="Which hand(s) the saved max applies to (default both)",
    )
    parser.add_argument(
        "--dry-run-keys",
        action="store_true",
        help="Exercise key handling without DDS (unit-test aid)",
    )
    args = parser.parse_args(argv)

    from teleop.utils.o6_grip_cap import default_path

    save_path = Path(args.save_path).expanduser() if args.save_path else default_path()
    _print_help(save_path)

    if args.dry_run_keys:
        class Dummy:
            def get_state(self):
                return np.zeros(6), np.zeros(6)

            def get_action(self):
                return np.zeros(6), np.zeros(6)

            def get_recording_snapshot(self):
                return {
                    "state": {
                        "left": {"torque": [0.0] * 6},
                        "right": {"torque": [0.0] * 6},
                    }
                }

            def update(self, *args, **kwargs):
                return None

        CupGripCalibrator(
            Dummy(),
            apply_to=args.side,
            save_path=save_path,
            cup_mouth_diameter_cm=args.cup_diameter_cm,
        )
        print("dry-run-keys mode: no robot commands", flush=True)
        return 0

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from teleop.robot_control.robot_hand_linker_o6 import LinkerO6Controller
    from sshkeyboard import listen_keyboard, stop_listening

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    # Calibration must be able to exceed any previous cap.
    controller = LinkerO6Controller(apply_grip_cap=False)
    controller.wait_until_ready(timeout=5.0)
    controller.activate()
    calibrator = CupGripCalibrator(
        controller,
        apply_to=args.side,
        save_path=save_path,
        cup_mouth_diameter_cm=args.cup_diameter_cm,
    )
    loop = threading.Thread(target=calibrator.run_loop, name="o6-cup-grip", daemon=True)
    loop.start()

    def on_press(key):
        if key in ("q", "Q"):
            print("[quit] releasing and exiting (not an e-stop)", flush=True)
            calibrator.open_hand()
            calibrator.request_stop()
            stop_listening()
            return
        if key in ("a", "A"):
            calibrator.arm()
        elif key in ("g", "G"):
            calibrator.load_saved()
        elif key in ("o", "O"):
            calibrator.open_hand()
        elif key in ("=", ".", "+"):
            calibrator.bump(+STEP)
        elif key in ("-", ","):
            calibrator.bump(-STEP)
        elif key == "]":
            calibrator.bump(+FINE_STEP)
        elif key == "[":
            calibrator.bump(-FINE_STEP)
        elif key in ("s", "S"):
            path = calibrator.record()
            if path is not None:
                print(f"[saved] {path}", flush=True)
        elif key == "1":
            calibrator.set_side("left")
        elif key == "2":
            calibrator.set_side("right")
        elif key in ("b", "B"):
            calibrator.set_side("both")
        elif key in AXIS_KEYS:
            calibrator.select_axis(AXIS_KEYS[key])
        elif key in ("h", "H", "?"):
            _print_help(save_path)
            return
        else:
            return
        print(f"[status] {calibrator.last_status}", flush=True)

    try:
        listen_keyboard(on_press=on_press, until=None, sequential=False)
    finally:
        calibrator.request_stop()
        loop.join(timeout=2.0)
        try:
            controller.stop()
        except Exception as error:
            print(f"[warn] controller stop: {error}", flush=True)
        if calibrator.error is not None:
            raise RuntimeError("calibration loop failed") from calibrator.error
    if calibrator.saved_path:
        print(f"[done] grip cap at {calibrator.saved_path}", flush=True)
    else:
        print(
            "[done] no save this session; teleop still uncapped if no prior file",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
