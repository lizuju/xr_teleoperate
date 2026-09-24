#!/usr/bin/env python3
"""Read-only teach-in for ice-cream machine fixed waypoints.

Operator moves the robot with Vision Pro teleop in another terminal.
This tool only SUBSCRIBES to ``rt/lowstate``, runs FK, and records wrist
poses into ``~/.config/xr_teleoperate/icecream_waypoints.json``.

Does NOT publish lowcmd, enter debug mode, close the hand, apply grip
caps, start teleop, or run any visual servo / VLA. ``q`` quits this tool
only — it is not an e-stop.

Schema ``icecream_waypoints_v2``:
- ``left_cup``: single grasp pose (wrist at dispenser).
- ``right_lever_*``: grasp pose, optional pulled pose, ``wait_s``, withdraw
  pose; legacy unit ``pull_direction_base`` kept if present.
- file-level ``turn_waist_yaw_rad``: waist yaw after the lever sequence.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (REPO, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from r1_hand_eye_common import (  # noqa: E402
    DEFAULT_URDF,
    HandEyeError,
    R1A7FK,
    motor_q_to_pinocchio_q,
    pinocchio_q_to_named,
    rt_from_se3,
)

SCHEMA = "icecream_waypoints_v2"
SCHEMA_V1 = "icecream_waypoints_v1"
DEFAULT_PATH = Path.home() / ".config" / "xr_teleoperate" / "icecream_waypoints.json"
MIN_PULL_DISTANCE_M = 0.01
DEFAULT_WAIT_S = 3.0
LEFT_FRAME = "left_wrist_yaw_link"
RIGHT_FRAME = "right_wrist_yaw_link"

SLOTS = {
    "left_cup": {"side": "left", "frame": LEFT_FRAME, "key": "1"},
    "right_lever_1": {"side": "right", "frame": RIGHT_FRAME, "key": "2"},
    "right_lever_2": {"side": "right", "frame": RIGHT_FRAME, "key": "3"},
    "right_lever_3": {"side": "right", "frame": RIGHT_FRAME, "key": "4"},
}
KEY_TO_SLOT = {meta["key"]: name for name, meta in SLOTS.items()}
LEVER_SLOTS = frozenset(name for name in SLOTS if name.startswith("right_lever_"))


class TeachError(RuntimeError):
    """Raised when a teach action is refused."""


def default_path():
    return Path(os.environ.get("XR_ICECREAM_WAYPOINTS", str(DEFAULT_PATH))).expanduser()


def empty_document():
    return {
        "schema": SCHEMA,
        "slots": {},
        "turn_waist_yaw_rad": None,
        "turn_waist_yaw_recorded_at": None,
        "updated_at": None,
        "notes": (
            "Fixed machine waypoints in URDF root / base frame. "
            "Recorded by teach_r1_icecream_waypoints.py (subscribe-only). "
            "Lever slots: grasp + pulled pose + wait_s + withdraw; "
            "file-level turn_waist_yaw_rad for post-lever waist yaw."
        ),
    }


def _finite_vec3(values, label):
    arr = np.asarray(values, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise TeachError(f"{label} must be finite")
    return [float(x) for x in arr]


def _finite_rot3(matrix, label):
    arr = np.asarray(matrix, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(arr)):
        raise TeachError(f"{label} must be finite")
    return [[float(x) for x in row] for row in arr]


def _optional_pinocchio_q(entry, label):
    if "pinocchio_q" not in entry or entry["pinocchio_q"] is None:
        return None
    q = np.asarray(entry["pinocchio_q"], dtype=float).reshape(17)
    if not np.all(np.isfinite(q)):
        raise TeachError(f"{label}.pinocchio_q must be finite")
    return [float(x) for x in q]


def _clean_pose_fields(entry, label, *, side, frame):
    cleaned = {
        "side": side,
        "frame": frame,
        "position_m": _finite_vec3(entry["position_m"], f"{label}.position_m"),
        "rotation": _finite_rot3(entry["rotation"], f"{label}.rotation"),
        "recorded_at": entry.get("recorded_at"),
    }
    pin_q = _optional_pinocchio_q(entry, label)
    if pin_q is not None:
        cleaned["pinocchio_q"] = pin_q
    return cleaned


def _clean_nested_pose(entry, label):
    if not isinstance(entry, dict):
        raise TeachError(f"{label} must be an object")
    cleaned = {
        "position_m": _finite_vec3(entry["position_m"], f"{label}.position_m"),
        "rotation": _finite_rot3(entry["rotation"], f"{label}.rotation"),
        "recorded_at": entry.get("recorded_at"),
    }
    pin_q = _optional_pinocchio_q(entry, label)
    if pin_q is not None:
        cleaned["pinocchio_q"] = pin_q
    return cleaned


def _clean_wait_s(value, label):
    try:
        wait = float(value)
    except (TypeError, ValueError) as error:
        raise TeachError(f"{label} must be a number") from error
    if not np.isfinite(wait) or wait <= 0.0:
        raise TeachError(f"{label} must be > 0")
    return float(wait)


def travel_length_m(grasp_position_m, pulled_position_m):
    """Euclidean distance between grasp and pulled wrist positions (meters)."""
    grasp = np.asarray(grasp_position_m, dtype=float).reshape(3)
    pulled = np.asarray(pulled_position_m, dtype=float).reshape(3)
    return float(np.linalg.norm(pulled - grasp))


def migrate_v1_to_v2(document):
    """In-memory upgrade. Does not write disk. Preserves all slot fields."""
    if not isinstance(document, dict):
        raise TeachError("document must be an object")
    schema = document.get("schema")
    if schema == SCHEMA:
        return document
    if schema != SCHEMA_V1:
        raise TeachError(f"unsupported schema {schema!r}; want {SCHEMA} or {SCHEMA_V1}")
    out = empty_document()
    out["updated_at"] = document.get("updated_at")
    if document.get("notes"):
        out["notes"] = str(document["notes"])
    if document.get("turn_waist_yaw_rad") is not None:
        out["turn_waist_yaw_rad"] = float(document["turn_waist_yaw_rad"])
        out["turn_waist_yaw_recorded_at"] = document.get("turn_waist_yaw_recorded_at")
    slots = document.get("slots") or {}
    if not isinstance(slots, dict):
        raise TeachError("slots must be an object")
    out["slots"] = {name: dict(entry) for name, entry in slots.items()}
    out["schema"] = SCHEMA
    return out


def validate_document(document):
    if not isinstance(document, dict):
        raise TeachError("document must be an object")
    schema = document.get("schema")
    if schema == SCHEMA_V1:
        document = migrate_v1_to_v2(document)
        schema = document.get("schema")
    if schema != SCHEMA:
        raise TeachError(f"unsupported schema {schema!r}; want {SCHEMA}")
    slots = document.get("slots")
    if not isinstance(slots, dict):
        raise TeachError("slots must be an object")
    out = empty_document()
    out["updated_at"] = document.get("updated_at")
    if document.get("notes"):
        out["notes"] = str(document["notes"])
    if document.get("turn_waist_yaw_rad") is not None:
        yaw = float(document["turn_waist_yaw_rad"])
        if not np.isfinite(yaw):
            raise TeachError("turn_waist_yaw_rad must be finite")
        out["turn_waist_yaw_rad"] = yaw
        out["turn_waist_yaw_recorded_at"] = document.get("turn_waist_yaw_recorded_at")
    for name, entry in slots.items():
        if name not in SLOTS:
            raise TeachError(f"unknown slot {name!r}")
        if not isinstance(entry, dict):
            raise TeachError(f"slot {name} must be an object")
        spec = SLOTS[name]
        side = entry.get("side", spec["side"])
        frame = entry.get("frame", spec["frame"])
        if side != spec["side"] or frame != spec["frame"]:
            raise TeachError(f"slot {name} side/frame mismatch")
        cleaned = _clean_pose_fields(entry, name, side=side, frame=frame)
        if name in LEVER_SLOTS:
            if "pull_direction_base" in entry and entry["pull_direction_base"] is not None:
                direction = np.asarray(entry["pull_direction_base"], dtype=float).reshape(3)
                if not np.all(np.isfinite(direction)):
                    raise TeachError(f"{name}.pull_direction_base must be finite")
                norm = float(np.linalg.norm(direction))
                if norm < 1e-9:
                    raise TeachError(f"{name}.pull_direction_base is zero")
                cleaned["pull_direction_base"] = [float(x) for x in (direction / norm)]
                cleaned["pull_direction_recorded_at"] = entry.get("pull_direction_recorded_at")
            if "pulled" in entry and entry["pulled"] is not None:
                cleaned["pulled"] = _clean_nested_pose(entry["pulled"], f"{name}.pulled")
            if "wait_s" in entry and entry["wait_s"] is not None:
                cleaned["wait_s"] = _clean_wait_s(entry["wait_s"], f"{name}.wait_s")
            if "withdraw" in entry and entry["withdraw"] is not None:
                cleaned["withdraw"] = _clean_nested_pose(entry["withdraw"], f"{name}.withdraw")
        else:
            for forbidden in ("pull_direction_base", "pulled", "wait_s", "withdraw"):
                if entry.get(forbidden) is not None:
                    raise TeachError(f"{name} cannot store {forbidden}")
        out["slots"][name] = cleaned
    return out


def load_document(path=None):
    target = Path(path) if path is not None else default_path()
    if not target.is_file():
        return empty_document()
    raw = json.loads(target.read_text(encoding="utf-8"))
    return validate_document(raw)


def save_document(document, path=None):
    """Atomically write a validated document. Returns the path used."""
    validated = validate_document(document)
    validated["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    target = Path(path) if path is not None else default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(validated, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix="." + target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def matrix_to_pose_entry(T_link, *, side, frame, pinocchio_q=None, recorded_at=None):
    R, t = rt_from_se3(T_link)
    entry = {
        "side": side,
        "frame": frame,
        "position_m": [float(x) for x in np.asarray(t, dtype=float).reshape(3)],
        "rotation": [[float(x) for x in row] for row in np.asarray(R, dtype=float).reshape(3, 3)],
        "recorded_at": recorded_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if pinocchio_q is not None:
        named = pinocchio_q_to_named(pinocchio_q)
        entry["pinocchio_q"] = named["pinocchio_q"]
    return entry


def nested_pose_from_matrix(T_link, *, pinocchio_q=None, recorded_at=None):
    R, t = rt_from_se3(T_link)
    entry = {
        "position_m": [float(x) for x in np.asarray(t, dtype=float).reshape(3)],
        "rotation": [[float(x) for x in row] for row in np.asarray(R, dtype=float).reshape(3, 3)],
        "recorded_at": recorded_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if pinocchio_q is not None:
        named = pinocchio_q_to_named(pinocchio_q)
        entry["pinocchio_q"] = named["pinocchio_q"]
    return entry


def compute_pull_direction(grasp_position_m, tip_position_m, min_distance_m=MIN_PULL_DISTANCE_M):
    """Unit vector in base frame from grasp toward tip. Refuses tiny motion."""
    grasp = np.asarray(grasp_position_m, dtype=float).reshape(3)
    tip = np.asarray(tip_position_m, dtype=float).reshape(3)
    delta = tip - grasp
    distance = float(np.linalg.norm(delta))
    if distance < float(min_distance_m):
        raise TeachError(
            f"pull sample too close ({distance * 1000:.1f} mm < "
            f"{float(min_distance_m) * 1000:.0f} mm); move further along the pull"
        )
    return [float(x) for x in (delta / distance)], distance


def _document_shell(document):
    return {
        "schema": SCHEMA,
        "slots": dict(document.get("slots") or {}),
        "turn_waist_yaw_rad": document.get("turn_waist_yaw_rad"),
        "turn_waist_yaw_recorded_at": document.get("turn_waist_yaw_recorded_at"),
        "updated_at": document.get("updated_at"),
        "notes": document.get("notes") or empty_document()["notes"],
    }


def _preserve_lever_extras(entry, previous):
    """Keep lever extras when rewriting the grasp pose."""
    for key in (
        "pull_direction_base",
        "pull_direction_recorded_at",
        "pulled",
        "wait_s",
        "withdraw",
    ):
        if key not in entry and previous.get(key) is not None:
            entry[key] = previous[key]
    return entry


def upsert_pose(document, slot_name, pose_entry):
    if slot_name not in SLOTS:
        raise TeachError(f"unknown slot {slot_name!r}")
    spec = SLOTS[slot_name]
    entry = dict(pose_entry)
    entry["side"] = spec["side"]
    entry["frame"] = spec["frame"]
    previous = (document.get("slots") or {}).get(slot_name) or {}
    if slot_name in LEVER_SLOTS:
        entry = _preserve_lever_extras(entry, previous)
    out = _document_shell(document)
    out["slots"][slot_name] = entry
    return validate_document(out)


def upsert_pull_direction(document, slot_name, direction_unit, *, recorded_at=None):
    if slot_name not in LEVER_SLOTS:
        raise TeachError(f"slot {slot_name} has no pull direction (levers only)")
    slots = dict(document.get("slots") or {})
    entry = slots.get(slot_name)
    if entry is None:
        raise TeachError(f"save grasp pose for {slot_name} before direction")
    entry = dict(entry)
    entry["pull_direction_base"] = list(direction_unit)
    entry["pull_direction_recorded_at"] = recorded_at or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    slots[slot_name] = entry
    out = _document_shell(document)
    out["slots"] = slots
    return validate_document(out)


def upsert_pulled_pose(document, slot_name, pulled_entry, *, min_distance_m=MIN_PULL_DISTANCE_M):
    if slot_name not in LEVER_SLOTS:
        raise TeachError(f"slot {slot_name} has no pulled pose (levers only)")
    slots = dict(document.get("slots") or {})
    entry = slots.get(slot_name)
    if entry is None:
        raise TeachError(f"save grasp pose for {slot_name} before pulled pose")
    entry = dict(entry)
    pulled = dict(pulled_entry)
    distance = travel_length_m(entry["position_m"], pulled["position_m"])
    if distance < float(min_distance_m):
        raise TeachError(
            f"pulled pose too close ({distance * 1000:.1f} mm < "
            f"{float(min_distance_m) * 1000:.0f} mm); pull further before p"
        )
    entry["pulled"] = pulled
    # Keep legacy unit vector in sync when a full pulled pose is recorded.
    direction, _ = compute_pull_direction(
        entry["position_m"], pulled["position_m"], min_distance_m=min_distance_m
    )
    entry["pull_direction_base"] = direction
    entry["pull_direction_recorded_at"] = pulled.get("recorded_at")
    slots[slot_name] = entry
    out = _document_shell(document)
    out["slots"] = slots
    return validate_document(out), distance


def upsert_wait_s(document, slot_name, wait_s):
    if slot_name not in LEVER_SLOTS:
        raise TeachError(f"slot {slot_name} has no wait_s (levers only)")
    slots = dict(document.get("slots") or {})
    entry = slots.get(slot_name)
    if entry is None:
        raise TeachError(f"save grasp pose for {slot_name} before wait_s")
    entry = dict(entry)
    entry["wait_s"] = _clean_wait_s(wait_s, f"{slot_name}.wait_s")
    slots[slot_name] = entry
    out = _document_shell(document)
    out["slots"] = slots
    return validate_document(out)


def upsert_withdraw_pose(document, slot_name, withdraw_entry):
    if slot_name not in LEVER_SLOTS:
        raise TeachError(f"slot {slot_name} has no withdraw pose (levers only)")
    slots = dict(document.get("slots") or {})
    entry = slots.get(slot_name)
    if entry is None:
        raise TeachError(f"save grasp pose for {slot_name} before withdraw")
    entry = dict(entry)
    entry["withdraw"] = dict(withdraw_entry)
    slots[slot_name] = entry
    out = _document_shell(document)
    out["slots"] = slots
    return validate_document(out)


def upsert_turn_waist_yaw(document, yaw_rad, *, recorded_at=None):
    yaw = float(yaw_rad)
    if not np.isfinite(yaw):
        raise TeachError("turn_waist_yaw_rad must be finite")
    out = _document_shell(document)
    out["turn_waist_yaw_rad"] = yaw
    out["turn_waist_yaw_recorded_at"] = recorded_at or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return validate_document(out)


class LowstateSubscribeQSource:
    """Subscribe-only rt/lowstate reader. Never creates a lowcmd publisher."""

    def __init__(self, interface="eno1", startup_timeout=5.0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC

        self._lock = threading.Lock()
        self._latest = None
        self._ready = threading.Event()
        self._crc = CRC()
        self._subscriber = None

        def handle(message):
            if self._crc.Crc(message) != int(message.crc):
                return
            full_q = [float(message.motor_state[index].q) for index in range(35)]
            if not all(np.isfinite(value) for value in full_q):
                return
            with self._lock:
                self._latest = np.asarray(full_q, dtype=float)
            self._ready.set()

        ChannelFactoryInitialize(0, interface)
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(handle, 10)
        if not self._ready.wait(float(startup_timeout)):
            raise TeachError(
                f"no rt/lowstate on {interface} within {startup_timeout:.1f}s "
                "(subscribe-only; no lowcmd was created)"
            )

    def get_motor_q(self):
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def close(self):
        if self._subscriber is not None:
            try:
                self._subscriber.Close()
            except Exception:
                pass
            self._subscriber = None


class IcecreamWaypointTeacher:
    """Keyboard-driven recorder. All motion comes from external teleop."""

    def __init__(self, q_source, fk, *, save_path=None, status_hz=5.0):
        self.q_source = q_source
        self.fk = fk
        self.save_path = Path(save_path) if save_path is not None else default_path()
        self.status_hz = float(status_hz)
        self.document = load_document(self.save_path)
        self.current_slot = "left_cup"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.last_status = "ready"
        self.last_left_xyz = None
        self.last_right_xyz = None
        self.error = None
        self.dirty = False

    def select_slot(self, slot_name):
        if slot_name not in SLOTS:
            raise TeachError(f"unknown slot {slot_name!r}")
        with self.lock:
            self.current_slot = slot_name
            self.last_status = f"slot={slot_name}"

    def _live_poses(self):
        full_q = self.q_source.get_motor_q()
        if full_q is None:
            raise TeachError("no joint q yet (waiting for rt/lowstate)")
        pin_q = motor_q_to_pinocchio_q(full_q)
        left = self.fk.link_pose(pin_q, LEFT_FRAME)
        right = self.fk.link_pose(pin_q, RIGHT_FRAME)
        return pin_q, left, right

    def save_pose(self):
        pin_q, left, right = self._live_poses()
        with self.lock:
            slot = self.current_slot
        spec = SLOTS[slot]
        T = left if spec["side"] == "left" else right
        entry = matrix_to_pose_entry(
            T, side=spec["side"], frame=spec["frame"], pinocchio_q=pin_q
        )
        with self.lock:
            self.document = upsert_pose(self.document, slot, entry)
            self.dirty = True
            xyz = entry["position_m"]
            self.last_status = (
                f"grasp saved {slot} xyz=[{xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f}] "
                f"(memory; press w to write)"
            )
        return entry

    def save_direction(self):
        """Legacy: unit pull_direction only. Prefer save_pulled_pose (p)."""
        with self.lock:
            slot = self.current_slot
            grasp = (self.document.get("slots") or {}).get(slot)
        if slot not in LEVER_SLOTS:
            raise TeachError("pull direction only for right_lever_1/2/3")
        if grasp is None:
            raise TeachError(f"save grasp pose for {slot} first (press s)")
        _pin_q, _left, right = self._live_poses()
        tip = right[:3, 3]
        direction, distance = compute_pull_direction(grasp["position_m"], tip)
        with self.lock:
            self.document = upsert_pull_direction(self.document, slot, direction)
            self.dirty = True
            self.last_status = (
                f"direction saved {slot} "
                f"u=[{direction[0]:.3f},{direction[1]:.3f},{direction[2]:.3f}] "
                f"Δ={distance * 1000:.1f} mm (legacy; prefer p for pulled pose)"
            )
        return direction, distance

    def save_pulled_pose(self):
        with self.lock:
            slot = self.current_slot
            grasp = (self.document.get("slots") or {}).get(slot)
        if slot not in LEVER_SLOTS:
            raise TeachError("pulled pose only for right_lever_1/2/3")
        if grasp is None:
            raise TeachError(f"save grasp pose for {slot} first (press s)")
        pin_q, _left, right = self._live_poses()
        pulled = nested_pose_from_matrix(right, pinocchio_q=pin_q)
        with self.lock:
            self.document, distance = upsert_pulled_pose(self.document, slot, pulled)
            self.dirty = True
            xyz = pulled["position_m"]
            self.last_status = (
                f"pulled saved {slot} xyz=[{xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f}] "
                f"travel={distance * 1000:.1f} mm (memory; press w to write)"
            )
        return pulled, distance

    def nudge_wait(self, delta_s):
        with self.lock:
            slot = self.current_slot
            entry = (self.document.get("slots") or {}).get(slot)
        if slot not in LEVER_SLOTS:
            raise TeachError("wait_s only for right_lever_1/2/3")
        if entry is None:
            raise TeachError(f"save grasp pose for {slot} first (press s)")
        current = entry.get("wait_s")
        if current is None:
            current = DEFAULT_WAIT_S
        new_wait = float(current) + float(delta_s)
        if new_wait <= 0.0:
            raise TeachError(f"wait_s must be > 0 (refused {new_wait})")
        with self.lock:
            self.document = upsert_wait_s(self.document, slot, new_wait)
            self.dirty = True
            self.last_status = f"wait_s={new_wait:.1f}s on {slot} (memory; press w to write)"
        return new_wait

    def set_wait_default(self):
        """Set wait_s to DEFAULT_WAIT_S (3.0) on the selected lever slot."""
        with self.lock:
            slot = self.current_slot
            entry = (self.document.get("slots") or {}).get(slot)
        if slot not in LEVER_SLOTS:
            raise TeachError("wait_s only for right_lever_1/2/3")
        if entry is None:
            raise TeachError(f"save grasp pose for {slot} first (press s)")
        with self.lock:
            self.document = upsert_wait_s(self.document, slot, DEFAULT_WAIT_S)
            self.dirty = True
            self.last_status = (
                f"wait_s={DEFAULT_WAIT_S:.1f}s on {slot} (memory; press [ / ] to nudge)"
            )
        return DEFAULT_WAIT_S

    def save_withdraw_pose(self):
        with self.lock:
            slot = self.current_slot
            grasp = (self.document.get("slots") or {}).get(slot)
        if slot not in LEVER_SLOTS:
            raise TeachError("withdraw pose only for right_lever_1/2/3")
        if grasp is None:
            raise TeachError(f"save grasp pose for {slot} first (press s)")
        pin_q, _left, right = self._live_poses()
        withdraw = nested_pose_from_matrix(right, pinocchio_q=pin_q)
        with self.lock:
            self.document = upsert_withdraw_pose(self.document, slot, withdraw)
            self.dirty = True
            xyz = withdraw["position_m"]
            self.last_status = (
                f"withdraw saved {slot} xyz=[{xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f}] "
                f"(memory; press w to write)"
            )
        return withdraw

    def save_turn_yaw(self):
        pin_q, _left, _right = self._live_poses()
        named = pinocchio_q_to_named(pin_q)
        yaw = float(named["waist_yaw"])
        with self.lock:
            self.document = upsert_turn_waist_yaw(self.document, yaw)
            self.dirty = True
            self.last_status = (
                f"turn_waist_yaw_rad={yaw:.4f} (file-level; memory; press w to write)"
            )
        return yaw

    def write_file(self):
        with self.lock:
            path = save_document(self.document, self.save_path)
            self.document = load_document(path)
            self.dirty = False
            n = len(self.document.get("slots") or {})
            self.last_status = f"wrote {path} ({n} slots, schema={SCHEMA})"
        return path

    def reload(self):
        with self.lock:
            self.document = load_document(self.save_path)
            self.dirty = False
            n = len(self.document.get("slots") or {})
            self.last_status = f"reloaded {self.save_path} ({n} slots)"

    def request_stop(self):
        self.stop_event.set()

    def run_status_loop(self):
        period = 1.0 / max(self.status_hz, 0.1)
        while not self.stop_event.is_set():
            try:
                pin_q, left, right = self._live_poses()
                left_xyz = [float(x) for x in left[:3, 3]]
                right_xyz = [float(x) for x in right[:3, 3]]
                yaw = float(pinocchio_q_to_named(pin_q)["waist_yaw"])
                with self.lock:
                    self.last_left_xyz = left_xyz
                    self.last_right_xyz = right_xyz
                    slot = self.current_slot
                    dirty = " *" if self.dirty else ""
                    entry = (self.document.get("slots") or {}).get(slot) or {}
                    wait = entry.get("wait_s")
                    wait_txt = f" wait={wait:.1f}s" if wait is not None else ""
                print(
                    f"\r[live] L=[{left_xyz[0]:6.3f},{left_xyz[1]:6.3f},{left_xyz[2]:6.3f}] "
                    f"R=[{right_xyz[0]:6.3f},{right_xyz[1]:6.3f},{right_xyz[2]:6.3f}] "
                    f"yaw={yaw:6.3f} slot={slot}{wait_txt}{dirty}   ",
                    end="",
                    flush=True,
                )
            except Exception as error:
                with self.lock:
                    self.last_status = f"live FK: {error}"
            time.sleep(period)


def _print_help(save_path):
    print(
        "\n".join(
            [
                "=== R1 ice-cream waypoint teach (read-only, v2) ===",
                "Move the robot with Vision Pro teleop in ANOTHER terminal.",
                "This tool only records wrist FK poses; it never commands arm/hand.",
                f"File: {save_path}",
                "Keys:",
                "  1          select left_cup (left wrist at dispenser mouth)",
                "  2 / 3 / 4  select right_lever_1 / _2 / _3",
                "  s          save grasp pose of selected slot",
                "  p          save pulled pose (levers): full wrist at bottom of pull",
                "  t          set wait_s = 3.0 s on selected lever",
                "  [ / ]      nudge wait_s by −1 / +1 s (must stay > 0)",
                "  o          save withdraw pose (hand left the lever)",
                "  y          save file-level turn_waist_yaw_rad from live lowstate",
                "  d          legacy: unit pull_direction only (prefer p)",
                "  w          write JSON (atomic; keeps other slots; migrates v1→v2)",
                "  r          reload JSON from disk",
                "  h / ?      help",
                "  q          quit this tool only (NOT an e-stop)",
                "Lever 3 order: (s) grasp → pull down → p → t/[ /] wait → withdraw → o → turn → y → w",
                "left_cup needs only s. Travel length = ||pulled − grasp||; push-back = return to grasp.",
            ]
        ),
        flush=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Teach fixed ice-cream machine waypoints (subscribe-only)."
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="JSON path (default ~/.config/xr_teleoperate/icecream_waypoints.json)",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--interface", default="eno1", help="DDS NIC for subscribe-only lowstate")
    parser.add_argument("--lowstate-timeout", type=float, default=5.0)
    parser.add_argument("--status-hz", type=float, default=5.0)
    parser.add_argument(
        "--q-json",
        type=Path,
        default=None,
        help="Offline fixed motor_q JSON (tests / dry runs; no DDS)",
    )
    return parser.parse_args(argv)


class StaticQSource:
    def __init__(self, full_q):
        self._full_q = np.asarray(full_q, dtype=float).reshape(-1)

    def get_motor_q(self):
        return self._full_q.copy()

    def close(self):
        return None


def open_q_source(args):
    if args.q_json:
        payload = json.loads(Path(args.q_json).expanduser().read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "full_q" in payload:
            return StaticQSource(payload["full_q"])
        if isinstance(payload, list):
            return StaticQSource(payload)
        raise TeachError("--q-json must be a list or an object with full_q")
    return LowstateSubscribeQSource(
        interface=args.interface, startup_timeout=args.lowstate_timeout
    )


def module_publishes_lowcmd(source_text=None):
    """Static check used by unit tests: this file must not publish lowcmd."""
    text = source_text if source_text is not None else Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden = ("lowcmd", "LowCmd", "ChannelPublisher", "Enter_Debug_Mode", "publish")
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            hits.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in ("lowcmd", "LowCmd", "publish"):
            hits.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Allow documentary mentions of lowcmd in strings/docstrings.
            continue
    return hits


def main(argv=None):
    args = parse_args(argv)
    save_path = Path(args.save_path).expanduser() if args.save_path else default_path()
    _print_help(save_path)

    try:
        fk = R1A7FK(args.urdf)
    except HandEyeError as error:
        print(f"[error] FK init failed: {error}", flush=True)
        return 2

    q_source = open_q_source(args)
    teacher = IcecreamWaypointTeacher(
        q_source, fk, save_path=save_path, status_hz=args.status_hz
    )
    status_thread = threading.Thread(
        target=teacher.run_status_loop, name="icecream-teach-status", daemon=True
    )
    status_thread.start()

    from sshkeyboard import listen_keyboard, stop_listening

    def on_press(key):
        try:
            if key in ("q", "Q"):
                print("\n[quit] exiting teach tool only (not an e-stop)", flush=True)
                teacher.request_stop()
                stop_listening()
                return
            if key in KEY_TO_SLOT:
                teacher.select_slot(KEY_TO_SLOT[key])
            elif key in ("s", "S"):
                teacher.save_pose()
            elif key in ("p", "P"):
                teacher.save_pulled_pose()
            elif key in ("t", "T"):
                teacher.set_wait_default()
            elif key == "]":
                teacher.nudge_wait(+1.0)
            elif key == "[":
                teacher.nudge_wait(-1.0)
            elif key in ("o", "O"):
                teacher.save_withdraw_pose()
            elif key in ("y", "Y"):
                teacher.save_turn_yaw()
            elif key in ("d", "D"):
                teacher.save_direction()
            elif key in ("w", "W"):
                path = teacher.write_file()
                print(f"\n[wrote] {path}", flush=True)
            elif key in ("r", "R"):
                teacher.reload()
            elif key in ("h", "H", "?"):
                print()
                _print_help(save_path)
                return
            else:
                return
            print(f"\n[status] {teacher.last_status}", flush=True)
        except TeachError as error:
            print(f"\n[refused] {error}", flush=True)
        except HandEyeError as error:
            print(f"\n[fk] {error}", flush=True)

    try:
        listen_keyboard(on_press=on_press, until=None, sequential=False)
    finally:
        teacher.request_stop()
        status_thread.join(timeout=2.0)
        q_source.close()
    if teacher.dirty:
        print(
            f"[done] unsaved changes in memory; re-run and press w, or edit {save_path}",
            flush=True,
        )
    else:
        print(f"[done] file {save_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
