"""Persist and apply an operator-confirmed Linker O6 max close (per axis).

The controllable grasp quantity on O6 is normalised finger close ``q`` in
``[0, 1]`` per axis (0 open, 1 fully closed). Hardware order matches teleop:

    thumb_pitch, thumb_yaw, index, middle, ring, pinky

Command ``tau`` is a separate 0–1 effort field that teleop currently sends as
``1.0`` when armed. Measured ``tau_est`` is motor current normalised the same
way — not Newtons — and is recorded only as metadata.

A missing or unreadable file means no cap: teleop behaviour is unchanged.
Absent or null ``sides.<hand>`` leaves that hand uncapped; limits are never
copied across hands. ``apply_to`` of ``left`` / ``right`` further restricts
which present side entries are active.

Schema ``o6_grip_cap_v2`` stores a length-6 ``max_close_q`` per side. Legacy
``o6_grip_cap_v1`` files with a scalar ``max_close_q`` are migrated in memory
to six equal axis limits (explicit, not silent scalar broadcast in clamp code).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np


SCHEMA = "o6_grip_cap_v2"
SCHEMA_V1 = "o6_grip_cap_v1"
DEFAULT_PATH = Path.home() / ".config" / "xr_teleoperate" / "o6_grip_cap.json"
QUANTITY = "normalized_close_q"
SIDES = ("left", "right")
AXIS_NAMES = (
    "thumb_pitch",
    "thumb_yaw",
    "index",
    "middle",
    "ring",
    "pinky",
)
AXIS_COUNT = len(AXIS_NAMES)


class GripCapError(ValueError):
    """Raised when a grip-cap file exists but cannot be trusted."""


def default_path():
    return Path(os.environ.get("XR_O6_GRIP_CAP", str(DEFAULT_PATH))).expanduser()


def _finite_unit(value, location):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GripCapError(f"{location}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise GripCapError(f"{location}: expected a finite number")
    if number < 0.0 or number > 1.0:
        raise GripCapError(f"{location}: expected a value in [0, 1], got {number}")
    return number


def _close_vector(value, location):
    """Parse a length-6 max_close_q, or migrate a legacy scalar to six equals."""
    if isinstance(value, (list, tuple)):
        if len(value) != AXIS_COUNT:
            raise GripCapError(
                f"{location}: expected {AXIS_COUNT} normalised axis values, got {len(value)}"
            )
        return [_finite_unit(item, f"{location}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        scalar = _finite_unit(value, location)
        return [scalar] * AXIS_COUNT
    raise GripCapError(f"{location}: expected a length-{AXIS_COUNT} list (or legacy scalar)")


def _side_entry(entry, location):
    if not isinstance(entry, dict):
        raise GripCapError(f"{location}: expected an object")
    if "max_close_q" not in entry:
        raise GripCapError(f"{location}.max_close_q: missing")
    max_close = _close_vector(entry.get("max_close_q"), f"{location}.max_close_q")
    out = {"max_close_q": max_close}
    if "q" in entry:
        out["q"] = _close_vector(entry["q"], f"{location}.q")
    if "command_torque" in entry and entry["command_torque"] is not None:
        out["command_torque"] = _finite_unit(
            entry["command_torque"], f"{location}.command_torque"
        )
    if "tau_est_at_save" in entry and entry["tau_est_at_save"] is not None:
        tau = entry["tau_est_at_save"]
        if not isinstance(tau, list) or len(tau) != AXIS_COUNT:
            raise GripCapError(
                f"{location}.tau_est_at_save: expected {AXIS_COUNT} values"
            )
        out["tau_est_at_save"] = [
            None
            if item is None
            else _finite_unit(item, f"{location}.tau_est_at_save[{i}]")
            for i, item in enumerate(tau)
        ]
    return out


def _migrate_v1_document(document):
    """Convert v1 scalar max_close_q entries into v2 length-6 vectors."""
    if not isinstance(document, dict):
        raise GripCapError("grip cap root must be an object")
    sides = document.get("sides")
    if not isinstance(sides, dict) or not sides:
        raise GripCapError("sides must be a non-empty object")
    migrated_sides = {}
    for side, entry in sides.items():
        if not isinstance(entry, dict):
            raise GripCapError(f"sides.{side}: expected an object")
        raw = entry.get("max_close_q")
        if isinstance(raw, (list, tuple)):
            vector = _close_vector(raw, f"sides.{side}.max_close_q")
            was_scalar = False
        else:
            vector = _close_vector(raw, f"sides.{side}.max_close_q")
            was_scalar = True
        new_entry = dict(entry)
        new_entry["max_close_q"] = vector
        if was_scalar:
            new_entry["migrated_from_scalar"] = True
        migrated_sides[side] = new_entry
    notes = document.get("notes") or ""
    migrate_note = (
        "Migrated from o6_grip_cap_v1 scalar max_close_q → length-6 equal axes; "
        "re-run cup calibration for per-axis limits."
    )
    if migrate_note not in str(notes):
        notes = (str(notes) + " " + migrate_note).strip() if notes else migrate_note
    return {
        **document,
        "schema": SCHEMA,
        "sides": migrated_sides,
        "notes": notes,
        "migrated_from": SCHEMA_V1,
    }


def validate_document(document):
    if not isinstance(document, dict):
        raise GripCapError("grip cap root must be an object")
    schema = document.get("schema")
    if schema == SCHEMA_V1:
        document = _migrate_v1_document(document)
        schema = document.get("schema")
    if schema != SCHEMA:
        raise GripCapError(f"unsupported schema: {schema!r}")
    if document.get("quantity") != QUANTITY:
        raise GripCapError(f"unsupported quantity: {document.get('quantity')!r}")
    sides = document.get("sides")
    if not isinstance(sides, dict) or not sides:
        raise GripCapError("sides must be a non-empty object")
    unknown = set(sides) - set(SIDES)
    if unknown:
        raise GripCapError(f"unknown sides: {sorted(unknown)}")
    parsed = {}
    for side in SIDES:
        if side not in sides:
            continue
        # Explicit null means "no cap for this side" — same as omitting the key.
        if sides[side] is None:
            continue
        parsed[side] = _side_entry(sides[side], f"sides.{side}")
    if not parsed:
        raise GripCapError("sides must include left and/or right")
    apply_to = document.get("apply_to", "both")
    if apply_to not in ("both", "left", "right"):
        raise GripCapError(f"apply_to must be both|left|right, got {apply_to!r}")
    diameter = document.get("cup_mouth_diameter_cm")
    if diameter is not None:
        if isinstance(diameter, bool) or not isinstance(diameter, (int, float)):
            raise GripCapError("cup_mouth_diameter_cm: expected a number")
        diameter = float(diameter)
        if not math.isfinite(diameter) or diameter <= 0.0:
            raise GripCapError("cup_mouth_diameter_cm: expected a positive finite number")
    command_torque = document.get("command_torque")
    if command_torque is not None:
        command_torque = _finite_unit(command_torque, "command_torque")
    return {
        "schema": SCHEMA,
        "quantity": QUANTITY,
        "units": document.get("units", "normalized_q_0_to_1"),
        "axis_names": list(AXIS_NAMES),
        "cup_mouth_diameter_cm": diameter,
        "apply_to": apply_to,
        "command_torque": command_torque,
        "sides": parsed,
        "recorded_at": document.get("recorded_at"),
        "notes": document.get("notes"),
        "migrated_from": document.get("migrated_from"),
    }


def load_grip_cap(path=None):
    """Return a validated document, or ``None`` if the file is absent."""
    target = Path(path) if path is not None else default_path()
    if not target.is_file():
        return None
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GripCapError(f"failed to read {target}: {error}") from error
    return validate_document(document)


def build_document(
    *,
    max_close_q,
    apply_to="both",
    cup_mouth_diameter_cm=9.0,
    command_torque=1.0,
    left_q=None,
    right_q=None,
    left_tau_est=None,
    right_tau_est=None,
    notes=None,
    recorded_at=None,
):
    close = _close_vector(max_close_q, "max_close_q")
    if apply_to not in ("both", "left", "right"):
        raise GripCapError(f"apply_to must be both|left|right, got {apply_to!r}")
    sides = {}
    targets = SIDES if apply_to == "both" else (apply_to,)
    for side in targets:
        entry = {"max_close_q": list(close), "command_torque": float(command_torque)}
        q = left_q if side == "left" else right_q
        if q is not None:
            entry["q"] = _close_vector(
                np.asarray(q, dtype=float).reshape(AXIS_COUNT).tolist(),
                f"{side}.q",
            )
        tau = left_tau_est if side == "left" else right_tau_est
        if tau is not None:
            entry["tau_est_at_save"] = [
                None if item is None else _finite_unit(float(item), f"{side}.tau_est")
                for item in list(tau)
            ]
        sides[side] = entry
    return {
        "schema": SCHEMA,
        "quantity": QUANTITY,
        "units": "normalized_q_0_to_1",
        "axis_names": list(AXIS_NAMES),
        "cup_mouth_diameter_cm": float(cup_mouth_diameter_cm)
        if cup_mouth_diameter_cm is not None
        else None,
        "apply_to": apply_to,
        "command_torque": _finite_unit(command_torque, "command_torque"),
        "sides": sides,
        "recorded_at": recorded_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": notes
        or (
            "Operator-confirmed per-axis max finger close for cup grasp. "
            "Not Newtons. Command torque left at teleop default unless overridden."
        ),
    }


def save_grip_cap(document, path=None):
    """Atomically write a validated grip-cap document. Returns the path used."""
    validated = validate_document(document)
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


def max_close_for_side(document, side):
    """Return a length-6 max close for ``side``, or ``None`` if uncapped.

    Absent or null ``sides.<side>`` means that hand is uncapped. Limits are
    never copied from the other hand — calibrate each side you want capped.
    ``apply_to`` further restricts which of the present side entries apply.
    """
    if document is None:
        return None
    apply_to = document.get("apply_to", "both")
    if apply_to not in ("both", side):
        return None
    entry = document.get("sides", {}).get(side)
    if not entry:
        return None
    return np.asarray(entry["max_close_q"], dtype=float).copy()


def command_torque_from_document(document):
    if document is None:
        return None
    if document.get("command_torque") is not None:
        return float(document["command_torque"])
    for side in SIDES:
        entry = document.get("sides", {}).get(side)
        if entry and entry.get("command_torque") is not None:
            return float(entry["command_torque"])
    return None


def clamp_close_q(values, max_close_q):
    """Elementwise clamp a length-6 close vector; ``None`` max leaves it unchanged.

    ``max_close_q`` may be a length-6 sequence. A legacy scalar is expanded to
    six equal limits (migration path only — new saves always store vectors).
    """
    result = np.asarray(values, dtype=float)
    if result.shape != (AXIS_COUNT,):
        raise ValueError(f"values must contain {AXIS_COUNT} axes")
    if max_close_q is None:
        return result.copy()
    limit = np.asarray(max_close_q, dtype=float)
    if limit.shape == ():
        limit = np.full(AXIS_COUNT, float(limit), dtype=float)
    if limit.shape != (AXIS_COUNT,):
        raise ValueError(f"max_close_q must contain {AXIS_COUNT} axes (or a scalar)")
    if not np.isfinite(limit).all():
        raise ValueError("max_close_q must be finite")
    if np.any(limit < 0.0) or np.any(limit > 1.0):
        raise ValueError("max_close_q must be within [0, 1]")
    return np.minimum(result, limit)


def limits_from_document(document):
    """Return ``{"left": ndarray|None, "right": ndarray|None, "command_torque": float|None}``."""
    return {
        "left": max_close_for_side(document, "left"),
        "right": max_close_for_side(document, "right"),
        "command_torque": command_torque_from_document(document),
    }
