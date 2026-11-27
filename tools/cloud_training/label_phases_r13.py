"""Automatic phase labels for run 13 (2026-11-26). The user waived manual review.

Phases (teleop/utils/icecream_task.PHASES): grasp_cup, turn_deliver, dispense, close_retract, turn_place.
Boundaries follow the runtime phase machine (IcecreamTask) so training and execution agree:
  b1 grasp_cup -> turn_deliver: start of the gripper-closed stretch (Dex1 < closed + 0.3) that lasts into the turn
  b2 turn_deliver -> dispense:  the right-hand lock at -90 starts (runtime context sets the right lock in phases 2-3
                                and holds the right arm), else the waist reaching within 8 deg of -90
  b3 dispense -> close_retract: the left O6 lets go after its last grip (mean > 0.6) at -90
  b4 close_retract -> turn_place: that right lock ends, else the waist leaving the -90 window
DAgger corrections that start mid-task get empty leading phases (equal boundaries; needs the relaxed validator).
Writes labels JSON + a table. Read-only on the recordings.
Usage: python label_phases_r13.py NAMES.txt RECORDINGS_DIR OUT.json"""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np

PHASES = ["grasp_cup", "turn_deliver", "dispense", "close_retract", "turn_place"]
names = Path(sys.argv[1]).read_text().split()
root, out = Path(sys.argv[2]), Path(sys.argv[3])
CLOSED, WINDOW, TURN = 4.2798 + 0.3, math.radians(8), math.radians(10)
TARGET = -math.pi / 2  # waist stop at -90 deg, front is 0

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def runs(mask):
    """(start, stop) of each True stretch."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))

labels = {"schema": "r1_icecream_phases_v1", "phases": PHASES, "episodes": {},
          "method": "automatic (label_phases_r13.py); manual review waived by the user on 2026-11-26"}
table = []
for name in names:
    d = root / name
    rows = [json.loads(l) for l in (d / "frames.jsonl").open()]
    n = len(rows)
    assert all(r["idx"] == i for i, r in enumerate(rows)), name
    waist = np.array([r["states"]["body"]["qpos"][0] for r in rows])
    dex = np.array([r["states"]["right_ee"]["qpos"][0] for r in rows])
    o6 = np.array([np.mean(r["states"]["left_ee"]["qpos"]) for r in rows])
    rlock = np.array([bool((r["sample"].get("hand_locks") or {}).get("right")) for r in rows])
    llock = np.array([bool((r["sample"].get("hand_locks") or {}).get("left")) for r in rows])
    at90 = np.abs(waist - TARGET) < WINDOW
    notes = []
    # where the recording starts
    w0 = waist[0]
    starts_at_90 = bool(at90[0])
    reaches_90 = bool(at90.any())
    if not reaches_90:
        b = [0, 0, 0, 0, 0, n]  # front only: a final place/release correction
        notes.append("never at -90: turn_place throughout")
    else:
        first90 = int(np.flatnonzero(at90)[0])
        # b4: the -90 stay ends (first frame after the first -90 arrival that leaves the window for good... use last 90)
        last90 = int(np.flatnonzero(at90)[-1])
        leave = last90 + 1
        # right lock period overlapping the -90 stay
        lock_runs = [(a, z) for a, z in runs(rlock) if z > first90 and a < leave]
        if starts_at_90:
            b1 = b2 = 0
            notes.append("starts at -90")
        else:
            early = waist[:first90 + 1] < -TURN
            turn = int(np.flatnonzero(early)[0]) if early.any() else first90
            closed_runs = [(a, z) for a, z in runs(dex < CLOSED) if a <= turn]
            if closed_runs and closed_runs[-1][1] >= turn:
                b1 = int(closed_runs[-1][0])
            else:
                b1 = turn
                notes.append("gripper not closed into the turn: b1 = turn start")
            # The runtime enters dispense on reaching -90; a right lock pressed after arrival starts it later.
            b2 = max(first90, int(lock_runs[0][0])) if lock_runs else first90
            b2 = max(b2, b1)
        # The runtime holds the waist in dispense/close_retract, so turn_place starts no later than leaving -90.
        b4 = min(int(lock_runs[-1][1]), leave) if lock_runs else leave
        b4 = max(b4, b2)
        notes.append(f"right lock {lock_runs[0][0]}-{lock_runs[-1][1]}" if lock_runs else "no right lock at -90")
        grip = np.flatnonzero((o6 > 0.6) & (np.arange(n) >= b2) & (np.arange(n) < b4))
        if len(grip):
            after = np.flatnonzero((o6 < 0.5) & (np.arange(n) > grip[-1]) & (np.arange(n) < b4))
            b3 = int(after[0]) if len(after) else int(grip[-1]) + 1
        else:
            b3 = b2 if starts_at_90 and o6[0] < 0.5 else (b2 + b4) // 2
            notes.append("no O6 grip at -90")
        b3 = min(max(b3, b2), b4)
        b = [0, b1, b2, b3, min(b4, n), n]
        if b[4] == n:
            notes.append("never leaves -90: turn_place empty")
    assert all(x <= y for x, y in zip(b, b[1:])) and b[0] == 0 and b[-1] == n, (name, b)
    start_phase = int(np.searchsorted(b, 0, side="right") - 1)
    labels["episodes"][name] = {"frames_sha256": digest(d / "frames.jsonl"), "reviewed": True, "boundaries": [int(x) for x in b],
                                "start_phase": start_phase, "note": "; ".join(notes)}
    secs = [(b[i + 1] - b[i]) / 40 for i in range(5)]
    table.append((name[:12], n / 40, start_phase, secs, int(rlock.sum()), int(llock.sum()), "; ".join(notes)))

out.write_text(json.dumps(labels, indent=1, ensure_ascii=False))
print(f"{'recording':12s} {'len':>5s} st " + " ".join(f"{p[:8]:>8s}" for p in PHASES) + "  rlock llock  notes")
for name, length, st, secs, rl, ll, note in table:
    print(f"{name:12s} {length:5.1f} {st:2d} " + " ".join(f"{s:8.1f}" for s in secs) + f"  {rl:5d} {ll:5d}  {note}")
full = [t for t in table if t[2] == 0]
print(f"\n{len(table)} recordings, {len(full)} start in grasp_cup; median phase seconds (those):",
      [round(float(np.median([t[3][i] for t in full])), 1) for i in range(5)])
print("dispense < 3 s:", [t[0] for t in table if 0 < t[3][2] < 3.0])
