import hashlib
import math

import numpy as np

PHASES = ["grasp_cup", "turn_deliver", "dispense", "close_retract", "turn_place"]
CONTEXT_DIM = len(PHASES) + 3


def source_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_context(phase, elapsed_s, locks=(False, False)):
    context = np.zeros(CONTEXT_DIM, dtype=np.float32)
    context[phase] = 1
    context[5:] = [min(max(elapsed_s, 0.0), 60.0) / 60.0, *locks]
    return context


def validate_annotation(annotation, directory, count, require_review=True):
    if require_review and annotation.get("reviewed") is not True:
        raise ValueError(f"{directory.name}: phase annotation needs human review")
    if annotation.get("frames_sha256") != source_digest(directory / "frames.jsonl"):
        raise ValueError(f"{directory.name}: phase annotation belongs to different frames")
    boundaries = annotation.get("boundaries")
    # Equal neighbours mark a phase the recording skips: a DAgger correction that starts at the lever or at the
    # final place has empty leading phases (run 13, 2026-11-26).
    if (not isinstance(boundaries, list) or len(boundaries) != len(PHASES) + 1
            or any(type(v) is not int for v in boundaries) or boundaries[0] != 0 or count < 1
            or boundaries[-1] != count or any(a > b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError(f"{directory.name}: need six non-decreasing frame boundaries from 0 to {count}")
    return boundaries


class IcecreamTask:
    """Phase transitions are confirmed by the operator or reviewed-model completion plus feedback."""

    def __init__(self, gripper_limits, auto=False, dispense_s=3.0):
        if not math.isfinite(dispense_s) or dispense_s < 0:
            raise ValueError("dispense_s must be finite and non-negative")
        self.gripper_limits, self.auto, self.dispense_s = gripper_limits, auto, dispense_s
        self.phase, self.started, self.completed_since = 0, None, None
        self.done, self.hold_pose, self.front = False, None, None
        self.events = []

    def start(self, now, pose):
        self.started = now
        self.front = float(pose[21])

    def context(self, now):
        return phase_context(self.phase, 0 if self.started is None else now - self.started,
                             (False, self.phase in (2, 3)))

    def advance(self, now, pose, reason="operator_confirmed"):
        if self.done:
            return
        self.events.append({"time": now, "phase": PHASES[self.phase], "reason": reason})
        if self.phase == len(PHASES) - 1:
            self.done = True
            return
        self.phase += 1
        self.started, self.completed_since = now, None
        if self.phase == 2:
            self.hold_pose = np.asarray(pose, dtype=np.float64).copy()
        elif self.phase == 4:
            self.hold_pose = None

    def observe(self, now, pose, completion):
        if not self.auto or self.done or self.started is None:
            return False
        low, high = self.gripper_limits
        ready = completion is not None and completion >= 0.9 and now - self.started >= 0.5
        if self.phase == 0:
            ready &= pose[20] < low + 0.3
        elif self.phase == 1:
            ready &= abs(pose[21] - (self.front - math.pi / 2)) < math.radians(8)
        elif self.phase == 2:
            ready &= now - self.started >= self.dispense_s
        elif self.phase == 4:
            ready &= abs(pose[21] - self.front) < math.radians(8) and pose[20] > high - 0.15
        self.completed_since = (now if self.completed_since is None else self.completed_since) if ready else None
        if self.completed_since is not None and now - self.completed_since >= 0.3:
            self.advance(now, pose, "model_and_feedback")
            return True
        return False

    def apply(self, target):
        target = np.asarray(target, dtype=np.float64).copy()
        if self.hold_pose is not None:
            target[7:14], target[20:22] = self.hold_pose[7:14], self.hold_pose[20:22]
        return target
