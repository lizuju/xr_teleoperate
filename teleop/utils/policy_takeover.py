"""Human takeover of a running ACT policy inside teleop (HG-DAgger).

g starts the policy from wherever the robot is held. Usually that is right after o put it on
the standard start pose. While the policy drives, teleop stays paused: there is no arm IK and no
hand retargeting, and the policy commands the arms, head, waist and both hands. It runs under the
same guards as run_r1_act_policy.py --execute: a blend in from the held pose, a limit on how far
the first arm target may be, fresh output, arm tracking error and a time limit. A guard trip
stops it and the robot holds.

s takes over and records. The policy stops, the pause realigns at the pose the policy left, so
there is no jump, following starts from there, and the episode is saved with info.intervention
describing the policy segment it corrects. r takes over without recording. p or o stops the
policy and holds. Only the human part is recorded. Every policy tick is logged to
policy-runs/<time>_takeover/ticks.jsonl in the run_r1_act_policy.py format, for analysis.
"""
import hashlib
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

POLICY_KEYS = ("g", "G")
ROOT = Path(__file__).resolve().parents[2]
O6_DEX1 = {"left": "linker_o6", "right": "dex1_1"}
FIRST_OUTPUT_SECONDS = 2.0


def policy_tools():
    """The policy runner and replay tool modules (tools/ is not a package)."""
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import replay_r1_episode_on_robot as replay
    import run_r1_act_policy as runner
    return runner, replay


class R1PolicyTakeover:
    """g key state (keyboard thread) and the policy's control ticks (main loop, while paused)."""

    def __init__(self, client, arm, hand, gravity, camera_config, log_dir, logger, *, checkpoint,
                 blend_seconds=2.0, max_seconds=120.0, max_start_delta=0.35, max_chunk_age=0.5,
                 max_tracking_error=0.5, ensemble_k=0.01, ensemble_window=0.0, waist_hold_deg=5.0, waist_capture_deg=2.0,
                 tools=None):
        self.runner, self.replay = tools or policy_tools()
        from teleop.utils.act_runtime import (ActionEnsemble, ObservationBuilder, PolicyGuard, WaistHold, ensemble_window_ticks,
                                              split_action)
        from teleop.utils.frame_sync import CameraSynchronizer

        meta = client.meta
        self.runner.check_policy_metadata(meta)
        self.runner.check_policy_cameras(meta["dataset"], camera_config)
        if meta["dataset"].get("end_effectors") != O6_DEX1:
            raise RuntimeError(f"takeover needs a policy for {O6_DEX1}, not {meta['dataset'].get('end_effectors')}")
        if meta.get("context_dim"):
            raise RuntimeError("takeover does not run phase-conditioned policies")
        for name, value in (("blend_seconds", blend_seconds), ("max_start_delta", max_start_delta),
                            ("ensemble_window", ensemble_window)):
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (("max_seconds", max_seconds), ("max_chunk_age", max_chunk_age),
                            ("max_tracking_error", max_tracking_error)):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        self.client, self.arm, self.hand, self.gravity, self.logger = client, arm, hand, gravity, logger
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.blend_seconds, self.max_seconds, self.max_start_delta = blend_seconds, max_seconds, max_start_delta
        self.split_action, self.WaistHold = split_action, WaistHold
        self.waist_window = (math.radians(waist_hold_deg), math.radians(waist_capture_deg))
        gripper_low, gripper_high = self.runner.validate_contract(meta["dataset"])
        self.new_guard = lambda: PolicyGuard(meta["normalization"], max_tracking_error=max_tracking_error,
                                             max_chunk_age_s=max_chunk_age, gripper_limits=(gripper_low, gripper_high))
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "run.json").write_text(json.dumps({
            "mode": "takeover", "checkpoint": str(self.checkpoint), "checkpoint_sha256": self.checkpoint_sha256,
            "dataset": meta["dataset"], "started_at": time.time(),
            "args": {"checkpoint": str(self.checkpoint), "blend_seconds": blend_seconds, "max_seconds": max_seconds,
                     "max_start_delta": max_start_delta, "max_chunk_age": max_chunk_age,
                     "max_tracking_error": max_tracking_error, "ensemble_k": ensemble_k, "ensemble_window": ensemble_window,
                     "waist_snap_deg": waist_hold_deg, "waist_capture_deg": waist_capture_deg},
        }, indent=2, ensure_ascii=False) + "\n")
        self._frames = (None, {"left": None, "right": None})
        builder = ObservationBuilder(arm, hand, CameraSynchronizer(["head", "left_wrist", "right_wrist"], anchor="head"),
                                     image_hw=tuple(meta["dataset"]["image_size_hw"]))
        self.run = self.runner.PolicyRun(arm, hand, lambda: self._frames, client, builder,
                                         ActionEnsemble(client.chunk_size, ensemble_k,
                                                        ensemble_window_ticks(ensemble_window, float(meta["dataset"]["fps"]))),
                                         self.new_guard(),
                                         self.log_dir / "ticks.jsonl")
        self.run.fps = float(meta["dataset"]["fps"])
        self._lock = threading.Lock()
        self._request = None   # "go", "stop" or "takeover", taken by the main loop
        self._busy = False     # a run is requested or active
        self._active = None    # the running segment
        self._pending = None   # the last segment, until an episode records its correction
        self.segments = 0

    # ---------------------------------------------------------------- keyboard thread
    def press_go(self):
        """g: start a run. False when one is already requested or running."""
        with self._lock:
            if self._busy:
                return False
            self._request, self._busy = "go", True
            return True

    def takeover(self):
        """s or r: stop the run for a human takeover. False when no run is requested or running."""
        return self._stop("takeover")

    def stop(self):
        """p or o: stop the run and hold. False when no run is requested or running."""
        return self._stop("stop")

    def _stop(self, kind):
        with self._lock:
            if not self._busy:
                return False
            self._request, self._busy = kind, False
            return True

    def driving(self):
        with self._lock:
            return self._busy

    def forget(self):
        """The robot left the pose the policy stopped at (o): the next episode is not its correction."""
        with self._lock:
            self._pending = None

    def take_intervention_info(self):
        """Episode info for a recording that starts after a policy segment ended, once; else None."""
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is None:
            return None
        return dict(pending, recording_started_after_s=round(time.time() - pending["ended_at"], 3))

    # ---------------------------------------------------------------- main loop
    def set_frames(self, head, wrists):
        self._frames = (head, dict(wrists))

    def step(self, now, generation):
        """While paused: start, advance or stop the run. True when it commanded the robot this tick."""
        with self._lock:
            request, self._request = self._request, None
        commanded = self._advance(now, generation, request)
        with self._lock:
            # A trip or a new pause ends a run on its own. A key pressed since then is left to
            # the next tick, so a fresh g is not cancelled here.
            if self._request is None:
                self._busy = self._active is not None
        return commanded

    def _advance(self, now, generation, request):
        if self._active is not None:
            if request == "go":
                self._end(now, "restarted by g", warn=False)
            elif request in ("stop", "takeover") or generation != self._active["generation"]:
                reason = {"stop": "stopped by the operator", "takeover": "operator took over"}.get(
                    request, "a new pause (p, o or input loss)")
                self._end(now, reason, warn=False)
                return False
        if request == "go":
            self._begin(now, generation)
        if self._active is None:
            return False
        return self._tick(now)

    def _held_pose(self):
        """The pose the robot holds now, from its requested targets, so the start does not move it."""
        snapshot = self.arm.get_recording_snapshot()
        requested = snapshot["requested"]
        left, right = self.hand.get_action()
        waist = requested.get("waist_q")
        if waist is None:
            waist = snapshot["state"]["waist_q"]
        ReplayPose = self.runner.ReplayPose
        return ReplayPose(arm_q=np.asarray(requested["arm_q"], dtype=np.float64),
                          arm_tau=np.asarray(requested["arm_tau"], dtype=np.float64),
                          head_q=np.asarray(requested["head_q"], dtype=np.float64), waist_q=float(waist),
                          left_q=np.atleast_1d(np.asarray(left, dtype=np.float64)),
                          right_q=np.atleast_1d(np.asarray(right, dtype=np.float64)), left_mode=1, right_mode=1)

    def _begin(self, now, generation):
        hold = self._held_pose()
        guard = self.new_guard()
        self.run.guard = guard
        self.run.reset_predictions()
        start_action = self.run.live_action(hold)
        self.segments += 1
        self._active = {
            "generation": generation, "hold": hold, "start_action": start_action, "guard": guard,
            "waist": self.WaistHold(*self.waist_window, start_action[self.runner.WAIST_INDEX]),
            "started": now, "started_at": time.time(), "first_output_by": now + FIRST_OUTPUT_SECONDS,
            "began": None, "deadline": now + self.max_seconds, "ticks": 0, "segment": self.segments,
        }
        self.run.write(t=now, tick=self.run.tick, mode="takeover_start", segment=self.segments,
                       live=start_action.tolist())
        self.logger.info("[R1 POLICY] run %d: the policy drives from the held pose. s takes over and records, "
                         "r takes over, p stops and holds.", self.segments)

    def _tick(self, now):
        active = self._active
        target, pose = self.run.step()
        if active["began"] is None:
            if target is None:
                if self.client.failed or now > active["first_output_by"]:
                    return self._trip(now, "no policy output after the start")
                self.replay.send_pose(self.arm, self.hand, active["hold"])
                return True
            delta = float(np.abs(self.split_action(target)["arm_q"] - pose.arm_q).max())
            if delta > self.max_start_delta:
                return self._trip(now, f"the first arm target is {delta:.3f} rad from the live pose "
                                       f"(limit {self.max_start_delta:.3f})")
            active["began"] = now
            self.logger.info("[R1 POLICY] commanding; blend %.1f s, limit %.0f s.", self.blend_seconds, self.max_seconds)
        snapshot = self.arm.get_recording_snapshot()
        tracking = float(np.abs(np.asarray(snapshot["requested"]["arm_q"]) - np.asarray(snapshot["state"]["q"])).max())
        guard = active["guard"]
        reason = guard.check(now, self.run.last_chunk_at, tracking, self.client.failed)
        if reason is None and target is None:
            reason = guard.tripped = "the last policy chunk ran out"
        if reason is None and now >= active["deadline"]:
            reason = f"time limit {self.max_seconds:g} s"
        if reason is not None:
            return self._trip(now, reason)
        alpha = (now - active["began"]) / self.blend_seconds if self.blend_seconds > 0 else 1.0
        command = self.runner.blended(active["start_action"], guard.clamp(active["waist"].apply(target)), alpha)
        self.replay.send_pose(self.arm, self.hand, self.runner.pose_from_action(command, self.gravity))
        self.run.write(t=now, tick=self.run.tick, mode="execute", target=target.tolist(), command=command.tolist(),
                       live=self.run.live_action(pose).tolist(), tracking=tracking, segment=active["segment"])
        active["ticks"] += 1
        return True

    def _trip(self, now, reason):
        self._end(now, reason, warn=True)
        return False

    def _end(self, now, reason, warn):
        active, self._active = self._active, None
        commanded = 0.0 if active["began"] is None else now - active["began"]
        self.run.write(t=now, tick=self.run.tick, mode="takeover_stop", segment=active["segment"], reason=reason)
        info = {"policy_checkpoint": str(self.checkpoint), "policy_checkpoint_sha256": self.checkpoint_sha256,
                "policy_log_dir": str(self.log_dir), "policy_segment": active["segment"],
                "policy_started_at": active["started_at"], "policy_seconds": round(now - active["started"], 3),
                "policy_commanded_seconds": round(commanded, 3), "policy_ticks": active["ticks"],
                "end_reason": reason, "ended_at": time.time()}
        with self._lock:
            self._pending = info
        message = (f"[R1 POLICY] run {active['segment']} ended after {commanded:.1f} s commanding: {reason}. "
                   "The robot holds; s takes over and records, r takes over, o returns to the start pose.")
        (self.logger.warning if warn else self.logger.info)(message)

    def close(self):
        """At exit: end a running segment, write summary.json and stop the inference worker."""
        try:
            if self._active is not None:
                self._end(time.monotonic(), "teleop exited", warn=False)
            summary = self.run.summary()
            summary["segments"] = self.segments
            (self.log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        finally:
            try:
                self.run.log.close()
            finally:
                self.client.close()


def create_policy_takeover(args, arm, hand, gravity, camera_config, logger, log_root):
    """Start the inference worker for --policy and wrap it for the g key."""
    runner, replay = policy_tools()
    from teleop.utils.act_runtime import PolicyClient

    logger.info("[R1 POLICY] loading %s for takeover (g runs it) ...", args.policy)
    client = PolicyClient([str(args.policy_worker_python), str(ROOT / "tools/act_inference_worker.py"),
                           str(args.policy), "--device", args.policy_device])
    try:
        return R1PolicyTakeover(client, arm, hand, gravity, camera_config,
                                Path(log_root) / (time.strftime("%Y%m%d_%H%M%S") + "_takeover"), logger,
                                checkpoint=args.policy, blend_seconds=args.policy_blend_seconds,
                                max_seconds=args.policy_max_seconds,
                                ensemble_window=getattr(args, "policy_ensemble_window", 0.0), tools=(runner, replay))
    except Exception:
        client.close()
        raise
