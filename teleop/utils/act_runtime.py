"""Runtime pieces for executing an ACT policy. No torch: this runs in the teleop environment.

The model itself lives in a separate worker process (tools/act_inference_worker.py)
so inference can use the GPU environment and can never stall the control loop.
"""

from collections import deque
import json
import math
from pathlib import Path
import queue
import select
import struct
import subprocess
import threading

import cv2
import numpy as np

from teleop.utils.frame_sync import feedback_target_ns
from teleop.utils.act_contract import ACTION_DIM, GROUP_SLICES


IMAGE_HW = (544, 640)
# Same limits the exporter applied to every training row.
OBSERVATION_SKEW_MS = 25.0
OBSERVATION_AGE_MS = 150.0
CLOCK_UNCERTAINTY_NS = 5_000_000
CLOCK_MAX_AGE_NS = 15_000_000_000


def prepare_training_image(bgr, image_hw=IMAGE_HW):
    """The one image transform shared by the exporter and live inference."""
    height, width = image_hw
    scale = min(height / bgr.shape[0], width / bgr.shape[1])
    resized = cv2.resize(bgr, (max(1, round(bgr.shape[1] * scale)), max(1, round(bgr.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    pixels[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)


def split_action(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (ACTION_DIM,):
        raise ValueError(f"expected {ACTION_DIM} action values, got {values.shape}")
    return {"arm_q": values[:14], "left_q": values[14:20], "right_q": values[20:21],
            "waist_q": float(values[21]), "head_q": values[22:24]}


def join_action(arm_q, left_q, right_q, waist_q, head_q):
    return np.concatenate((arm_q, left_q, right_q, [waist_q], head_q)).astype(np.float64)


def group_errors(difference):
    return {name: float(np.abs(difference[part]).max()) for name, part in GROUP_SLICES.items()}


# Teleop turns the waist only with keys: the arrows step between stops 90 deg apart (counted from
# the activation heading, which the recordings keep within about 1 deg of 0), and the number keys
# go straight to 0, +-90 or +-148 deg (r1_head_waist.WAIST_ANGLE_KEYS). So every recorded waist
# action is a stop or part of a turn between two stops.
WAIST_STOPS = (0.0, -math.pi / 2, math.pi / 2, -math.radians(148.0), math.radians(148.0))
WAIST_INDEX = GROUP_SLICES["body"].start


def nearest_waist_stop(waist, window):
    """The waist stop within window radians of waist, or None."""
    stop = min(WAIST_STOPS, key=lambda value: abs(waist - value))
    return stop if abs(waist - stop) <= window else None


class WaistHold:
    """Holds the commanded waist on a waist stop (0 or +-90 deg) until the policy turns away.

    ACT reproduces the stops with a few degrees of drift (-86 deg instead of -90 while pulling the
    lever). While a stop is held, a policy target within release radians of it is commanded as
    exactly the stop. A target further out ends the hold and passes unchanged: the policy begins a
    turn by creeping off the stop (-2, -4, -6 deg) and commits to it only once it sees the waist
    move, so holding until the turn was further along kept the robot facing forward. A target
    within capture radians of a stop starts a new hold, and so does a starting waist that close.
    release 0 disables the hold.
    """

    def __init__(self, release, capture, waist):
        if not (math.isfinite(release) and 0.0 <= release < math.pi / 4):
            raise ValueError("waist hold release must be at least 0 and under 45 deg")
        if not (math.isfinite(capture) and capture >= 0.0 and (release == 0.0 or capture <= release)):
            raise ValueError("waist hold capture must be at least 0 and no more than release")
        self.release, self.capture = release, capture
        self.stop = nearest_waist_stop(float(waist), capture) if release > 0.0 else None

    def apply(self, action):
        """A copy of action with its waist target on the held stop, if any. Nothing else changes."""
        action = np.array(action, dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"expected {ACTION_DIM} action values, got {action.shape}")
        if self.release == 0.0:
            return action
        waist = float(action[WAIST_INDEX])
        if self.stop is not None and abs(waist - self.stop) > self.release:
            self.stop = None
        if self.stop is None:
            self.stop = nearest_waist_stop(waist, self.capture)
        if self.stop is not None:
            action[WAIST_INDEX] = self.stop
        return action


def ensemble_window_ticks(seconds, fps):
    """An ensemble window in seconds as control ticks: 0 stays off, any positive window keeps at least one tick."""
    return 0 if seconds <= 0 else max(1, int(round(seconds * fps)))


class ActionEnsemble:
    """ACT temporal ensembling: every chunk covering a tick votes, older chunks weigh more.

    window > 0 lets only the chunks started in the last `window` ticks vote (the newest covering chunk always does),
    so a change of plan reaches the target within that window instead of being averaged against a whole chunk
    length of older plans. 0 keeps every covering chunk, the ACT default."""

    def __init__(self, chunk_size, k=0.01, window=0):
        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise ValueError(f"window must be a whole number of ticks, at least 0, not {window!r}")
        self.chunk_size, self.k, self.window = chunk_size, k, window
        self.chunks = deque()

    def reset(self):
        self.chunks.clear()

    def add(self, start_tick, chunk):
        if len(chunk) != self.chunk_size:
            raise ValueError(f"expected a chunk of {self.chunk_size}, got {len(chunk)}")
        self.chunks.append((start_tick, chunk))

    def get(self, tick):
        while self.chunks and self.chunks[0][0] + self.chunk_size <= tick:
            self.chunks.popleft()
        covering = [(start, chunk) for start, chunk in self.chunks if 0 <= tick - start]
        if self.window:
            covering = [(start, chunk) for start, chunk in covering if tick - start < self.window] or covering[-1:]
        votes = [chunk[tick - start] for start, chunk in covering]
        if not votes:
            return None
        weights = np.exp(-self.k * np.arange(len(votes)))
        return (np.asarray(votes, dtype=np.float64) * (weights / weights.sum())[:, None]).sum(axis=0)


def rtc_weights(chunk_size, delay, overlap):
    """Real-time chunking guidance weights for a chunk requested while `overlap` steps of the current one remain.

    The first `delay` steps run from the current chunk while the new one is computed, so they must match it (1).
    From there the weight falls exponentially to 0 at the end of the current chunk (the soft mask of Black et al.
    2025, c (e^c - 1) / (e - 1)), and the steps after it are left to the policy (0)."""
    weights = np.zeros(chunk_size, dtype=np.float32)
    overlap = max(0, min(int(overlap), chunk_size))
    delay = max(0, min(int(delay), overlap))
    weights[:delay] = 1.0
    if overlap > delay:
        c = (overlap - np.arange(delay, overlap)) / (overlap - delay + 1)
        weights[delay:overlap] = c * np.expm1(c) / (math.e - 1)
    return weights


def smooth_chunk(chunk, window):
    """Centered moving average along a chunk's own time axis (edge steps repeated), so it adds no lag.
    window is a whole, odd number of steps; 1 leaves the chunk as it is."""
    chunk = np.asarray(chunk, dtype=np.float64)
    if window <= 1:
        return chunk
    pad = window // 2
    padded = np.concatenate((np.repeat(chunk[:1], pad, axis=0), chunk, np.repeat(chunk[-1:], pad, axis=0)))
    sums = np.cumsum(np.concatenate((np.zeros((1, chunk.shape[1])), padded)), axis=0)
    return (sums[window:] - sums[:-window]) / window


def odd_ticks(seconds, fps):
    """A smoothing window in seconds as an odd number of ticks; 0 s is 1 tick (no smoothing)."""
    ticks = max(1, int(round(seconds * fps)))
    return ticks if ticks % 2 else ticks + 1


class ChunkExecutor:
    """Chunk execution instead of temporal ensembling: every tick follows the newest chunk alone, and a new chunk
    is requested every `interval` ticks. A flow policy's chunk jitters from step to step, which ensembling used to
    average away, so each chunk is smoothed along its own time axis (`smooth` steps, centered, no lag) and the
    first `fade` ticks after a switch crossfade from the old chunk to the new one.

    guide_steps turns on real-time chunking (Black, Galliker and Levine 2025): each request also carries the rest
    of the current chunk, and the policy is guided to continue it over the frozen delay plus guide_steps steps.
    None sends no guide. Same add/get/reset as ActionEnsemble."""

    def __init__(self, chunk_size, interval, action_dim=ACTION_DIM, smooth=1, fade=0, guide_steps=None):
        def whole(value, low, high, name):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be a whole number of ticks from {low} to {high}, not {value!r}")
        whole(interval, 1, chunk_size - 1, "interval")
        whole(smooth, 1, chunk_size, "smooth")
        if smooth % 2 == 0:
            raise ValueError(f"smooth must be odd, not {smooth}")
        whole(fade, 0, chunk_size - 1, "fade")
        if guide_steps is not None:
            whole(guide_steps, 0, chunk_size, "guide_steps")
        self.chunk_size, self.interval, self.action_dim = chunk_size, interval, action_dim
        self.smooth, self.fade, self.guide_steps = smooth, fade, guide_steps
        self.reset()

    @property
    def guided(self):
        return self.guide_steps is not None

    def reset(self):
        self.current = self.previous = None
        self.requested_at = self.switched_at = None
        self._switching = False

    def add(self, start_tick, chunk):
        if len(chunk) != self.chunk_size:
            raise ValueError(f"expected a chunk of {self.chunk_size}, got {len(chunk)}")
        if self.current is None or start_tick >= self.current[0]:
            self.previous = self.current
            self.current = (start_tick, smooth_chunk(chunk, self.smooth))
            self._switching = True

    @property
    def source(self):
        """Request tick of the chunk being followed, or None."""
        return None if self.current is None else self.current[0]

    def get(self, tick):
        if self.current is None:
            return None
        if self._switching:
            self.switched_at, self._switching = tick, False
        start, chunk = self.current
        if not 0 <= tick - start < self.chunk_size:
            return None
        value = chunk[tick - start].copy()
        since = tick - self.switched_at
        if self.previous is not None and 0 <= since < self.fade:
            old_start, old = self.previous
            if 0 <= tick - old_start < self.chunk_size:
                alpha = (since + 1) / (self.fade + 1)
                value = (1 - alpha) * old[tick - old_start] + alpha * value
        return value

    def due(self, tick):
        """Whether a request should go out at this tick: at once without a chunk, then every interval ticks."""
        return self.current is None or self.requested_at is None or tick - self.requested_at >= self.interval

    def guide(self, tick, delay):
        """(prefix, weights) for a request sent at `tick` whose chunk is expected `delay` ticks later.
        Without a current chunk the weights are all 0, which samples exactly as without guidance."""
        prefix = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        if self.current is None:
            return prefix, np.zeros(self.chunk_size, dtype=np.float32)
        start, chunk = self.current
        overlap = max(0, self.chunk_size - max(0, tick - start))
        if overlap:
            prefix[:overlap] = chunk[self.chunk_size - overlap:]
        span = overlap if self.guide_steps is None else min(overlap, delay + self.guide_steps)
        return prefix, rtc_weights(self.chunk_size, delay, span)

    def sent(self, tick):
        self.requested_at = tick


class ObservationBuilder:
    """Rebuilds one training row from live data, under the same rules as the exporter."""

    def __init__(self, arm, hand, sync, image_hw=IMAGE_HW):
        self.arm, self.hand, self.sync, self.image_hw = arm, hand, sync, image_hw
        self._prepared = {}

    def observe(self, head, wrists):
        self.sync.offer("head", head)
        for side, frame in wrists.items():
            self.sync.offer(f"{side}_wrist", frame)

    def _image(self, name, sequence, bgr):
        cached = self._prepared.get(name)
        if cached is None or cached[0] != sequence:
            cached = (sequence, prepare_training_image(bgr, self.image_hw))
            self._prepared[name] = cached
        return cached[1]

    def build(self, now_ns):
        """(qpos, images, None) when usable, else (None, None, reason)."""
        pairing = self.sync.pair(now_ns, complete_only=True)
        if pairing is None:
            return None, None, "cameras_incomplete"
        if pairing.timestamp_basis != "mapped_source":
            return None, None, "camera_clock_invalid"
        stamps = []
        for frame in pairing.frames.values():
            timing = getattr(frame.payload, "timing", None) or {}
            measured = timing.get("clock_measured_monotonic_ns")
            uncertainty = timing.get("clock_uncertainty_ns")
            if (timing.get("clock_valid") is not True or type(measured) is not int or type(uncertainty) is not int
                    or not 0 <= now_ns - measured <= CLOCK_MAX_AGE_NS or uncertainty > CLOCK_UNCERTAINTY_NS):
                return None, None, "camera_clock_invalid"
            stamps.append(timing["mapped_monotonic_ns"])
        head = pairing.frames["head"].payload
        stereo_skew = head.timing.get("stereo_skew_ns")
        if type(stereo_skew) is not int or stereo_skew < 0:
            return None, None, "stereo_unaligned"
        stamps.append(head.timing["mapped_monotonic_ns"] + stereo_skew)
        target_ns = feedback_target_ns(pairing)
        robot = self.arm.get_recording_samples(target_ns, None, now_ns)["nearest"]
        hands = self.hand.get_recording_states_at(target_ns, now_ns)
        if robot is None or hands.get("left") is None or hands.get("right") is None:
            return None, None, "feedback_missing"
        stamps += [robot["monotonic_ns"], hands["left"]["monotonic_ns"], hands["right"]["monotonic_ns"]]
        if (max(stamps) - min(stamps)) / 1e6 > OBSERVATION_SKEW_MS:
            return None, None, "observation_skew"
        if not 0 <= (now_ns - min(stamps)) / 1e6 <= OBSERVATION_AGE_MS or max(stamps) > now_ns:
            return None, None, "observation_age"
        width = head.bgr.shape[1] // 2
        wrists = [pairing.frames[f"{side}_wrist"].payload for side in ("left", "right")]
        images = np.stack((self._image("head_left", head.sequence, head.bgr[:, :width]),
                           self._image("head_right", head.sequence, head.bgr[:, width:]),
                           self._image("left_wrist", wrists[0].sequence, wrists[0].bgr),
                           self._image("right_wrist", wrists[1].sequence, wrists[1].bgr)))
        qpos = np.array([*robot["q"], *hands["left"]["q"], *hands["right"]["q"], robot["waist_q"], *robot["head_q"]],
                        dtype=np.float32)
        if qpos.shape != (ACTION_DIM,) or not np.isfinite(qpos).all():
            return None, None, "feedback_invalid"
        return qpos, images, None


class PolicyGuard:
    """Decides, each tick, whether the policy may keep commanding. A trip latches until exit."""

    def __init__(self, normalization, range_margin=0.1, max_tracking_error=0.5, tracking_hold_s=0.5,
                 max_chunk_age_s=0.5, gripper_limits=(0.0, 5.4)):
        low, high = (np.asarray(normalization[key], dtype=np.float64) for key in ("action_min", "action_max"))
        margin = (high - low) * range_margin + 0.02
        self.low, self.high = low - margin, high + margin
        self.low[GROUP_SLICES["left_ee"]] = np.maximum(self.low[GROUP_SLICES["left_ee"]], 0.0)
        self.low[GROUP_SLICES["right_ee"]] = np.maximum(self.low[GROUP_SLICES["right_ee"]], gripper_limits[0])
        self.high[GROUP_SLICES["left_ee"]] = np.minimum(self.high[GROUP_SLICES["left_ee"]], 1.0)
        self.high[GROUP_SLICES["right_ee"]] = np.minimum(self.high[GROUP_SLICES["right_ee"]], gripper_limits[1])
        if np.any(self.low > self.high):
            raise ValueError("Training action range does not intersect the configured gripper limits")
        qlow, qhigh = (np.asarray(normalization[key], dtype=np.float64) for key in ("qpos_min", "qpos_max"))
        qmargin = (qhigh - qlow) * range_margin + 0.05
        self.qpos_low, self.qpos_high = qlow - qmargin, qhigh + qmargin
        self.max_tracking_error, self.tracking_hold_s = max_tracking_error, tracking_hold_s
        self.max_chunk_age_s = max_chunk_age_s
        self.tracking_since = None
        self.clamped_ticks = 0
        self.tripped = None

    def clamp(self, target):
        clamped = np.clip(target, self.low, self.high)
        self.clamped_ticks += bool(np.any(clamped != target))
        return clamped

    def out_of_training_range(self, qpos):
        return [int(index) for index in np.flatnonzero((qpos < self.qpos_low) | (qpos > self.qpos_high))]

    def check(self, now, last_chunk_at, tracking_error, worker_failed=None):
        """Returns the reason to stop commanding, or None."""
        if self.tripped is not None:
            return self.tripped
        reason = None
        if worker_failed:
            reason = f"inference worker failed: {worker_failed}"
        elif last_chunk_at is None or now - last_chunk_at > self.max_chunk_age_s:
            reason = f"no fresh policy output for over {self.max_chunk_age_s:.2f}s"
        elif tracking_error > self.max_tracking_error:
            if self.tracking_since is None:
                self.tracking_since = now
            if now - self.tracking_since >= self.tracking_hold_s:
                reason = (f"arm tracking error {tracking_error:.3f} rad stayed above "
                          f"{self.max_tracking_error:.3f} rad for {self.tracking_hold_s:.2f}s")
        else:
            self.tracking_since = None
        self.tripped = reason
        return reason


class GravityModel:
    """Same feed-forward the IK computes for a target: RNEA at rest, waist and head locked."""

    def __init__(self, urdf=None):
        import pinocchio as pin
        self.pin = pin
        urdf = Path(__file__).resolve().parents[2] / "assets/r1/r1_a7.urdf" if urdf is None else urdf
        full = pin.buildModelFromUrdf(str(urdf))
        locked = [full.getJointId(name) for name in ("waist_yaw_joint", "head_pitch_joint", "head_yaw_joint")]
        self.model = pin.buildReducedModel(full, locked, np.zeros(full.nq))
        self.data = self.model.createData()
        self.zero = np.zeros(self.model.nv)

    def __call__(self, arm_q):
        return np.array(self.pin.rnea(self.model, self.data, np.asarray(arm_q, dtype=np.float64), self.zero, self.zero))


def guide_size(guide_shape):
    """Bytes of a real-time chunking guide: the prefix (chunk x action) and its weights (chunk), float32."""
    return 0 if guide_shape is None else (guide_shape[0] * guide_shape[1] + guide_shape[0]) * 4


def request_size(state_dim, cameras, image_hw, context_dim=0, guide_shape=None):
    """guide_shape (chunk, action) for a real-time chunking worker, whose every request ends with a guide."""
    return (state_dim + context_dim) * 4 + cameras * image_hw[0] * image_hw[1] * 3 + guide_size(guide_shape)


def response_size(chunk_size, action_dim):
    return 16 + chunk_size * action_dim * 4


def encode_request(qpos, images, context=None, guide=None):
    """guide = (prefix, weights) is appended after the images, for a real-time chunking worker."""
    task = b"" if context is None else np.asarray(context, dtype=np.float32).tobytes()
    tail = b"" if guide is None else (np.asarray(guide[0], dtype=np.float32).tobytes()
                                      + np.asarray(guide[1], dtype=np.float32).tobytes())
    return np.asarray(qpos, dtype=np.float32).tobytes() + task + np.asarray(images, dtype=np.uint8).tobytes() + tail


def decode_request(payload, state_dim, cameras, image_hw, context_dim=0):
    split = state_dim * 4
    end = split + context_dim * 4
    image_end = end + cameras * image_hw[0] * image_hw[1] * 3
    return (np.frombuffer(payload[:split], dtype=np.float32).copy(),
            np.frombuffer(payload[end:image_end], dtype=np.uint8).reshape(cameras, *image_hw, 3).copy(),
            np.frombuffer(payload[split:end], dtype=np.float32).copy())


def decode_guide(payload, chunk_size, action_dim):
    """The (prefix, weights) a real-time chunking request ends with."""
    tail = payload[len(payload) - guide_size((chunk_size, action_dim)):]
    split = chunk_size * action_dim * 4
    return (np.frombuffer(tail[:split], dtype=np.float32).reshape(chunk_size, action_dim).copy(),
            np.frombuffer(tail[split:], dtype=np.float32).copy())


def encode_response(inference_ms, chunk, completion=None):
    return struct.pack("<dd", inference_ms, -1.0 if completion is None else completion) + np.asarray(chunk, dtype=np.float32).tobytes()


def decode_response(payload, chunk_size, action_dim):
    ms, complete = struct.unpack("<dd", payload[:16])
    return (ms, np.frombuffer(payload[16:], dtype=np.float32).reshape(chunk_size, action_dim).astype(np.float64),
            None if complete < 0 else complete)


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        part = stream.read(size - len(data))
        if not part:
            return None
        data += part
    return bytes(data)


class PolicyClient:
    """Non-blocking handle on the inference worker. A hung worker can never block the caller."""

    def __init__(self, command, startup_timeout=120.0):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        readable, _, _ = select.select([self.process.stdout], [], [], startup_timeout)
        line = self.process.stdout.readline() if readable else b""
        if not line:
            self.process.kill()
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            raise RuntimeError("inference worker did not start")
        self.meta = json.loads(line)
        self.chunk_size = self.meta["chunk_size"]
        self.action_dim = self.meta["action_dim"]
        self._outbox = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._responses = deque()
        self._pending = None
        self.failed = None
        self._writer_thread = threading.Thread(target=self._writer, name="act-worker-writer", daemon=True)
        self._reader_thread = threading.Thread(target=self._reader, name="act-worker-reader", daemon=True)
        self._writer_thread.start()
        self._reader_thread.start()

    def _writer(self):
        try:
            while True:
                payload = self._outbox.get()
                if payload is None:
                    return
                self.process.stdin.write(payload)
                self.process.stdin.flush()
        except (OSError, ValueError) as error:
            self.failed = self.failed or f"write failed: {error}"

    def _reader(self):
        size = response_size(self.chunk_size, self.action_dim)
        while True:
            try:
                payload = read_exact(self.process.stdout, size)
            except (OSError, ValueError) as error:
                self.failed = self.failed or f"read failed: {error}"
                return
            if payload is None:
                self.failed = self.failed or f"worker exited with status {self.process.poll()}"
                return
            inference_ms, chunk, complete = decode_response(payload, self.chunk_size, self.action_dim)
            with self._lock:
                self._responses.append((self._pending, inference_ms, chunk, complete))
                self._pending = None

    @property
    def idle(self):
        with self._lock:
            return self._pending is None and not self._responses

    def send(self, tick, sent_at, qpos, images, context=None, guide=None):
        with self._lock:
            self._pending = (tick, sent_at)
        self._outbox.put_nowait(encode_request(qpos, images, context, guide))

    def poll(self):
        """((tick, sent_at), inference_ms, chunk, completion) for a finished request, else None."""
        with self._lock:
            return self._responses.popleft() if self._responses else None

    def close(self):
        try:
            self._outbox.put_nowait(None)
        except queue.Full:
            pass
        # Terminate before closing a pipe whose writer may be blocked on a full image payload.
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3.0)
        self._writer_thread.join(timeout=1.0)
        self._reader_thread.join(timeout=1.0)
        self.process.stdin.close()
        self.process.stdout.close()


def summarize(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {"count": len(ordered), "p50": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], "max": ordered[-1]}
