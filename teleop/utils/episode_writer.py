from copy import deepcopy
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread

import cv2
import numpy as np


logger = logging.getLogger(__name__)
CLOSE_TIMEOUT = 5.0
OUTCOMES = {"unspecified", "success", "failure", "discarded"}


class EpisodeWriter:
    def __init__(self, task_dir, task_goal=None, task_desc=None, task_steps=None,
                 frequency=30, image_size=(640, 480), depth_size=None, rerun_log=True, metadata=None,
                 queue_capacity=60, quality_report=False):
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self.task_dir = Path(task_dir)
        self.text = {
            "goal": task_goal if task_goal is not None else "Pick up the red cup on the table.",
            "desc": task_desc if task_desc is not None else "task description",
            "steps": task_steps if task_steps is not None else "step1: do this; step2: do that; ...",
        }
        self.info = {
            "version": "2.0.0",
            "date": datetime.date.today().isoformat(),
            "author": "unitree",
            "image": {"width": image_size[0], "height": image_size[1], "fps": frequency},
            # Declared only when depth frames are actually recorded. A manifest
            # that advertises depth while every sample carries depths=null makes
            # downstream consumers believe a modality is present that is not.
            "depth": None if depth_size is None else {"width": depth_size[0], "height": depth_size[1],
                                                      "fps": frequency},
            "audio": {"sample_rate": 16000, "channels": 1, "format": "PCM", "bits": 16},
            "joint_names": {name: [] for name in ("left_arm", "left_ee", "right_arm", "right_ee", "body")},
            "tactile_names": {"left_ee": [], "right_ee": []},
            "sim_state": "",
        }
        self.info.update(deepcopy(metadata or {}))
        self._base_info = deepcopy(self.info)
        self.rerun_log = rerun_log
        self.rerun_logger = None
        self.item_data_queue = Queue(queue_capacity)
        self._lock = Lock()
        self._state = "idle"
        self._create_pending = False
        self._closed = False
        self._error = None
        self.quality_report = quality_report
        self._quality_process = None
        self._outcome = "unspecified"
        self._started_at = None
        self._saved_at = None
        self.item_id = -1
        self.episode_id = -1
        self.episode_dir = None
        self._frames = None
        self._frame_count = 0
        #: (field, colour key, source sequence) -> the relative path already written,
        #: plus the counters the final manifest reports the measured rates from.
        self._image_paths = {}
        self._written_images = {}
        self._reused_images = 0
        self._first_timestamp_ns = None
        self._last_timestamp_ns = None
        self.worker_thread = Thread(target=self.process_queue, name="episode-writer", daemon=True)
        self.worker_thread.start()

    def raise_if_failed(self):
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Episode recording failed: {error}") from error

    def is_ready(self):
        self.raise_if_failed()
        with self._lock:
            return self._state == "idle" and not self._closed

    def create_episode(self):
        self.raise_if_failed()
        with self._lock:
            if self._closed or self._state != "idle":
                return False
            self._state = "recording"
            self._create_pending = True
            self._outcome = "unspecified"
            self._started_at = datetime.datetime.now().astimezone()
            self._saved_at = None
            self.item_id = -1
        return True

    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None,
                 audios=None, sim_state=None, sample=None, color_sequences=None):
        """Queue one sample.

        ``color_sequences`` maps a colour key to the source frame's sequence number.
        A key whose sequence was already written in this episode is stored once and
        referenced again, which is what keeps a 40 Hz sample loop over a 10 Hz camera
        from writing the same JPEG four times. Keys without a sequence are always
        written.
        """
        self.raise_if_failed()
        with self._lock:
            if self._closed or self._state != "recording":
                raise RuntimeError("No episode is accepting frames")
            if self.item_data_queue.full():
                self._error = RuntimeError("recording queue is full; episode is incomplete")
            else:
                self.item_id += 1
                # The producer may reuse image buffers or nested state dictionaries.
                item = deepcopy({
                    "idx": self.item_id,
                    "colors": colors,
                    "depths": depths,
                    "states": states,
                    "actions": actions,
                    "tactiles": tactiles,
                    "audios": audios,
                    "sim_state": sim_state,
                    "sample": sample,
                    "color_sequences": color_sequences,
                })
                self.item_data_queue.put_nowait(item)
        self.raise_if_failed()

    def save_episode(self, outcome="unspecified"):
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
        self.raise_if_failed()
        with self._lock:
            if self._state == "recording":
                self._outcome = outcome
                self._state = "saving"

    def abort(self, error):
        with self._lock:
            if self._state in ("recording", "saving") and self._error is None:
                self._error = error if isinstance(error, Exception) else RuntimeError(str(error))

    def _start_episode(self):
        self.episode_dir = None
        self.task_dir.mkdir(parents=True, exist_ok=True)
        existing = [int(path.name[8:].split("_", 1)[0]) for path in self.task_dir.iterdir()
                    if path.is_dir() and path.name.startswith("episode_")
                    and path.name[8:].split("_", 1)[0].isdigit()]
        self.episode_id = max(existing, default=-1) + 1
        episode_dir = self.task_dir / f"episode_{self.episode_id:04d}"
        episode_dir.mkdir()
        self.episode_dir = episode_dir
        self.info = deepcopy(self._base_info)
        self.info["date"] = self._started_at.date().isoformat()
        self._frame_count = 0
        self._image_paths = {}
        self._written_images = {}
        self._reused_images = 0
        self._first_timestamp_ns = None
        self._last_timestamp_ns = None
        for name in ("colors", "depths", "audios"):
            if name == "depths" and self.info.get("depth") is None:
                continue
            (episode_dir / name).mkdir()
        self._frames = (episode_dir / "frames.jsonl").open("x", encoding="utf-8")
        self._write_manifest("recording")
        self.raise_if_failed()
        if self.rerun_log and self.rerun_logger is None:
            from .rerun_visualizer import RerunLogger
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary=60, memory_limit="300MB")
        logger.info("Recording episode: %s", episode_dir)

    def _write_manifest(self, status):
        with self._lock:
            outcome, error = self._outcome, self._error
        manifest = {
            "schema": "xr_teleop_episode_v2",
            "status": status,
            "info": self.info,
            "text": self.text,
            "episode_id": self.episode_id,
            "frame_count": self._frame_count,
            "frames": "frames.jsonl",
            "outcome": outcome,
            "started_at": self._started_at.isoformat(),
            "saved_at": self._saved_at.isoformat() if self._saved_at is not None else None,
        }
        if error is not None:
            manifest["error"] = str(error)
        temporary = self.episode_dir / ".episode.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.episode_dir / "episode.json")

    def _process_item_data(self, item):
        idx = item["idx"]
        sequences = item.pop("color_sequences", None) or {}
        sample = item.get("sample") or {}
        timestamp = sample.get("timestamp_ns")
        if isinstance(timestamp, int) and not isinstance(timestamp, bool):
            if self._first_timestamp_ns is None:
                self._first_timestamp_ns = timestamp
            self._last_timestamp_ns = timestamp
        for field in ("colors", "depths"):
            for key, image in (item[field] or {}).items():
                # A modality key may be present with a null payload: the camera
                # is configured for this run but did not deliver a usable frame
                # for this sample. Keep the key so the schema stays stable and
                # the matching sample.sources entry stays checkable.
                if image is None:
                    continue
                sequence = sequences.get(key) if field == "colors" else None
                if sequence is not None:
                    previous = self._image_paths.get((field, key, int(sequence)))
                    if previous is not None:
                        # Same camera frame as an earlier sample, so it is already
                        # byte for byte on disk. Point at it instead of encoding and
                        # writing a second copy; sample.sources still carries the
                        # sequence and the repeated flag for consumers that need to
                        # tell a reused frame from a fresh one.
                        item[field][key] = previous
                        self._reused_images += 1
                        continue
                suffix = ".jpg" if field == "colors" else ".png"
                relative = Path(field) / f"{idx:06d}_{key}{suffix}"
                # `depths/` only exists when the manifest declares depth, so a
                # legacy caller that supplies one anyway still lands on disk.
                (self.episode_dir / field).mkdir(exist_ok=True)
                if not cv2.imwrite(str(self.episode_dir / relative), image):
                    raise OSError(f"Failed to save {relative}")
                item[field][key] = str(relative)
                self._written_images[key] = self._written_images.get(key, 0) + 1
                if sequence is not None:
                    self._image_paths[(field, key, int(sequence))] = str(relative)
        for key, audio in (item["audios"] or {}).items():
            relative = Path("audios") / f"audio_{idx:06d}_{key}.npy"
            np.save(self.episode_dir / relative, audio.astype(np.int16))
            item["audios"][key] = str(relative)
        line = json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self._frames.write(line + "\n")
        self._frames.flush()
        self._frame_count += 1
        if self.rerun_logger is not None:
            self.rerun_logger.log_item_data(item)

    def _record_measured_image_rates(self):
        """Replace the declared frame rates with the ones the episode holds.

        The camera config states what the publisher is asked for, not what the
        device delivers: the head stereo stream runs near 10 Hz against a
        configured 30, and a 40 Hz sample loop reuses each of those frames about
        four times. A manifest that repeats the configured number makes every
        downstream timeline wrong, so the rate is measured from the images that
        were actually written, and the configured value is kept beside it.
        """
        if self._first_timestamp_ns is None or self._last_timestamp_ns is None:
            return
        duration_s = (self._last_timestamp_ns - self._first_timestamp_ns) / 1e9
        if duration_s <= 0.0 or not self._written_images:
            return
        images = self.info.get("images")
        if not isinstance(images, dict):
            images = {}
        for key, written in self._written_images.items():
            spec = images.get(key)
            if not isinstance(spec, dict):
                continue
            spec.setdefault("declared_fps", spec.get("fps"))
            spec["fps"] = written / duration_s
            spec["unique_frames"] = written
            spec["frames"] = self._frame_count
        head = images.get("color_0")
        if not isinstance(head, dict):
            head = None
        if isinstance(self.info.get("image"), dict):
            if head is not None:
                self.info["image"]["declared_fps"] = self.info["image"].get("fps")
                self.info["image"]["fps"] = head["fps"]
            elif "color_0" in self._written_images:
                # A caller that supplies no per-key roster still gets an honest
                # `image.fps`; only the declared value beside it is missing.
                self.info["image"]["declared_fps"] = self.info["image"].get("fps")
                self.info["image"]["fps"] = self._written_images["color_0"] / duration_s
        self.info["image_storage"] = {
            "duration_s": duration_s,
            "written_images": sum(self._written_images.values()),
            "reused_images": self._reused_images,
            "note": ("a colour key whose camera frame is unchanged from an earlier "
                     "sample is stored once and referenced again, so the files under "
                     "colors/ are fewer than the frame count; sample.sources carries "
                     "each key's sequence and repeated flag"),
        }

    def _finish_episode(self):
        self._frames.flush()
        os.fsync(self._frames.fileno())
        self._frames.close()
        self._frames = None
        self.raise_if_failed()
        self._record_measured_image_rates()
        self._saved_at = datetime.datetime.now().astimezone()
        saved_dir = self.task_dir / f"episode_{self.episode_id:04d}_{self._saved_at:%Y%m%d_%H%M%S_%f}"
        self.episode_dir.rename(saved_dir)
        self.episode_dir = saved_dir
        self._write_manifest("complete")
        if self.quality_report:
            try:
                if self._quality_process is None or self._quality_process.poll() is not None:
                    checker = Path(__file__).resolve().parents[2] / "tools/check_teleop_episode.py"
                    self._quality_process = subprocess.Popen(
                        [sys.executable, "-u", str(checker), "--worker"],
                        stdin=subprocess.PIPE, text=True, encoding="utf-8",
                        start_new_session=True)
                self._quality_process.stdin.write(json.dumps(str(saved_dir.resolve())) + "\n")
                self._quality_process.stdin.flush()
            except (OSError, ValueError) as error:
                logger.warning("[QUALITY] 无法启动质检；数据已保存：%s (%s)", saved_dir, error)
        with self._lock:
            if self._error is not None:
                raise self._error
            self._state = "idle"
        logger.info("Saved episode: %s (%d frames)", self.episode_dir, self._frame_count)

    def process_queue(self):
        try:
            while True:
                try:
                    item = self.item_data_queue.get(timeout=0.05)
                except Empty:
                    item = None
                with self._lock:
                    create = self._create_pending
                    self._create_pending = False
                try:
                    if create:
                        self._start_episode()
                    self.raise_if_failed()
                    if item is not None:
                        self._process_item_data(item)
                finally:
                    if item is not None:
                        self.item_data_queue.task_done()
                with self._lock:
                    save = self._state == "saving" and self.item_data_queue.empty()
                if save:
                    self._finish_episode()
                with self._lock:
                    if self._error is not None:
                        raise self._error
                    if self._closed and self._state == "idle":
                        return
        except Exception as error:
            with self._lock:
                active = self._state in ("recording", "saving")
                if self._error is None:
                    self._error = error
                self._state = "failed"
            if self._frames is not None:
                try:
                    self._frames.close()
                except Exception:
                    logger.exception("Failed to close partial episode frames")
                self._frames = None
            if active and self.episode_dir is not None:
                try:
                    self._write_manifest("incomplete")
                except Exception:
                    logger.exception("Could not mark episode incomplete: %s", self.episode_dir)
            while True:
                try:
                    self.item_data_queue.get_nowait()
                    self.item_data_queue.task_done()
                except Empty:
                    break
            logger.error("Episode recording failed: %s", self._error)

    def close(self):
        with self._lock:
            self._closed = True
            if self._state == "recording":
                self._state = "saving"
        self.worker_thread.join(timeout=CLOSE_TIMEOUT)
        if not self.worker_thread.is_alive() and self._quality_process is not None:
            try:
                self._quality_process.stdin.close()
            except OSError:
                pass
        if self.worker_thread.is_alive():
            with self._lock:
                if self._error is None:
                    self._error = TimeoutError(f"recording worker did not stop within {CLOSE_TIMEOUT}s")
        self.raise_if_failed()
