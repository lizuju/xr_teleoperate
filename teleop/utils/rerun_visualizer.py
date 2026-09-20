import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import rerun as rr
import rerun.blueprint as rrb

os.environ["RUST_LOG"] = "error"

V2_SCHEMA = "xr_teleop_episode_v2"
COLOR_KEYS = ("color_0", "color_1", "color_2", "color_3")
COLOR_VIEW_NAMES = {
    "color_0": "head_left",
    "color_1": "head_right",
    "color_2": "left_wrist",
    "color_3": "right_wrist",
}
QPOS_PARTS = ("left_arm", "right_arm", "left_ee", "right_ee", "body")
BODY_JOINTS = ("waist_yaw", "head_pitch", "head_yaw")
TRACKING_SOURCES = ("left_hand_tracking", "right_hand_tracking")


def _is_executable(path):
    try:
        path = Path(path)
    except TypeError:
        return False
    return path.is_file() and os.access(path, os.X_OK)


def _scripts_directory(python_executable=None):
    return Path(python_executable or sys.executable).expanduser().resolve().parent


def _prepend_path_entry(path_value, directory):
    directory = str(directory)
    parts = [part for part in str(path_value or "").split(os.pathsep) if part]
    if directory in parts:
        parts.remove(directory)
    parts.insert(0, directory)
    return os.pathsep.join(parts)


def _bundled_rerun_cli():
    try:
        from importlib.resources import files as resource_files
    except ImportError:
        resource_files = None
    if resource_files is not None:
        for package in ("rerun_cli", "rerun_sdk.rerun_cli"):
            try:
                candidate = resource_files(package).joinpath("rerun")
            except (ModuleNotFoundError, FileNotFoundError, TypeError, ValueError):
                continue
            try:
                if not candidate.is_file():
                    continue
                path = Path(os.fspath(candidate))
            except (OSError, TypeError, ValueError, FileNotFoundError):
                continue
            if path.is_file():
                return path
    try:
        import rerun_cli
    except ImportError:
        return None
    path = Path(rerun_cli.__file__).resolve().parent / "rerun"
    return path if path.is_file() else None


def resolve_rerun_viewer(python_executable=None, environ=None):
    """Locate the Rerun Viewer without a global `rerun` on PATH.

    Callers that invoke venv Python by absolute path never get `.venv/bin` on
    PATH. `rr.spawn()` still looks up `rerun` via PATH, so prepend the scripts
    directory next to `sys.executable`, then try that directory, the bundled
    `rerun_cli` resource, and `shutil.which`.
    """
    env = dict(os.environ if environ is None else environ)
    scripts = _scripts_directory(python_executable)
    env["PATH"] = _prepend_path_entry(env.get("PATH", ""), scripts)
    candidates = [scripts / "rerun"]
    if os.name == "nt":
        candidates.append(scripts / "rerun.exe")
    bundled = _bundled_rerun_cli()
    if bundled is not None:
        candidates.append(bundled)
    which = shutil.which("rerun", path=env["PATH"])
    if which:
        candidates.append(Path(which))
    seen = set()
    for candidate in candidates:
        try:
            resolved = Path(candidate).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if not _is_executable(resolved):
            continue
        env["PATH"] = _prepend_path_entry(env["PATH"], resolved.parent)
        return resolved, env
    return None, env


def prepare_rerun_spawn_path():
    """Put the venv (or bundled) viewer on PATH for `rr.spawn()` and return it."""
    viewer, env = resolve_rerun_viewer()
    os.environ["PATH"] = env["PATH"]
    if viewer is not None:
        return viewer
    scripts = _scripts_directory()
    version = getattr(rr, "__version__", None)
    pin = f"rerun-sdk=={version}" if version else "rerun-sdk"
    raise RuntimeError(
        "Failed to find the Rerun Viewer next to this Python or in the rerun-sdk package. "
        f"Looked for {scripts / 'rerun'} and importlib.resources (rerun_cli). "
        "A global `rerun` install is not required. "
        f"Install the matching SDK into this venv: {sys.executable} -m pip install '{pin}'"
    )


def episode_directory(path):
    """Resolve an episode folder from a path to the folder, episode.json, or data.json."""
    path = Path(path).expanduser().resolve()
    if path.name in ("episode.json", "data.json"):
        return path.parent
    if path.is_dir():
        if (path / "episode.json").is_file() or (path / "data.json").is_file():
            return path
        children = sorted(
            child for child in path.iterdir()
            if child.is_dir() and child.name.startswith("episode_")
        )
        if len(children) == 1:
            return children[0]
        if len(children) > 1:
            names = ", ".join(child.name for child in children[:8])
            raise FileNotFoundError(
                f"{path} contains multiple episodes ({names}); pass one episode_* directory"
            )
    raise FileNotFoundError(
        f"{path} is not a v2 episode (episode.json) or a v1 episode (data.json)"
    )


def detect_episode_format(directory):
    directory = Path(directory)
    manifest_path = directory / "episode.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{manifest_path}: {exc}") from exc
        if isinstance(manifest, dict) and manifest.get("schema") == V2_SCHEMA:
            return "v2", manifest
        if isinstance(manifest, dict) and (directory / "frames.jsonl").is_file():
            return "v2", manifest
    data_path = directory / "data.json"
    if data_path.is_file():
        return "v1", None
    raise FileNotFoundError(
        f"{directory} has no episode.json (v2) or data.json (v1)"
    )


def _contained_file(directory, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    try:
        path.relative_to(directory)
    except ValueError:
        return None
    return path


def _read_color(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _finite_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def summarize_episode(path):
    directory = episode_directory(path)
    fmt, manifest = detect_episode_format(directory)
    summary = {
        "episode": str(directory),
        "format": fmt,
        "schema": None if manifest is None else manifest.get("schema"),
        "status": None if manifest is None else manifest.get("status"),
        "outcome": None if manifest is None else manifest.get("outcome"),
        "episode_id": None if manifest is None else manifest.get("episode_id"),
        "frame_count": None if manifest is None else manifest.get("frame_count"),
        "episode_json": (directory / "episode.json").is_file(),
        "data_json": (directory / "data.json").is_file(),
        "frames_jsonl": (directory / "frames.jsonl").is_file(),
        "color_keys": [],
        "has_hands": False,
        "has_body": False,
        "modes": {},
        "tracking_hold_frames": 0,
    }
    if fmt == "v2":
        color_keys = set()
        modes = Counter()
        has_hands = has_body = False
        frames_path = directory / (manifest.get("frames") if manifest else "frames.jsonl")
        if not frames_path.is_file():
            frames_path = directory / "frames.jsonl"
        count = 0
        with frames_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                frame = json.loads(line)
                count += 1
                color_keys.update(key for key, value in (frame.get("colors") or {}).items() if value)
                states = frame.get("states") or {}
                has_hands = has_hands or any(states.get(name, {}).get("qpos") for name in ("left_ee", "right_ee"))
                has_body = has_body or bool((states.get("body") or {}).get("qpos"))
                mode = (frame.get("sample") or {}).get("mode")
                modes[mode] += 1
        summary.update(
            frame_count=count if summary["frame_count"] is None else summary["frame_count"],
            color_keys=sorted(color_keys, key=lambda key: (COLOR_KEYS.index(key) if key in COLOR_KEYS else 99, key)),
            has_hands=has_hands,
            has_body=has_body,
            modes={str(key): value for key, value in modes.items()},
            tracking_hold_frames=int(modes.get("tracking_hold", 0)),
        )
        return summary

    with (directory / "data.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    rows = payload.get("data") or []
    color_keys = set()
    has_hands = has_body = False
    for item in rows:
        color_keys.update(key for key, value in (item.get("colors") or {}).items() if value)
        states = item.get("states") or {}
        has_hands = has_hands or any(states.get(name, {}).get("qpos") for name in ("left_ee", "right_ee"))
        has_body = has_body or bool((states.get("body") or {}).get("qpos"))
    summary.update(
        schema=payload.get("info", {}).get("version") if isinstance(payload.get("info"), dict) else "v1",
        frame_count=len(rows),
        color_keys=sorted(color_keys),
        has_hands=has_hands,
        has_body=has_body,
    )
    return summary


class RerunEpisodeReader:
    def __init__(self, task_dir=".", json_file="data.json"):
        self.task_dir = task_dir
        self.json_file = json_file

    def return_episode_data(self, episode_idx):
        episode_dir = Path(self.task_dir) / f"episode_{int(episode_idx):04d}"
        return list(self.iter_episode_data(episode_dir))

    def iter_episode_data(self, episode_dir, load_images=True, max_frames=None):
        episode_dir = Path(episode_dir)
        fmt, manifest = detect_episode_format(episode_dir)
        if fmt == "v2":
            yield from self._iter_v2(episode_dir, manifest, load_images=load_images, max_frames=max_frames)
            return
        yield from self._iter_v1(episode_dir, load_images=load_images, max_frames=max_frames)

    def _iter_v2(self, episode_dir, manifest, load_images=True, max_frames=None):
        frames_name = manifest.get("frames") if isinstance(manifest, dict) else None
        frames_path = episode_dir / (frames_name if isinstance(frames_name, str) else "frames.jsonl")
        if not frames_path.is_file():
            raise FileNotFoundError(f"{episode_dir} is v2 but frames.jsonl is missing")
        last_paths = {}
        last_images = {}
        with frames_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if max_frames is not None and line_number > max_frames:
                    return
                if not line.strip():
                    continue
                item = json.loads(line)
                colors = {}
                if load_images:
                    for key, relative in (item.get("colors") or {}).items():
                        if not relative:
                            continue
                        if relative == last_paths.get(key):
                            colors[key] = last_images[key]
                            continue
                        path = _contained_file(episode_dir, relative)
                        image = _read_color(path) if path is not None and path.is_file() else None
                        if image is None:
                            continue
                        last_paths[key] = relative
                        last_images[key] = image
                        colors[key] = image
                yield {
                    "idx": item.get("idx", line_number - 1),
                    "colors": colors,
                    "states": item.get("states") or {},
                    "actions": item.get("actions") or {},
                    "sample": item.get("sample") or {},
                    "tactiles": item.get("tactiles") or {},
                    "audios": {},
                    "depths": {},
                }

    def _iter_v1(self, episode_dir, load_images=True, max_frames=None):
        json_path = Path(episode_dir) / self.json_file
        if not json_path.is_file():
            raise FileNotFoundError(
                f"{episode_dir} has no {self.json_file} (v1) and no episode.json (v2)"
            )
        with json_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ValueError(f"{json_path}: expected a v1 object with a data list")
        for index, item in enumerate(rows):
            if max_frames is not None and index >= max_frames:
                return
            colors = {}
            if load_images:
                colors = self._process_images(item, "colors", episode_dir)
            yield {
                "idx": item.get("idx", index),
                "colors": colors,
                "states": item.get("states") or {},
                "actions": item.get("actions") or {},
                "sample": item.get("sample") or {},
                "tactiles": item.get("tactiles") or {},
                "audios": {},
                "depths": {},
            }

    def _process_images(self, item_data, data_type, dir_path):
        images = {}
        for key, file_name in (item_data.get(data_type) or {}).items():
            if not file_name:
                continue
            path = _contained_file(dir_path, file_name)
            if path is None or not path.is_file():
                continue
            image = _read_color(path)
            if image is not None:
                images[key] = image
        return images

    def _process_audio(self, item_data, data_type, episode_dir):
        return {}


class RerunLogger:
    def __init__(self, prefix="", IdxRangeBoundary=30, memory_limit=None,
                 spawn=True, save_path=None, application_id=None):
        self.prefix = prefix
        self.IdxRangeBoundary = IdxRangeBoundary
        rr.init(application_id or datetime.now().strftime("Runtime_%Y%m%d_%H%M%S"))
        if save_path:
            rr.save(str(save_path))
        elif spawn:
            prepare_rerun_spawn_path()
            if memory_limit:
                rr.spawn(memory_limit=memory_limit, hide_welcome_screen=True)
            else:
                rr.spawn(hide_welcome_screen=True)
        self.setup_blueprint()

    def _time_ranges(self):
        if not self.IdxRangeBoundary:
            return None
        return [
            rrb.VisibleTimeRange(
                "idx",
                start=rrb.TimeRangeBoundary.cursor_relative(seq=-self.IdxRangeBoundary),
                end=rrb.TimeRangeBoundary.cursor_relative(),
            )
        ]

    def setup_blueprint(self):
        time_ranges = self._time_ranges()
        cameras = []
        for key, name in COLOR_VIEW_NAMES.items():
            kwargs = {"origin": f"{self.prefix}colors/{key}", "name": name}
            if time_ranges:
                kwargs["time_ranges"] = time_ranges
            cameras.append(rrb.Spatial2DView(**kwargs))
        plots = []
        for origin, name in (
            (f"{self.prefix}left_arm", "left_arm"),
            (f"{self.prefix}right_arm", "right_arm"),
            (f"{self.prefix}left_ee", "left_hand"),
            (f"{self.prefix}right_ee", "right_hand"),
            (f"{self.prefix}waist", "waist"),
            (f"{self.prefix}head", "head"),
            (f"{self.prefix}sample", "tracking"),
        ):
            kwargs = {"origin": origin, "name": name, "plot_legend": rrb.PlotLegend(visible=True)}
            if time_ranges:
                kwargs["time_ranges"] = time_ranges
            plots.append(rrb.TimeSeriesView(**kwargs))
        layout = rrb.Vertical(
            contents=[
                rrb.Grid(contents=cameras, grid_columns=2, name="cameras"),
                rrb.Grid(contents=plots, grid_columns=2, name="qpos"),
            ],
            row_shares=[1.15, 1],
        )
        rr.send_blueprint(layout)

    def _scalar(self, path, value):
        number = _finite_number(value)
        if number is None:
            return
        rr.log(path, rr.Scalar(number))

    def _log_qpos(self, kind, part, values):
        if not values:
            return
        if part == "body":
            for index, value in enumerate(values):
                name = BODY_JOINTS[index] if index < len(BODY_JOINTS) else str(index)
                if name.startswith("waist"):
                    self._scalar(f"{self.prefix}waist/{kind}/qpos/{name}", value)
                elif name.startswith("head"):
                    self._scalar(f"{self.prefix}head/{kind}/qpos/{name}", value)
                else:
                    self._scalar(f"{self.prefix}body/{kind}/qpos/{index}", value)
            return
        for index, value in enumerate(values):
            self._scalar(f"{self.prefix}{part}/{kind}/qpos/{index}", value)

    def log_item_data(self, item_data: dict):
        rr.set_time_sequence("idx", item_data.get("idx", 0))
        sample = item_data.get("sample") or {}
        timestamp_ns = sample.get("timestamp_ns")
        if type(timestamp_ns) is int:
            rr.set_time_nanos("timestamp", timestamp_ns)
        monotonic_ns = sample.get("monotonic_ns")
        if type(monotonic_ns) is int:
            rr.set_time_nanos("monotonic", monotonic_ns)

        states = item_data.get("states") or {}
        for part in QPOS_PARTS:
            info = states.get(part) or {}
            if info:
                self._log_qpos("states", part, info.get("qpos") or [])

        actions = item_data.get("actions") or {}
        for part in QPOS_PARTS:
            info = actions.get(part) or {}
            if info:
                self._log_qpos("actions", part, info.get("qpos") or [])

        colors = item_data.get("colors") or {}
        for key, value in colors.items():
            if value is None or isinstance(value, str):
                # Live recording has already replaced pixels with relative JPEG
                # paths; re-reading them would add disk I/O on the teleop loop.
                continue
            if getattr(value, "ndim", None) == 3:
                rr.log(f"{self.prefix}colors/{key}", rr.Image(value))

        mode = sample.get("mode")
        if isinstance(mode, str) and mode:
            rr.log(f"{self.prefix}sample/mode", rr.TextLog(mode))
            self._scalar(f"{self.prefix}sample/tracking_hold", 1.0 if mode == "tracking_hold" else 0.0)
            self._scalar(f"{self.prefix}sample/following", 1.0 if mode == "following" else 0.0)
            self._scalar(f"{self.prefix}sample/paused", 1.0 if mode == "paused" else 0.0)

        sources = sample.get("sources") or {}
        for name in TRACKING_SOURCES:
            source = sources.get(name) or {}
            if not source:
                continue
            if "fresh" in source and source["fresh"] is not None:
                self._scalar(f"{self.prefix}sample/{name}/fresh", 1.0 if source["fresh"] else 0.0)
            if source.get("age_ms") is not None:
                self._scalar(f"{self.prefix}sample/{name}/age_ms", source.get("age_ms"))

        alignment = sample.get("camera_alignment") or {}
        if isinstance(alignment, dict) and "aligned" in alignment:
            self._scalar(f"{self.prefix}sample/camera_aligned", 1.0 if alignment.get("aligned") else 0.0)

    def log_episode_data(self, episode_data: list):
        for item_data in episode_data:
            self.log_item_data(item_data)


def close_recording():
    if hasattr(rr, "disconnect"):
        rr.disconnect()


if __name__ == "__main__":
    raise SystemExit("Use tools/rerun_teleop_episode.py <episode_dir>  (never starts teleop)")
