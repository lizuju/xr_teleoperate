#!/usr/bin/env python3
"""Open a recorded teleop episode in Rerun. Never sends robot commands."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop.utils.rerun_visualizer import (
    RerunEpisodeReader,
    RerunLogger,
    close_recording,
    episode_directory,
    summarize_episode,
)


def display_available(environ=None):
    env = os.environ if environ is None else environ
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def missing_display_message():
    return (
        "No DISPLAY (or WAYLAND_DISPLAY) is set, so the Rerun Viewer GUI cannot open. "
        "Use --save foo.rrd to write a recording file, or --summary for a text inventory. "
        "To open the GUI, run this on a machine with a display "
        "(local console, ssh -X, or export DISPLAY=:0 on the Ubuntu host)."
    )


def open_episode(path, max_frames=None, spawn=True, save_path=None, memory_limit="2GB", prefix="offline/"):
    directory = episode_directory(path)
    summary = summarize_episode(directory)
    reader = RerunEpisodeReader(task_dir=str(directory.parent))
    logger = RerunLogger(
        prefix=prefix,
        IdxRangeBoundary=None,
        memory_limit=memory_limit,
        spawn=spawn,
        save_path=save_path,
        application_id=f"teleop_{directory.name}",
    )
    logged = 0
    for item in reader.iter_episode_data(directory, load_images=True, max_frames=max_frames):
        logger.log_item_data(item)
        logged += 1
    close_recording()
    summary["logged_frames"] = logged
    if save_path is not None:
        summary["save"] = str(Path(save_path).resolve())
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="View a recorded R1 teleop episode in Rerun. Never sends robot commands."
    )
    parser.add_argument("episode", type=Path, help="episode_* directory, episode.json, or a task dir with one episode")
    parser.add_argument("--summary", action="store_true",
                        help="Print JSON inventory and exit; do not open Rerun")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Log only the first N frames")
    parser.add_argument("--save", type=Path, default=None,
                        help="Write an .rrd file instead of spawning the viewer")
    parser.add_argument("--no-spawn", action="store_true",
                        help="Do not open the Rerun viewer")
    parser.add_argument("--memory-limit", default="2GB")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")
    if args.summary:
        print(json.dumps(summarize_episode(args.episode), indent=2, ensure_ascii=False))
        return 0
    spawn = not args.no_spawn and args.save is None
    if spawn and not display_available():
        print(missing_display_message(), file=sys.stderr)
        return 2
    result = open_episode(
        args.episode,
        max_frames=args.max_frames,
        spawn=spawn,
        save_path=args.save,
        memory_limit=args.memory_limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
