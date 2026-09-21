#!/usr/bin/env bash
# Replay a recorded R1_A7 + Linker O6 episode onto the real robot.
#
# This is not Rerun. --check-only never sends commands. Without --check-only,
# the process connects, prints the live-to-start error, and waits for r.
# q / Ctrl+C stop the program; they are not an e-stop.
#
# Example:
#   ./teleop/run_r1_a7_replay.sh /home/hnh/unitree_r1_dev/teleop-recordings/这次测试名/episode_0000 --check-only
#   ./teleop/run_r1_a7_replay.sh /home/hnh/unitree_r1_dev/teleop-recordings/这次测试名/episode_0000
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EPISODE_DIR [--check-only] [options...]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dev_root="$(cd -- "${script_dir}/../.." && pwd)"
python="${dev_root}/.venv-xr/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Missing XR Python environment: $python" >&2
  exit 1
fi

cd "$script_dir/.."
exec "$python" -u tools/replay_r1_episode_on_robot.py "$@"
