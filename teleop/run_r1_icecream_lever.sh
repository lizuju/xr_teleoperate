#!/usr/bin/env bash
# Replay the taught right-lever ice-cream sequence on the real R1.
#
# --check-only never sends commands. Without --check-only, the process connects,
# prints live-to-grasp deltas, and waits for r. q / Ctrl+C = software e-stop for
# this tool (stop trajectory, freeze LIVE, release right hand, stop publisher,
# exit debug when possible). Hardware power cut is still the robot button.
#
# Example:
#   ./teleop/run_r1_icecream_lever.sh --check-only
#   ./teleop/run_r1_icecream_lever.sh
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dev_root="$(cd -- "${script_dir}/../.." && pwd)"
python="${dev_root}/.venv-xr/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Missing XR Python environment: $python" >&2
  exit 1
fi

cd "$script_dir/.."
exec "$python" -u tools/replay_r1_icecream_lever.py "$@"
