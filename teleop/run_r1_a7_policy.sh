#!/usr/bin/env bash
# Run a trained ACT policy on the R1_A7 + left O6 + right Dex1.
#
# Default is shadow mode: reads feedback and cameras, logs what the policy would
# command, sends nothing. Only --execute can move the robot, and only after r.
# q / Ctrl+C stop the program; they are not an e-stop.
#
# Example:
#   ./teleop/run_r1_a7_policy.sh /home/hnh/unitree_r1_dev/training-runs/这次运行/best.pt
#   ./teleop/run_r1_a7_policy.sh /home/hnh/unitree_r1_dev/training-runs/这次运行/best.pt --execute
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [--execute] [options...]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export R1_DEV_ROOT="${R1_DEV_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}"
dev_root="$R1_DEV_ROOT"
python="${dev_root}/.venv-xr/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Missing XR Python environment: $python" >&2
  exit 1
fi

for argument in "$@"; do
  if [[ "$argument" == "--execute" ]]; then
    R1_RIGHT_EE=dex1 R1_DEX1_TRANSPORT=lowcmd R1_DEX1_SLOT=33 R1_PREFLIGHT_REQUIRE_CAMERA_CLOCK=1 \
      "$python" -u "${script_dir}/../tools/check_r1_teleop.py"
    break
  fi
done

cd "$script_dir/.."
exec "$python" -u tools/run_r1_act_policy.py "$@"
