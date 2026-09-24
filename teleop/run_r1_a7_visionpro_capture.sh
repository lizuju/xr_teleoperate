#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 || "$1" == -* ]]; then
  echo "Usage: $0 VISION_PRO_IP TASK_NAME TASK_GOAL [teleop options...]" >&2
  exit 2
fi
visionpro_ip="$1"
task_name="$2"
task_goal="$3"
shift 3
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/run_r1_a7_capture.sh" "$task_name" "$task_goal" "$@" \
  --tracking-source visionpro --visionpro-ip "$visionpro_ip" \
  --display-mode pass-through --wrist-display off --hand-torque-hud off
