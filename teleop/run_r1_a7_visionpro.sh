#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "Usage: $0 VISION_PRO_IP [teleop options...]" >&2
  exit 2
fi
visionpro_ip="$1"
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--check-only" ]]; then
  exec bash "${script_dir}/run_r1_a7_vector.sh" --check-only
fi
exec bash "${script_dir}/run_r1_a7_vector.sh" "$@" \
  --tracking-source visionpro --visionpro-ip "$visionpro_ip" \
  --display-mode pass-through --wrist-display off --hand-torque-hud off
