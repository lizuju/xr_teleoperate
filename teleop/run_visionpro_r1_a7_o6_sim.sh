#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dev_root="$(cd -- "${script_dir}/../.." && pwd)"

export XR_TELEOP_CERT=/home/hnh/.config/xr_teleoperate/cert.pem
export XR_TELEOP_KEY=/home/hnh/.config/xr_teleoperate/key.pem

cd "${dev_root}/xr_teleoperate"
exec "${dev_root}/.venv-xr/bin/python" -u \
  "${script_dir}/visionpro_r1_a7_o6_sim.py" \
  --linker-o6-urdf-root "${dev_root}/linkerhand-urdf/O6" \
  --linker-o6-calibration \
  "${dev_root}/xr_teleoperate/teleop/robot_control/linker_o6_visionpro_calibration.json" \
  --frequency 30 \
  --tracking-timeout 0.25
