#!/usr/bin/env bash
set -euo pipefail

cd /home/hnh/unitree_r1_dev/xr_teleoperate
exec /home/hnh/unitree_r1_dev/.venv-xr/bin/python -u \
  teleop/r1_a7_arm_live_bridge.py \
  --mode dry-run \
  "$@"
