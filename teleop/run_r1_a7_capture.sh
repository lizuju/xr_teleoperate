#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TASK_NAME TASK_GOAL [teleop options...]" >&2
  exit 2
fi
if [[ -z "$1" || "$1" == */* || "$1" == "." || "$1" == ".." || -z "$2" ]]; then
  echo "TASK_NAME must be a single directory name and TASK_GOAL must be nonempty." >&2
  exit 2
fi
task_name="$1"
task_goal="$2"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dev_root="$(cd -- "${script_dir}/../.." && pwd)"
bash "${script_dir}/run_r1_a7_vector.sh" --check-only
check_only=false
if [[ "${3:-}" == "--check-only" ]]; then
  check_only=true
  shift 3
else
  shift 2
fi
if [[ "$check_only" == true ]]; then
  exit 0
fi

export XR_TELEOP_CERT="${HOME}/.config/xr_teleoperate/cert.pem"
export XR_TELEOP_KEY="${HOME}/.config/xr_teleoperate/key.pem"
diagnostics=()
if [[ -n "${ARM_DIAG_HZ:-}" ]]; then
  diagnostics+=(--arm-diagnostic-hz "${ARM_DIAG_HZ}")
fi
if [[ -n "${ARM_DIAG_DIR:-}" ]]; then
  mkdir -p "${ARM_DIAG_DIR}"
  diagnostics+=(--arm-diagnostic-dir "${ARM_DIAG_DIR}")
fi

cd "$script_dir"
exec "${dev_root}/.venv-xr/bin/python" -u teleop_hand_and_arm.py \
  --input-mode hand \
  --display-mode immersive \
  --arm R1_A7 \
  --ee linker_o6 \
  --linker-o6-method vector \
  --linker-o6-urdf-root "${dev_root}/linkerhand-urdf/O6" \
  --waist-follow \
  --waist-follow-threshold-deg "${WAIST_FOLLOW_THRESHOLD_DEG:-20}" \
  --waist-follow-speed-deg "${WAIST_FOLLOW_SPEED_DEG:-40}" \
  --waist-follow-accel-deg "${WAIST_FOLLOW_ACCEL_DEG:-90}" \
  --waist-follow-compensation "${WAIST_FOLLOW_COMPENSATION:-torso}" \
  --wrist-display "${WRIST_DISPLAY:-off}" \
  --arm-translation-scale "${ARM_TRANSLATION_SCALE:-0.87}" \
  --arm-limit-softness "${ARM_LIMIT_SOFTNESS:-0.1}" \
  --arm-posture-weight "${ARM_POSTURE_WEIGHT:-0.02}" \
  --arm-velocity-limit "${ARM_VELOCITY_LIMIT:-30.0}" \
  --arm-dq-feedforward "${ARM_DQ_FEEDFORWARD:-on}" \
  --arm-dq-limit "${ARM_DQ_LIMIT:-6.0}" \
  --arm-target-velocity-limit "${ARM_TARGET_VELOCITY_LIMIT:-6.0}" \
  --arm-target-accel-limit "${ARM_TARGET_ACCEL_LIMIT:-40.0}" \
  --camera-calibration "${CAMERA_CALIBRATION:-}" \
  --record-max-tracking-age-ms "${RECORD_MAX_TRACKING_AGE_MS:-100}" \
  --network-interface eno1 \
  --img-server-ip 192.168.124.147 \
  --headless \
  --record \
  --task-dir "${dev_root}/teleop-recordings" \
  --task-name "$task_name" \
  --task-goal "$task_goal" \
  "${diagnostics[@]}" \
  "$@"
