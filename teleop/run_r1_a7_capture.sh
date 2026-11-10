#!/usr/bin/env bash
# R1_A7 录制入口（Vector + --record）
# 右 Dex1-1 最小开口沿用 ~/.config/xr_teleoperate/dex1_min_q，可用 R1_DEX1_MIN_Q 覆盖。
# 一次启动可连续多条：r 跟随，s 开录，y/n/x 结束并标记成功/失败/丢弃（文件保留）。
# p 暂停保持姿态；暂停后 s 等双手稳定并重新对齐，再恢复跟随和开始下一条。
# 每条保存目录带系统时间；再次 s 可开始下一条，无需重启。
#
# 用法：
#   ./teleop/run_r1_a7_capture.sh TASK_NAME "TASK_GOAL" [teleop options...]
#   ./teleop/run_r1_a7_capture.sh TASK_NAME "TASK_GOAL" --check-only
#
# 显示模式默认 immersive（硬编码）。可在任务名/目标之后追加后缀覆盖，例如：
#   ./teleop/run_r1_a7_capture.sh 任务名 "任务目标" --display-mode ego
#   ./teleop/run_r1_a7_capture.sh 任务名 "任务目标" --display-mode pass-through
# 不要加单独的裸 "--"；--check-only 必须紧跟在 TASK_GOAL 之后（第 3 个参数）。
# 其余 teleop 选项在 shift 任务参数后经末尾 "$@" 转发（后写覆盖先写）。
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TASK_NAME TASK_GOAL [--check-only] [teleop options...]" >&2
  echo "  e.g. $0 my_task \"pick cup\" --display-mode ego" >&2
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
right_ee_file="${HOME}/.config/xr_teleoperate/right_ee"
right_ee="${R1_RIGHT_EE:-}"
if [[ -z "$right_ee" && -r "$right_ee_file" ]]; then
  right_ee="$(tr -d '[:space:]' < "$right_ee_file")"
fi
right_ee="${right_ee:-linker_o6}"
export R1_RIGHT_EE="$right_ee"
R1_PREFLIGHT_REQUIRE_CAMERA_CLOCK=1 bash "${script_dir}/run_r1_a7_vector.sh" --check-only
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
dex1_min_q_file="${HOME}/.config/xr_teleoperate/dex1_min_q"
dex1_min_q="${R1_DEX1_MIN_Q:-}"
if [[ -z "$dex1_min_q" && -r "$dex1_min_q_file" ]]; then
  dex1_min_q="$(tr -d '[:space:]' < "$dex1_min_q_file")"
  if [[ -z "$dex1_min_q" ]]; then
    echo "Saved Dex1 minimum opening is empty: $dex1_min_q_file" >&2
    exit 2
  fi
fi
dex1_min_q="${dex1_min_q:-0}"

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
  --right-ee "$right_ee" \
  --dex1-transport "${R1_DEX1_TRANSPORT:-lowcmd}" \
  --dex1-slot "${R1_DEX1_SLOT:-33}" \
  --dex1-min-q "$dex1_min_q" \
  --wrist-stabilizer "${WRIST_STABILIZER:-on}" \
  --wrist-stabilizer-position-mm "${WRIST_STABILIZER_POSITION_MM:-3}" \
  --wrist-stabilizer-rotation-deg "${WRIST_STABILIZER_ROTATION_DEG:-3}" \
  --linker-o6-method vector \
  --linker-o6-urdf-root "${dev_root}/linkerhand-urdf/O6" \
  --waist-follow \
  --waist-follow-threshold-deg "${WAIST_FOLLOW_THRESHOLD_DEG:-20}" \
  --waist-follow-speed-deg "${WAIST_FOLLOW_SPEED_DEG:-40}" \
  --waist-follow-accel-deg "${WAIST_FOLLOW_ACCEL_DEG:-90}" \
  --waist-follow-compensation "${WAIST_FOLLOW_COMPENSATION:-torso}" \
  --waist-turn-speed-deg "${WAIST_TURN_SPEED_DEG:-30}" \
  --waist-turn-accel-deg "${WAIST_TURN_ACCEL_DEG:-60}" \
  --wrist-display "${WRIST_DISPLAY:-off}" \
  --arm-translation-scale "${ARM_TRANSLATION_SCALE:-0.87}" \
  --arm-limit-softness "${ARM_LIMIT_SOFTNESS:-0.1}" \
  --arm-posture-weight "${ARM_POSTURE_WEIGHT:-0.01}" \
  --arm-velocity-limit "${ARM_VELOCITY_LIMIT:-30.0}" \
  --arm-dq-feedforward "${ARM_DQ_FEEDFORWARD:-on}" \
  --arm-dq-limit "${ARM_DQ_LIMIT:-6.0}" \
  --arm-target-velocity-limit "${ARM_TARGET_VELOCITY_LIMIT:-6.0}" \
  --arm-target-accel-limit "${ARM_TARGET_ACCEL_LIMIT:-40.0}" \
  --camera-calibration "${CAMERA_CALIBRATION:-}" \
  --record-max-tracking-age-ms "${RECORD_MAX_TRACKING_AGE_MS:-100}" \
  --network-interface eno1 \
  --img-server-ip 192.168.5.10 \
  --headless \
  --record \
  --task-dir "${dev_root}/teleop-recordings" \
  --task-name "$task_name" \
  --task-goal "$task_goal" \
  "${diagnostics[@]}" \
  "$@"
