#!/usr/bin/env bash
# R1_A7 实机遥操（Vector）— 2026-09-15
#
# 这里固定了 R1_A7 的启动参数，并把「大臂限速」做成默认值：
#   ARM_DQ_FEEDFORWARD  默认 on —— 把目标速度作为速度前馈下发给电机（原来 dq 恒为 0，
#                       纯位置控制，所以手臂追不上快速的手部动作，表现为一卡一卡）。
#                       要退回“削命令”的保守方案：ARM_DQ_FEEDFORWARD=off ARM_VELOCITY_LIMIT=3.0
#   ARM_DQ_LIMIT        前馈速度上限，默认 6.0 rad/s（安全网，不是跟踪上限）
#   ARM_VELOCITY_LIMIT  位置目标的安全限速，默认 30.0 rad/s（约等于不限）
#   ARM_DIAG_HZ         设了就按该频率记录诊断（复测大臂问题建议 40）
#   ARM_DIAG_DIR        设了就把诊断 JSONL 写进该目录
#   WAIST_FOLLOW                on/off，默认 **off**（腰跟随会废掉手臂可达空间，见下）
#   WAIST_FOLLOW_THRESHOLD_DEG  仅在 WAIST_FOLLOW=on 时生效，默认 15 度
#   WAIST_FOLLOW_DWELL          仅在 WAIST_FOLLOW=on 时生效，默认 0.3 秒
#
#   为什么默认关：腰跟随会转躯干，而腕部目标是补偿到「世界坐标」的，于是腰一转，
#   手相对躯干就被推出可达范围。2026-09-16 实测（r1-diag-15deg）：腰偏离参考 5 度内
#   工作空间饱和 42.6%，5-10 度 90%，**超过 10 度 100% 饱和** —— 而饱和正是大臂发卡的
#   直接原因（90% 的命令跳变发生在饱和时）。头部仍会跟着你的视线转。
#   CAMERA_CALIBRATION  相机标定 JSON 路径；留空则用 assets/r1/camera_calibration.json（存在才读）。
#                       标定结果会写进每个 episode 的 info.camera_calibration，供后面数采/训练使用。
#
# 例：
#   ./teleop/run_r1_a7_vector.sh                        # 日常遥操（限速 3.0 已生效）
#   ARM_DIAG_HZ=40 ARM_DIAG_DIR=$HOME/r1-diag ./teleop/run_r1_a7_vector.sh
set -euo pipefail

check_only=false
if [[ $# -eq 1 && "$1" == "--check-only" ]]; then
  check_only=true
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dev_root="$(cd -- "${script_dir}/../.." && pwd)"
python="${dev_root}/.venv-xr/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Missing XR Python environment: $python" >&2
  exit 1
fi

export XR_TELEOP_CERT="${HOME}/.config/xr_teleoperate/cert.pem"
export XR_TELEOP_KEY="${HOME}/.config/xr_teleoperate/key.pem"

"$python" -u "${script_dir}/../tools/check_r1_teleop.py"
if [[ "$check_only" == true ]]; then
  exit 0
fi

# Waist following is OFF by default. It turns the torso to follow head yaw, but
# the wrist targets are compensated to stay in world space, so rotating the waist
# drags the arms across their envelope and out of it. Measured 2026-09-16:
# workspace saturation is 42.6% while the waist sits within 5 deg of its
# reference, 90% at 5-10 deg, and 100% beyond 10 deg -- and saturated targets are
# exactly what the operator feels as the upper arm catching. The head still
# points where you look: head_q_target is computed independently and the head
# joint has roughly +-115 deg of travel of its own.
#   Set WAIST_FOLLOW=on to bring it back.
waist=()
if [[ "${WAIST_FOLLOW:-off}" != "off" ]]; then
  waist+=(--waist-follow)
  waist+=(--waist-follow-threshold-deg "${WAIST_FOLLOW_THRESHOLD_DEG:-15}")
  waist+=(--waist-follow-dwell "${WAIST_FOLLOW_DWELL:-0.3}")
fi
diagnostics=()
if [[ -n "${ARM_DIAG_HZ:-}" ]]; then
  diagnostics+=(--arm-diagnostic-hz "${ARM_DIAG_HZ}")
fi
if [[ -n "${ARM_DIAG_DIR:-}" ]]; then
  mkdir -p "${ARM_DIAG_DIR}"
  diagnostics+=(--arm-diagnostic-dir "${ARM_DIAG_DIR}")
  echo "[run] 诊断日志目录: ${ARM_DIAG_DIR}"
fi

cd "$script_dir"
exec "$python" -u teleop_hand_and_arm.py \
  --input-mode hand \
  --display-mode immersive \
  --arm R1_A7 \
  --ee linker_o6 \
  --linker-o6-method vector \
  --linker-o6-urdf-root "${dev_root}/linkerhand-urdf/O6" \
  "${waist[@]}" \
  --arm-velocity-limit "${ARM_VELOCITY_LIMIT:-30.0}" \
  --arm-dq-feedforward "${ARM_DQ_FEEDFORWARD:-on}" \
  --arm-dq-limit "${ARM_DQ_LIMIT:-6.0}" \
  --camera-calibration "${CAMERA_CALIBRATION:-}" \
  --network-interface eno1 \
  --img-server-ip 192.168.124.147 \
  --headless \
  "${diagnostics[@]}" \
  "$@"
