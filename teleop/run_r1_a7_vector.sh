#!/usr/bin/env bash
# R1_A7 实机遥操（Vector）— 2026-09-15
#
# 这里固定了 R1_A7 的启动参数，并把「大臂限速」做成默认值：
#   ARM_POSTURE_WEIGHT  把手臂姿态拉回激活时姿态的权重，默认 0.02 —— 这是防「手肘扭住」的那一项。
#                       7 自由度只有腕部位姿目标，存在一个自由的零空间，不锚住它肘就会慢慢漂到怪角度。
#                       实测（validate_redundancy.py，真实 URDF，120 帧扫掠）：0.0 时肘行程 0.685 rad，
#                       0.02 降到 0.629 而位置跟踪基本不变（+8%）、腕部姿态误差 0.089→0.131 rad；
#                       0.05 能降到 0.575 但姿态误差到 0.187 且开始贴关节限位。0 = 关闭。
#   ARM_LIMIT_SOFTNESS  软关节限位壁垒的权重，默认 0.1（0 = 关闭）。
#   ARM_TRANSLATION_SCALE  手部位移的比例，默认 0.87。机器人手臂比人臂短，按比例映射就是两者之比：
#                       R1_A7 肩到腕约 0.65 m，成人手臂约 0.75 m → 65/75 = 0.87。旋转从不缩放。
#                       1.0 会让手臂去够比操作者远 15% 的位置，腕部更容易顶到工作空间边界；
#                       0.7 少走 19%，手要划得比机器人动得还多，手感发涩。
#   ARM_DQ_FEEDFORWARD  默认 off —— 与 2026-09-14 的生产行为一致：dq 恒为 0，纯位置控制。
#                       设 on 则把目标速度作为速度前馈下发给电机（原来 dq 恒为 0，
#                       纯位置控制，所以手臂追不上快速的手部动作，表现为一卡一卡）。
#                       要退回“削命令”的保守方案：ARM_DQ_FEEDFORWARD=off ARM_VELOCITY_LIMIT=3.0
#   ARM_DQ_LIMIT        前馈速度上限，默认 6.0 rad/s（安全网，不是跟踪上限）
#   ARM_VELOCITY_LIMIT  位置目标的安全限速，默认 30.0 rad/s（约等于不限）
#   ARM_DIAG_HZ         设了就按该频率记录诊断（复测大臂问题建议 40）
#   ARM_DIAG_DIR        设了就把诊断 JSONL 写进该目录
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
  --waist-follow \
  --arm-translation-scale "${ARM_TRANSLATION_SCALE:-0.87}" \
  --arm-limit-softness "${ARM_LIMIT_SOFTNESS:-0.1}" \
  --arm-posture-weight "${ARM_POSTURE_WEIGHT:-0.02}" \
  --arm-velocity-limit "${ARM_VELOCITY_LIMIT:-30.0}" \
  --arm-dq-feedforward "${ARM_DQ_FEEDFORWARD:-off}" \
  --arm-dq-limit "${ARM_DQ_LIMIT:-6.0}" \
  --camera-calibration "${CAMERA_CALIBRATION:-}" \
  --network-interface eno1 \
  --img-server-ip 192.168.124.147 \
  --headless \
  "${diagnostics[@]}" \
  "$@"
