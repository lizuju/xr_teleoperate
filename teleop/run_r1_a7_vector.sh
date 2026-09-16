#!/usr/bin/env bash
# R1_A7 实机遥操（Vector）— 2026-09-15
#
# 这里固定了 R1_A7 的启动参数，并把「大臂限速」做成默认值：
#   ARM_DQ_FEEDFORWARD  默认 on —— 把目标速度作为速度前馈下发给电机（原来 dq 恒为 0，
#                       纯位置控制，所以手臂追不上快速的手部动作，表现为一卡一卡）。
#                       要退回“削命令”的保守方案：ARM_DQ_FEEDFORWARD=off ARM_VELOCITY_LIMIT=3.0
#   ARM_DQ_LIMIT        前馈速度上限，默认 6.0 rad/s（安全网，不是跟踪上限）
#   ARM_VELOCITY_LIMIT  位置目标的安全限速，默认 30.0 rad/s（约等于不限）
#   FREQUENCY           控制环频率，默认 40 Hz。手臂每拍消费一个排队样本，样本平均要等半拍，
#                       40→60 能把这段等待从约 12 ms 降到约 8 ms，代价是 IK 每秒多解 50%（每次约 2.5 ms）。
#                       2026-09-16 实测：60 Hz 下循环平均确实到了 59.6 Hz，但 body_max=21.3 ms 已经超过
#                       16.7 ms 的预算，循环没有余量；而同期样本年龄中位数是 196 ms，省 4 ms 是噪声。
#                       所以默认留在 40。想试就设 60，但先看退出汇总里的 body_max。
#   ARM_TARGET_VELOCITY_LIMIT  关节「参考轨迹」限速，默认 5.0 rad/s —— 治「一顿一顿」的主开关。
#                       手部数据到达控制环是不均匀的（2026-09-16 实测：每秒仅 11 个新样本，
#                       样本年龄中位 45 ms、p95 251 ms、最差 446 ms）。一个过期样本被替换时，
#                       整段手的位移会在一个 25 ms 控制周期内一次性下达：p95 22.8 度、
#                       最大 84.5 度关节位移，隐含 46 rad/s，而关节实测只有 5-7 rad/s 的能力。
#                       伺服因此长期落后 0.25-0.45 rad 再猛追，这就是卡顿。
#                       4.0 跟得上正常手速（稳态约 0.8、峰值约 2 rad/s），只把「追赶」拉长。
                       2026-09-16 实测：4.0 时操作者反馈手感好；提到 5.0 后紧接着两次都变回「一顿一顿」，
                       所以默认退回 4.0。当初提上去的理由（峰值 5.01 对能力 7.19）是错的 —— 7.19 是
                       没有整形器时、在猛烈追赶中测到的，不是「手感平滑」的速度。
#                       更跟手就调大（6-8），更顺就调小（2-3），0 = 关闭整形、恢复原始目标。
#   ARM_TARGET_ACCEL_LIMIT     参考速度变化率上限，默认 40.0 rad/s^2（0 = 只限速不限加速度）
#   ARM_TRANSLATION_SCALE      手部位移比例，默认 0.87。机器人手臂比人臂短，按比例映射就是两者之比：
#                       R1_A7 肩到腕约 0.65 m，成人手臂约 0.75 m → 65/75 = 0.87。旋转不缩放。
#                       1.0 会让手臂去够比操作者远 15% 的位置，腕部更容易顶到工作空间边界；
#                       0.7 少走 19%，手要划得比机器人动得还多，手感发涩。
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
#   ./teleop/run_r1_a7_vector.sh                        # 日常遥操（参考轨迹限速 4.0 已生效）
#   ARM_TARGET_VELOCITY_LIMIT=0 ./teleop/run_r1_a7_vector.sh          # 关掉整形，恢复原始目标
#   ARM_TARGET_VELOCITY_LIMIT=6 ./teleop/run_r1_a7_vector.sh          # 更跟手，但更冲
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
  --frequency "${FREQUENCY:-40}" \
  --arm-translation-scale "${ARM_TRANSLATION_SCALE:-0.87}" \
  --arm-velocity-limit "${ARM_VELOCITY_LIMIT:-30.0}" \
  --arm-dq-feedforward "${ARM_DQ_FEEDFORWARD:-on}" \
  --arm-dq-limit "${ARM_DQ_LIMIT:-6.0}" \
  --arm-target-velocity-limit "${ARM_TARGET_VELOCITY_LIMIT:-4.0}" \
  --arm-target-accel-limit "${ARM_TARGET_ACCEL_LIMIT:-40.0}" \
  --camera-calibration "${CAMERA_CALIBRATION:-}" \
  --network-interface eno1 \
  --img-server-ip 192.168.124.147 \
  --headless \
  "${diagnostics[@]}" \
  "$@"
