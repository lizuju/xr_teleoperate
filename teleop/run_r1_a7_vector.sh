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
#   ARM_DQ_FEEDFORWARD  默认 on —— 把目标速度写入电机 dq，让手臂跟着走而不是纯位置追。
#                       Python / 控制器默认本来就是 on。脚本曾为对齐 09-14 写成 off，
#                       同时位置限速已从 3.0 撤回到 30（约等于不限），于是既没有削尖峰
#                       也没有前馈。09-15 实测 dq=0 时跟随比 p50 只有 0.22–0.69。
#                       关掉：ARM_DQ_FEEDFORWARD=off
#                       要退回“削命令”的保守方案：ARM_DQ_FEEDFORWARD=off ARM_VELOCITY_LIMIT=3.0
#   ARM_DQ_LIMIT        前馈速度上限，默认 6.0 rad/s（安全网，不是跟踪上限）
#   ARM_VELOCITY_LIMIT  位置目标的安全限速，默认 30.0 rad/s（约等于不限）
#   WRIST_DISPLAY       头显上要不要显示左右腕部相机小窗，默认 **off**（不显示）。
#                       为什么默认关：两路面板每 30 Hz 无条件重新编码重发，而画面内容只有约 12 Hz
#                       是新的，实测单块 base64 60.7 KB、两块合计约 3.5 MB/s，全压在同一条 Wi-Fi 上，
#                       和手部跟踪的上行抢信道；Vuer 子进程也因此常年吃满一个核。
#                       实测同一天：链路空闲时手部数据年龄 p50 24 ms（不卡），链路忙时 p50 106 ms、
#                       7% 的帧超时导致手臂冻结（卡）。关掉面板省下这部分下行，给上行让路。
#                       取值 off(默认) / auto(相机在就显示) / both / left / right。
#                       不影响录制：腕部画面照样进 episode 的 color_2 / color_3。
#   WAIST_FOLLOW_THRESHOLD_DEG  腰跟随的触发门槛，默认 20 度。
#                       注意比的不是头的绝对偏航角，而是「残差」= 你累计转了多少头 − 腰已经跟了多少。
#                       残差超过这个值并保持 0.2 s 才开始跟；跟到残差落到 5 度以下就停（迟滞带）。
#                       调大 = 小幅度转头完全不动腰；调小 = 更容易带动腰。
#   WAIST_FOLLOW_SPEED_DEG  跟随时腰目标的最大角速度，默认 40 度/秒（旧值写死 20）。
#                       旧值追一个 45 度的转身要 2.2 s，体感就是机器人拖在你后面。
#   WAIST_FOLLOW_ACCEL_DEG  腰目标的加速度上限，默认 90 度/秒²（旧值写死 28.6）。
#                       40 度/秒 配 90 度/秒² → 0.44 s 到全速。
#   WAIST_FOLLOW_COMPENSATION  腰部转动时手腕目标保持在哪个坐标系，默认 torso。
#                       torso = 手臂相对躯干的姿态完全按你命令的来，腰一转整条手臂跟着身体一起转，
#                               关节一动不动 —— 对应真人「腰转、手跟着身体走」的关系。
#                       world = 把目标绕骨盆轴反向转，让手留在世界坐标里不动；手不动时腰一转，
#                               手臂只能靠折叠去补偿，实测肘的活动量放大 35 倍（手静止 2mm 内：
#                               腰不动 0.10°/周期，腰转 >2° 时 3.51°/周期），这就是「手臂乱扭」的来源。
#                       world 只在「有东西必须钉在世界坐标里」时才有意义，而腰跟随是看头触发的，
#                       不属于这种情况。设成 world 只为了复现旧行为做对比。
#   ARM_DIAG_HZ         设了就按该频率记录诊断（复测大臂问题建议 40）
#   ARM_DIAG_DIR        设了就把诊断 JSONL 写进该目录
#   CAMERA_CALIBRATION  相机标定 JSON 路径；留空则用 assets/r1/camera_calibration.json（存在才读）。
#                       标定结果会写进每个 episode 的 info.camera_calibration，供后面数采/训练使用。
#
# 例：
#   ./teleop/run_r1_a7_vector.sh                        # 日常遥操（dq 前馈 on，位置限速 30 ≈ 不限）
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
  --camera-calibration "${CAMERA_CALIBRATION:-}" \
  --network-interface eno1 \
  --img-server-ip 192.168.124.147 \
  --headless \
  "${diagnostics[@]}" \
  "$@"
