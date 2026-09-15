#!/usr/bin/env bash
# 大臂跟随专项复测：只比日常脚本多两件事 —— 40 Hz 诊断 + 自动建日志目录。
#
# 控制行为与 ./teleop/run_r1_a7_vector.sh 完全一致（速度前馈默认开、位置限速 30）。
# 日常遥操直接用 vector 脚本即可；这个脚本是给"改完要量化对比"用的。
#
#   ./teleop/run_r1_a7_follow_check.sh
#   ARM_DQ_FEEDFORWARD=off ARM_VELOCITY_LIMIT=3.0 ./teleop/run_r1_a7_follow_check.sh   # 方案 A
#
# 跑完把打印出来的 JSONL 路径给我，我用同一套脚本对比新旧数据
# （关键指标：命令>1.5×实际 的帧比例、停顿帧比例、跟随比中位数）。
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stamp="$(date +%Y%m%d-%H%M%S)"
diag_dir="${ARM_DIAG_DIR:-$HOME/r1-diag-follow-$stamp}"
mkdir -p "$diag_dir"

echo "[follow-check] 速度前馈=${ARM_DQ_FEEDFORWARD:-on} (上限 ${ARM_DQ_LIMIT:-6.0} rad/s)" \
     "位置限速=${ARM_VELOCITY_LIMIT:-30.0} rad/s 诊断=${ARM_DIAG_HZ:-40} Hz"
echo "[follow-check] 日志目录=$diag_dir"
echo "[follow-check] 建议动作：先慢速小幅活动，再做几次 2~3 秒的快速往返，最后静置 5 秒"
ARM_DIAG_HZ="${ARM_DIAG_HZ:-40}" ARM_DIAG_DIR="$diag_dir" \
  exec bash "$script_dir/run_r1_a7_vector.sh" "$@"
