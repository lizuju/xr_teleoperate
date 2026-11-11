#!/usr/bin/env bash
# R1_A7 实机遥操（Vector）— 2026-09-15
#
# 这里固定了 R1_A7 的启动参数，并把「大臂限速」做成默认值：
#   ARM_POSTURE_WEIGHT  把手臂姿态拉回激活时姿态的权重，默认 0.01 —— 这是防「手肘扭住」的那一项。
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
#   ARM_TARGET_VELOCITY_LIMIT  IK 目标整形的巡航上限，默认 6.0 rad/s。
#                       dq 前馈从这条整形后的轨迹求导，避免 Vision Pro/IK 突发台阶被伺服猛追。
#                       0 = 关掉整形。不是已撤销的 3.0 位置限速。
#   ARM_TARGET_ACCEL_LIMIT  整形加速度，默认 40 rad/s^2（0.15 s 到 6 rad/s）。这是去快动作顿挫的那一项。
#                       0 = 不限加速度。
#   WRIST_DISPLAY       头显上要不要显示左右腕部相机小窗，默认 **off**（不显示）。
#                       为什么默认关：两路面板每 30 Hz 无条件重新编码重发，而画面内容只有约 12 Hz
#                       是新的，实测单块 base64 60.7 KB、两块合计约 3.5 MB/s，全压在同一条 Wi-Fi 上，
#                       和手部跟踪的上行抢信道；Vuer 子进程也因此常年吃满一个核。
#                       实测同一天：链路空闲时手部数据年龄 p50 24 ms（不卡），链路忙时 p50 106 ms、
#                       7% 的帧超时导致手臂冻结（卡）。关掉面板省下这部分下行，给上行让路。
#                       取值 off(默认) / auto(相机在就显示) / both / left / right。
#                       不影响录制：腕部画面走 ZMQ 55556/55557，进 episode 的 color_2 / color_3。
#                       腕部 WebRTC（60002/60003）头显不用，PC2 上可以关，数采不受影响。
#   RECORD_MAX_TRACKING_AGE_MS  按 [s] 开录前，XR 手部数据允许的最大年龄，默认 100 ms。
#                       09-18 实测：链路好时 left_age_ms p50=24 ms，晚上差时 106 ms 且 7% 超 500 ms。
#                       超限会拒绝开录并打 [RECORD] not started；已经在录的 episode 只告警不停。
#                       关掉门控：RECORD_MAX_TRACKING_AGE_MS=0
#   WAIST_FOLLOW_THRESHOLD_DEG  腰跟随的触发门槛，默认 20 度。
#                       注意比的不是头的绝对偏航角，而是「残差」= 你累计转了多少头 − 腰已经跟了多少。
#                       残差超过这个值并保持 0.2 s 才开始跟；跟到残差落到 5 度以下就停（迟滞带）。
#                       调大 = 小幅度转头完全不动腰；调小 = 更容易带动腰。
#   WAIST_FOLLOW_SPEED_DEG  跟随时腰目标的最大角速度，默认 40 度/秒（旧值写死 20）。
#                       旧值追一个 45 度的转身要 2.2 s，体感就是机器人拖在你后面。
#   WAIST_FOLLOW_ACCEL_DEG  腰目标的加速度上限，默认 90 度/秒²（旧值写死 28.6）。
#                       40 度/秒 配 90 度/秒² → 0.44 s 到全速。
#   HAND_OCCLUSION_GUARD  默认 on：一只手挡住另一只手大半（头显看不到）、追踪姿态瞬间跳变、或两只手腕贴在一起时，
#                       把那只手当作丢失，机械臂和夹爪/灵巧手原地保持，看清楚 0.15 s 后恢复跟随。off 关闭。
#   WAIST_HEAD_FOLLOW  默认 off：腰只由方向键一键转身控制，人转头或转身不再带动机器人的腰（机器人的头仍跟着你的头转）。
#                       设为 on 恢复转头带腰；下面的 WAIST_FOLLOW_THRESHOLD/SPEED/ACCEL 只在 on 时起作用。
#   WAIST_TURN_SPEED_DEG  方向键一键转身（← 左转、→ 右转，到下一个 90 度档位：正前 / ±90 度）的最大角速度，默认 30 度/秒。
#                       比跟随慢，因为转身时手里可能端着杯子。转身时遥操员面朝前别动。
#   WAIST_TURN_ACCEL_DEG  一键转身的加速度和刹车，默认 60 度/秒²；30 度/秒配 60 度/秒² 转 90 度约 3.5 s。
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
#   R1_RIGHT_EE         右手腕装的是什么：linker_o6（默认）或 dex1（宇树 Dex1-1 夹爪，PC2 上跑 dex1_1_service）。
#                       不设时读 ~/.config/xr_teleoperate/right_ee 这个文件（一行，linker_o6 或 dex1），
#                       所以换手以后改一次文件即可，启动命令不用变。预检会按它检查右手的反馈话题。
#   R1_DEX1_MIN_Q       右 Dex1-1 最小目标角度（rad）；优先使用环境变量，否则读取
#                       ~/.config/xr_teleoperate/dex1_min_q。未设置为 0；用 tools/set_dex1_min_opening.py 保存杯子开口。
#                       手指合拢时到该位置停止，张开仍可全开；这不是毫米或夹持力标定。
#   WRIST_STABILIZER    默认 on：手指在动时对手腕目标做轻量平滑，时间常数 40 ms。
#                       不再设置小动作死区，细微位移和转动每帧都会响应。
#                       WRIST_STABILIZER_POSITION_MM（默认 3 mm）和 WRIST_STABILIZER_ROTATION_DEG（默认 3 度）
#                       限制的是滤波落后量，不是保持不动的范围；设为 0 则该分量直通。
#                       指尖速度低于 0.08 m/s 后保持 0.1 s，再用 0.1 s 退出平滑。
#                       可减轻快速抖动，无法消除与主动转腕同频的缓慢追踪偏移。
#                       关掉：WRIST_STABILIZER=off
#   R1_DEX1_GRIP_DELTA  夹爪命令最多领先实测角度多少（rad），默认 0.5（约 2.5 N·m 驱动力矩，开合更快）。
#                       原来是 0.18（约 0.9 N·m）。设了最小开口时，夹杯子的力由最小开口决定，不随这个值变大。
#   R1_DEX1_TRANSPORT   Dex1-1 的接法：lowcmd（默认，内走线；宇树 App 灵巧手类型设为 Dex1-1，主控把它当 rt/lowcmd 的一个关节）
#                       或 service（外接 485 转 USB 板，PC2 上跑 dex1_1_service）。R1_DEX1_SLOT 是内走线的关节位，默认 33。
#                       标定结果会写进每个 episode 的 info.camera_calibration，供后面数采/训练使用。
#
# 例：
#   ./teleop/run_r1_a7_vector.sh                        # 日常遥操（默认 immersive；dq 前馈 on，目标整形 6 rad/s / 40 rad/s^2）
#   ./teleop/run_r1_a7_vector.sh --display-mode ego           # 后缀覆盖：第一人称小窗 + 周围透传
#   ./teleop/run_r1_a7_vector.sh --display-mode pass-through  # 后缀覆盖：纯透传（无机器人相机平面）
#   ./teleop/run_r1_a7_vector.sh --check-only                 # 只跑预检，不启遥操
#   ARM_DIAG_HZ=40 ARM_DIAG_DIR=$HOME/r1-diag ./teleop/run_r1_a7_vector.sh
#
# 显示模式：脚本硬编码 --display-mode immersive，额外参数经末尾 "$@" 转发；
# argparse 后写覆盖先写。直接在命令后追加 --display-mode ego|pass-through，
# 不要加单独的裸 "--"（会结束选项解析，后面的 mode 进不了 argparse）。
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

right_ee_file="${HOME}/.config/xr_teleoperate/right_ee"
if [[ -z "${R1_RIGHT_EE:-}" && -r "$right_ee_file" ]]; then
  R1_RIGHT_EE="$(tr -d '[:space:]' < "$right_ee_file")"
fi
R1_RIGHT_EE="${R1_RIGHT_EE:-linker_o6}"
case "$R1_RIGHT_EE" in
  linker_o6|dex1) ;;
  *) echo "R1_RIGHT_EE must be linker_o6 or dex1, not '$R1_RIGHT_EE'" >&2; exit 2 ;;
esac
export R1_RIGHT_EE

export XR_TELEOP_CERT="${HOME}/.config/xr_teleoperate/cert.pem"
export XR_TELEOP_KEY="${HOME}/.config/xr_teleoperate/key.pem"

"$python" -u "${script_dir}/../tools/check_r1_teleop.py"
if [[ "$check_only" == true ]]; then
  exit 0
fi

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
  echo "[run] 诊断日志目录: ${ARM_DIAG_DIR}"
fi

cd "$script_dir"
exec "$python" -u teleop_hand_and_arm.py \
  --input-mode hand \
  --display-mode immersive \
  --arm R1_A7 \
  --ee linker_o6 \
  --right-ee "$R1_RIGHT_EE" \
  --dex1-transport "${R1_DEX1_TRANSPORT:-lowcmd}" \
  --dex1-slot "${R1_DEX1_SLOT:-33}" \
  --dex1-grip-delta "${R1_DEX1_GRIP_DELTA:-0.5}" \
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
  --waist-head-follow "${WAIST_HEAD_FOLLOW:-off}" \
  --waist-turn-speed-deg "${WAIST_TURN_SPEED_DEG:-30}" \
  --hand-occlusion-guard "${HAND_OCCLUSION_GUARD:-on}" \
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
  "${diagnostics[@]}" \
  "$@"
