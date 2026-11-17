# 雪糕任务：左 O6＋右 Dex1 的 24 维模仿学习

这套流程使用已有成功示教，从离线训练、独立评估、只读推理走到实机运行与纠错。基础 ACT 和带阶段条件的 ACT 共用同一套 24 维动作、图像处理和执行接口。IMU 不进入模型。既有 Safari／Vision Pro 遥操启动方式不变；历史右腕图像的原地颜色修正见文末。

当前任务是**右 Dex1 夹杯、左 O6 操作拉杆**。阶段模型会在接料与关杆阶段固定右臂、右夹爪及腰部目标；左臂、左手和头部继续由策略输出。

## 1. 数据与动作约定

| 索引（从 0 开始） | 内容 | 维数 | 单位 |
| --- | --- | ---: | --- |
| 0–6 | 左臂 | 7 | rad |
| 7–13 | 右臂 | 7 | rad |
| 14–19 | 左 O6 | 6 | 0–1 归一化 |
| 20 | 右 Dex1 | 1 | rad |
| 21 | 腰 yaw | 1 | rad |
| 22–23 | 头 pitch、yaw | 2 | rad |

观测是四路 RGB（头左／头右／左腕／右腕）和时间对齐的 24 维实际关节位置；输出是 24 维**绝对目标位置**。右 Dex1 不被错误压缩到 0–1。部署从 checkpoint 读取录制时的夹爪行程、lowcmd 33 号槽位、kp、kd 和 grip_delta。

图像统一完整缩放并补黑边到 384×320，不做中心裁剪或左右翻转。左 O6 保留其相机所需的红蓝交换，右 Dex1 相机已经是正常 BGR，不能再套用 O6 的交换。新录制带有 `image_color_profile=r1_wrist_device_rgb_v1`；已核验旧版 Dex1 记录在导出时仅修正 `color_3`。四路相机原始尺寸、相机顺序、关节顺序和单位都必须匹配。

当前不依赖手眼标定来学习 RGB＋关节到动作的映射，但这不代表能随意改变机位、相机方向、机器、夹爪或桌面布局。右腕几何标定尚不足以支持可信的几何定位；不把 IMU 当作有效输入。

数据只取完整且标注 success 的原始 episode。图像时间、反馈时间、目标接受时间、断更、控制模式和配对质量按现有质检规则筛选。无效观测不能作为训练输入；某一行的观测无效而已接受动作仍有效时，动作可作为此前有效观测的未来监督。

40 Hz 是动作查询网格，不是每秒 40 张新相机图像。动作块根据实际接受时间在 `t+k/40` 上因果取样；沿用导出器根据连续控制帧、实际时刻和控制序号核验的片段，在片段终点或阶段边界后遮罩。夹爪更新慢于录制时，保留已接受目标的正常保持；不能把重复目标错误地当成丢帧。不同片段之间不插值、不制造缺失动作。

## 2. 固定划分与模型

2026-11-16 这一版冻结 15 条成功记录，按完整原始 episode 分组，不能把同一条的片段随机拆进不同集合：

- 训练 10 条：0000、0002、0003、0004、0006、0009、0013、0014、0016、0022。
- 验证 3 条：0001、0005、0018，用于选择 checkpoint 和早停。
- 测试 2 条：0010、0020，保留作最终报告，不能根据测试分数反复调参。

`dataset.json` 保存源文件哈希、来源、剔除原因、分段、动作定义、图像变换、划分和仅由训练集计算的归一化量。训练拒绝跨集合重复的源内容；评估根据源文件哈希和路径重新判断独立性。这些仍是同一天、同场景的独立 episode，不能证明跨天或换布局泛化。

基础模型：ImageNet 预训练 ResNet18（四相机共享主干、独立相机嵌入）＋ACT CVAE／Transformer；hidden 512、4 层 encoder、1 层 decoder、40 步动作块、32 维 latent。训练用 batch 2、累计 4 次梯度（有效 batch 8）、bf16、AdamW、梯度裁剪、轻微颜色扰动。不开随机几何裁剪。按原始 episode 均衡采样；阶段模型按 episode＋阶段均衡采样。

阶段条件模型另外输入 8 维上下文：5 阶段 one-hot、当前阶段经过时间、左右手锁定标志。它同时预测阶段完成概率；完成分支只看当前图像、关节和上下文，不能看包含未来动作的 VAE latent。动作块不跨人工标注的阶段边界。

## 3. 路径与环境

本次隔离源码：

```bash
cd /home/hnh/unitree_r1_dev/work/icecream-il24-20261116/xr_teleoperate
export R1_DEV_ROOT=/home/hnh/unitree_r1_dev
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH=.:tools:teleop/teleimager/src:teleop/televuer/src:teleop/robot_control/dex-retargeting/src
```

训练 Python 为 `$R1_DEV_ROOT/.venv-train/bin/python`，实机接口 Python 为 `$R1_DEV_ROOT/.venv-xr/bin/python`。隔离目录的 `assets` 和父目录的 `linkerhand-urdf` 链接到现有模型文件，仅供读取。`R1_DEV_ROOT` 使策略启动脚本复用既有 Python 环境，不需要把代码覆盖到生产目录。

当前实测状态（2026-11-16）：15 条成功记录导出为 211 个连续片段，合格观测 25,287 个，其中 17,188 个有完整 40 步监督。训练／验证／测试分别为 16,237／5,211／3,839 个观测。初始审计曾发现把正常夹爪保持误判成缺口的问题，加载器已修正并重新训练。

已导出的基础数据集：

`/home/hnh/unitree_r1_dev/training-datasets/icecream_24d_20261117_rgb_v2`

生成另一份数据集时，必须用新的输出目录；原始数据、已有导出和已有训练运行均不覆盖。复现固定来源的导出示例：

```bash
"$R1_DEV_ROOT/.venv-train/bin/python" - <<'PY'
from tools.export_r1_training_dataset import export_dataset
ids = [0,1,2,3,4,5,6,9,10,13,14,16,18,20,22]
export_dataset(
    '/home/hnh/unitree_r1_dev/teleop-recordings/icecream_dex1',
    '/home/hnh/unitree_r1_dev/training-datasets/icecream_24d_reexport',
    include_sources=[f'episode_{i:04d}_20261116_*' for i in ids],
    val_sources=[f'episode_{i:04d}_20261116_*' for i in [1,5,18]],
    test_sources=[f'episode_{i:04d}_20261116_*' for i in [10,20]],
    image_hw=(320,384))
PY
```

## 4. 基础 ACT 训练、恢复与评估

正式训练的命令如下；不要把本次短流程测试的 checkpoint 当作可用雪糕策略：

```bash
"$R1_DEV_ROOT/.venv-train/bin/python" tools/train_r1_act.py \
  "$R1_DEV_ROOT/training-datasets/icecream_24d_20261117_rgb_v2" \
  --output "$R1_DEV_ROOT/training-runs/icecream_act24_baseline_rgb_v2" \
  --steps 50000 --batch-size 2 --accumulate 4 --chunk-size 40
```

训练输出 `run.json`（配置、源码和数据哈希）、`metrics.jsonl`、`best.pt`、`latest.pt`。模型只用验证集选优。验证连续 10 次不改善时早停。checkpoint 原子写入，包含模型、归一化、硬件约定；`latest.pt` 额外保留优化器、随机数和采样位置。

恢复时重复原命令并加 `--resume`。数据哈希、batch、累计次数、上下文、增强、seed、KL 权重必须一致。每个微批次有确定性的采样种子；Ctrl+C 在梯度累计中途不会把不完整优化步骤覆盖成恢复点，恢复从最近一次完整保存继续。浮点运算不承诺跨 GPU／驱动逐位一致。

```bash
"$R1_DEV_ROOT/.venv-train/bin/python" tools/eval_r1_act.py \
  "$R1_DEV_ROOT/training-runs/icecream_act24_baseline_rgb_v2/best.pt" \
  --split val --batch-size 2

# 配置选定后，再做一次最终 test 报告。
"$R1_DEV_ROOT/.venv-train/bin/python" tools/eval_r1_act.py \
  "$R1_DEV_ROOT/training-runs/icecream_act24_baseline_rgb_v2/best.pt" \
  --split test --batch-size 2
```

报告保留 24 个关节的真实单位 MAE、左右臂／左右末端／身体分组、多个预测时长，与“保持当前姿态”和“训练动作均值”对比；另有移动／保持两类误差，避免等待接料时的静止帧掩盖运动错误。阶段模型还有分阶段误差和完成判断的 TP／FP／FN。离线误差不是闭环成功率。

```bash
"$R1_DEV_ROOT/.venv-train/bin/python" tools/report_r1_il.py \
  --dataset "$R1_DEV_ROOT/training-datasets/icecream_24d_20261117_rgb_v2" \
  --output "$R1_DEV_ROOT/work/icecream-il24-20261116/dataset-audit.json"
```

## 5. 阶段标注和条件训练

五阶段边界采用原始帧编号，左闭右开：

1. `grasp_cup`：开始到杯子夹稳并提起。
2. `turn_deliver`：转身，把杯子送到出口。
3. `dispense`：操作拉杆并接料，直到准备关杆。
4. `close_retract`：关杆、左手退出，右手保持杯子。
5. `turn_place`：右手收回、转回、放杯松开。

`tools/annotate_r1_icecream.py` 已生成 `phase-review/review.html` 和 `candidates.json`。页面展示头部、左腕、右腕抽帧，编辑四个内部边界后逐条勾选已核验，再下载 `reviewed_phases.json`。候选来自夹爪、腰部和锁手状态；接料结束仅是占位建议，必须核看图像。抽帧不足以判断时回看原视频。

候选默认 `reviewed=false`。导出会核对原始 `frames.jsonl` 哈希和边界范围，拒绝未核验标注。将第 3 节导出命令改用新的输出目录（例如 `icecream_24d_phased_v1`），并加入：

```python
annotations='/home/hnh/unitree_r1_dev/work/icecream-il24-20261116/reviewed_phases.json'
```

随后重复训练命令，更换数据／运行目录并加 `--conditioned`。它是同一个共享策略，不是五个互不相关的模型。阶段输出和完成概率目前需要独立验证；无需等待标注就可以先训练基础 ACT。

## 6. 推理与实机运行

先只读检查，不需要戴头显；模型直接读取机器人摄像头和实际关节。

```bash
./teleop/run_r1_a7_policy.sh \
  "$R1_DEV_ROOT/training-runs/icecream_act24_baseline_rgb_v2/best.pt" \
  --shadow-seconds 60 --img-server-ip 192.168.50.165
```

默认 shadow：不激活、不发送动作。记录预测、实际状态、输入拒绝原因、推理耗时和往返耗时。训练与推理不可争用同一块 GPU 做正式延迟测量。网络地址是当前摄像头网关，拓扑变化后需调整。

实机运行需在模型离线评估和只读检查通过、现场就绪后显式使用 `--execute`：

```bash
./teleop/run_r1_a7_policy.sh /path/to/validated/best.pt \
  --execute --max-seconds 90 --img-server-ip 192.168.50.165
```

- `r`：开始或从手动暂停恢复；先检查起始关节范围和首个目标差。
- `p`：保持当前命令目标并暂停。
- `n`：阶段模型人工确认下一阶段；基础模型无阶段切换。
- `q`：退出程序，不等同机器人硬件急停。

推理在单独进程，主循环按 40 Hz 消费预测动作块并做时序融合。返回值按发起请求的时间计龄，过期预测不执行；延迟时按真实时间前进，阶段切换和暂停恢复清空旧块。首段和暂停恢复做渐入，普通阶段切换不重复两秒渐入。整次运行的时限不会被换阶段重置。

默认阶段切换由 `n` 确认。加 `--auto-stage` 才启用“模型完成概率＋关节反馈＋持续确认”切换；只允许用于有核验阶段标签训练的模型。当前逻辑是特定雪糕场景：转向起始腰角减 90°，返回起始腰角，夹爪闭合／释放等。它不能凭关节判断杯子确实夹住、雪糕已流出或拉杆确实复位。

`--dispense-min-phase-seconds 3` 仅代表接料**阶段**至少持续 3 秒，**不是拉杆到位后精确接料 3 秒的计时器**。需要精确时序时，应增加经过实测的拉杆位置或流量事件判据；在此之前用人工阶段确认验证。阶段模型的完成信号也不能直接当作任务成功标签。

夹爪使用录制行程与力限制，手指保留已有平滑，手臂保留当前目标整形与速度前馈。姿态超范围、观测过旧、推理失败、持续跟踪误差等情况下保持目标，不能用放大超时掩盖坏输入。

## 7. 实机评估、纠错和迭代

每次运行生成独立 `policy-runs/<时间>/run.json`、`ticks.jsonl`、`summary.json`，包含 checkpoint 哈希、动作、实测状态、阶段事件及耗时。成功与失败由实际观察评分：

```bash
"$R1_DEV_ROOT/.venv-train/bin/python" tools/record_r1_policy_result.py \
  "$R1_DEV_ROOT/policy-runs/<本次运行>" \
  --outcome failure --failure-phase dispense --reason "拉杆没有拉到位"
```

需要纠错时先退出策略进程，再使用原有遥操采集脚本录一条人类示教；不能让两个进程同时写电机。成功示教完成后可通过 `--correction /path/to/episode` 关联到该次 intervention。这里只记录人工反馈和示教来源，**没有实现无缝同时接管，也不会把模型自己输出的动作自动当成人类真值**。

```bash
"$R1_DEV_ROOT/.venv-train/bin/python" tools/report_r1_il.py \
  --policy-runs "$R1_DEV_ROOT/policy-runs" \
  --output "$R1_DEV_ROOT/work/icecream-il24-20261116/task-results.json"
```

按 checkpoint 统计成功、失败、干预、失败阶段；shadow 不计入成功率。新增纠错示教进训练集，保留原验证／测试来源；新一天、新杯位的独立测试另建版本。训练不负责自动发现已挂满的输出盘，不自动上传私人录制或权重。

## 8. 如何判断值得上实机

先能过拟合少量成功示教，再在独立验证的移动帧上超过保持／均值基线；阶段完成误报尤其要检查。固定机位和桌面，先做一阶段，再两阶段连接，最后全流程；每种策略记录相同任务的真实成功／干预和失败原因。

现有 15 条能搭流程和建立初始基线，但动作和场景变化有限。根据失败集中补采杯位偏移、接近角度、夹杯失败后恢复、拉杆接触、放杯等示教。不要只靠更多重复等待帧扩大数据量。先把模仿学习闭环做实，再考虑仿真或强化学习；当前没有需要接入 RL 的环节。

## 首轮流程验证结果（2026-11-16，旧颜色，仅作历史记录）

- RTX 4060 8 GB 实际完成基础模型 100 步训练，并从 `latest.pt` 恢复到第 101 步；有效 batch 8，峰值 PyTorch allocated 显存约 1,338 MB。
- 短测输出为 `/home/hnh/unitree_r1_dev/training-runs/icecream_act24_smoke_v2_20261116`。这是流程检查模型，不是正式训练模型，不能据此宣称任务成功。
- 单独推理进程对真实输入连续执行 50 次：动作块均为 40×24；推理 P50 7.05 ms、P95 8.60 ms、最大 9.50 ms；进程往返 P95 12.41 ms。这不含实机图像传输、机械响应或闭环控制延迟。
- 实机 shadow 尝试在 DDS 初始化时退出：当时 `eno1` 为 DOWN 且无 IP。未改网络，也未发机器人动作。完整实机只读观测和闭环效果待恢复机器人连接后验证。
- 人工阶段标注尚未核验；阶段导出、条件模型训练／完成分支、评估和协议已用合成夹具回归验证，尚无真实阶段模型。
- 完整离线评估仅使用验证集；测试集继续保留，等待正式模型选定后评估。

101 步短测的独立验证集结果（5,211 个观测）：

| 分组 | 单位 | 模型 MAE | 保持当前姿态 MAE | 训练均值 MAE |
| --- | --- | ---: | ---: | ---: |
| 左臂 | rad | 0.1346 | 0.0350 | 0.2220 |
| 右臂 | rad | 0.1708 | 0.0454 | 0.1800 |
| 左 O6 | 0–1 | 0.1180 | 0.0284 | 0.1440 |
| 右 Dex1 | rad | 0.2248 | 0.0790 | 0.2865 |
| 腰与头 | rad | 0.1102 | 0.0199 | 0.2780 |

短测模型尚未超过保持基线，不能上线自主操作。完整报告另含移动／保持分层误差；正式训练应继续观察独立验证分数和实际任务效果。

最终回归：训练环境 99 项（89 通过、10 因机器人 SDK／Pinocchio 环境缺失跳过）；运行环境 41 项（32 通过、9 因训练模型依赖缺失跳过）。两个环境共同覆盖模型、带阶段标注的导出→训练前向／反向→离线评估、二进制协议，以及模拟控制器的只读／停顿／超时／阶段计时行为。真实机器人未执行。

## 2026-11-17 右腕颜色修复

右 Dex1 相机曾错误套用 O6 的红蓝交换，导致红色物体呈蓝紫色。现已按相机所在的末端设备处理：左 O6 交换，右 Dex1 不交换；新的实时推理和采集保持一致。

先在原始录制不变的情况下，使用 `tools/correct_r1_dataset_colors.py` 从旧导出创建新的 `icecream_24d_20261117_rgb_v2`，仅对确认受影响的右腕 RGB 张量交换红蓝通道，不再压缩 JPEG；其他相机、动作、归一化和划分保持相同。迁移时来源的 episode／frames 文件哈希已匹配。重复修正已带颜色标记的数据集会被拒绝。

新录制的颜色版本标记用于避免二次交换。没有颜色标记且来源代码版本未核验的旧 Dex1 记录不会被盲目处理，需先确认图像颜色。旧 29 维双 O6 数据按原有颜色处理。

**重训用本文的新数据集和新运行目录，不加 `--resume`。** 旧的流程短测和 `icecream_act24_baseline_v1` 模型使用错误颜色，仅留作历史产物。新运行接口拒绝没有正确颜色版本标记的旧 checkpoint。

标注页面在 `phase-review-rgb/review.html`。随后按用户要求修正完整原始录制时，将右腕图片引用从 JPEG 改为 PNG，因此 `frames.jsonl` 哈希已变化，阶段帧号和边界不变。工作目录与本地输出中的候选 JSON／标注页面已同步更新哈希；此前自行下载的 JSON 需按 `raw-color-correction.json` 中的旧新哈希映射更新，不能直接沿用旧哈希。

验证记录：训练环境颜色／导出／模型回归 104 项（94 通过、10 跳过）；隔离运行环境 108 项（99 通过、9 跳过）；生产采集／导出相关回归 108 项全部通过。只读抓取当前右腕 640×480 图像，直接保留 Dex1 正常 BGR 后红色正常显示；没有启动机器人控制。

训练集迁移时全量核对已通过：211 个片段、25,287 个右腕训练图像逐帧验证为原张量的红蓝互换；其他三路共 75,861 个压缩图像块逐块一致；12,238 个非图像数组一致。训练／验证／测试仍为 16,237／5,211／3,839 个观测。冻结训练集继续记录导出时的源 manifest／frames 哈希；与随后原地修正版本的映射记录在 `raw-color-correction.json` 及各原始 episode 的 `info.image_color_correction`。

已检查 `teleop-recordings` 下的全部任务：`icecream_dex1` 共 23 条均可识别为旧右腕换色版本，其中 15 条 success 已生成修正训练数据；7 条 discarded 和 1 条 unspecified 不进入训练。`dance` 4 条和 `icecream` 3 条为双 O6，不应套用 Dex1 反向修正。

2026-11-17 已按用户要求删除旧颜色训练导出 `icecream_24d_20261116_v1`，释放 24,810,910,437 字节，未另做备份；保留 `icecream_24d_20261117_rgb_v2`。修正版的 `derived_from` 继续保留旧目录及其 manifest 哈希作为历史来源记录。

随后按用户选择，将全部 23 条原始录制的 24,121 张右腕 JPEG 解码后交换红蓝通道，写为无损 PNG，逐张读回核对像素一致后删除旧 JPEG；完整 36,636 帧的非图像内容不变，其他三路共 74,629 张图像哈希不变。颜色版本标记已写入每条原始 episode，后续导出不再重复交换。抽查全部 211 个训练片段的 633 个观测，与修正原图经训练预处理后的像素一致。未额外保留旧颜色副本。PNG 占用更多空间，清理及修正后磁盘剩余约 22.07 GiB。

用户中断的 `training-runs/icecream_act24_baseline_rgb_v2` 测试运行已删除，当时只有配置及日志，尚未保存 checkpoint；可直接用本文命令在同一路径重新开始训练。
