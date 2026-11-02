# ACT 训练与策略执行

流程：录示范 → 导出 → 训练 → 离线评估 → 影子模式 → 实机执行。前四步不连接机器人。IMU 不参与。

下面的命令都在 Ubuntu 的 `/home/hnh/unitree_r1_dev/xr_teleoperate` 里运行。训练会占满机器，**不要在遥操或采集时训练**。

## 1. 导出

```bash
/home/hnh/unitree_r1_dev/.venv-xr/bin/python tools/export_r1_training_dataset.py \
  /home/hnh/unitree_r1_dev/teleop-recordings/任务名 \
  --output /home/hnh/unitree_r1_dev/training-datasets/任务名_act_日期
```

只导出标记为成功的录像。输出目录不能已存在。

- 新格式是 `r1_act_hdf5_v3`。旧的 v1/v2 不能直接读，要从原始录像重新导出。
- 一个片段是一段连续可用的**动作**，默认至少 40 帧。片段里观察也合格的行才作为训练样本。观察不合格的行不训练，但它的动作仍留在前面样本的动作序列里。
- 2026-09 之前的录像没有时钟映射信息，导不出样本。
- 只有一条来源录像时没有独立验证集。积累到多个采集场次后，用 `--val-source '*_20261107_*'` 把一整个场次留作验证。

## 2. 训练

```bash
/home/hnh/unitree_r1_dev/.venv-train/bin/python tools/train_r1_act.py \
  /home/hnh/unitree_r1_dev/training-datasets/任务名_act_日期 \
  --output /home/hnh/unitree_r1_dev/training-runs/任务名_日期
```

- 训练环境是独立的 `.venv-train`（GPU），依赖见 `tools/requirements-training.txt`。遥操的 `.venv-xr` 不变。
- 输出：`best.pt`（验证误差最低）、`latest.pt`（可续训）、`metrics.jsonl`、`run.json`。
- 中断或续训：Ctrl+C 会保存 `latest.pt`，之后同一命令加 `--resume`。
- 默认 50000 步、批大小 8、学习率 1e-5。`--chunk-size` 默认 40 帧，即 40 Hz 下的 1 秒。

## 3. 离线评估

```bash
/home/hnh/unitree_r1_dev/.venv-train/bin/python tools/eval_r1_act.py \
  /home/hnh/unitree_r1_dev/training-runs/任务名_日期/best.pt --split val
```

报告按身体部位给出真实单位的误差（弧度；手指是 0–1），并和两个基线对比：一直保持当前姿态、一直输出训练均值。**没有明显好于这两个基线的模型不要上实机。** 离线误差低不代表闭环能完成任务。

## 4. 影子模式（不会动）

```bash
./teleop/run_r1_a7_policy.sh /home/hnh/unitree_r1_dev/training-runs/任务名_日期/best.pt
```

读取真实反馈和相机，计算策略想发什么指令并记录，**不激活、不发送任何指令**。你可以用手或遥控器摆姿势，看输出是否合理。按 `q` 退出。结果在 `/home/hnh/unitree_r1_dev/policy-runs/时间/`：

- `summary.json`：推理延迟、观察不合格原因、策略目标和当前姿态的差。
- `ticks.jsonl`：每一帧的目标和实际姿态。

## 5. 实机执行

```bash
./teleop/run_r1_a7_policy.sh /home/hnh/unitree_r1_dev/training-runs/任务名_日期/best.pt --execute
```

先预检，再进入影子预览；按 `r` 才开始动。**`q` 和 Ctrl+C 不是急停，急停是机器人上的硬件按钮。** 手放在急停上，清空手臂活动范围。

启动时：
- 当前姿态超出训练范围会拒绝启动（`--allow-out-of-range-start` 可越过）。
- 策略的第一个目标离当前姿态超过 0.35 rad 会拒绝开始（`--max-start-delta`）。
- 从当前姿态过渡 2 秒到策略输出（`--blend-seconds`）。

运行时遇到以下情况会**保持当前姿态，且不会自动恢复**，要按 `q` 退出：
- 超过 0.5 秒没有新的策略输出（观察不合格或推理卡住）。从发出请求算起超过 0.5 秒才回来的输出不会执行，也不算新输出，数量记在 `summary.json` 的 `stale_chunks`。
- 手臂跟踪误差持续 0.5 秒超过 0.5 rad，例如撞到桌子。
- 推理进程退出，或运行满 60 秒（`--max-seconds`）。

目标还会被限制在训练时的动作范围内。手臂目标整形和遥操相同（6 rad/s、40 rad/s²）：训练用的动作本身就是经过它之后的目标，它的作用是消除速度前馈的抖动。`--arm-target-velocity-limit 0` 可以关掉，但指令会变成阶跃，动作会更抖。

结构上，模型跑在独立进程里（`tools/act_inference_worker.py`），推理卡住或崩溃都不会阻塞控制循环。
