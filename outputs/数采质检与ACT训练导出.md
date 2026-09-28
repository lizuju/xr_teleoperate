# 数采质检与 ACT 训练导出

原启动方式和按键不变。在 Ubuntu 项目目录运行：

```bash
cd /home/hnh/unitree_r1_dev/xr_teleoperate
./teleop/run_r1_a7_capture.sh 这次测试名 "这次测试目标"
```

`r → s → 动作 → y/n/x → s → 下一条`。`p` 仍是可选暂停，不需要每条都按。

## 每条保存后的自动质检

保存完成后，独立后台进程会依次检查每条数据，在同一个采集终端打印 `[QUALITY]`，并在该条录像目录生成 `quality.json`。下一条采集无需等质检结束。按 `q` 正常退出后，已提交的质检会继续完成。

报告区分操作结果和数据质量：

- 任务标签：成功、失败、丢弃、未标记。不会改写你的标签，也不会删除录像。
- 文件是否完整、视觉与关节合格样本比例、追踪保持数量、相机未对齐数量、连续片段数量、实际图像 FPS。
- 实际时间基准、观察跨度/年龄、双臂与手指请求时间差的 P95，以及各筛除原因的数量。
- `training_no_imu` 是本次导出器的筛选口径。原检查器的 `trainable` 保留其原含义，不能直接当作“整条可不经筛选投入训练”。
- 质检自身出错会打印“质检未完成”，不会把已保存录像当作写盘失败。

旧数据可手动补报告：

```bash
/home/hnh/unitree_r1_dev/.venv-xr/bin/python tools/check_teleop_episode.py \
  /home/hnh/unitree_r1_dev/teleop-recordings/icecream/episode_0000 --summary
```

## 导出成功示范

录制结束后，按任务目录导出到一个新的目标目录：

```bash
cd /home/hnh/unitree_r1_dev/xr_teleoperate
/home/hnh/unitree_r1_dev/.venv-xr/bin/python tools/export_r1_training_dataset.py \
  /home/hnh/unitree_r1_dev/teleop-recordings/这次测试名 \
  --output /home/hnh/unitree_r1_dev/training-datasets/这次测试名_act
```

也可以将源路径指定为某一条 `episode_*` 目录。输出目录必须不存在，且不能放在原始录像目录内；重复导出时使用新名字。这样不会覆盖原始数据或已有训练集。

默认只导出完整、结构校验通过、标记为 `success` 的 R1_A7 + O6 录像。`failure`、`discarded`、`unspecified` 会保留在源目录，导出清单记录跳过原因。不同任务应使用不同源目录；同一任务内每条的原始目标文字保存在清单中。

样本筛选条件：

1. 正在跟随；XR 与双手追踪年龄不超过 100 ms。
2. 机器人与双手反馈不超过 250 ms；四路图像完整、新鲜且相机配对合格。
3. 已发布的双臂/双手命令有效且不超过 250 ms，双手处于控制模式；双臂和手指请求年龄不超过 100 ms，两者时间差不超过 25 ms。两只手的成功发布记录必须对应同一个手部请求序号。
4. 图像必须具有有效的源时钟映射，相机时钟不确定度不超过 5 ms；双目和历史反馈对齐有效。包含双眼的图像与反馈总时间跨度不超过 25 ms，最旧观察年龄不超过 150 ms。仅有接收时间、缺少映射或手部请求时间/序号的旧样本会被筛除。
5. 不使用 IMU 的数值、有效标志或 IMU 包缺失数量作为视觉/关节样本的训练条件。原始文件的结构和质量标志一致性仍会被检查。

每个无效样本都会切断片段；采样间隔超过标称周期的 1.5 倍也会切段。默认保留至少 40 帧的连续片段，可用 `--min-frames` 修改。不会把追踪中断前后的动作直接拼起来，也不会插值补造丢失动作。

## 输出内容与时间含义

```text
目标目录/
  dataset.json
  quality/source_0000.json
  episode_0.hdf5
  episode_1.hdf5
  ...
```

当前输出 schema 为 `r1_act_hdf5_v2`。读取器拒绝旧版 `r1_act_hdf5_v1`，因为观察状态的时间含义已改变；请从原始录像重新导出到新目录，不能只改 schema 字符串。缺少源端时间映射、历史反馈或手部请求信息的旧录像仍能质检，但可能没有合格训练片段，无法通过重新导出补造这些信息。

HDF5 使用 ACT 常见的 `observations/qpos`、`observations/images/*`、`action` 布局，另保存原始帧索引和时间作为溯源信息。

| 项目 | 约定 |
| --- | --- |
| 状态/动作维度 | 29：左臂 7、右臂 7、左手 6、右手 6、腰/头 3 |
| 单位 | 手臂和腰/头为弧度；手指为设备归一化位置 0–1；具体顺序在 dataset.json |
| 输入状态 | 图像源时间附近的 `sample.aligned_states` 历史关节反馈，按记录的相机锚点配对；不使用该行最新 `states` 替代 |
| 动作标签 | 同一行的 `actions` 请求位置，保留其在发布限幅/平滑前的含义；没有自动前后错移一帧 |
| 部署约定 | 推理时按同样源时间规则配对图像与历史反馈、检查观察年龄；策略输出接入示范所用目标整形、限幅和手部平滑入口，不可直接作为裸电机指令 |
| 四路图像 | head_left、head_right、left_wrist、right_wrist；统一缩放为 320×240 RGB，保留原图方向。原始标定与缩放比例存入清单 |
| 采样时间 | 保留原始控制样本和实际时间间隔，标称通常为 40 Hz；不把重复图像当作新的视觉帧，不做重采样 |
| 观测延迟 | 保留观察锚点、各相机映射时间、历史反馈时间和请求时间，并记录观察跨度、年龄与观察到请求的延迟；软件配对不表示硬件曝光同步，也不补偿未知的人类反应延迟 |
| 手部请求时间 | 保存接受一对手部目标时的单调时间和请求序号，并与两手成功发布的 `request_sequence` 关联；发布时间不是物理执行确认 |
| IMU | 不写入 HDF5 训练数据，不传给训练读取器；原始录像中的 IMU 保留 |
| qvel | 对配对后的历史关节位置按控制决策时间做有限差分，可能包含重复观察；不是物理速度测量，读取器不将其作为模型输入 |

训练/验证按原始整条录像划分，固定随机种子；同一条拆出的所有片段只属于一个集合。归一化统计只使用训练集。只有一条可用原始录像时，不伪造验证集，验证读取器返回 `None`。

## 训练读取

项目已提供读取器，支持不同长度的片段、动作块补齐及掩码：

```python
from teleop.utils.act_dataset import load_act_data

train_loader, val_loader, stats, metadata = load_act_data(
    "/home/hnh/unitree_r1_dev/training-datasets/这次测试名_act",
    batch_size=8,
    chunk_size=40,
)
images, qpos, actions, is_pad = next(iter(train_loader))
# images: [B, 4, 3, 240, 320]，RGB，范围 0–1
# qpos: [B, 29]，用训练集统计归一化
# actions: [B, 40, 29]，用训练集统计归一化
# is_pad: [B, 40]，True 的补齐位置不得计入损失
```

模型的状态维度和动作维度都应设置为 `29`，相机名称使用 `metadata['camera_names']`。ACT 网络要求的图像归一化仍由网络端处理。

不要直接套用原始 ALOHA ACT 的全部默认配置：其中部分实现固定 14 维、假设每条等长，并对实机动作自动偏移一帧。使用这里的读取器可保留本数据集的维度、片段边界、划分与时间约定；模型本身仍需按 29 维配置。本次实现数据导出与读取，没有启动模型训练或机器人策略执行。

导出新增依赖为 `h5py==3.14.0`，已安装在当前 Ubuntu 的 `.venv-xr` 中；现有 NumPy、OpenCV、PyTorch 沿用原环境。换训练机器时，在其相应 Python 环境安装依赖：

```bash
python -m pip install -r tools/requirements-training.txt
```
