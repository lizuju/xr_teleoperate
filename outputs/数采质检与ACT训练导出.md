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
3. 已发布的双臂/双手命令有效且不超过 250 ms，双手处于控制模式；手臂请求目标新鲜。
4. 新版录像要求相机时钟映射、双目配对、反馈时间对齐有效。旧录像只能使用已有的接收时间配对，会标记 `legacy_host_receive`，不假称完成源端同步。
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

HDF5 使用 ACT 常见的 `observations/qpos`、`observations/images/*`、`action` 布局，另保存原始帧索引和时间作为溯源信息。

| 项目 | 约定 |
| --- | --- |
| 状态/动作维度 | 29：左臂 7、右臂 7、左手 6、右手 6、腰/头 3 |
| 单位 | 手臂和腰/头为弧度；手指为设备归一化位置 0–1；具体顺序在 dataset.json |
| 输入状态 | 同一个控制采样时刻的 `states` 实测关节位置；没有把 `aligned_states` 的历史反馈混进来 |
| 动作标签 | 同一行的 `actions` 请求位置，保留其在发布限幅/平滑前的含义；没有自动前后错移一帧 |
| 部署约定 | 将策略输出接到与示范一致的目标整形、限幅和手部平滑入口；不可把这些标签直接当作裸电机指令 |
| 四路图像 | head_left、head_right、left_wrist、right_wrist；统一缩放为 320×240 RGB，保留原图方向。原始标定与缩放比例存入清单 |
| 采样时间 | 保留原始控制样本和实际时间间隔，标称通常为 40 Hz；不把重复图像当作新的视觉帧，不做重采样 |
| 观测延迟 | 图像和关节仍保留各自接收时间。软件配对合格不表示硬件曝光同步，也没有自动补偿未知的相机/人类反应延迟 |
| 手部请求时间 | 原数据没有独立的手部请求时间戳，使用采样时刻定位快照；不把发布时间伪装成请求时间 |
| IMU | 不写入 HDF5 训练数据，不传给训练读取器；原始录像中的 IMU 保留 |
| qvel | 为常用 HDF5 布局提供关节位置的离线有限差分值，明确标为估计量；默认读取器不将其作为模型输入 |

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
