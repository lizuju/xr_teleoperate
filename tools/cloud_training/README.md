# 云 GPU 训练工具包

在租来的 GPU 机器（优云智算、AutoDL 等）上训练 24 维 ACT 模型，训练完把模型取回 Ubuntu 上机。

## 文件

| 文件 | 在哪运行 | 作用 |
| --- | --- | --- |
| `make_package.sh` | Ubuntu | 按划分文件打包：划分、episode 列表、代码、脚本 |
| `upload.sh` | Ubuntu | 上传训练包、ResNet18 权重和原始录制，断了重新运行即可续传 |
| `run_training.sh` | GPU 机器 | 一键执行：环境检查（自动对齐 torch 2.5.1）→ 解包 → 导出 → 训练 → 验证集评估，可续跑 |
| `download.sh` | Ubuntu | 取回 `best.pt` 等结果 |
| `make_filtered_dataset.py` | GPU 机器 | 从已有导出中去掉部分训练录制，不用重新导出 |
| `split.example.json` | — | 划分文件模板（第 2 轮实际用的划分） |

## 划分文件

格式见 `split.example.json`：

| 字段 | 内容 |
| --- | --- |
| `train`、`val`、`test` | episode 目录名列表 |
| `exportable_frames` | 每条 episode 的可导出帧数，用于估算导出需要的磁盘空间 |
| `created`、`source_task`、`rule` | 说明：哪天、哪个任务、按什么规则选的 |

- 按**整条录制**划分，同一条 episode 只能在一个部分里。
- 不同轮次之间，验证集和测试集保持不变，结果才能比较。
- 测试集在选定最终模型之前不用。

模板里第 2 轮的划分：
- 52 条可导出帧 ≥ 90% 的成功示教，加上 2026-11-20 新录的 19 条好数据，共 71 条。
- 训练 61、验证 5、测试 5。

## 1. 打包（Ubuntu）

```bash
cd ~/unitree_r1_dev/work/pr-20261110/xr_teleoperate   # 任何 xr_teleoperate 的 git 副本
tools/cloud_training/make_package.sh my-split.json ~/unitree_r1_dev/autodl/<名字>
```

- 打包的是当前 git 副本**已提交**的代码（`CODE_COMMIT` 记录提交号）。先确认这个提交和 Ubuntu 上部署的一致。
- 原始录制从 `~/unitree_r1_dev/teleop-recordings/icecream_dex1` 读取。

## 2. 租机器

**GPU：** RTX 4090（24G 或 48G）、3090，或 A800、A100、H20。

不要选以下这些：
- **RTX 50 系、RTX 6000D、RTX PRO 6000**：太新，需要 PyTorch 2.7+，和 Ubuntu 上运行模型的 2.5.1 对不上。脚本遇到会直接停下。
- **P40、V100S、2080**：太旧，不支持训练用的 bf16。

模型很小，只用约 1.4 GB 显存，24G 就足够。

**其他配置：**
- 1 卡，按量计费。
- CPU 选核数多的一档。读训练数据靠 CPU，≥16 核时加 `NUM_WORKERS=8`。
- 镜像：任意 PyTorch 镜像。版本不是 2.5.1 时，第一次运行会自动装成 2.5.1（下载约 2.5 GB）。
- 磁盘：320×384 时每条录制约需原始 0.55 GB 加导出 2.2 GB。52 条至少 160 GB（建议 200 GB），71 条至少 200 GB（建议 240 GB）。240×288 大约省一半导出空间。`run_training.sh` 导出前会检查空间，不够就停下。
  - 优云智算：调大“系统盘”。
  - AutoDL：用数据盘，上传时加 `REMOTE_DIR=/root/autodl-tmp/icecream52`。

## 3. 免密登录（只需一次，密码由你自己输入）

```bash
ssh-copy-id -p 端口 用户@主机
```

端口、用户、主机都来自平台给的 SSH 登录命令。

## 4. 上传（Ubuntu）

```bash
cd ~/unitree_r1_dev/autodl/<名字>
./upload.sh -p 端口 用户@主机
```

## 5. 训练（GPU 机器）

```bash
cd ~/icecream52
NUM_WORKERS=8 TAG=71 NAME=icecream_act24_71_320x384_pause4 \
  EXTRA_TRAIN_ARGS="--max-pause-seconds 4" nohup bash run_training.sh > run.log 2>&1 &
tail -f run.log
```

- 关掉 SSH 也会继续跑。中途停了，再运行同一条命令：已完成的导出直接复用，训练从 `latest.pt` 续跑。
- 验证分数连续 10 次没有提高就提前停。
- 常用设置：

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `TAG` / `NAME` | `52` / `icecream_act24_<TAG>_<IMAGE_HW>` | 数据集和训练结果的名字 |
| `IMAGE_HW` | `320x384` | 导出图像尺寸 |
| `STEPS` `BATCH` `ACCUM` `CHUNK` | 50000 2 4 40 | 训练步数、batch、梯度累积、动作块长度 |
| `EXTRA_TRAIN_ARGS` | 空 | 传给 `train_r1_act.py` 的其他参数，如 `--max-pause-seconds 4`、`--qpos-noise 0.05` |
| `NUM_WORKERS` | 4 | 读数据进程数 |
| `WAIT_PID` | 空 | 导出完后等这个进程（比如上一轮训练）结束再开始训练 |

- 只想去掉部分训练录制、复用已有导出时，先用 `make_filtered_dataset.py` 生成新的数据集目录，再用对应的 `TAG` 运行。

结束时在验证集上评估，结果写到 `runs/<名字>/eval_val.json`。先看模型误差是否低于“保持当前姿态”和“训练均值”两个基线。

## 6. 取回模型（Ubuntu）

```bash
./download.sh -p 端口 用户@主机 <名字>
```

- 文件放到 `~/unitree_r1_dev/training-runs/<名字>/`：`best.pt`、`run.json`、`metrics.jsonl`、`eval_*.json`。
- **用完记得关机或释放实例。**

## 7. 上机（先影子模式）

```bash
cd ~/unitree_r1_dev/xr_teleoperate
./teleop/run_r1_a7_policy.sh ~/unitree_r1_dev/training-runs/<名字>/best.pt \
  --shadow-seconds 60 --img-server-ip 192.168.50.165
```

- 影子模式只读机器人状态和相机、只记录模型的预测，不发送任何动作。
- 离线评估和影子模式都通过后，才由操作者在现场用 `--execute` 实际运行（见 `docs/icecream_il24.md`）。
