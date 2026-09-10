# O6 dex-retargeting 迁移（2026-09-09）

O6 已改用基于 URDF 的优化求解，不再运行原来的弯曲角/启发式映射，也不再加载个人 min/max 标定文件。机械臂 IK、头腰算法、实物 O6 驱动和安装模型未修改。

## 三种模式

启动主程序或手部仿真源时选择一个模式，默认 `vector`：

```bash
--linker-o6-method vector
--linker-o6-method position
--linker-o6-method dexpilot
```

- Vector：匹配腕部到五个指尖的向量，作为首次真人测试基线。
- Position：匹配腕局部坐标系中的五个指尖位置，不是世界坐标绝对位置。
- DexPilot：匹配指尖间及腕到指尖的向量，包含捏合时的距离投影；不等于给六自由度 O6 增加人体手指自由度。

当前固定腕基座且缩放相同，Vector 与 Position 的目标几何很接近，但优化损失不同。不能仅凭算法名字断言哪种最适合你的手型。

原命令中的 `--linker-o6-calibration ...` 必须删除。旧 `LinkerO6Calibration`、标定 JSON 和 `tools/validate_linker_o6_calibration.py` 已从运行代码移除，不提供兼容分支；历史记录未删除。电机顺序、硬件方向、超时看门狗和模型关节边界不是个人标定，仍然保留。

## 先做无动作验证

Ubuntu 中使用现有虚拟环境，先确认没有另一个程序占用 8012。以下是手部隔离 dry-run，不初始化机械臂、DDS 发布器或串口：

```bash
cd /home/hnh/unitree_r1_dev/xr_teleoperate/teleop
/home/hnh/unitree_r1_dev/.venv-xr/bin/python teleop_hand_and_arm.py \
  --input-mode hand --ee linker_o6 --hand-only --dry-run \
  --display-mode pass-through --img-server-ip 192.168.124.147 \
  --linker-o6-method vector --record \
  --task-dir /home/hnh/unitree_r1_dev/o6-dex-recordings --task-name vector
```

图像服务地址应与当前图像服务一致，不因算法迁移而更改证书或网络配置。Vision Pro 连入后，双手有效时按 `r` 输出目标；按 `s` 开始/停止 JSONL 录制；按 `q` 退出。切换算法需退出后以另一 `--linker-o6-method` 重启，避免叠开会话。此处的 dry-run 按 `r` 不驱动实物；不要与普通真机主程序的 `r` 混淆。

已有 JSON 仿真预览的输入源也已迁移：

```bash
bash /home/hnh/unitree_r1_dev/xr_teleoperate/teleop/run_visionpro_r1_a7_o6_sim.sh --linker-o6-method vector
```

这条命令只启动 Vision Pro 输入源，Isaac 接收端需要单独运行。两个现有 `r1_a7_o6_live_ik_preview.py` 默认接受三种新的 `linker_o6_dex_*_v1` mapping，并校验 `retargeting_method`；原预览命令若显式传入旧 `--expected-hand-mapping`，应删除该参数。独立键盘/合成源可显式指定其自身 mapping，但不会恢复旧的 Vision Pro 手部算法。

## 求解模型与依赖

- 复用 Ubuntu 当前 Unitree vendored `dex-retargeting 0.4.7`、Pinocchio 3.1.0 和 NumPy 1.26.4；本次未安装或升级依赖。不要直接用最新上游包覆盖现有环境：上游新版本配置 API 与当前分支不同。
- 输入是 TeleVuer 已转换的 25 个腕局部点，单位米，指尖索引 `[4, 9, 14, 19, 24]`。不能使用 MediaPipe 21 点默认索引。
- 输入到 O6 根坐标的旋转为 `[[-1,0,0],[0,0,-1],[0,-1,0]]`。不再次套用腕部装配的左右 ±90° 旋转，否则会重复变换。
- 官方 URDF 的 distal 原点不是指尖。专用临时求解模型增加五个固定 tip，位置来自官方 STL 的远端顶部；这是几何任务点近似，不是实测接触垫中心。源 URDF、STEP 转接件及实体装配未改变。
- 只优化六个主动关节，五个 mimic 关节由耦合关系计算。主动范围取自身及被动关节范围的交集。例如左拇指 `2.29 × pitch <= 1.08 rad`，不能只检查主动关节的 `0.58 rad` 上限。硬件归一化仍用原范围，未将交集重新放大到 1。
- 输出仍按拇指 pitch、拇指 yaw、食指、中指、无名指、小指排列，左右各六轴，0 张开、1 闭合。求解器保留关节约束和 warm start；后续低延迟更新已将求解器低通设为直通（`low_pass_alpha=1.0`），实机 O6 控制器仍保留 40 ms 时间常数的平滑，避免重复滤波。断流重置求解状态，失败不冒充新鲜目标。

## 离线评估与后续验收

```bash
cd /home/hnh/unitree_r1_dev/xr_teleoperate
/home/hnh/unitree_r1_dev/.venv-xr/bin/python tools/evaluate_linker_o6_retargeting.py \
  --frames 377 --output /tmp/o6-dex-synthetic.json
```

已完成 377 帧合成 FK 输入测试，双手求解 p95：Vector 2.102 ms、Position 0.895 ms、DexPilot 2.254 ms。全部输出有限，主动及 mimic 关节均在模型边界内。详见 `reports/o6_dex_retargeting_synthetic_20260909.json`。这是求解器耗时，不是实际循环频率、网络延迟或真人映射精度；合成 FK 使用同一机器人模型，不能作为真人动作验收。

新增 dry-run 录制保存明确的 `hand_points_frame` 和左右 `25×3` 原始输入，可用同一段动作比较三算法：

```bash
/home/hnh/unitree_r1_dev/.venv-xr/bin/python tools/evaluate_linker_o6_retargeting.py \
  --recording /替换为新录制文件.jsonl --output /tmp/o6-dex-human.json
```

过去 377 帧真人标定记录只有 12 轴目标，没有原始 25 点，不能用来回放评估新算法。三种模式的指尖距离是诊断量，不是完全相同的优化目标，尤其 DexPilot 会主动改变捏合距离。

下一步录制张开、握拳、拇指对食指/中指、无名指、小指、连续快慢动作，并在 Isaac 检查手指方向、捏合及遮挡恢复。此次没有启动 Vision Pro、Isaac 或真机动作，也未验证新算法的真实手感。实物 O6 首次验收仍需低速、空载、清空夹伤区域；这次迁移不取消原有硬件安全流程。

## 已部署验证

已更新 Ubuntu `/home/hnh/unitree_r1_dev/xr_teleoperate` 及两个现有仿真快照接收端。部署前按文件 SHA-256 检查，保留远端已有头位置补偿、固定参考恢复及逐侧手部 watchdog 改动；没有以本地旧版本覆盖远端机械臂代码。

- 主工程 160 项相关测试通过，包含三算法真实求解、真实 NLopt 失败注入、手部隔离、头腰和机械臂回归。
- 两个仿真接收端 20 项离线测试通过；真实 producer 字典的三模式 × 两个接收端共 6 项 payload 集成检查通过。
- Python 语法、启动脚本语法和两个入口 `--help` 检查通过。上述接收端测试没有启动 Isaac 物理仿真。
- 改动前文件及移出的旧配置/校验器备份：`/home/hnh/unitree_r1_dev/o6-dex-backup.nExyoS`。没有删除历史真人记录。

实现参考：[dex-retargeting](https://github.com/dexsuite/dex-retargeting)、[Unitree 的接入实现](https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/hand_retargeting.py)、[Linker O6 模型](https://github.com/linker-bot/linkerhand-urdf/tree/main/O6)。
