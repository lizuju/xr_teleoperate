配套原生头显源码：https://github.com/lizuju/VisionProTeleop/tree/production-20260924 。新电脑请从该仓库克隆并按其 R1 安装说明编译。

# R1 Vision Pro 原生透视接入

Ubuntu 端和原生客户端已部署；Xcode 27 编译、签名、安装与头显开发者信任已完成。头显 IP 为 `192.168.124.68`。实际头手输入、单手失追与恢复、数据过期无效化均已验证；无线仍有延迟波动，尚未启动或验证机器人运动。

## 两种模式

旧网页 VR 使用原来的 `teleop/run_r1_a7_vector.sh` 和 `teleop/run_r1_a7_capture.sh`。这两个文件和现有 IK、Linker O6 映射文件经过 SHA-256 检查，接入没有修改它们。默认输入仍为 WebXR。此前并行更新的手腕工作空间代码已经合并保留。

原生模式新增：

- `teleop/run_r1_a7_visionpro.sh VISION_PRO_IP`
- `teleop/run_r1_a7_visionpro_capture.sh VISION_PRO_IP 这次测试名 "这次测试目标"`

`VISION_PRO_IP` 必须替换成头显 App 显示的 IP，不是 Ubuntu 的 `192.168.124.147`。两种模式每次只启动一个；采集脚本已经包含遥操。脚本仍由用户启动。

## 安装头显客户端

Mac 工程是本文件同目录的 `Tracking Streamer.xcodeproj`，Scheme 选择 `VisionProTeleop`，target 为 `Tracking Streamer`。

1. Xcode 登录自己的 Apple ID，在 Signing & Capabilities 选择自己的 Team。本机已经选择检测到的 Personal Team，并设置独立 Bundle ID `local.r1.TrackingStreamer.srvjvmz2ws`；显示名为 **R1 Tracking Streamer**，避免混淆原版客户端。
2. Mac 和 Vision Pro 连接同一个 `HuaNanHuNo.1_5G`，开启 Wi-Fi 和蓝牙，保持头显唤醒并靠近 Mac。头显停留在“设置 → 通用 → 远程设备”。Xcode 27 使用 **Xcode → Open Developer Tool → Device Hub**，点左上角 **＋ → Pair Nearby Device…**，选择 Apple Vision Pro / visionOS，选中发现的头显并输入配对码；允许本地网络发现，按系统提示信任 Mac、启用开发者模式。创建 Simulator 的弹窗应取消。配对安装不要求两端使用同一个 Apple ID；Xcode 登录的开发账号用于 App 签名。
3. 在 Xcode 选择自己的 Vision Pro 为运行设备，完成编译并安装。本次个人签名有效期至北京时间 **2026-10-01 13:53**，到期后需用 Xcode 重新签名安装。首次打开若提示未信任开发者，请到头显“设置 → 通用 → VPN 与设备管理”信任自己的开发者 App。
4. 打开 R1 Tracking Streamer，允许手部追踪、世界感知和本地网络。点 Start 进入 mixed 透视空间，看到真实环境，保持双手可见。
5. 在 App 设置中将 Hand Tracking → Prediction Offset 设为 **0 ms**。新安装默认零，已有设置可能保留其他数值。

本分支专用于本地机器人输入：移除了原作者账号的 iCloud/共享钥匙串等签名权限与设置云同步入口，跳过云登录引导；使用原生 mixed 空间。机器人相机小窗和力矩 HUD 暂不在此模式显示，机器人端相机采集仍按现有流程保存。

**不要使用未经修改的 App Store Tracking Streamer 开启机器人跟随。** 上游在失追后继续发送缓存姿态，原协议没有真实追踪有效性和采样时刻；本分支增加 protocol v1 元数据，Ubuntu 端会拒绝未提供这些字段的客户端。

## 先检查数据，不连接机器人控制

在 Ubuntu 的 `/home/hnh/unitree_r1_dev/xr_teleoperate` 执行（替换 IP）：

```bash
../.venv-xr/bin/python tools/check_visionpro_input.py 头显IP --seconds 10
```

此工具只接收头手数据，不导入机器人控制器，也不创建机器人命令发布器。正常应看到 `source=visionpro`、`protocol_version=1`、`head_tracking=true`，双手可见时左右 tracking 为 true。遮住一只手后，该侧应失效；恢复可见后收到新数据。退出透视或头部追踪中断后，双手都应失效。

这一步通过后，由用户运行：

```bash
./teleop/run_r1_a7_visionpro.sh 头显IP
```

或直接采集：

```bash
./teleop/run_r1_a7_visionpro_capture.sh 头显IP 这次测试名 "这次测试目标"
```

按键沿用现有流程：`r` 跟随；`s` 开始一条；`y/n/x` 保存并标记成功/失败/丢弃（文件保留）；再次 `s` 开始下一条。`p` 暂停保持，`r` 或 `s` 按现有稳定检测流程重新对齐，`q` 退出。

新模式在头部丢失、网络超时后会暂停，并要求明确按 `r/s` 重新对齐；单手失追时该手保持，另一只手继续沿用既有控制逻辑。网络断开后接收器会自动重连；重连会重置时钟估计并保留暂停要求，仍需明确按 `r/s` 重新对齐。协议或时钟异常仍会拒绝输入。头显 App 后台停止姿态服务，回到前台或再次 Start 时恢复监听。接收端在矩阵转换前合并积压姿态，保留失效事件；机械臂和手指线程分别观察单侧失追，250 ms 超时不变。

## 实现与数据兼容

链路：原生 ARKit → gRPC 接收进程 → 与 WebKit 相同的关节轴转换 → 现有 TeleVuerWrapper → 原来的手指映射、IK、暂停及采集逻辑。

- 使用前 25 个手关节，排除两个前臂关节。每个关节执行 `handAnchor × anchorFromJoint × C_side`，腕部取第 0 个关节，头部使用原始 ARKit 矩阵。
- 单独环境 `/home/hnh/unitree_r1_dev/.venv-visionpro` 只承担协议接收。旧 `.venv-xr` 没有安装/升级 `avp_stream`、grpc、numpy 等依赖。
- 手部有效性来自真实 `anchorUpdates` 事件：排除 removed，检查手锚点 `isTracked` 与骨架；每侧使用 `AnchorUpdate.timestamp`，预测查询只影响姿态，不能刷新有效时间；与旧 WebXR 相同，个别手指被遮挡时接受 ARKit 提供的关节估计，不将整手直接判为失效。重复发送同一事件不会刷新有效时间，静止但仍被追踪的手通过新的更新事件保持有效。
- Ubuntu 与头显时钟采用收到包的最小时间差估计；诊断年龄包含源数据年龄和相对额外延迟，不是经过校准的单向网络延迟，也不是硬件同步。
- episode 元数据记录输入来源、协议版本、源码哈希及时间语义，诊断记录实际 prediction offset。原有自动质量报告与 ACT 导出继续使用；IMU 仍不作为有效训练输入。

## 验证与恢复

最终真机验证使用 `anchorUpdates`：左手移到背后时该侧有效标志变为 false、时间戳清零，右手仍有效；左手回到视野后双手恢复。退出/头部失追以及数据过期时，双手都无效并锁存重新对齐要求。事件源版本的最终 20 秒观测约 90.65 Hz，协议错误为 0，头部有效 96.7%、左手 84.5%、右手 95.9%。这包含用户实际姿态变化和传输停顿，不能当作持续无中断能力指标。

最后一轮接收间隔年龄最大约 336 ms；有效手部数据年龄中位数约 66–68 ms、95 分位约 173–187 ms。Ubuntu 到头显的 30 次 ping 往返 3.3–175.5 ms、无丢包，Mac 同时测得最高 415.8 ms；说明无线存在抖动。超过 250 ms 的旧数据会被判无效并要求按 r/s 重新对齐，未放宽保护阈值。此前查询预测姿态版本的 20 秒 100% 有效结果已被失追测试推翻，不能作为最终版本结论。

真机证据：`reference/protocol/live-anchor-event-loss.json`、`live-final-check.json`、`visionos-install-status.json`。

离线测试总计 314 项，其中 313 项通过，另有 1 项旧的测试文本断言在接入前已失败（仍查找旧版 `s` 按键代码）。34 项新增接口/启动/gRPC 测试全部通过；其中包含实际本地 gRPC 序列化、未改客户端拒绝、活跃连接反复发送旧姿态及断连。还验证了暂停恢复、连续采集、质量与训练导出，以及最新工作空间代码。没有启动遥操、录制或机器人运动。

Ubuntu 部署备份：`/home/hnh/unitree_r1_dev/backups/visionpro-input-20260924-133425`。普通切回旧模式直接使用原脚本即可，无需恢复文件。若需要撤回共享文件更改，应先确认备份后是否还有其他编辑，避免覆盖后续修改。

上游基础：Improbable-AI/VisionProTeleop，commit `4c549905c2a8b214d79f7cd88e535101a1ce32af`，MIT。完整原生补丁为同任务目录 `reference/protocol/tracking-validity-v1.patch`，协议及来源许可证随 Ubuntu 工具部署。Swift 生成文件及追踪有效性代码已实际编译通过。针对 Xcode 27 的并发检查，依赖锁文件只把 swift-async-algorithms 1.1.0 更新为官方已修复此问题的 1.1.2（PR 399）；没有关闭并发检查。签名、安装和真机数据接收已成功，实际机器人运动仍由用户启动与验证。
