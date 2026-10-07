# production-20261007

发布当前 R1 原生第一人称及遥操代码。原生输入短暂中断后，机器人先保持当前目标；输入恢复连续有效更新后自动重新对齐并继续跟随。接收端改为读取最新完整姿态快照，减少本机逐帧管道排队造成的额外延迟。

## 配套版本

- [xr_teleoperate](https://github.com/lizuju/xr_teleoperate/tree/production-20261007)：本次从 `10808c110e7a4acd9ce89a4ef3a41dff16e81398` 更新，包含之前的连续数采、质量报告、ACT 导出和控制前馈时序修复。
- [VisionProTeleop](https://github.com/lizuju/VisionProTeleop/tree/production-20261007)：`12be52ae679977f6ebdd04968fc6837b4e1ca8db`，原生第一人称取景及恢复提示。
- teleimager 固定 `97025042acd1f8d05cd5de7259d72b1c6b4bde73`，televuer 固定 `184b469597c4f94b8b4099c492fb85ee29567bb2`，均沿用已发布版本。

```bash
git clone --recurse-submodules --branch production-20261007 https://github.com/lizuju/xr_teleoperate.git
```

原生 App 安装见配套仓库的 `README-R1-安装.md`；视频服务安装见 [网关说明](../../deploy/visionpro-video/README.md)。

## 使用与变化

头显先进入 R1 Tracking Streamer →「机器人第一人称」，Ubuntu 运行其中一个入口：

```bash
./teleop/run_r1_a7_visionpro.sh VISION_PRO_IP
./teleop/run_r1_a7_visionpro_capture.sh VISION_PRO_IP 这次测试名 "这次测试目标"
```

- 首次按 `r` 跟随。输入中断触发的保持，在双手至少 5 次不同的新样本、持续至少 0.35 秒有效更新后自动以保持目标重新对齐；这里不判断双手空间静止。人工 `p` 暂停仍需 `r/s`，不会被自动恢复解除。
- `s` 开始一条；`y/n/x` 结束并标记成功/失败/排除，`x` 保留文件；可继续按 `s` 开始下一条。自动恢复不会自动开启新一条采集。
- Ubuntu 桥接使用 `/dev/shm` 私有目录中的原子最新快照，慢消费者跳过中间快照；源时间、累计头部/单手失追信息与重连状态保留。诊断区分上游数据年龄、本机写入和接收处理耗时。`snapshots_skipped` 不是网络丢包计数。
- 持续无新鲜源样本时重建 gRPC 连接；保留原有时钟映射，防止重连后的积压数据被误判为新数据。源时钟回退或致命协议错误仍要求重启接收脚本。
- 原生默认取景 120°，显示范围上限 80°，「…」中可选最大取景。保留双目顺序、相机标定与 HUD 位置；头部相机仍固定 10 FPS。
- 当前原生启动脚本的 500 ms 输入保护不放宽。原 Safari 入口、关节映射和恢复行为保持原样。IMU 继续仅作原始诊断信息，不作为有效训练输入。

## 双手夹杯上限配置

本次设备已将 `~/.config/xr_teleoperate/o6_grip_cap.json` 移至备份，解除左右手保存的每轴夹杯闭合上限。没有该文件时，现有控制器自然使用完整的正常映射范围；关节 `[0,1]` 校验、电机限位和平滑逻辑仍有效。原设备指令扭矩为 1.0，与默认值一致。

该文件属于用户目录配置，Git 更新不会删除其他设备上的旧文件。需要相同设置时，先退出遥操，再将旧文件备份移走；如果设置了 `XR_O6_GRIP_CAP`，应检查它指向的实际文件。独立雪糕拉杆回放仍可通过 `--grip-cap` 指定原配置备份。

## 验证

精确文件校验及本次测试结果见同目录 `production-20261007-validation.json`。原生显示回归 87 项通过，发布 Swift 文件与已安装并完成编译的版本一致。Ubuntu 在隔离发布目录运行接收、桥接、自动恢复、暂停、旧模式、连续采集、时间信息及双手范围相关测试，共 204 项通过，0 失败、0 跳过。

发布过程没有启动机器人、头显 App 或重启视频服务；保留现场运行目录。此前只读验证证明本机排队耗时降低，但仍观测到超过 500 ms 的无线接收间隔。本次发布不代表消除无线断流，也未执行真机运动验收。
