# production-20260928-visionpro

发布 R1 Tracking Streamer 原生机器人第一人称：双目 H.264 视频、源时间关联、光学校准与宽视场、操作状态及暂停原因；同时发布姿态发送/接收积压优化和数采训练时间信息更新。Safari 网页第一人称继续使用原有入口与关节映射。

## 配套仓库

- [xr_teleoperate](https://github.com/lizuju/xr_teleoperate/tree/production-20260928-visionpro)：遥操、连续采集、质量报告、训练导出及 Ubuntu 网关。
- [VisionProTeleop](https://github.com/lizuju/VisionProTeleop/tree/production-20260928-visionpro)：原生 App，安装见该仓库 `README-R1-安装.md`。
- [teleimager](https://github.com/lizuju/teleimager/tree/production-20260928-visionpro)：PC2 视频源元数据与 Ubuntu 接收；主仓库 gitlink 固定精确提交。
- televuer 沿用主仓库已有 gitlink，本次无需更新。

```bash
git clone --recurse-submodules --branch production-20260928-visionpro https://github.com/lizuju/xr_teleoperate.git
```

原生视频的网络、依赖、证书与部署要求见 [视频服务说明](../../deploy/visionpro-video/README.md)。发布和设备部署是独立操作；本次 Git 发布没有切换现场运行目录、重启相机服务或启动机器人。

## 启动

头显先打开 R1 Tracking Streamer →「机器人第一人称」，地址填写 Ubuntu host/IP。然后在 Ubuntu 主仓库目录按需要运行其中一个：

```bash
./teleop/run_r1_a7_visionpro.sh VISION_PRO_IP
./teleop/run_r1_a7_visionpro_capture.sh VISION_PRO_IP 这次测试名 "这次测试目标"
```

采集脚本已包含遥操。`r` 对齐并跟随，`s` 开始一条，`y/n/x` 结束并标为成功/失败/排除；`x` 保留文件。可以连续 `s` → `y/n/x`，无需重启；`p` 可选暂停保持。暂停后 `r/s` 仍走重新对齐流程。Safari 使用 `run_r1_a7_vector.sh` / `run_r1_a7_capture.sh`，一次仅启动一个控制模式。

## 行为与边界

- 头部相机固定 10 FPS，不提高源帧率或插帧。左右相机固定，HUD 保持原位置。视角范围 60–120°，110°按钮提供宽视野，已保存的视角可由操作者切换。
- 原生发送读取最新样本后等待写入，最高 120 Hz，变化或心跳才发送；接收端合并积压，但保留追踪失效计数和原始时间。写入耗时表示本地提交等待，不是 RTT。
- 区分头部/单手失追、接收超时、源数据过旧及视频故障；全局暂停原因保留至操作者成功重新对齐。250 ms 姿态与 500 ms 视频保护不放宽。
- IMU 保留为诊断信息，仍不作为有效训练输入。ACT 导出升级到 `r1_act_hdf5_v2`，观测使用与图像源时间对应的历史反馈，记录手部请求时间与发布序号。已有 v1 导出需从原始 episode 重新导出；缺少必要源时间/请求信息的旧记录会被筛除，不能自动补造。

## 验证

发布前已完成 visionOS 27 / Xcode 27 编译、签名、安装与头显只读验证。原生发送检查 29 项、显示检查 74 项、状态解码检查 10 项通过；Ubuntu 已部署版本 132 项检查通过。合并最新 main 后的独立发布树测试结果与精确文件校验见同目录 `production-20260928-visionpro-validation.json`。

头显 60 秒只读观测中，头部/视频源有效性保持正常，发送约 88.8 Hz；仍观察到一次稳态约 283.5 ms 接收间隔，保护正常生效。本版不能宣称消除无线抖动，未进行本次发布版本的机器人运动验收。
