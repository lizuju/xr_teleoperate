# production-20261008-runtime-reconnect

在 [production-20261008](production-20261008.md) 基础上，增加实际运行源码核对，并修复原生第一人称累计故障导致后续重连长期等待 5 秒的问题。

- [xr_teleoperate](https://github.com/lizuju/xr_teleoperate/tree/production-20261008-runtime-reconnect)：运行版本预检与回归测试。
- [VisionProTeleop](https://github.com/lizuju/VisionProTeleop/tree/production-20261008-runtime-reconnect)：视频重连修复与回归测试。

## 运行版本核对

现有启动预检显示 Python 路径、checkout HEAD/tag/dirty、实际 `teleimager.client` 和 `televuer.tv_wrapper` 导入路径及 SHA256。checkout 身份明确不代表已部署源码；已有 `.runtime-release.json` 时，逐文件核对本机部署快照。源码修改、缺失或仓外导入会警告，不改变既有启动保护。

部署快照属于本机元数据，不进入 Git。按 tag 新克隆的代码以 Git checkout 身份和实际导入路径为依据；没有本机快照时会如实提示 unavailable。已有生产部署的 97 个受控 Python/Shell 源文件全部匹配。快照不覆盖外部 SDK、系统包、标定或 YAML。

Ubuntu 实际使用的 teleimager 客户端已同步既有子模块版本 `97025042` 中的清理修复：publisher 不再引用不存在的 clock_socket，subscriber 关闭自己的时钟请求 socket。三个子模块 gitlink 均保持原样。

## 原生视频重连

累计重连次数继续用于显示，重试等待按连续失败次数计算为 1–5 秒。只有同一源 epoch 中序列和源时间持续递增、更新时间间隔均不超过 500 ms、源与接收时间都持续至少 2 秒，才清零连续失败；之后偶发故障等待 1 秒。重复、未来、过期、缺失或时钟无效的帧不能触发重置。

保留 500 ms 视频新鲜度保护和 3 秒无帧重连检查。现有重连日志增加 `retry_delay_s`、`consecutive_failures` 与累计 `reconnects`，不增加日志频率。

Safari、IK、自动追踪恢复、采集标签和 IMU 排除逻辑沿用已有版本。启动方式不变：

```bash
./teleop/run_r1_a7_visionpro.sh 192.168.124.68
./teleop/run_r1_a7_visionpro_capture.sh 192.168.124.68 这次测试名 "这次测试目标"
```

## 验证

29 项生产 Python 检查、31 项原生 Swift 重连检查通过，完整 visionOS 编译与真机安装成功。已只读确认 Safari 的 47 个已安装前端资源与发布资源一致。本次没有启动机器人或遥操，没有测量真实无线断线恢复耗时；重连退避修复不消除无线断流。
