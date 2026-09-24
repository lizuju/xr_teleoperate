# production-20260924

保存 2026-09-24 现场版本：手臂回收与换向修正、连续采集和结果标签、IMU/时钟同步/坐标系记录、数据质检与 ACT 导出、原生 Vision Pro 接收与恢复，以及雪糕机示教工具。

## 配套版本

- 主仓库：https://github.com/lizuju/xr_teleoperate/tree/production-20260924
- 头显原生客户端：https://github.com/lizuju/VisionProTeleop/tree/production-20260924
- `teleop/televuer` 和 `teleop/teleimager` 的修订由主仓库子模块固定，两个仓库也使用同名标签。

```bash
git clone --recurse-submodules --branch production-20260924 https://github.com/lizuju/xr_teleoperate.git
```

## Safari 网页资源

本版 TeleVuer 依赖修改后的 Vuer 0.0.60 网页资源，已随仓库保存在 `vendor/vuer-xr-session-fix.tar.gz`，许可证见 `vendor/LICENSE.vuer`。完成原有 Python 环境安装后，用遥操所用的解释器执行：

```bash
../.venv-xr/bin/python tools/install_webxr_client.py
```

此命令只安装网页资源，不启动遥操。Vuer 资源修复的是页面和 XR 会话行为，不代表 Safari 在设备不支持 immersive-ar 时能够获得透视能力。原生透视客户端的安装见 `outputs/VisionProTeleop透视接入.md` 与对应头显仓库的 `README-R1-安装.md`。

## 范围

版本保留已有运动限制和追踪丢失保护。本次现场的一般动作卡顿与手部输入断更对应；用户复测已恢复流畅，尚未证明无线网络是唯一原因。本标签不包含额外的网络参数或追踪恢复算法修改。

未纳入采集数据、运行日志、部署备份、签名证书和两个空的临时文件。发布操作在独立副本中完成，现场运行目录不切换版本。

## 发布检查

Python 917 项测试：916 项通过，1 项依赖历史 episode 的可视化检查因样本不可用跳过。测试模块分进程运行，避免旧测试夹具之间的模块注入互相影响。Safari 网页 9 项行为检查通过，归档的 47 个资源与现场安装内容逐文件相同；资源安装脚本已在临时目录验证。测试明细见同目录 validation JSON。

VisionProTeleop 的 10 个修改文件与先前通过 visionOS 27 编译、安装与生命周期验证的源码一致；本次发布未重新安装头显 App，也未启动机器人运动。
