# 原生第一人称视频服务

配套发布：`production-20260928-visionpro`。原生 App 见 [VisionProTeleop](https://github.com/lizuju/VisionProTeleop/tree/production-20260928-visionpro)。主仓库 `teleop/teleimager` 的 gitlink 固定本次 PC2 视频服务和 Ubuntu 图像客户端源码。Safari 仍使用已有网页与启动脚本，左右相机顺序固定为左、右。

## 数据路径与前提

PC1 头部源固定输出 **10 FPS**。PC2 的 GStreamer helper 接收并组成 `1088×448` 的左右 SBS 图像，每眼 `544×448`；配置中的 `fps: 30` 是读取上限，不能提高源帧率。原生 App 通过 WebRTC H.264 接收图像，通过 `r1-video-meta-v1` data channel 接收与 RTP 帧对应的源时间和时钟信息。普通上游 WebRTC 服务没有这条元数据协议，不能直接代替本版服务。

Ubuntu 的 HTTPS 网关仅代理 SDP/网页并解析 mDNS ICE 地址，**不是 TURN/媒体中继**。现场 Ubuntu Wi-Fi 为 `192.168.124.147`、有线为 `192.168.123.222`，PC2 为 `192.168.123.164`。需要保留现有跨网段路由/IP forwarding；仅启动网关不足以建立媒体连接。新部署须按实际网络配置可达路由，不能改掉机器人有线地址。

- 头部 HTTPS：Ubuntu `60001` → PC2 `60001`。
- `/r1/status`：只在头部网关提供，读取同 uid 的 `/tmp/r1-teleop-status-<uid>.json`；超过 1 秒的状态不再显示为有效。
- 头显姿态：Ubuntu 连接 Vision Pro 的 gRPC `12345`，与视频反向传输独立。
- 现有 `60000` 配置/时钟转发与 `55555–55557` JPEG 转发继续供桌面预览和录制使用。手腕网关 `60002/60003` 也可运行同一源码，不提供遥操状态接口。

## 安装与版本固定

以下是新部署/维护步骤，发布操作本身不会执行服务重启。已运行的机器先备份源码、Python 环境及配置，并安排好停止遥操的维护窗口。

1. 主仓库执行 `git submodule update --init --recursive`，获取本标签固定的图像服务源码。PC2 使用 `lizuju/teleimager` 同名标签，勿把 PC2 的脏目录直接 reset 或切分支。
2. PC2 使用 Python 3.10，按 teleimager 仓库的 server 安装说明安装。精确帧关联依赖已验证的 `aiortc==1.15.0`、`av==17.1.0`；更换版本必须重跑该仓库的 `test_video_metadata_aiortc.py`。GStreamer helper 使用系统 Python/PyGObject/GStreamer，不是 Conda 的 gi。
3. Ubuntu 网关依赖可安装到现有网关环境：`python -m pip install -r deploy/visionpro-video/requirements-gateway.txt`。系统需 `avahi-resolve-host-name` 和 `timeout`。根据实际 home 路径检查提供的 user service 模板；默认假定 `~/unitree_r1_dev/.venv-xr` 和 `~/unitree_r1_dev/xr_teleoperate`。
4. 新部署把 `r1-camera-forward-60001.service` 放入 `~/.config/systemd/user/` 后才执行 `systemctl --user daemon-reload` 和 `systemctl --user enable --now r1-camera-forward-60001.service`。若已有运行服务，确认其 ExecStart 指向希望使用的源码再安排重启。PC2 共享服务重启会同时中断 Safari、原生与录制视频；不要顺带替换 USB recovery unit/drop-in。

现场实测 PC2 依赖还包括 aiohttp 3.14.3、numpy 1.26.4、pyzmq 27.2.0、PyTurboJPEG 2.5.0、opencv-python 4.11.0.86、psutil 7.2.2、PyYAML 6.0.3、logging_mp 0.2.2。Ubuntu 网关为 aiohttp 3.10.5。其余环境依赖和既有安装步骤由 teleimager 仓库管理。

## 配置与证书

PC2 实际配置位于 `~/.config/teleimager/teleimager_server.yaml`。仓库 YAML 是初始化模板，不能覆盖当前相机、设备或校准配置。

网关读取 `~/.config/xr_teleoperate/{rootCA.pem,cert.pem,key.pem}`。证书必须匹配 App 填入的 Ubuntu host/IP；当前网关访问 PC2 时使用 bind host 校验其证书身份。App 中 `RobotVideoRootCA.der` 是公开 CA；新环境使用自己的公开 CA 替换并重新构建。不要提交、复制进 App 或公开 `key.pem` / `rootCA.key`，也不要为发布重建在用 CA。

网关参数：`--bind-host`、`--bind-port`、`--upstream-host`、`--upstream-port`。App 地址栏填 Ubuntu host/IP（不含协议或端口）；遥操脚本参数填 Vision Pro IP。

## 使用与验收

先开 R1 Tracking Streamer →「机器人第一人称」，看到正立双目视频与状态栏后由操作者启动遥操。60–120° 视场可调，110°适合尽量看全桌面和双手。光学资源针对当前真实相机标定，不能视为任意 R1 通用标定。

检查真实源帧率约 10 FPS、元数据时钟有效、画面年龄正常、失追/锁屏后保护与恢复仍有效。250 ms 姿态/500 ms 视频保护没有放宽；无线仍可能出现真实超时。回滚时恢复旧源码及其依赖版本，只重启受影响服务，再做只读验收。
