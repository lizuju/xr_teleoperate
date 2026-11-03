# 头部原图高清视频

接收头部相机的 1280×1088 原图，同时提供 Vision Pro 第一人称画面和录制、训练用的图像。`r1-teleimager` 继续负责腕部相机和端口 60000 的相机列表。

```
PC1 原图 (RTP JPEG, UDP 5004/5005)
  → PC2 r1-head-hd：解码后分两路
      ZMQ 55555：每眼 640×544 JPEG，30 fps，用于录制和策略
      WebRTC 60004：每眼 960×816，H.264 30 fps 10 Mbps，用于头显
  → Ubuntu r1-camera-forward-60004：转发 WebRTC 协商
  → Vision Pro App：按 head-optics.json 去畸变和双目校正
```

App 代码和 `head-optics.json` 在 `lizuju/VisionProTeleop`。服务和 App 要一起更新：App 只接受 `head-optics.json` 里写明尺寸的视频，也只连 60004。

## 安装

1. 在 `http://192.168.123.161:9080` 打开“左右目原图”，其他开关不动，保存后整机断电重启。页面需要 `video_hub` 关闭、`stereo_patch_pc1` 开启。
2. 把本目录除 `r1-camera-forward-60004.service` 以外的文件放到 PC2 的 `~/unitree_r1_dev/r1_head_hd`，然后运行 `install.sh`，它会要 `sudo` 密码。
3. 在 Ubuntu 上安装网关，不需要 `sudo`：

```bash
cp r1-camera-forward-60004.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now r1-camera-forward-60004.service
```

4. 重新编译并安装 App。

## 把录制和训练切到原图

先在 PC2 上把 teleimager 更新到带 `external_publisher` 的版本，然后运行：

```bash
~/miniconda3/envs/teleimager/bin/python3.10 switch_head_source.py raw
./apply_raw_head.sh
```

Ubuntu 上的录制、训练和预检代码要一起更新。切换前录的数据和训练的模型不能和新数据混用，策略程序会拒绝旧模型。

退回旧画面用 `rollback_raw_head.sh`，Ubuntu 上的代码也要一起退回。

## 验证

在 PC2 上看服务日志。`published` 是录制帧率，`viewer` 是发给头显的帧率，头显连上后 `viewers` 为 1：

```bash
journalctl -u r1-head-hd -n 5 --no-pager
```

## 相机参数

更换相机或画面左右眼不舒适时，用同一时刻的左右眼原图重新生成，场景要静止，并且有不同远近的细节：

```bash
python fit_head_optics.py left.jpg right.jpg head-optics.json
```

## 已知的坑

- `rtpjpegdepay` 会在相机的完整 JPEG 前面再加一个 JPEG 头。直接解码得到纯灰色画面，尺寸和帧率却都正常，要从第二个 SOI 开始解码。
- App 只接受 `timestamp_kind` 为 `pc2_rtp_decoded_receive` 的时间信息。
- 码率是固定的。aiortc 的带宽估计曾把编码器压到最低档，所以没有跟随它调整。网络较弱时改 `BITRATE_BPS`。
