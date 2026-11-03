# 开机设置头部相机服务

应用里的开关重启后不保留。这个工具在 PC2 开机时关闭 `video_hub`、开启 `stereo_patch_pc1`，头部原图才能发到 UDP 5004/5005。

把本目录放到 PC2 的 `~/unitree_r1_dev/r1_camera_services`，然后运行：

```bash
./build.sh
./install.sh
```

`install.sh` 会要 `sudo` 密码。`r1_camera_services eth10 list` 可以查看当前服务状态。
