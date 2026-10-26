# 从普通视频到相机运动的第一条真实路线（COLMAP SfM）

这不是“普通视频已经重建了整个有尺寸的空间”。程序用 FFmpeg 从 MP4/MOV 抽取实际彩色帧，再实际调用 COLMAP 的 SIFT 特征提取、时间顺序特征匹配、相机定位和稀疏三维结构求解。

## 使用

前置条件：Python 3.12、已安装 echo4d 项目依赖、FFmpeg/ffprobe 与系统 COLMAP，且命令可执行。

```bash
python -m echo4d.video_sfm --video ./my-scene.mp4 --output ./outputs/my-sfm --fps 4 --max-frames 120
```

如果拍摄设备已校准，可指定相机内参（务必对应**实际视频的像素分辨率、裁剪和镜头模式**；错误参数会影响求解）：

```bash
python -m echo4d.video_sfm --video ./my-scene.mp4 --output ./outputs/my-calibrated-sfm --camera-model PINHOLE --camera-params 525,525,319.5,239.5
```

第二条入口是有严格时间戳的彩色 PNG/JPEG 图像序列：

```bash
python -m echo4d.video_sfm --images ./frames --timestamps ./times.csv --output ./outputs/my-sequence-sfm
```

`times.csv` 表头为 `name,t`，时间为秒，名称不得包含路径或重复时间。原始视频使用 FFmpeg 均匀重采样的相对时间序列（不冒称保留了变帧率视频原始每帧 PTS）；每次限制处理最多 240 帧，默认 120 帧；可以分片处理长视频。输出目录必须不存在，不会覆盖旧数据。程序失败也在指定的新输出目录保存带 `status=failed` 的报告，**不能把部分结果当作有效轨迹**。

成功后输出目录包含 `trajectory-scale-unknown.json`（只含相机**相对**轨迹）、`report.json`（版本、外部命令、失败信息、输入摘要、注册率）、模型 text 和时间戳。RGB 单目 SfM 的物理尺度和绝对方向不确定；文件明确记为 `units=unknown`、`metric_scale_estimated=false`。

**禁止**把这份未校准轨迹直接标记成 `cameraToWorld` 米制，或喂入 echo4d 的 TSDF。要得到真实尺寸 3D Mesh，还需独立的物理尺度来源与和照片严格配准的米制深度，且针对畸变相机应先使用 COLMAP 的去畸变工具，同步变换全部图像和内参。当前的 `colmap.py` 适配器只接受已经满足这些条件的外部输入。

## 真实公开数据验证

`tools/tum_rgb_sfm.py` 从已固定 SHA-256 的 TUM Freiburg1 XYZ 官方数据集选取 RGB 图像；用 FFmpeg 制造**时间均匀的测试 MP4**；再独立执行完全相同的 RGB SfM。最后才读取 TUM 真实相机轨迹，用仅供评价的 Sim(3)（统一旋转、平移、缩放）匹配得到注册率、位置 ATE 和相机旋转误差。**事后 Sim(3) 对齐不能作为产品已从视频推算实际物理尺度的证明；训练/评价使用同一批匹配相机，ATE 乐观。**

代码、测试、公共数据下载、运行命令、硬件成本、量化报告都由 `.github/workflows/tum-rgb-sfm.yml` 进行保存。CI 通过只说明定位程序真实运行且结果可评价，不等于纹理稀疏/动态场景全部适用。

任务：Issue #11；输入适配接口：PR #6；独立几何误差基准：Issue #3。