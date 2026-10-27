# 从普通视频到相机运动的第一条真实路线（COLMAP SfM）

> **棋盘模块标注（2026-10-27）：M00/M01/M04。** 共享视觉位姿模块，可服务P1及其几何/旧Mesh组合；本文定位链不是全项目唯一路径。 [路线策略](STRATEGY_CHESSBOARD.md) · [模块合同](STRATEGY_MODULES.md) · [逐格验证](STRATEGY_VALIDATION.md)。下文历史状态/数字不因该标注改变，最新证据以GitHub为准。

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

## VFR / 原始视频时间戳（PR #34 B 独立候选）

默认 `--timestamp-mode resampled` 保持过去行为：FFmpeg 按 `--fps` 均匀重采样。**这不会保留 VFR 原始时间间隔**，但维持旧实验输入和结果的解释方式。

若原视频帧率随时间变化，且需要保留来源视频内解码帧的时间关系，可显式改用：

```bash
python -m echo4d.video_sfm --video ./my-scene.mp4 \
  --output ./outputs/pts-sfm --timestamp-mode source-pts \
  --max-frames 120 --camera-model PINHOLE \
  --camera-params 525,525,319.5,239.5
```

该模式先从 FFprobe 读取`best_effort_timestamp_time`，核查其有限、非负、严格递增；按解码帧序号和预算固定间隔抽取，FFmpeg 的 select + passthrough 不会为了固定帧率重复补帧。抽取的每张图仍命名 `frame_00000.png` 等；现有 `trajectory-scale-unknown.json` 中的 `frames[].t_s`、`registered_timestamps.csv` 与 `all_input_timestamps.csv` 均保留名称和相对秒数格式。新 `report.json` 增加 `source_pts_sampling.frame_map`，列出每张图的解码帧号、视频原始 PTS 和以第一解码帧为零的相对 PTS；失败会记录到 report 且不发布成功轨迹。

**注意：**两种模式选中的帧不一定相同。即使固定同一个 MP4、相机参数和最大帧数，也不能把不同的输入帧集合/不同的 3D Mesh 的注册率和几何误差直接比较成算法优劣。`--fps` 仅用于 `resampled` 模式；`source-pts` 模式以 decoded-frame stride 控制最多抽多少帧。后续 #17/#23 以帧名、相机标定和单调相对时间匹配，字段格式保持不变，但**不能凭文件名推断两个模式抽取了同一张源照片**。如用实测深度做时间配准，必须核查来源时基、第一帧绝对时间偏移及实际同步误差；时间对齐不等于已经获得米制尺度。

不支持不存在/重复/倒序的解码 PTS，不会静默生成近似时间；这类输入需先修复源记录并留下变更证据。多 disconnected COLMAP sparse-model 选择不在本轮迁移中。


## 多模型安全与 Issue #47 B 合流候选

当 COLMAP 产生多个互不连通的模型时，程序分别检查每个模型的相机参数、合法位姿和已抽取图像身份；保留每个模型的拒绝原因。只选择注册帧最多的单一合规模型，注册数并列按数字模型编号升序，绝不混合两个世界坐标系。成功报告中 `result.sfm_model_dir` 对应 `text-model-N`，下游 #17/#23 必须从该目录读取 `images.txt/cameras.txt/points3D.txt` 并验证 SHA，不可假定模型始终为 0。

正常使用 `--timestamp-mode source-pts` 时，`frames[].t_s` 是视频容器 PTS 相对第一解码帧的秒数，而不是与外部深度传感器天然同步的绝对 UTC 或 TUM 时戳。采集设备之间的零点偏移需要独立对应关系；#17/#23 的传感器 RGB/深度时间差仍需逐帧实测核对，不能因为 PTS 严格递增就宣称达到 30ms 同步门槛。默认 `resampled` 保持旧时间含义。

组合回归以 [Issue #47 B 的合流记录](ISSUE47_B_VIDEO_INTEGRATION.md) 为准，包含原有 #37/#40 独立真实 CI 与当前组合分支的新 CI。源 MP4 的人工重编码 CFR/VFR 检测不能冒充原生手机视频的传感器硬件 PTS 验证；两者均不从单目 RGB 推导米制尺度。
