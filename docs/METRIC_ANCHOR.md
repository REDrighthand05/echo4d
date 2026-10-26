# RGB 估计轨迹 + 少量真实测深 → 米制轨迹（并非 RGB 自动测距）

这个实验解决的是单目 SfM 的 **未知整体尺度**。COLMAP 从图片自动估计相机相对运动与稀疏 3D 特征点，但输出任意单位。**外部相机深度传感器**在同一张画面中测出若干被 COLMAP 识别的特征的米制距离，就能计算比例并把轨迹缩放到米。

**不能理解成**“普通 MP4 自己恢复真实距离”。当前还需要实际传感器提供 3 张近时间、注册到 RGB 的米制 Z 深度帧；使用普通单目视频本身并无这项额外测量。

## 工程形式

输入包括先前 `python -m echo4d.video_sfm` 生成的 `trajectory-scale-unknown.json`，以及它所对应的真实 COLMAP TEXT `images.txt` / `cameras.txt` / `points3D.txt`。

传感器测量清单 `measured-depth-evidence.json` 内容类似：

```json
{
  "source_kind": "measured_registered_z_depth",
  "source_description": "Depth measured by an external calibrated camera, registered to RGB pixels",
  "depth_numeric_units_per_meter": 5000,
  "anchors": [
    {"name": "frame_00015.png", "depth": "depth/anchor15.png", "timestamp_delta_seconds": 0.005},
    {"name": "frame_00045.png", "depth": "depth/anchor45.png", "timestamp_delta_seconds": -0.008},
    {"name": "frame_00075.png", "depth": "depth/anchor75.png", "timestamp_delta_seconds": 0.012}
  ],
  "holdouts": [
    {"name":"frame_00005.png", "depth":"depth/holdout05.png", "timestamp_delta_seconds":0.008},
    {"name":"frame_00010.png", "depth":"depth/holdout10.png", "timestamp_delta_seconds":0.008},
    {"name":"frame_00025.png", "depth":"depth/holdout25.png", "timestamp_delta_seconds":0.008},
    {"name":"frame_00035.png", "depth":"depth/holdout35.png", "timestamp_delta_seconds":0.008},
    {"name":"frame_00055.png", "depth":"depth/holdout55.png", "timestamp_delta_seconds":0.008},
    {"name":"frame_00065.png", "depth":"depth/holdout65.png", "timestamp_delta_seconds":0.008},
    {"name":"frame_00085.png", "depth":"depth/holdout85.png", "timestamp_delta_seconds":0.008}
  ]
}
```

示例里的深度文件/时间差**只是接口示例，必须替换成真正测得的数值**。

```bash
python -m echo4d.metric_anchor \
  --trajectory ./sfm/trajectory-scale-unknown.json \
  --colmap-model ./sfm/text-model-0 \
  --depth-root ./sensor-data \
  --depth-spec ./measured-depth-evidence.json \
  --output ./metric-trajectory-from-sensor.json
```

### 方法

把 COLMAP 图像中**实际观测到**的 2D/SfM3D 匹配点变换到相机坐标得到 `Z_sfm`；从对应的传感器深度图取得 `Z_m`（米）；每个点形成 `s=Z_m/Z_sfm`。固定过滤 3x3 邻域有明显深度跳变、失效、偏出图像或重投影明显错误的样本。每帧用中位数，各帧中位数再取中位数；每帧少于 10 个有效点或三帧比例分歧超 20% 时明确失败。记录原始证据文件 SHA 和每一步排除个数。

额外 7 帧传感器深度**不用于计算比例**，单独报告稀疏特征的深度误差。输出的相机轨迹是世界坐标任意旋转/平移，但位移数值以米计，仍不是稠密深度图，无法直接喂给原有 TSDF 内核。请勿把模型的推测深度冒称为已测传感器深度。

公开实测脚本：`tools/tum_metric_anchor.py`，自动下载 SHA 固定的 TUM、实际运行 COLMAP，随后才读取指定 3 帧真实深度，完成尺度和 7 帧未校准深度的验证；**直到尺度锁定后才读取 TUM 的 groundtruth.txt**。GT 评估只借用第15帧真实位姿对齐任意世界系的方向和原点，**不进行尺度拟合**。这只是实验读数而非真实设备的独立世界定位能力。

本实验预注册 [Issue #14](https://github.com/REDrighthand05/echo4d/issues/14)，依赖之前的 [PR #12](https://github.com/REDrighthand05/echo4d/pull/12)。
