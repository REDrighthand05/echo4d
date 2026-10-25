# B 路输入桥接：实测深度 + 传感器定尺度的 SfM 相机轨迹

状态：**可运行的数据适配器 + 数值/错误反例测试**，不是已验收的真实 Mesh，也不是单目 RGB 自带深度。属于 [Issue #15](https://github.com/REDrighthand05/echo4d/issues/15) 的 B 路首阶段，叠在 [PR #17](https://github.com/REDrighthand05/echo4d/pull/17) 之上。

## 原理与来源纪律

- **RGB SfM**：之前真实 COLMAP 推出的 `trajectory-scale-unknown.json`；坐标系与尺度任意。
- **米制轨迹**：`metric-trajectory-from-sensor.json`；来自外部实测深度锚点，**不是** TUM GT 的评估后 Sim(3)。
- **训练深度**：必须是图像同一相机 Z 轴、与 RGB 像素配准的传感器深度（PNG / float NPY）；可指定深度数值单位。模型预测深度要走以后单独的 C 路，不得标为实测。
- **留出帧**：数据划分预先冻结。适配器只读取训练深度；不打开、计算或输出留出深度，哪怕留出文件不存在也应能成功转换。
- **安全契约**：核对 SfM 轨迹 SHA、COLMAP images/cameras SHA、尺度输出是否为原始轨迹 **纯平移缩放**、相机单内参、逐帧 RGB/深度尺寸、时间差 ≤30ms、路径界限，异常不发布半成品。
- **仍未证明**：时间同步与像素配准是否真正准确、重建器在真实数据上的误差、独立点云的几何真值；SHA 证明字节相同而非设备诚实。

## 用法

提供已经运行过的未知尺度 SfM/三锚点米制实验输出、对应 COLMAP 模型文件夹，原始注册 RGB 图像及训练段深度图。先冻结划分 `train-holdout.json`：

```json
{
  "source_kind": "measured_registered_z_depth",
  "source_description": "Registered physical RGB-D sensor; original public dataset and calibration identified separately",
  "depth_numeric_units_per_meter": 5000,
  "train": [
    {"name": "frame_00000.png", "depth": "depth/000000.png", "timestamp_delta_seconds": -0.008},
    {"name": "frame_00002.png", "depth": "depth/000016.png", "timestamp_delta_seconds": 0.009}
  ],
  "holdout": ["frame_00001.png"]
}
```

以上仅为**格式例子**，并非实际配对数据、可直接用的深度地址或验证结果；三锚点的帧不能列入 holdout。每个训练 name 必须对应真实注册的 SfM frame。实际实验应采用更多训练/留出帧并预先登记索引与哈希。

```bash
python -m echo4d.metric_rgbd \
  --metric-trajectory ./metric-trajectory-from-sensor.json \
  --sfm-trajectory ./trajectory-scale-unknown.json \
  --colmap-model ./sfm/text-model-0 \
  --images ./sfm/extracted-rgb \
  --depth-root ./sensor-dataset \
  --split ./train-holdout.json \
  --output ./outputs/estimated-pose-measured-depth \
  --downsample 4

echo4d validate ./outputs/estimated-pose-measured-depth/recording.json
echo4d reconstruct ./outputs/estimated-pose-measured-depth/recording.json \
  --output ./outputs/estimated-pose-measured-depth-mesh \
  --bounds -2 -2 -2 2 2 2 --voxel .04
```

此处 `--bounds` 是占位示范！实际 SfM 任意世界系的包围盒需按 **米制相机轨迹和测得深度** 确定，不可直接复制 TUM GT 世界盒。体素规模受限，重建可能失败，应保留输出与报错。

转换后会得到 `echo-dataset-1` 的 `recording.json`、RGB/米制 Z NPY 及 `sensor-provenance.json`（源 SHA、时间差、留出名单）。输出只能写入不存在的目录，失败回滚。

## B 路真实验证尚需

1. 以既定 TUM RGB+深度划分运行此适配器，确认真实 depth→recording→PLY，不可借 GT 位姿或拟合后 scale 给生产过程。
2. 与 A=GT pose + 同样实测深度使用**同一组训练与留出帧**比较覆盖率/MAE/RMSE/P95 和失败；此前 #9 的 48/12 采样方式不同，不能把它的 11.33 cm 直接当可比 A 基线。
3. 重新对不同场景实测并保留失败、原始耗时、内存和机器产物。最终独立几何真值仍需 [#3](https://github.com/REDrighthand05/echo4d/issues/3)。

本文件是协议描述，不是新的 E3/E4 实验报告。
