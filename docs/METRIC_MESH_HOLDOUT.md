# ECHO4D B 路：12 帧留出测深的 Mesh 视角一致性评分

此工作叠加在 [Draft PR #23](https://github.com/REDrighthand05/echo4d/pull/23) 之上，服务 [Issue #15](https://github.com/REDrighthand05/echo4d/issues/15)。它**不修改**三锚点尺度、48/12 原始划分、TSDF 的边界、体素和输出 Mesh，只在 Mesh 生成后对 **12 张未参与融合的实测深度图** 进行数值验证。

## 实际计算

- 直接复用 [Draft PR #8](https://github.com/REDrighthand05/echo4d/pull/8) 的三角形 z-buffer 投影算法（而不是只计算顶点位置）。
- 使用 B 路自己的**估计 SfM 相机位姿**和来自独立传感器锚点的米制尺度，**绝不读取 TUM groundtruth.txt 的相机位姿来评分**。
- 仅在输出 mesh 之后，从源 TUM depth.txt 找到固定留出 12 张 RGB 的最近原始实测 Z 深度，要求时间差不超过 30ms，并保证与 48 张训练深度完全不重复。
- 以相同的固定缩放采样 4× 的相机内参、深度栅格计算 z-buffer。缺失网格像素单独计入覆盖率，**不当作低误差像素**。
- 输出 `mesh-heldout-depth.json`、每帧 RMSE/MAE/P95/覆盖率、全部 12 张源深度 SHA-256、整体加权（按重叠像素）结果、厘米级残差比例和两个探索筛选标志（coverage ≥80%、overlap RMSE ≤15cm）。
- 若留出图找不到、重复使用训练测深、相机内参不符、模型/文件损坏、某视角无法量化，实验 FAIL 并保留 `report.json`；**不得默默改验证名单**。

## 运行及证据等级

```bash
PYTHONPATH=tools python -m pytest tests/python/test_metric_holdout.py tests/python/test_metric_rgbd.py tests/python/test_tum_sensor_metric_mesh.py -q
python tools/tum_sensor_metric_mesh.py --work /tmp/echo4d-B-holdout-work --report-dir artifacts/tum-sensor-metric-mesh
```

运行前目标路径必须不存在。机器需要真实 FFmpeg/COLMAP、Python 锁定依赖；下载 TUM 原始 RGB-D（固定 SHA256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`）。真实结果以工作流 Artifacts 和 Run 为准，不以提交代码宣称成功。

**严格边界：** 这不是从普通单目 RGB 推出深度的结果；48 帧训练和 12 帧留出来自**同一 TUM RGB-D 传感器、同一段环境**，不能当作独立激光扫描的表面真值。留出相机位置来自估计 SfM（不是 GT pose），也不代表真正新场景泛化。#9 的旧 48/12 帧划分不一样，不能直接将它的 11.33cm RMSE 当作这次 A/B 公平对照。下一步仍需在**相同帧名单、相同深度**上复算 A（GT pose）基线、增加第二场景并做独立表面测量。
