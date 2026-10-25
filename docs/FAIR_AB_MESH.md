# 公平 A/B：完全相同训练 RGB+D、留出实测深度，只改变相机位姿来源

关联 [Issue #15](https://github.com/REDrighthand05/echo4d/issues/15)、[Draft PR #23](https://github.com/REDrighthand05/echo4d/pull/23)、[Draft PR #26](https://github.com/REDrighthand05/echo4d/pull/26)。本 Draft PR 叠加在 #26 上；**不能冒充已合并 main，也不能以单数据集冒充产品级精度**。

## 冻结相同输入与真正改变的变量

固定公开 TUM Freiburg1 XYZ 原包 SHA256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`。固定原 Issue #15 的 RGB 抽取、前60样本、留出帧索引 `[4,5,9,10,14,19,24,25,29,34,35,55]`，48 张训练未留出。每组使用**完全相同**的 RGB、真正实测 camera-Z 深度、图像内参、深度÷5000、降采样4×、0.07m TSDF voxel、max_depth=4m、min_observations=2、max_voxels=1.5M 与训练点云分位数定界策略。A/B 的包围盒可以因为各自的实际轨迹世界系不同而有不同坐标，但**只从对应48张训练测深计算**，不能窥看留出数据。

**A（GT 姿态）**：TUM groundtruth.txt 插值后的真实位姿，只替换 48 张训练 manifest 中的 `cameraToWorld`；从 **B 同一份训练数据目录**读取原 RGB、depth/float NPY、时间戳。再次执行 CPU TSDF 生成独立 A Mesh。在留出视角也使用同一批对应 TUM GT camera pose。

**B（实际 SfM）**：先运行 #26 已完成的真实 MP4→CPU COLMAP→三张独立 TUM depth 锚点求米制尺度→48张实测 Z TSDF→12张估计相机姿态 Mesh 留出评分。B 重建/评分全部完成之后，A 实验才把 GT 姿态用于相机轨迹的替换和独立评价。**B 不能借 A 的 GT 相机位姿或评分 Sim3 scale 定产品尺度。** 注意，B 上游 #14 为了后验诊断也会读取 GT；这是诊断信息但并未用于 B 尺度/位姿/TSDF。

**同一计分方法**：A 和 B 都用 #8/#26 的原始三角形 z-buffer、同一 camera-Z 深度匹配、同一 >=0.15m <=4m 有效像素范围、按重叠像素加权的 coverage / MAE / RMSE / P95 和 3/5/10/30cm 残差比例。12 帧必须全部出现，逐帧比较传感器原始深度 SHA 与有效像素数。两者独立运行 80% coverage + 15cm overlap RMSE **研究筛选门槛**；一组 FAIL 必须保留失败日志，不能悄悄删帧、重调体素。

## 执行

```bash
PYTHONPATH=tools python -m pytest tests/python/test_tum_fair_ab_mesh.py tests/python/test_tum_sensor_mesh_holdout.py tests/python/test_holdout.py -q
PYTHONPATH=tools python tools/tum_fair_ab_mesh.py \
  --work /tmp/echo-fair-ab-work \
  --report-dir artifacts/tum-fair-ab
```

工作/报告目标必须尚不存在；需要有 Python 锁定依赖、FFmpeg、实际 CPU COLMAP，约 448MB 公开 TUM 数据。Actions 配置见 `.github/workflows/tum-fair-ab-mesh.yml`，上报 A/B 机器 JSON、Mesh SHA、逐帧传感器 SHA、使用的分组、实测覆盖率和错误、失败日志；所有关键成绩随后必须另外 commit 一份小 JSON 到 `docs/evidence`，附件会过期。

## 尚未证明

代码提交、Mock 测试及示例命令**不等于**完成真正的 A/B 实验；必须以 CI 最终输出为准。即便成功，A/B 在同一段物理 RGB-D 测深上进行，仅验证同传感器视角一致性，**不是**第二独立场景、激光扫描表面真值或动态人物建模。B 与 A 网格差异同时包含位姿造成的 TSDF 非线性变化、包围盒及重投影变化，不能直接宣称误差差值全由 SfM 定位一项造成。官方设备深度仍有对齐/遮挡误差。后续需要另一运动场景及独立几何真值。
