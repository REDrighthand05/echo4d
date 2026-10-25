# ECHO4D B 路 12 张未融合深度视角验证

**状态：算法与可复现实验入口；是否通过 80% coverage / 15cm RMSE 必须看真实机器数据。**

本工作依赖 [PR #23](https://github.com/REDrighthand05/echo4d/pull/23) 成功的 B 路：RGB-only COLMAP 位姿 + 3 帧*外部传感器*深度锚点定比例 + 48 张真实传感器 Z 深度 TSDF。针对 [Issue #15](https://github.com/REDrighthand05/echo4d/issues/15) **事先冻结**的 12 张不融合深度图检验。

## 实验冻结条件

- 官方 TUM `fr1_xyz` 源 SHA256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`。
- 先重新运行上游 B-48/12 构建新 Mesh；通过后才打开留出 12 张实测深度做评估；不让残差改变网格/相机/尺度。
- 验证视角**必须使用 SfM 自己估计且传感器定尺度**的 `camera_to_metric_sfm_world`。不用 TUM `groundtruth.txt` 姿态做验证视角投影；上游 #14 的 GT 后验相机路线评分属于另一个指标。
- 限定 camera-Z 0.15–4m 范围，深度÷5000 转米；视角内参来自真正 COLMAP cameras.txt，验证测深/RGB 同近时间配对最多 0.03s，颜色和深度同像素注册在数据来源中假定而非在此得到独立认证。
- 三角形 Z-buffer / 比例正确的深度内插函数 **原样复用 [Draft PR #8](https://github.com/REDrighthand05/echo4d/pull/8) 的 `src/echo4d/holdout.py` 和几何反例测试**；此分支临时携带该模块以避免主分支尚未合入 #8 时无法运行。未来合并 PR #8 和本 PR 时应去掉重复的文件变更、共享同一个模块，避免两个 diverging 实现。
- 输出全部 12 帧原始传感器有效像素数、Mesh 可见重叠数、缺失数、每帧 MAE/RMSE/P95，汇总以**像素计数加权**（而不是无权平均每帧 RMSE）记录 MAE/RMSE、coverage、bias、3/5/10/30cm 界。
- 有任何留出帧缺数据/无预测交集即报告失败；不清除难例。首轮 **coverage≥80% 且 overlap RMSE≤0.15m** 才满足 #15 初步探索性指标，失败留机器 JSON，不允许事后更改首轮验收线。
- 如果这条 real data CI 是绿色的，也仅证明某个 TUM 序列的同传感器观测深度视角一致性。不是独立激光表面 E4 真值；原 12 留出里有部分曾被 #14 当 sparse-depth *事后*分析，不是假装从未被查看。

运行：

```bash
PYTHONPATH=tools python tools/tum_sensor_mesh_holdout.py \
  --work /tmp/echo4d-12-heldout-work \
  --report-dir artifacts/echo4d-12-heldout-result
```

失败后会保留 `holdout-run-report.json` 和 `HOLDOUT_SUMMARY.md`；如已完成全部帧，还会有 `holdout-depth-comparison.json`，即使**未达预设精度门槛也保留数据**。全部机器数据公开 Actions Artifact（有默认保留期限）。后续应归档关键指标、运行 commit、SHA 到 Issue #15；真正公平的 A/B 比较，需要在同一个 48/12 源集合上重跑使用真实 GT poses 的 A 组。
