# ECHO 真正静态场景：同一段视频、同样48张实测深度建模、同样12张检查（公平A/B）

*预注册协议：* [GitHub Issue #28](https://github.com/REDrighthand05/echo4d/issues/28)，尤其是 AB-V2-final **运行前修正**。旧 B-V1 的 48/12 不被改写。

- **A（参考相机）**：实际 TUM 彩色视频画面+48张传感器测距，摄像机具体在哪里来自 TUM **真实参考位姿**。
- **B（自动相机）**：完全相同的48张视频画面+48张传感器测距，但相机位置是 COLMAP 从RGB自己算的；整个空间尺寸另来自3张外部传感器测距图，不使用TUM GT拟合尺度。
- 评估：事先保留的**完全相同12张传感器测距图**，通过同一份模型三角形深度投影代码比较A和B，必须报告模型能覆盖多少有效测量像素，以及覆盖区内的绝对偏差、平方均值开根号、95%大偏差分位、3/5/10/30cm误差比例、每张结果和丢失的表面。
- 研究筛选：覆盖>=80%，覆盖像素中的RMSE<=15cm；任一方案差则完整保存失败报告，不能临时调阈值或剔除难画面。

## 为什么有新的AB-V2-final？

原 #15 的12张检查中 `frame_00025.png` 缺少达到项目原定时间同步要求的GT位姿。若把它从检查换到训练仍会造成 A 训练用到缺失参考。因此新协议**在运行前**从索引0..60中移除25、补入60，共60张，留出 `[4,5,9,10,14,19,24,26,29,34,35,55]`，剩余48张训练。**旧实验的原48/12和误差不与V2横向比较。**若V2的第60张/其他帧也缺GT，实验直接失败并留下原因，不再换样本。

## 复现

需要Python 3.12、安装仓库 `requirements.lock`、COLMAP 3.9 CPU、FFmpeg和TUM Freiburg1 XYZ官方压缩包下载；从仓库根目录：

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q tests/python/test_tum_static_ab.py tests/python/test_metric_holdout.py
python tools/tum_static_ab.py --work ./scratch/tum-static-ab-v2 --report-dir ./artifacts/tum-static-ab-v2
```

输出 `ab-run-report.json`, `ab-comparison.json`, `a-known-pose-holdout.json`, `mesh-heldout-depth.json`, 48/12列表、A/B `.ply`、逐条来源哈希。`GITHUB_SHA`、实际下载SHA、时间/输入/输出命令放在报告中。机器真实 Actions 为准；绿色仅表示此公共测试场景达到探索性筛选线，并不等于有独立激光场景真值，更不能证明仅凭RGB能知道每个像素距离。

本次只研究静态场景和可复现的技术门禁，不涉及动态人/眼镜。
