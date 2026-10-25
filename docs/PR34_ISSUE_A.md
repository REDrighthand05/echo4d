# PR #34 · 问题 A / P0：同一 Mesh 评分交叉验证

## 审核结论与实验边界（2026-10-26）

目标不是比较两份历史报告谁的重建误差较低，而是在 **同一个已经完成的 Mesh、同一个 COLMAP 相机和估计米制位姿、同一份 TUM 传感器深度、同一帧顺序和同一投影阈值** 上运行三个不同提交中的评分入口：

- 旧 #25：`072186fe7989efb300759b9e5442e2ca56d208f7:src/echo4d/metric_holdout.py`
- 旧 #26：`454eaaa8d2d99d19adbeb40210930b360d6d219a:tools/tum_sensor_mesh_holdout.py`
- 集成 #34：本工作流 checkout 中的 `src/echo4d/metric_holdout.py`（源于 #34 head `d030f615aae94b70218badb28c29300da7894285`）

此检查只冻住**两个旧评分入口的源文件**，复用集成分支中相同的 `holdout.py` 三角形光栅化、传感器读取、COLMAP 相机和公共数据工具。它有能力发现评分入口/汇总/协议分歧，**不能**独立证明共享光栅化实现没有算法 bug。额外的离线相机方向 oracle 验证 camera-to-world 对应正向相机 Z 和相机中心移动效果。

### 为什么专门使用旧 B-V1

旧 #26 `validate_frozen_holdout_contract` 硬编码来自 Issue #15 的 B-V1 48/12：测试视角包括 `frame_00025.png`。最终 #34 AB-V2-final 预登记在 Issue #28 中，因参考 GT 缺失预先用 `frame_00026.png` 替换 `frame_00025.png`。所以不能把 AB-V2-final 的 JSON 丢给原 #26 后绕过失败，也不能把两个协议的历史分数直接横向比较。

`tools/cross_score_pr34_a.py` 只接受 `protocol.ab_version=legacy-B-V1`。真实 B-V1 数据由一个重建进程获取/定尺/融合一次，在 Mesh 完成后才打开留出深度。随后三份评分逻辑访问完全相同的本地文件；代码还检查 3 个锚点不与留出帧重叠、深度原始单位是 5000 units/m、米制位姿不得来自 GT、训练深度路径无复用。

### 验收门槛

1. 十二个视角的名字和顺序严格一致；深度文件名/哈希一致；训练 48 张唯一。缺失/零交集失败，不静默跳过。
2. 所有原始传感器有效像素和重叠像素完全一致，coverage、MAE/RMSE/P95 差异绝对值不得超过 `1e-9`。
3. 汇总按重叠像素加权，独立从逐帧 MAE 和 RMSE 重算；汇总 P95 只能从完整残差重算，不得错误地加权平均逐帧 P95。三个评分器均实际对原始残差求整体 P95。
4. 机器证据 `cross_score.json` 保留逐帧 5 类数字、逐帧两组 delta、单 Mesh SHA-256、两个旧源码 SHA-256 和异常原因；同时附原始 Mesh、划分、定尺轨迹与 #34 原始评分供审计。
5. 仅当 Actions 中完整实测成功后，才记录 `passed_same_mesh_B_V1_cross_score`。编写工作流与通过离线测试不代表实测交叉验证通过。

## 复现

在完整 checkout（可读取两个固定历史提交）中：

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest tests/python/test_cross_score_pr34_a.py -q
git show 072186fe7989efb300759b9e5442e2ca56d208f7:src/echo4d/metric_holdout.py > /tmp/legacy25.py
git show 454eaaa8d2d99d19adbeb40210930b360d6d219a:tools/tum_sensor_mesh_holdout.py > /tmp/legacy26.py
python tools/tum_sensor_metric_mesh.py --work /tmp/pr34-a-work --report-dir artifacts/pr34-a-source
python tools/cross_score_pr34_a.py --work /tmp/pr34-a-work --report-dir artifacts/pr34-a-source --legacy25 /tmp/legacy25.py --legacy26 /tmp/legacy26.py --output artifacts/pr34-a-cross
```

第一条真实数据命令需要 CPU COLMAP、FFmpeg 和联网下载经 SHA 固定的官方 TUM Freiburg1 XYZ；`--work` 必须是尚不存在的路径。正式 CI 在 `.github/workflows/pr34-score-crosscheck.yml`，它保存 PASS 与 FAIL 机器记录。

## 与已归档 AB-V2-final 证据的关系

集成 PR #34 原有[永久证据](https://github.com/REDrighthand05/echo4d/tree/bf7155ee91a1e6dc51c979d7913e32cd6c155555/docs/evidence/integration-20261026)使用 **两份不同 Mesh**：A 参考位姿 Mesh `9422e006…` 与 B 估计位姿 Mesh `123f39a1…`；A/B 同一组 12 个实测深度视角、177475 有效像素，经身份验证。A 的 92.8660% 覆盖率/12.8042cm RMSE、B 的 91.8293%/11.2555cm RMSE 是 **A/B 不同构建结果**，而不是同一 Mesh 的 #25/#26/#34 评分器差异；禁止拿这组数字冒充本任务完成。

另外，这些图像是同一 RGB-D 传感器的留出测量，不是外部激光扫描真值。该实验即使通过，也不能推出消费级几何精度或纯 RGB 稠密深度能力。

**状态：** 代码与 CI 触发已提交到隔离审查分支；真实结果必须按工作流输出更新，合并 #34/main 仍由维护者单独决定。
