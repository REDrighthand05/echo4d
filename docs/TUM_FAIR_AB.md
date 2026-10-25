# Echo4D 同输入、同指标的真实 A/B 比较（第一轮）

状态：**代码/实际数据工作流已提交；不能在机器完成前报告 A 的精度**。上级：[Issue #27](https://github.com/REDrighthand05/echo4d/issues/27)，协议：[Issue #15](https://github.com/REDrighthand05/echo4d/issues/15)。这是叠加在 [#25](https://github.com/REDrighthand05/echo4d/pull/25) 正式评分模块之上的 Draft；并不更改用户的主分支。

## 为什么同一份输入？

目前两条 12-frame B 真实报告都已通过研究探索门槛，但在不同的 COLMAP / Mesh SHA 上运行；因此不能比较哪条评分实现更准。要归因定位与重建的影响，必须在一轮机器运行内固定真正相同的输入。

- 公共真实 TUM `fr1_xyz` 官方压缩包 SHA256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`。
- 继续 #15 已冻结的前 60/48 train/12 holdout，真实 RGB 90 帧 COLMAP；3 张外部实测深度锚点定米制。**即使 A 有 GT 姿态，B 也不能读取它来求轨迹或调米制尺度**。
- A = **TUM GT camera pose** + 相同的48帧实测传感器 camera-Z、RGB、内参；B = 真实 RGB COLMAP pose + 独立深度比例 + 同样 RGB/Z。
- A 的 GT pose 经一个训练集内已经冻结的 `frame_00015.png` **刚性 SE(3) 世界系定位**进入 B 的米制坐标系。只允许旋转与位移，不允许全局相似尺度拟合。这个参考不会用于优化 B 的尺度或 12-frame 投影。
- A 与 B **字节相同** RGB / 真实 Z 深度；一致 TSDF voxel=0.07m、min observations=2、max depth=4m、内参、world bounds（用 B 训练输入计算且锁定；A 不用 GT 再做另一套 bbox）。
- 两路均用 [#25](https://github.com/REDrighthand05/echo4d/pull/25) 的同一个三角形 z-buffer + `evaluate_sensor_metric_holdouts` 和 12 帧数据；A 通过**显式仅供评价**的 GT pose override，B 保持默认估计相机视角。
- 同时报告 full sensor pixels、overlap/missing、coverage、MAE/RMSE/P95、3/5/10/30cm 残差；**按重叠像素计数聚合**，每帧不消失。两组 holdout depth source SHA 必须逐帧匹配，否则 FAIL。
- 研究用首轮 screening gate 仍是 coverage ≥80%、重叠区域 RMSE ≤0.15m；失败也必须保留机器 JSON，不能因 A 或 B 不好看而调整本轮阈值。
- `B-A` 是整个「姿态 + 融合 + 评价姿态」引起的深度一致性差异；不是统计意义上纯粹位姿误差的因果估计。需要进一步逐环节冻结/消融才能严格归因。
- 所有测量均来自同一 TUM RGB-D 传感器，**不是独立扫描 E4 几何真值**；TUM MP4 是实拍 RGB 重编码，不是原生手机录像。

## 可执行方法

```bash
PYTHONPATH=tools python -m pytest tests/python/test_tum_fair_ab.py -q
PYTHONPATH=tools python tools/tum_fair_ab.py \
  --work /tmp/tum-same-48-12-ab \
  --report-dir artifacts/tum-fair-ab
```

CLI 自动真实下载 TUM，运行 B=RGB COLMAP+外部实测尺度+48实测深度→Mesh→12帧评分，**B完成后**才读取 TUM GT 创建 A 的 48帧训练记录，保留相同 bytes 并用同边界 TSDF 重建 A，随后相同评价器对 12 帧分别打分并交叉比对原始观测/文件 SHA。

主要产物：`fair-ab-run.json`、`fair-ab-comparison.json`、`FAIR_AB_SUMMARY.md`、`mesh/mesh.ply` 与 `a-mesh/mesh.ply`，失败也保留已产生的相关报告。CI 真实运行入口见 `.github/workflows/tum-fair-ab.yml`，长期证据索引将回填 Issue #27/#15，绝不能用 14 天 Actions artifact 代替永久机器小报告。

## 通过后还缺什么？

不同运动/设备条件的 A/B 统计、ROI 敏感性、VFR 手机视频、真实像素注册误差、外部独立几何真值。完成本脚本不授权直接合并 #6/#12/#17/#23/#25 的整条依赖链。
