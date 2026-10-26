# 验证范围与复现

`python tools/verify.py` 运行全部 Python 数值/输入/CLI 测试，然后生成合成 RGB-D、重建 Mesh，并用 Node 检查 Python 导出的场景契约。建议使用 Python 3.12 和 `requirements.lock`；浏览器检查需要 Node 22+。

## 真实数据与独立性

- 主干历史 TUM 已知位姿结果：[机器报告](validation/tum-report.json)。
- RGB 估计位姿、实测深度定尺度与 Mesh：[B 路接口](METRIC_RGBD_BRIDGE.md)。
- 相同输入的参考位姿/估计位姿对照：[AB-V2-final](STATIC_FAIR_AB.md)。机器报告与两组历史 Mesh 保存在 [证据目录](evidence/tum-static-ab-v2/README.md)，报告中的 commit/run 才是其对应版本。
- 第二场景测试只证明 RGB 定位实验范围，见 [说明](CROSS_SCENE_DESK.md)；不能写成第二场景完整米制 Mesh 已验收。

旧 B-V1 划分含缺少合法参考位姿的第25帧。新 AB-V2-final 在运行前登记排除它，补入第60帧，同时重跑两组；两版本数字不直接比较。原协议失败仍保留在 PR #29/#32 和 Actions 中。

研究筛选线为总体覆盖率至少80%、覆盖区域内 RMSE 至多15厘米。这不是产品精度标准。没有覆盖的区域仍是缺失，不能记为准确；同一传感器的留出深度一致性不等于独立激光扫描表面真值。

## 本轮整合验收

整合分支需通过 Linux/Windows、Python 3.12/3.13 的通用检查和新的真实 AB-V2-final 回归。新的 A/B 报告必须逐帧核验源深度 SHA、时间关联和有效像素数一致，并记录实际 checkout SHA、PR head 和 base。历史成功报告不代替整合版本的 CI。

来源、迁移范围及尚未吸收的 PR 见 [整合清单](INTEGRATION_20261026.md)。
