# TUM B-48/12：首次真实 RGB SfM + 外部实测深度 → Mesh

**证据类别：E3（公共真实数据上，代码完整运行），不是 E4（外部独立表面几何真值）。** 首轮真实执行完成时间 2026-10-25。最初预注册见 [Issue #15](https://github.com/REDrighthand05/echo4d/issues/15)，提交见 [PR #23](https://github.com/REDrighthand05/echo4d/pull/23)。

## 原始来源与机器运行

- 成功 Actions：[run #40205153574](https://github.com/REDrighthand05/echo4d/actions/runs/40205153574)，运行 HEAD **`31c4a376cca9c3caa565030d5892058eb307aa88`**，输出名 `tum-sensor-metric-mesh-40205153574`，artifact id `12603831924`（保留期 14 天）。
- 已下载原始 artifact ZIP，检查包含 `report.json`、`metric/report.json`、`mesh/report.json`、`mesh/mesh.ply`、`mesh/scene.echo.json`、`mesh/evidence.npz` 等 11 项；Mesh 的 SHA256 通过原始二进制内容复算。**本 README 只是机器记录的人工核查索引，不是将 artifact ZIP 伪装为永久附件。**
- TUM `freiburg1_xyz` 官方 RGB-D 压缩包：448,204,271 bytes，SHA256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`。
- `ECHO_HEAD_SHA` = `31c4a376cca9c3caa565030d5892058eb307aa88`。真实 COLMAP 注册 **90/90** 帧，得到 **9,161** 个稀疏 SfM 点。
- 三帧独立传感器锚点求米制：`scale_m_per_sfm_unit = 0.05026938735588676`；与 TUM GT 独立定一个世界原点/方向（不再缩放）时的其他 88 帧位置 RMSE **0.0153761615m**。GT 后验分数不能反馈至 B 路重建。
- 训练/验证 split：冻结 60 帧中的 **48 train / 12 holdout**。划分源 SHA256 `5280afe5fbc24ddbee04979a92cad897c2e2ba3d7bf7d7e7c3a5ea4e0e4be25d`。验证深度**未**进入 TSDF；早先 #14 的独立稀疏深度诊断触及一些相同帧，不能宣称“从未分析过的全盲留出”。
- `recording.json` SHA256 `92b2d0ab2d8415714bf9ee434ee7d791011b74153f41dbc74fb5f894230bff10`。传感器尺度来源、每帧 RGB+depth 哈希见 artifact 的 `sensor-provenance.json` 原始输出 SHA `a090dd26baabe6d6e500903c0082e36a575c72c39aee7ec66f59cd77a7b636c3`。

## 实测 Mesh（原始机器 report.json）

| 固定条件 | 实测结果 |
|---|---|
| 输入 | 估计 SfM 位姿（传感器求比例）+ 48 帧真实 TUM Z-depth |
| TSDF | 7 cm voxel，21 cm truncation，max depth 4m，min observation 2 |
| 训练帧 | **48** |
| 顶点 / 三角面 | **2,456 / 4,006** |
| 报告表面积 | **6.609065366 m²** |
| 开放边 / 非流形边 | **848 / 0** |
| Fusion 内核用时（不含下载、COLMAP） | **0.234761301 s** |
| 完整 B runner 用时 | **122.639 s** |
| `mesh.ply` SHA256 | `32782aa04d24e5ad9054880b2d721fd6f70701ce13525696f71350cfd4bf3a25` |
| `scene.echo.json` SHA256 | `0576ce5c4b4d84c73a91de22c9582b698f1f8608558a9f27bfd043e31dc4eb7f` |
| `mesh/report.json` SHA256 | `c05a79f070d36fae20d76d85e5ce10b26c11f5cb11c522686c3e34dacb6498e9` |

**诊断提醒**：记录显示 B 路成功，但嵌套的 `metric/report.json` 状态曾误标 `running`；在本结果产生*之后*修复了调用方的子状态，修复不应追溯改写第一轮原始机器报告。未来若重复运行，应同时复核本修复。

## 结果可以证明、还不能证明什么

✅ 真实公开 RGB 输入到 COLMAP 估计轨迹，外部实测尺度，实测稠密深度经 TSDF 生成**非空 Mesh 和 scene**的闭环已跑通。

❌ 不支持“从纯单目普通视频恢复真实米制稠密几何”：这里有 3 个外部深度锚点和 48 张外部真实传感器深度图。

❌ 顶点数、面积、GT 辅助的相机路径 RMSE 均不构成**独立表面几何精度**；尚未计算本次 12 帧完整图像的 withheld-depth 投影误差、第二场景 B 路、设备泛化。不能直接比较 #9 的旧 48/12 11.33cm RMSE（不同采样）。

下一门禁：基于 **本次冻结训练/留出名单**，通过三角形 Z-buffer 把 B Mesh 投影到 12 个未融合的实测深度视角，不采用 TUM GT 位姿进行评价投影，报告覆盖率、MAE、RMSE、P95 和 3/5/10/30cm 尾部。然后同样数据跑 A（GT pose + measured depth）作为合理基线。参见 Issue #15。
