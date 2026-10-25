# Echo4D 静态场景端到端整合交接（2026-10-25）

**这是协同决策建议 + 已执行改动的交接，不是自动合并授权。** 上游要求：实拍 RGB 视频 + **额外实测、配准的相机 Z 深度** → COLMAP SfM → 三帧传感器尺度锚点 → 48 帧 TSDF Mesh → 12 帧非融合测深误差评分；需要公开/可重复运行的源数据、哈希、失败与两种对照。完成静态场景前，不转向动态人物/穿戴设备。

本项目统一说明：单目 RGB 给的是**相对**位置与任意尺度；独立实测测距后才有物理米制。模型覆盖率与重叠区域误差必须**同时**报告，绿色 CI 不等于独立几何 E4。

## 责任分工与整合决策

| 事项 / 负责人（按 PR，不假定真实人名） | 版本、文件、结果 | 决策和剩余条件 |
| --- | --- | --- |
| **12 帧 Mesh 验收**：主执行 PR [#26](https://github.com/REDrighthand05/echo4d/pull/26)，当前贡献者负责 #25 检查迁移；主干维护者最终审查 | 基线 #26 `6850af60`，`tools/tum_sensor_mesh_holdout.py`、`tests/python/test_tum_sensor_mesh_holdout.py`、`.github/workflows/tum-b-mesh-holdout.yml`；真实 [Run 40206826106](https://github.com/REDrighthand05/echo4d/actions/runs/40206826106) PASS。补丁 `454eaaa8` 含防复用训练深度、固定 split、相机内参、深度单位、三锚点和 GT 误用检测及对应反例。| #26 为**正式评分候选**，保留明确的整体 80% coverage + 15cm overlap RMSE 阈值；[#25](https://github.com/REDrighthand05/echo4d/pull/25) 为检查来源，不双合 `src/echo4d/holdout.py`；合并前检查 #26 补丁真实 CI，保留 #25 的 provenance/按帧输出优势，重复模块去重后再关 #25。无论首次实验 FAIL/PASS 都留证据。 |
| **COLMAP 输入**：#6 模块负责人；#19 CLI 迁移负责人；#24 缺陷修复负责人；主干维护者审查 | [#6](https://github.com/REDrighthand05/echo4d/pull/6) `src/echo4d/colmap.py`、`docs/COLMAP_IMPORT.md`、`tests/python/test_colmap.py`：已有真实实验依赖此契约。[#24](https://github.com/REDrighthand05/echo4d/pull/24) `53474659` 在 #6 之上修复 POINTS2D 缺行及伪装 NPZ；[#19](https://github.com/REDrighthand05/echo4d/pull/19) `7e438bf4` 增加 `src/echo4d/cli.py` 的 `import-colmap` 命令，但同名 `colmap.py` 是不兼容的另一实现。 | **以 #6 + #24 为核心解析和单位契约**，把 #19 的用户入口**改为调用该核心**，不得以整体文件覆盖。固定外部尺度 `meters_per_colmap_unit`（米/SfM unit），按 `R_cw = R_wcᵀ; C = -R_wcᵀ t_wc; C_m = s C` 转换，方向矩阵不乘尺度。外部 depth 为 camera Z、须已配准为米（#6 当前要求 float NPY 米制）；若需要 #19 的 PNG/深度÷系数，单独经过验证转换，不能一边换位姿尺度、一边默默重复换算深度。#24 子分支 CI 与同一输入/错误路径回归通过后再统一 CLI。 |
| **视频 SfM**：#12 兼容主线执行；#13 帧时间/评分子功能负责人；主干维护者审查选优 | [#12](https://github.com/REDrighthand05/echo4d/pull/12) `3315b6d1` `src/echo4d/video_sfm.py`、`src/echo4d/sfm_eval.py`、`tools/tum_rgb_sfm.py`：#17 三锚点和 #23 B 路都直接依赖其 `trajectory-scale-unknown.json`、原始 COLMAP TXT 和 camera/帧名；[#13](https://github.com/REDrighthand05/echo4d/pull/13) `dd1b4be5` `src/echo4d/video_sfm.py`、`sfm_evaluation.py`、`tools/tum_video_sfm.py` 记录原始解码 PTS、完整模型候选与分离的 GT 25% 评分。 | **先保持 #12 输出协议作为下游兼容基线**，从 #13 按功能迁移 PTS、模型候选日志和不参与 Sim(3) 拟合的验证集；不得把 #13 的 `trajectory.json` 直接冒充 #12 格式。对两实现用**同一输入视频/采样帧/标定/runner** 比较注册率、ATE/旋转、错误、耗时、VFR 时间戳，保存版本/命令；目前不同数据选择的旧成绩不能判定谁更优。 |
| **公平 A/B**：Issue [#15](https://github.com/REDrighthand05/echo4d/issues/15) 验收负责人；#23 B 路实现维护者；另设 A 路 GT-pose 基线执行责任 | 具体数据/划分：TUM `fr1_xyz` SHA-256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`，前60采样，留出索引 `[4,5,9,10,14,19,24,25,29,34,35,55]`，其余48训练，深度 PNG÷5000→Z(m)。[B 起点 Run 40205153574](https://github.com/REDrighthand05/echo4d/actions/runs/40205153574) E3 Mesh 成功。 | **待做**：A=冻结 48 张同测深+独立 TUM GT pose；B=完全相同 48 张测深+实际 COLMAP estimated poses+三传感器尺度锚点。各自在自身世界坐标（无需相互对齐来美化分数）求来自训练深度的 bbox，均设 voxel=0.07m、max_depth=4m、min_observations=2、max_voxels=1.5M；对**相同12张源深度**投影，A 用独立 GT pose、B 用 estimated metric pose，各自报告按传感器有效像素加权的 coverage、重叠 MAE/RMSE/P95、3/5/10/30cm、每帧和失败名单；禁止把留出 GT / Z 给 B 的尺度、TSDF 或模型选择。不同路径的 Mesh hash、内参、同步误差和运行时间均需归档。由于还没有这组 A 的机器 Run，**不得声称已完成公平 A/B**。 |
| **证据归档**：工作台 [#18](https://github.com/REDrighthand05/echo4d/pull/18) 执行；各实验 PR 输出机器原始件 | `docs/evidence/B_HOLDOUT_25_26_20261025.json` 固化了 **#25 与 #26 两套确实跑过的** B 路数字、输入/分组 SHA、Mesh/报告哈希、Actions 和复现命令。 | 所有后续成功/失败都要 commit 小型 JSON（不上传大原始数据或人物/位置隐私），并记录 Runner/head SHA、阈值、mesh SHA、时长、错误；过期附件只作补充，不作唯一证据源。 |

## 2026-10-25 · A/B 同输入代码现已提交（结果验收中）

A/B 执行负责人：本轮负责 [Draft PR #32](https://github.com/REDrighthand05/echo4d/pull/32) 的贡献者（base #26）；主干维护者负责最终结果审查。当前代码版本 `50aadf697a14d41a8b52471513acb750f9b8793e`，涉及 `tools/tum_fair_ab_mesh.py`、`tests/python/test_tum_fair_ab_mesh.py`、`.github/workflows/tum-fair-ab-mesh.yml` 和 `docs/FAIR_AB_MESH.md`。代码先执行原 B 实验（确保不把 GT 相机位姿给 B 的训练/融合），再从 B **同一份48张实际测深/RGB训练 manifest** 中仅替换 `cameraToWorld` 为 A 的 TUM GT pose，另建 A Mesh；之后两个 Mesh 都在同一12张留出原始传感器 Z 上跑相同的 z-buffer / 像素加权误差，验证每帧传感器 SHA、有效像素数和名字相同。各自的包围盒严格只来自本路径训练深度，保持固定资源参数。

最初合成 GT 缺失反例因前几张**测试假矩阵无效**而失败，已提交预验证修正（先查缺失再变更）。上述测试 FAIL 和新版本运行均须留证据。新 head 的 [真实 A/B 工作流](https://github.com/REDrighthand05/echo4d/actions/runs/40209172069) 与 [主验证](https://github.com/REDrighthand05/echo4d/actions/runs/40209172070) 在记档时未完成；**不要把已成功的 B-only Run 冒充新的 A/B 成绩**。即便 A/B 通过，也只检验同传感器同场景留出一致性，尚非外部激光/多设备 E4。

## 两个已确认 B 路真实结果，不是 A/B 对照

* #25：旧提交 `072186fe`，[Run 40206602087](https://github.com/REDrighthand05/echo4d/actions/runs/40206602087)，12 张，177457 实测有效像素/163529 重叠，**覆盖率 92.151%**，重叠 MAE **3.577cm**，RMSE **10.763cm**，Mesh SHA `e2f13717c719...`。
* #26：旧提交 `6850af60`，[Run 40206826106](https://github.com/REDrighthand05/echo4d/actions/runs/40206826106)，12 张，177457/163010，**覆盖率 91.859%**，重叠 MAE **3.533cm**，RMSE **10.522cm**，Mesh SHA `46cb962637f3...`。
* 这两次 **Mesh SHA 不同，顶点/面数不同**，因此小幅误差差别可能来自重跑 SfM/融合的结果差异，而不是评分代码质量优劣。**不能把 #26 数字较低直接当胜利。** 两者都满足当前探索筛选线，但不得宣称产品级精度或跨场景泛化。

## Gate / 交接给主干看护者

1. 先审查 [#26 的 #25 安全检查迁移](https://github.com/REDrighthand05/echo4d/pull/26)（commit `454eaaa8`）及其**新 head**真实 CI；未通过则先修复，#25 不关闭。
2. 接收 #24 到 #6 核心；改造 #19 保留 CLI 而不覆盖 `colmap.py`；正反例包括 `mm↔m`、误把 COLMAP `tvec` 当中心、缺行、路径逃逸、重复计算比例。
3. 做 #12 与 #13 的同视频配对实验；选定稳定的 `trajectory-scale-unknown` 输出契约才允许 #17/#23 更新依赖。
4. 按冻结 48/12 产出 **A 和 B 两个独立的 Mesh/评分报告**；未执行的新 Run、失败与 GT 信息泄漏必须直说。所有合并、关闭重复 PR 和改变主干接口需维护者审查，不由本交接文件自动批准。
