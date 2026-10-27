# ECHO4D 视频定位整合：PR #12 / #13（GPT B）

**范围**：此候选堆叠在 [Draft #34](https://github.com/REDrighthand05/echo4d/pull/34) 的固定 head `d030f615aae94b70218badb28c29300da7894285` 上；#34 已继承 #12→#17→#23。独立工作分支 `integrate/video-pr12-pr13-gptb-20261026`；不修改 main、不覆盖原 PR、不自行合并。

## 开工前去重

- [#12](https://github.com/REDrighthand05/echo4d/pull/12) head `3315b6d14853d72076706d36a1c06f9be513e7da`：主导实现与下游输出格式。
- [#13](https://github.com/REDrighthand05/echo4d/pull/13) head `dd1b4be5d8b85b4bdf1fe11661876c8e15d0cdb0`：另一份独立 SfM 实现；有效模型选择、失败诊断、PTS 时间戳的思路有价值；**不可直接用其 `trajectory.json` 替换 #12 格式**。
- [#37](https://github.com/REDrighthand05/echo4d/pull/37) 已在另一独立 Draft 分支实现可选原始 decoded PTS / CFR/VFR 抽帧并安排真实 A/B，**此处不复制该功能，也不修改它的分支**。#37 的两种模式并非未经修改的 #12 与 #13 代码整体对照。
- [#17](https://github.com/REDrighthand05/echo4d/pull/17) / [#23](https://github.com/REDrighthand05/echo4d/pull/23) 均沿 #12 依赖链；#23 直接拒绝缺少来源与错误标记的未定尺度轨迹。

## 迁移判断与代码变化

| PR #13 功能 | 本候选决定 | 理由与回归 |
|---|---|---|
| 多个断开的 COLMAP 模型择优 | **有条件迁入** | 与 #12 一样按注册数最大；非法候选可被记录并跳过；并列按最小数字模型 ID，最多 32 个。每个模型独立坐标系，不拼接。 |
| 模型相机归属、异常位姿、转换失败的诊断 | **迁入** | `report.json.model_candidates` 包含有效及被拒模型与原因；所有都失败则不写成功轨迹。保持 #12 的真实工具命令、标准模型路径和原 `result` 字段。 |
| decoded-frame PTS、VFR 抽帧 | **不在此重复** | 已由 #37 单独负责；其时间基与 #17 实测深度必须做额外对齐验证。 |
| #13 的独立评估器 | **暂不迁移** | 已有 #12 `sfm_eval.py`，换评分器会改变误差含义。使用同一后验评估器做独立对照，不能将 Sim(3) 评估尺度用于部署。 |
| #13 的 `trajectory.json`、`camera_to_world_row_major` | **不迁移** | #17 / #23 的输入契约使用 `trajectory-scale-unknown.json`、列优先 16 数、`metric_scale_estimated=false`、COLMAP hash 与帧名。 |

**保留的不可变输出：** `trajectory-scale-unknown.json`（`echo-sfm-trajectory-1`、`units=unknown`、`frames[].camera_to_sfm_world`）、`registered_timestamps.csv`、`all_input_timestamps.csv`、`report.json.result.sfm_model_dir` 和 `text-model-N/{images,cameras,points3D}.txt`。选择 N≠0 时，下游根据 `sfm_model_dir` 找到正确模型，而非猜 0。

**注意兼容性边界：** #12 原本支持 `SIMPLE_RADIAL` 的非米制 SfM，仍予保留；#17/#23 的实测深度标定严格要求受支持的 PINHOLE 无畸变配置。此候选不会把带畸变轨迹谎称能直接进行米制融合。

## 固定真实公平对照（结果不能预填）

机器入口：`.github/workflows/tum-video-pr12-pr13-ab.yml` 与 `tools/tum_video_pr12_pr13_ab.py`。

**事先冻结：**

1. 同一个官方 TUM `freiburg1_xyz` tar，SHA-256=`a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`；从官方 RGB 索引 `[::8][:90]` 提取 90 张真实帧。深度、GT 不向 COLMAP 提供。
2. 只编码**一份** 4fps CFR MP4（libx264/CRF15），使用同一视频 SHA；两个进程都抽取固定的全部 90 张 decoded frames。#12 使用 4fps FPS resample，#13 使用 decoded-index stride=1。逐索引 PNG SHA 必须 90/90 一致；两组**相对视频时间戳**偏差不得超过 0.001s，不一致就 **FAIL / 不比姿态误差**。这是人为重编码的测试视频，不代表原生手机 VFR 时间。
3. 两个 CLI 均使用单相机 `PINHOLE` 与相同**明确固定内参** `525,525,319.5,239.5`。#13 CLI 本身不接受内参参数，因此只在比较运行时通过公开保存 SHA 的 COLMAP wrapper 注入 `ImageReader.camera_params`，不改变 #13 Python 源码；并同样固定特征 4096、匹配 overlap=10、CPU 双线程。这个透明受控 harness 与 #13 历史原生运行不是字节等同条件。
4. 固定 #13 源码 commit `dd1b4be5d8b85b4bdf1fe11661876c8e15d0cdb0`；用独立 checkout 执行，不引用 #12 的代码作为 #13 实现。代码路径 / 源码版本均记录。
5. 两次真正的 COLMAP 求解完成且输入 SHA/时序校验通过，**之后**才读 TUM GT。仅用相同 #12 后验评估器计算注册数/比例、同帧 Sim(3) ATE RMSE（米，仅评估）、旋转误差。失败保留原始两组 report，不替换数据或事后换帧调门槛。
6. 本比较不把 #13 的轨迹发布到 #17/#23，避免其行优先/时间基不兼容。实际生产仍使用 #12 未定尺度格式并依靠 #17 实测深度才可定米。

**测量证据状态**：既有 #12 [真实 TUM 机器档案](https://github.com/REDrighthand05/echo4d/tree/feat/rgb-video-to-real-colmap-sfm/docs/evidence/tum-rgb-sfm-v1) 记录 90/90 注册与后验 ATE RMSE 约 1.2394 cm；#13 的[独立 CI](https://github.com/REDrighthand05/echo4d/actions/runs/40205348583) 成功，但历史两者不同输入/采样/内参，**不能据此判定谁更准**。新工作流需在 Draft PR 上真实运行，并读取归档中的 `report.json` 才能填写**同协议**测量值；若运行失败则保留 FAIL，不改变预注册规则。

## 验证门禁与交接

- **静态和单元**：原 `tests/python/test_video_sfm.py` + 新 `test_video_sfm_models.py` 涵盖多模型择优、非法首模型退让、错误来源帧/相机、非正常旋转、相机配置、数量上限、并列确定性及原列优先轨迹字段。
- **#17/#23 兼容**：下游读取的文件名、hash、字段、单位和 `sfm_model_dir` 保留；自动运行 `test_metric_anchor.py` / `test_metric_rgbd.py`。这是**输入结构与单元验证**，不是对新结果已完成的米制 Mesh 重跑。
- **真实校验**：比较工作流还须真实通过，并由人阅读两个原始报告、抽样 SHA/时间、注册比例、ATE 和失败日志；真实动作未完成前不得写 PASS。
- **未知限制**：原生手机 VFR、滚动快门、动态物体、大视差不足、极端运动、曝光漂移、镜头内参变化、速度/资源回归；未提供独立米制误差或真实 Mesh 验收。
- **交接**：本 Draft 请求 #27/#28 主干负责人审阅。同时间戳功能与 #37 冲突时，先保留 #37 的 opt-in PTS 接口，然后合入本分支不重叠的模型安全检查；无授权不动 main。

