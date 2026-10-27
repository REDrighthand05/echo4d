# Issue #47 · GPT B 视频相机交付与验收账本

**性质：独立 Draft 合流候选；未经主干看护审查不得合入 main。**

## 不可变来源

- 主干基线 `d9f5ce39001e4fc88619ce66e83e736eae5614a7`（此前 #34 已合并）；本次没有覆盖旧几何/测深程序。
- PTS 来源：[#37](https://github.com/REDrighthand05/echo4d/pull/37) head `f5f039a8eeb56f69c185a57ca9c663bbd0824fa3`（`video_pts.py` + 专项测试/公开数据 CFR A/B）。
- 多模型来源：[#40](https://github.com/REDrighthand05/echo4d/pull/40) head `1653ddbf0aa3386c1d03e321fff54f46f7b2b38b`（位姿/相机验证、模型诊断与 #12/#13 同输入实测）。
- 两来源基于同一 #34 祖先修改 `video_sfm.py`；本合流**人工按函数/语义结合**，以 #40 的合法单模型选择为主体，将 #37 的可选 PTS 和报告小段迁入。未使用 git 盲合并整个文件。
- 不修改 `src/echo4d/colmap.py`、`metric_anchor.py`、`metric_rgbd.py`；不代替 #17/#23 的测深传感器、标定与物理尺度来源。

## 三类独立验证，不能混写结论

| 门槛 | 固定输入、检查与产物 | 验收含义 |
| --- | --- | --- |
| 组合 Python + 四平台 | `issue47-video-compat.yml`，`test_issue47_b_video.py` + #37/#40 视频测试 + #17/#23 原有回归 | 严格 PTS 和合法多模型一起被调用，`trajectory-scale-unknown.json`、`text-model-N`、hash、`registered_timestamps.csv`、`all_input_timestamps.csv` 继续存在；错误模型仅有失败报告，无成功轨迹。模拟 COLMAP 的测试不等于现实 SfM 准确。 |
| 真实同输入、两种实现对照 | #40 `tum-video-pr12-pr13-ab.yml` 在 pinned TUM RGB 90 帧同一 CFR MP4 运行原 PR#13 与主实现，并在输入 SHA/PTS 一致后才评估。#37 `tum-video-pts-ab.yml` 在同一 MP4 上比较 legacy fps 与 source-pts，要求 90 张逐帧 PNG SHA 相同。 | 真实 COLMAP 配置、注册数、失败与后验相机误差可比较；试验轨迹的 Sim(3) 配准尺度**仅用于评分**。两者是人为 CFR 重新编码，不是原始摄像头 VFR。 |
| 原始照片时间 → 重新编码 VFR → 实际定位 | `issue47-tum-vfr.yml`，`tum_video_vfr_pts_check.py`；固定官方 TUM SHA，从真实 RGB 对应原始时间计算 FFconcat duration，编码 VFR MP4，逐帧读 FFprobe decoded PTS，再用独立 `source-pts` 跑真 COLMAP 并保留每张映射、注册、位姿与误差。 | **原视频**在这里严格指*本实验实际输入的 VFR MP4*，不是原装手机视频。账本同时报告真实 TUM 原始时间与重新编码 PTS 的差值，明确量化编码器舍入/变更，拒绝漂移超过 80ms 的测试编码。 |

## 工程合流的强制接口

1. `--timestamp-mode resampled` 默认行为保持：FFmpeg `fps` 固定相对时间，可能重复/跳帧，不宣称原始 sensor PTS。显式 `--timestamp-mode source-pts` 按解码帧序号有界采样；拒绝丢失、非有限、负数、重复/非递增 decoded PTS 与实际图片数量不符。
2. `source_pts_sampling.frame_map` 包含每个导出文件名、源解码帧号、视频中的原始 PTS 和相对 PTS。下游轨迹 `frames[].t_s` 为**第一解码帧相对秒数**，不会自动转换为原 TUM 绝对捕获时钟；需要对齐传感器才可声称实测深度同步。
3. #40 的 `select_sparse_model` 校验相机、右手正交刚体位姿、来源帧与注册下限，最多 32 个有效编号候选；仅选择一个合规模型，保留所有候选诊断 `model_candidates`、`selected_model_id`，不拼接不同行坐标系。所有失败均退出并留下 `report.json`，不会发布合法 `trajectory-scale-unknown.json`。
4. 输出协议不变：`echo-sfm-trajectory-1`、`units=unknown`、`metric_scale_estimated=false`、相机矩阵 **column-major 16数**，模型 `cameras.txt/images.txt` SHA，`result.sfm_model_dir`，标准时间戳 CSV。这是 #17/#23 的可读取条件，但不是已验证的三维米制精度或物理尺度。
5. `--images` 只能配合来源 CSV `name,t`；显式 `--timestamp-mode source-pts` 必须有 `--video`，否则 CLI 拒绝，避免用户把同名 `frame_N` 在两套时基之间静默互换。

## 已存在的上游 GitHub 证据（仅证实其各自提交）

- #37 [跨平台 verify 40238064158](https://github.com/REDrighthand05/echo4d/actions/runs/40238064158)，[真实相同输入 CFR PTS/FPS 40238064205](https://github.com/REDrighthand05/echo4d/actions/runs/40238064205) — 上游 CI 完成 success；不等于本合流分支已通过。
- #40 [跨平台 verify 40241925414](https://github.com/REDrighthand05/echo4d/actions/runs/40241925414)，[真实 PR12 vs PR13 同输入比较 40241925569](https://github.com/REDrighthand05/echo4d/actions/runs/40241925569) — 上游 CI 完成 success；该测试需要固定 COLMAP wrapper 以对齐 #13 缺失的内参 CLI，不能当作未修改的 #13 原生默认性能。
- 原 #34 [静态实测 A/B 40213189930](https://github.com/REDrighthand05/echo4d/actions/runs/40213189930) 不是本次 VFR/模型整合成果，且对真实表面绝对真值仍有限制。

**验收仍待本合流提交实际 GitHub Actions 后填写：** 组合四平台、真实对照原始机器 artifact、重新编码 VFR 误差、注册与后验 ATE。失败不重写预注册 cohort / 阈值，必须保留失败工件、运行号和发生阶段。

## 不纳入本次完成声明

原生手机或网络流容器里的硬件时钟校正、B 帧/异常容器跨设备差异、外部设备绝对时间同步、动态/低纹理空间鲁棒性、独立房间表面真值与无深度的米制 Mesh。即使全部 CI 通过，也仅能证明上述固定公共数据和单位/失败契约；#47 A、C、D 与双层界面另行验收。
