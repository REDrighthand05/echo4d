# ECHO4D: RGB 视频真正自动恢复相机轨迹（公开 TUM 基准）

**证据类别：真实 COLMAP RGB-only SfM + 事后 TUM 相机轨迹评分；没有从普通视频得到真实米制尺度或真实深度。**

## 实际 GitHub 工作台

- [Issue #11: 视频自动相机轨迹](https://github.com/REDrighthand05/echo4d/issues/11)
- [Draft PR #12: 实际运行程序与测试](https://github.com/REDrighthand05/echo4d/pull/12)
- [首个真正成功的 GitHub Actions #40194684285](https://github.com/REDrighthand05/echo4d/actions/runs/40194684285)
- PR 分支首次成功源码 SHA: `2d9a334fd0970aa9f7156aff1e29ee0be94d7159`（`feat/rgb-video-to-real-colmap-sfm`，主分支未改）

## 1. 公共数据与算法输入

- TUM 官方实拍 RGB-D 包，实际已从 [官方镜像](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz) 下载 **448,204,271 bytes**，SHA-256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`。
- 只取 TUM RGB 序列每隔 8 张一帧，共 **90 张 640×480 真正实拍的彩色帧**，FFmpeg **重新编码成模拟 MP4**（固定 4 fps，约 22.5 秒，H.264，文件 SHA-256 `d766da92599180d9ba575fae03af868bf17be9eba39a4c93fda27313fd2d77c9`）。这是**公开实拍 RGB 图像重新打包的视频，不是原生手机拍摄的 MP4**。时间轴是重新采样的相对时间，不是源相机原始每帧 PTS。
- 向 COLMAP 输入的是视频抽取的 RGB 和 TUM Freiburg1 已知相机内参 PINHOLE `525,525,319.5,239.5`；**没有把 TUM 的深度或 groundtruth.txt 提供给 SfM 求位姿**。
- 真实执行 `feature_extractor → sequential_matcher → mapper → model_converter`，CPU 计算；CLI 命令、参数、版本兼容性与 stderr 记录在机器 `report.json`。

## 2. 真正观测到的结果（不是估计值）

| 检查量 | 实际结果 |
|---|---:|
| FFmpeg 提取 RGB 帧 | **90** |
| COLMAP 成功注册相机 | **90 / 90 (100.0%)** |
| COLMAP 稀疏三维点 | **9,078** |
| 可用 TUM 参考轨迹比较的帧 | **89**（一个 TUM 帧缺 GT 插值，未纳入精度评分） |
| 事后 Sim(3) 最优对齐位置 ATE RMSE | **1.239 cm** |
| 事后 Sim(3) 位置误差中位数 | **1.089 cm** |
| 事后 Sim(3) 位置误差 P95 | **2.049 cm** |
| 事后 Sim(3) 相机朝向误差中位数 | **2.286°** |
| 事后 Sim(3) 相机朝向误差 P95 | **3.501°** |
| 评分时使用的 Sim(3) 比例 | `0.05015481494771` **m/SfM 单位，仅供真值评价** |
| 下载、视频合成、COLMAP、评分合计 | **182.511 秒** |

**ATE 是在用于评分的同一组 89 帧上拟合 Sim(3) 后计算的，属于偏乐观的事后轨迹形状误差；不是真正部署能力，也不能作为已测出米制尺度的证明。**

## 3. 历史失败记录

- [第一次 #40193741585](https://github.com/REDrighthand05/echo4d/actions/runs/40193741585)：COLMAP 3.9.1 不接受 `--version`，安装检测阶段失败；已改 `colmap help`。
- [第二次 #40193839258](https://github.com/REDrighthand05/echo4d/actions/runs/40193839258)：完成环境安装与真实图像输入，外部 SfM 失败详情未向外层透传；已加强嵌套失败 JSON。
- [第三次 #40194292241](https://github.com/REDrighthand05/echo4d/actions/runs/40194292241)：**90 帧真实 SfM 已解出几何**，但导出器遇到首个不足三帧的小子模型便提前终止；已修复遍历所有模型再取注册最多的有效候选。
- **第四次 #40194684285：实际注册 90/90，报告状态 `passed_rgb_only_sfm_with_posthoc_evaluation`**。

## 4. 实际证据文件

- `report.json`：GitHub Actions 机器完整报告（约 85KB），含下载 SHA、90 个源 RGB 文件哈希、真实命令、各阶段耗时和 stderr、90 位姿事后精度评分。不是 AI 按印象编写。
- `trajectory-scale-unknown.json`：COLMAP 真正输出的 90 个相机 camera-to-SfM-world 变换，**未标米制**。文件 SHA `2779d7641a6eb2303f6ad4eb60e21c4de52b81e51c9d39582c9903faa0236605`。
- `cameras.txt`：被选中实际 COLMAP 子模型的相机内参文本。
- `registered_timestamps.csv`：90 个 SfM 已注册帧的时间戳。
- `camera-path-scale-unknown.png`：根据机器输出轨迹绘制的示意图，坐标轴明确是 **SfM 任意单位**，不是米。
- 完整 `images.txt` 和 `points3D.txt` 较大，附在 [原始 GitHub Actions Artifact #12599208468](https://github.com/REDrighthand05/echo4d/actions/runs/40194684285/artifacts/12599208468)；各文件的 SHA-256 及稀疏点数已在报告中长期保存，原始 Artifact 可能按 Actions 保留策略过期。

## 5. 尚未完成

没有自动米制尺度标定、稠密 Z 深度、真实三角表面、手机摄像头多焦段适配、动态物体掩码、稀疏视角的漂移检测或眼镜实时化。**后续将这个 RGB-only 轨迹与独立获得且配准的真实深度和尺度连起来，再做原有 Issue #3 的独立精度验收。**
