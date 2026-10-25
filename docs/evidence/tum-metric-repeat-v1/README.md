# TUM 米制尺度标定：第二次独立 COLMAP 运行（2026-10-25）

这是第二轮真实运行 [Actions #40197656598](https://github.com/REDrighthand05/echo4d/actions/runs/40197656598) 的**未经修改的机器原始报告**与完整相机轨迹。关联 [Issue #14](https://github.com/REDrighthand05/echo4d/issues/14)、[Draft PR #17](https://github.com/REDrighthand05/echo4d/pull/17)。第一轮证据在 `docs/evidence/tum-metric-anchor-v1/`。单目求轨迹只用 RGB、已知内参；外部传感器三张深度图才用于米制标定，GT 姿态仅在全部尺度锁定后用于评分。

- 原始 TUM tar SHA256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`，与首次完全相同；90/90 张 RGB 成功注册。
- 本轮稀疏特征 **9,107**，第一次 **9,139**：COLMAP 的重复求解有数值/初始化波动，文件不会逐字节相同。
- 三张测深锚点依次得到 **0.050263521 / 0.049078874 / 0.048850368** m/SfM unit，中位数 **0.049078873963**。
- 这份SfM世界系自身尺度与第一次不一样；**不能**直接将这次 `0.049078874` 与第一次 `0.050998167` 相减解释为实际空间尺寸变化。
- 为了只作审计，对本轮 GT 姿态做**事后** Sim3 评价的参照比例为 `0.050411469898`；独立测深比例相对参照偏差 **-2.6434%**，第一次为 **−2.5443%**，两次相差约 **−0.0991 个百分点**。评价真值比例未用于程序定尺度。
- 米制轨迹以单个第15帧 GT 姿态做 SE3 世界方向/原点对齐（**无尺度调整**），另外 **88** 帧相机位置 RMSE **1.5377cm**，第一次 **1.5399cm**。
- 7 帧未参与标定的实测深度共 **5,462** 个可比稀疏特征点，按点数加权 MAE **3.8048cm**、RMSE **10.3218cm**；仍存在明显长尾，不是稠密几何精度认证。
- 本轮含数据下载与 COLMAP、传感器标定和事后验证总用时 **183.867s**。

机器证明文件：`report.json`（真实命令、尺度锚点/留出深度、SHA、全程结果）、`trajectory-scale-unknown.json`（RGB-only轨迹）、`metric-trajectory-from-sensor.json`（有外部测量的米制轨迹）、`measured-depth-evidence.json`（深度图时间对应）。均保持 Actions artifact 原字节，通过 Git hash-object 逐个核对后写入 Git。源码并未合并主分支。
