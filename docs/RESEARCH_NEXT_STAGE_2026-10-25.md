> 历史设计备忘录，2026-10-27归档接收。下文保留2026-10-25的事实快照与主观概率，不代表当前PR状态；实时进度与新的验收以#15/#47为准。A/B链已有接收，普通RGB预测深度C、独立表面真值与部署级泛化仍未完成。

# Echo4D 研发研究备忘录：从真实 SfM 到可审计的米制 Mesh

日期：2026-10-25。状态：**设计与证据审查；不是已通过的实验报告**。关联 [Issue #15](https://github.com/REDrighthand05/echo4d/issues/15)。

## 一、决策摘要

首先不扩建 UI，不提前做动态人物或实时眼镜。保留 `main` 的已知深度＋位姿 CPU TSDF 作为 A 组基线，以真实公共数据把以下层连接：

`RGB / 视频 → 估计 camera poses（尺度未知） → 独立测量尺度 → 传感器或预测的 Z-depth → echo-dataset-1 → TSDF → 留出视角评估`

这个过程需要**三条严格区分的轨道**：

| 实验轨道 | 位姿 | 深度 | 可证明什么 |
|---|---|---|---|
| A | TUM 真值 | TUM 实测 | 已知位姿条件下静态融合质量的基线 |
| B | 真 COLMAP 估计 + 独立锚点求米制比例 | TUM 实测（训练段） | 不依赖位姿真值，估计轨迹进入融合的能力；**不是纯 RGB 重建** |
| C | 真 COLMAP 估计 + 有据可查的尺度来源 | 从 RGB 预测且核验为 camera-Z 的深度 | 单目估计进入 Mesh 的实用性；预测误差与实测必须分开报告 |

同一场景必须冻结训练/尺度标定/验证输入集合及 metric；测量并保留失败，不能做事后择优。

## 二、仓库核对（2026-10-25）

- [main README](https://github.com/REDrighthand05/echo4d/blob/main/README.md) 写明已知位姿的 TUM 60 帧重建 10,810 顶点 / 18,497 三角形，并明确没有自动 SLAM。此为仓库记录，并非本次重新运行 TUM 的独立结果。
- [#9](https://github.com/REDrighthand05/echo4d/issues/9) 记录 TUM 48 训练帧、12 留出帧的重建 Mesh 深度一致性：覆盖率 95.876%、重叠像素 MAE 3.18 cm、RMSE 11.33 cm；这**不是独立激光表面真值**。
- [#11](https://github.com/REDrighthand05/echo4d/issues/11) 记录独立两次真实 RGB COLMAP 实验的 90/90 注册和 1.2394/1.2502 cm 事后 Sim(3) ATE；真实输入是 TUM 实拍 RGB 重新编码的 MP4，并非原生手机片段，且这个 ATE 的比例由评分 GT 拟合，**没有由软件自动解决米制尺度**。
- [PR #6](https://github.com/REDrighthand05/echo4d/pull/6) 的 `colmap.py` **要求**外部提供已配准米制深度和 `meters_per_colmap_unit`；拒绝“无标定”冒充米制。仍是 Draft。
- [PR #12](https://github.com/REDrighthand05/echo4d/pull/12) 与 [PR #13](https://github.com/REDrighthand05/echo4d/pull/13) 皆以 PR #6 分支为 base，皆包含 `src/echo4d/video_sfm.py`、`docs/VIDEO_SFM.md`、`tests/python/test_video_sfm.py`；是互相竞争的重叠实现，**不是可以按文件一起直接合并的两条独立功能链**。必须先审查区别。连接到主分支前需重新核对 base、提交及 CI。
- `main` 的 `docs/FORMAT.md` 定义列主序 cameraToWorld、米制相机 Z-depth、显示世界 Y-up、图像和深度严格配准；写适配器时应以此契约作为唯一目标。

### 容易混淆的三种“厘米”

1. *事后 Sim(3) 相机 ATE*：模型尺度以真值拟合的**轨迹形状**误差；不能证明实物米制。
2. *留出深度 RMSE*：预测 Mesh 相机视图与未融合但同设备采集的深度像素误差；不能自动证明独立表面几何真值。
3. *独立扫描的几何误差*：另行取得带校准的点云/尺寸并事前规定对齐的外部测量；尚须实验。

## 三、固定的实验协议

1. **选择数据**：先继承 #9 的分帧方法，把 A/B/C 中的尺度锚定样本也预先固定并排除 12 个留出帧；另一组不同运动轨迹的数据须在最终产品结论前验证，防止单场景过拟合。公共数据哈希、序列索引和版本写机器报告。
2. **分离信息流**：SfM 阶段只读取训练 RGB 与公开相机标定；米制锚点阶段只读取预注册的部分训练深度像素及对应 SfM 3D 点；后验评分阶段才能读取 GT poses 与留出深度。不得使用事后 Sim(3) scale 喂给产品适配器。
3. **求尺度**：COLMAP 注册的多视角稀疏点与已校准训练帧深度关联，反投影到相机 Z，以足够空间分布的对应关系拟合全局正尺度；固定鲁棒估计规则，报告原始对应数、拒绝比例、中位偏差与误差分布。若几何退化、锚点不够或尺度不稳定，明确失败而非回落 GT。
4. **配准与物理单位**：校验 FFmpeg 的 PTS/帧映射；如果使用去畸变图像，同步更新相机内参与所有 RGB/深度图；COLMAP 世界坐标到 echo cameraToWorld 的变换不得重复转置或引入反射，最终位姿 3x3 应满足正交和 det≈+1。单目预测的绝对尺度若需要外部锚点，明确其来源。
5. **三路冻结对照**：A 继续使用 TUM GT pose + measured Z，B 使用估计 pose + measured Z，C 使用估计 pose + predicted Z，复用同一个 Mesh 生成器/留出评分器。不能把不同输入或不同覆盖范围的结果只按一个 RMSE 排名。
6. **反例门禁**：有意错开时间戳、深度单位 mm↔m、位姿旋转、标定主点、模型预测偏移、少量 outlier、极少平移；明确记录哪类被阻断，哪类仅分数下降。锁定所有报告字段和失败日志。
7. **两项对外验收**：公开样例从训练数据进入真正可打开 Mesh；另外对实物尺寸/独立扫描进行几何误差验收。二者应分开宣称。

### 建议的先验筛选阈值（待证实）

- 训练/留出协议固定后，留出深度覆盖率 >=80%、重叠区域 RMSE <=0.15 m：作为**B/C 首次实验筛选线**，不是产品精度保证。原始长尾不得删除。
- 事后 GT **评估**米制比例误差 <=10%：仅用于识别实验失败，不准传回求解器。
- 输出完整 registered/failed frame 数、coverage、MAE、median、RMSE、P95、0.03/0.05/0.1/0.3 m residual 分布、runtime、peak RAM/VRAM 与重建/评估输入哈希；不能只写“通过”。
- 至少对第二种运动轨迹做一组 holdout，错误单位/故意退化至少一例被 FAIL 明确阻挡。

这些阈值只是为避免“实验成功”定义漂移，真正的对外发布指标需有多场景统计和独立基准支持。

## 四、先审后改的 PR 分叉处理

PR #12 的证据文件已纳入其分支；PR #13 声称保留原始 decoded PTS，并使用另一个 `sfm_evaluation.py`，两者的时间、失败处理和算分模块存在方案差异。不得因为任何一个有绿 CI 就假设优越：

- 核查输入：原生视频 VFR 的时间戳、精确帧对应、相机畸变与已知内参是否用相同数据解释。
- 核查输出：完整注册率、外部命令与失败日志、未知尺度显式类型、图像路径安全、机器可读的不可变数据契约。
- 对相同冻结输入运行两版，比较轨迹 ATE/旋转/注册帧数/运行时间；任何一版的 GT 后验尺度仅供评价。
- 在评审结果前 **保留两支 Draft**；最终只指定一版主干，再逐项迁移胜出模块。不能让它们互相覆盖。

## 五、路线与时间盒

**0–30 天**：完成重叠 PR 的逐项比较、来源证据索引和两组 RGB-only SfM 基准；同时定义尺度锚点与 A/B/C 的正式 JSON 结果模式。交付：决策文档、两个真实 run URL、一个反例 FAIL。

**31–90 天**：完成 B 路独立尺度 + 已测深度到 TSDF，公布 A/B 同协议的留出误差；C 路完成至少一种预测深度候选的许可、输入单位与推理资源审计和第一份真实失败/成功结果。交付：独立可打开的 PLY/scene + 报告及原始数据出处。

**91–180 天**：扩展到不同设备/运动及失败输入，做独立表面真值/实物尺寸测量，再考虑包装成普通用户的一键导入应用。动态 4D、长时人物追踪与可穿戴实时化不在此阶段的验收范围内。

## 六、工程可行性：主观先验，不是统计学保证

在拥有可用公共 RGB-D 数据、每周持续开发、GitHub Actions 可执行且 AI 辅助编码的条件下，以 **90 天**为期限：

| 目标 | 主观可行性估计 | 最大不确定性 |
|---|---|---|
| 复跑现有两条 SfM 版本并形成审计比较 | 75%–90% | CI 依赖版本与执行环境 |
| B 路 (SfM + 独立部分训练深度锚点 + TUM measured depth + Mesh) 一场景闭环 | 50%–75% | 位姿、尺度及深度配准 |
| C 路 (估计位姿 + 预测 Z-depth) 一场景 Mesh 且达预设留出门禁 | 20%–45% | 单目深度尺度和跨视角一致性 |
| 面向广泛普通手机视频的健壮 4D 系统 | <5% | 动态长期跟踪、观测不足、设备及算力 |

区间仅用于安排优先级，非依据样本频率计算；若硬件/投入/数据可用性不同需调整。

## 七、资料与交付约束

- 项目愿景：[Issue #1](https://github.com/REDrighthand05/echo4d/issues/1)
- 普通视频路线：[Issue #2](https://github.com/REDrighthand05/echo4d/issues/2)
- 独立精度验收：[Issue #3](https://github.com/REDrighthand05/echo4d/issues/3)
- 原始实验：[Issue #7](https://github.com/REDrighthand05/echo4d/issues/7)、[#9](https://github.com/REDrighthand05/echo4d/issues/9)、[#11](https://github.com/REDrighthand05/echo4d/issues/11)
- 新一阶段实施清单：[Issue #15](https://github.com/REDrighthand05/echo4d/issues/15)
- 上游候选源码（只列研究候选、尚未做许可证/适配实测）：[COLMAP](https://github.com/colmap/colmap)、[Open3D](https://github.com/isl-org/Open3D)、[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)、[MapAnything](https://github.com/facebookresearch/map-anything)

这是一份**研究与验收设计**，不是新算法实现、性能突破或测试通过证明。所有新指标应由独立公开 run 生成后回填 Issue。
