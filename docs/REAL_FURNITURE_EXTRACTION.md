# ECHO GPT C — 第一件真实家具：照片证据到米制三角面的安全提取

**交接 Issue:** [#42](https://github.com/REDrighthand05/echo4d/issues/42) · **上游:** [#34](https://github.com/REDrighthand05/echo4d/pull/34) merged, [#43](https://github.com/REDrighthand05/echo4d/pull/43) geometry gate, [#44](https://github.com/REDrighthand05/echo4d/pull/44) source scene inventory. 此候选**独立分支**叠加于 #44 固定 head `e3400420`，不写 main/#43/#44 的分支。没有重复实现 #43 的测量评分或 #44 的库存/Hi3D 归一化工具。

## 事先固定的验收协议（不能为模型外观调整）

- ECHO 真实输入只检查 **同一批** `docs/evidence/tum-static-ab-v2/a-known-pose-mesh.ply` 和 `b-estimated-pose-mesh.ply`；SHA-256分别锁定 `7c83972f794dc1ad90041d36475be9d1a46d9916c576f8c754efd132310170b4` 和 `a5273c0d38117d6f2063b6edc9c49adb9c71f8e38c2f2bc35cfb3d63a33d5897`。前者 2270顶点/3773面，后者 2518顶点/4064面；几何连通片 24 / 34，**不是家具件数**。请看 #44 原始机器报告。
- 目标一件 **chair 或 table**，至少 **2 个不同来源 RGB 画面**，各有与该画面逐像素配准的、已核对“同一个物体实例”的**二值前景 mask**、**实测且以米为单位的相机Z-depth**、同画面内参和 world-from-camera 刚体位姿；全部文件逐字节 SHA-256。不得根据断开的面片自行标注家具，人工确认也须保留原始遮罩及依据。允许合法类别 sofa/cabinet，为未来样本预留，但首轮不因此扩大目标。
- 对每个真实原场景顶点用相机位姿/内参投影，在原始像素 Z 与测得 Z 差 **≤3cm（固定测试默认值）**、同实例 mask 认领时记一个肯定票。至少两张肯定票且零张在深度一致处明确否定，才保留此顶点。只有三个顶点都合格的**原三角面**被保留：不生成新三角面、不封孔、不平滑、不做世界坐标缩放/平移，也不拟合留出数据。
- 导出 `observed-object-world.ply`、`selected-source-face-indices.npy`、`report.json`，包含来源输入每张 RGB/mask/depth SHA、原场景 SHA、选出三角面索引/包围盒/尺寸以及**identity 世界变换**。**这只是待人工独立审核的候选**；自动工具没有能力认证审阅者和原始照片语义，因此即使机器 PASS 也将 `verified_furniture=false`。
- 缺已审核 mask / 无两视角对应 / 来源 hash 异常 / 未定米制尺度 / 动态移动物体 / 深度不一致时 FAIL 或 BLOCKED，不输出已经证实的家具网格。原 `mesh.ply` 永不覆盖，Hi3D 也不在此运行。
- 真实尺度范围只能来自原重建传感器校准、不能由单目 RGB 或 Hi3D 的视觉效果恢复；请保留精度和 source claim 分离。

## 机器对照：从真实 ECHO 场景开始，公开数据兜底

`python tools/real_furniture_scene_trial.py --repo-root . --out /tmp/echo-object-trial --fetch-public-fallback` 在 CI 真正读取/验证这两份 ECHO PLY，用 #44 的 `inventory()` 复用已有连通块统计。由于**没有可审核的同物体源图 mask 对应**，这里必须记录 `blocked_missing_furniture_instance_evidence`，而非把最大网格片称作桌椅。

备选公开物体：Stanford Object Sensing `single-chair` Kinect RGB-D，官方原始资料 [source](https://graphics.stanford.edu/projects/objectsensing/)（`single-chair.zip`, 约130MB，CC BY-NC-SA 4.0）。CI 尝试下载，限定180MiB、限定 ZIP 解压数据预算、拒绝路径逃逸；保存源包 SHA、RGB/depth/pose 配对数与**一帧原始照片 SHA**供复核。若下载失败应如实留原因。源数据虽标注单椅子，但是每帧哪些像素归属椅子、是否包含背景仍须确认。**绝不将同网站提供的 ShapeNet 检索模型冒充真实扫描**。若无一致实例 mask，只作为下一步数据源，不输出假椅子模型。此类外部公开资料还没有映射到 ECHO 原房间位置；不得复用 ECHO 世界坐标或者虚构物理放置。

机器脚本将**审计运行 PASS**与**物体未确认 BLOCKED**明确分开。测试文件 `tests/python/test_real_furniture_extract.py` 使用可计算答案的小型虚构平面来检查投影、原面索引、保存米制坐标、负例和不覆盖，**不应被统计为真实家具成功**。

## 已调查、可复用的项目（截至 2026-10-26）

| 组件 | 作用/输入 | 运行资源 | 许可证/发布注意 | 不能证明什么 |
|---|---|---|---|---|
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | 用 `RGB + table/chair 文字` 找候选二维框 | PyTorch；官方支持CPU，常用GPU推理 | Apache-2.0 源码，权重另审 | 检测框不等于像素 mask，同名的两把椅子也可能混淆 |
| [SAM 2](https://github.com/facebookresearch/sam2) | 输入 RGB/视频与人指定框/点，分割并追踪实例 | PyTorch，推荐Linux/NVIDIA CUDA；无需自己训练大模型 | 模型/源码 Apache-2.0，版本/权重需固定 | 自动画的“像椅子”并不是几何深度证据，也可能跨画面跳实例 |
| [Open3D](https://github.com/isl-org/Open3D) | 测量深度反投影/点云 RANSAC、DBSCAN | CPU可用，密点云耗内存 | MIT | 点云聚类/连通性本身不证明物体身份 |
| [trimesh](https://github.com/mikedh/trimesh) | 原三角面子集导出、场景 PLY/GLB IO | CPU、轻量 | MIT | 不自行识别家具；本分支复用 #44 锁定版本 4.11.1 |
| [CO3D](https://ai.meta.com/datasets/co3d-downloads/) | 多视角真实物体（chair等）RGB + foreground masks + cameras/point cloud 的有监督备选 | 数据 TB 级，类别单包很大 | CC BY-NC 4.0，仅实验候选 | 即使物体类别已标注，不等于在 ECHO 房间中观察到它 |
| [ScanNet](https://www.scan-net.org/ScanNet/) | 带实例语义标签的实测室内RGB-D+Mesh，对象级基准 | 大容量，下载需机构申请使用协议 | 专门条款，不能未经许可镜像/转发 | 获取权限前不可把其标签当已测试输入 |
| [Stanford single-chair RGB-D](https://graphics.stanford.edu/projects/objectsensing/) | 备选公开单椅子序列：真实照片、深度、相机 pose | 下载约130MB，CPU审计 | CC BY-NC-SA 4.0 | 需再核对每帧对象实例掩膜；ShapeNet检索几何非实测原物体 |

**排序**：先原始照片人工看候选，之后 Grounding DINO→SAM2（若GPU/许可允许）形成*候选* mask，追踪同一件家具；再把 mask、实测深度和原姿态投射到 ECHO Mesh，**不按纯3D连通块推断家具**；最后将经审查的粗模型交给 #44 prepare 和 #43 gate。无需从零训练大型模型。暂不向源码捆绑任何大权重。

## Hi3D 负责人交接

只有得到**有源照片/多视角实例 mask/米制来源/可逆世界坐标证据**的真正家具候选，才允许调用 #44 `prepare`，再将未来 Hi3D 生成几何交给 #43 在不参与生成的独立测量数据上评价。Hi3D 生成部分必须在单独层标为 AI 推测，**不能覆盖已观测的真实表面**。当前真实物体未证实则不进入 GPU 细化。


## Issue #47 C→D interface audit (additional experimental candidate)

Mainline ECHO uses a **strict binary PLY reader** for independent holdout
geometry scoring. Trimesh's default xyz+face PLY cannot be fed to that
reader. The candidate now writes the **same bounded vertex/face binary
layout**, with face-derived normals and neutral gray (127) **placeholder**
RGB. The extra fields exist **only for compatibility**; they are **not
measured sensor colors, observed TSDF fusion, reconstructed texture, or
verified furniture**. The `report.json` explicitly records this caveat;
the original triangle indices, world vertices and source mesh SHA persist.
The C→D reader regression calls the actual `echo4d.holdout.read_echo_ply`,
not a toy substitute.

Two identical source photographs (including the same photo copied to a
different filename) cannot count as two independent observations. Two
different bytes captured at effectively the **same camera pose** also
cannot count as independent viewpoints; at least a 1cm camera translation
or 0.5° orientation baseline is required in the current research gate.
These checks are conservative software safeguards, **not cryptographic
proof of actual different capture times, correct masks or identity**.

The world-frame identifier in the candidate report is based on the
**original scene SHA**, not an assertion of TUM/world GT alignment.
GPT D's physical scoring still requires independently verified
camera-to-world registration and dimension anchors before treating this
mesh as compatible with the measured heldout depth. PLY parse
compatibility alone does not mean the full D proof passes.
