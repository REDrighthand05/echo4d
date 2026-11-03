# echo_super 技术大盘点 · 第一批（经典几何链）

总则：
- 只收目标函数写得明白、解确定的正经技术；投票/共识/平均式黑箱手法（RANSAC、TSDF、Hough 等）一律「排除」，理由写在状态列。
- 学习/神经网络类方法单独关「笼子：估计类」，证据等级标"估计/生成"，只许作参考输入，永不冒充实测几何进入主线。
- 证据等级：实测＝由几何/光学测量与显式优化得出；估计＝模型推断；生成＝模型合成。
- 状态：主线在用／已验待注册／待审（逐个验，验过才注册）／待替换／排除。归属：DS 写＝派 DeepSeek Flash 按工单实现、Muse 审核；Muse 写＝Muse 亲手；外部接入＝成熟库/程序以声明的出入类型接入。

| 技术 | 年代 | 开源与代表实现 | 性能档 | consumes → produces | 证据等级 | 状态 | 归属 |
|---|---|---|---|---|---|---|---|
| 视频抽帧（ffmpeg 定间隔采样） | 2000 起 | 开源，ffmpeg | CPU轻 | video → frames | 实测 | 主线在用（stage_extract） | 外部接入 |
| 帧清晰度挑选（拉普拉斯方差） | 2000 | 开源，Pech-Pacheco 法，本项目 quality.py | CPU轻 | frames → frames_selected | 实测 | 主线在用（stage_select，DS 已交付） | DS写 |
| SIFT 特征 | 1999/2004 | 开源，OpenCV / COLMAP | CPU重 | frames → features | 实测 | 主线在用（stage_poses 内部） | 外部接入 |
| SURF 特征 | 2006 | 开源，OpenCV nonfree | CPU中 | frames → features | 实测 | 待审 | DS写 |
| ORB 特征 | 2011 | 开源，OpenCV | CPU轻 | frames → features | 实测 | 待审 | DS写 |
| AKAZE 特征 | 2013 | 开源，OpenCV | CPU轻 | frames → features | 实测 | 待审 | DS写 |
| 最近邻比率检验（Lowe ratio test） | 2004 | 开源，OpenCV BFMatcher | CPU轻 | features → features（匹配对） | 实测 | 待审 | DS写 |
| 五点法本质矩阵 + 非线性最小二乘精化 | 2006 | 开源，Nistér 法，OpenCV / COLMAP | CPU轻 | features → poses（相对位姿） | 实测 | 待审 | Muse写 |
| RANSAC 家族（随机采样一致性） | 1981 | 开源，多实现 | — | — | — | 排除：随机抽样＋共识计数即投票，违反无投票原则；由确定性全点最小二乘＋显式残差门限替代 | — |
| 自定义配对匹配（时间带±30＋首尾回环＋稀疏全局胶水） | 2026 本项目 | 开源，pycolmap ImportedPairingOptions | CPU重 | frames → poses（匹配图） | 实测 | 主线在用（182 帧整轨即此法） | Muse写 |
| 增量式 SfM | 2000s | 开源，COLMAP | CPU重 | frames_selected → poses + points（稀疏） | 实测 | 主线在用（stage_poses） | 外部接入 |
| 全局式 SfM | 2013 | 开源，Theia / OpenMVG | CPU重 | frames_selected → poses | 实测 | 待审 | 外部接入 |
| 稀疏束调整（Levenberg-Marquardt） | 2000s | 开源，Ceres / g2o / COLMAP 内置 | CPU重 | poses + points → poses + points（精化） | 实测 | 待审（COLMAP 内部已含；全局 BA 待单独注册为可调用内核） | 外部接入 |
| PatchMatch MVS | 2009/2016 | 开源，COLMAP | GPU 必需（pycolmap 无 CUDA/HIP 直接拒绝，2026-10-31 实测） | frames + poses → depth（稠密） | 实测 | 环境不可行（本机无 GPU）；替代路线＝双目 SGBM 实测深度 | 外部接入 |
| SGBM 双目立体 | 2008 | 开源，Hirschmüller 法，OpenCV | CPU中 | frames（双目对） → depth | 实测 | 方法不适用（2026-10-31 实测：几何/极线校验 0.0000px 无误，但步行前向运动使校正视差达数千像素，远超 SGBM 搜索范围，输出星芒状伪点云）；仅适用于近似平行侧移的双目对 | DS写 |
| 单目深度 Depth Anything V2 | 2024 | 开源权重，onnx-community | CPU重 | frames_selected → depth | 估计 | 主线在用（stage_depth，标签为估计，不得作实测引用） | 外部接入 |
| 学习式特征 SuperPoint + SuperGlue | 2018/2020 | 开源权重 | GPU | frames → features | 估计 | 待审（笼子内评估） | 外部接入 |
| 逐帧深度仿射对齐（稀疏锚点最小二乘拟合） | 2026 本项目 | 开源，本项目 fuse.py | CPU轻 | poses + depth → depth（已对齐） | 实测 | 主线在用（stage_fuse）。全局联合版已实测（2026-10-31，experiments/optimize_refine.py：全部帧 (a,b) 联合最小二乘，锚点项+跨帧一致性项，雅可比已数值验证）：跨帧散布 0.0838→0.0834，增益≈0.5%——独立逐帧拟合已榨干该参数族，剩余弯曲是帧内深度畸变，(a,b) 表达不了；下一杠杆＝逐帧形变场 | Muse写 |
| 逐帧深度形变场（4×4 双线性网格 exp 校正，全局最小二乘） | 2026 本项目 | 开源，本项目 experiments/optimize_field.py | CPU中 | poses + depth → depth（形变校正） | 实测 | 已验不采用（2026-10-31）：锚点残差中位 −27%（帧内畸变模型是真的），但跨帧散布 0.0834→0.0856 不降反升——深度侧参数族（仿射+形变场）双双到顶，表面弯曲主因不在深度侧，下一杠杆＝位姿侧联合精化 | Muse写 |
| PCA 最小二乘法向估计 | 经典 | 开源，Open3D | CPU轻 | points → normals | 实测 | 待审（泊松链已在用其法） | DS写 |
| 法向定向 orient_normals（朝向来源相机） | 经典规则 | 开源，本项目工单 | CPU轻 | points + normals → normals | 实测 | 已注册 | DS写 |
| 点到面 ICP 配准 | 1992/1994 | 开源，Open3D | CPU中 | points + points → points（已对齐） | 实测 | 待审（多片段合并路线） | Muse写 |
| 平面拟合（总最小二乘 TLS） | 经典 | 开源，本项目 / numpy | CPU轻 | points → points（平面参数存 meta） | 实测 | 已注册 | DS写 |
| 泊松表面重建（Kazhdan） | 2006/2013 | 开源，Open3D | CPU重 | points + normals → mesh | 实测 | 已注册 | Muse写 |
| Ball-Pivoting 表面重建 | 1999 | 开源，Open3D | CPU中 | points + normals → mesh | 实测 | 待审 | DS写 |
| Delaunay 三角化 + 图割表面提取 | 2004 | 开源，Labatut 法，CGAL | CPU重 | points → mesh | 实测 | 待审 | 外部接入 |
| Alpha Shapes | 1983/1994 | 开源，CGAL / Open3D | CPU中 | points → mesh | 实测 | 待审 | DS写 |
| Marching Cubes 等值面提取 | 1987 | 开源，scikit-image | CPU中 | occupancy → mesh | 实测 | 待审 | DS写 |
| 移动最小二乘 MLS 曲面拟合 | 2003 | 开源，PCL / Open3D | CPU中 | points → points | 实测 | 待审（注：局部加权最小二乘拟合，目标函数明确，非投票） | 外部接入 |
| TSDF 体积融合 | 1996/2011 | 开源，多实现 | — | — | — | 排除：多帧加权平均/投票式融合，违反无投票原则；由全局优化＋泊松重建替代 | — |
| Hough 变换（直线/平面检测） | 1962/1972 | 开源，多实现 | — | — | — | 排除：参数空间投票计数，违反无投票原则；由 TLS 平面拟合替代 | — |
| 体素平均合并（现 fuse 末步） | — | 本项目 fuse.py | CPU轻 | points → points | — | 待替换：平均式合并违反原则，换确定性选取并入泊松链 | Muse写 |
| ORB-SLAM3 | 2021 | 开源，官方仓库 | CPU重 | video + imu → poses | 实测 | 待审 | 外部接入 |
| DSO 直接法单目里程计 | 2016 | 开源，官方仓库 | CPU中 | video → poses + points | 实测 | 待审 | 外部接入 |
| LSD-SLAM | 2014 | 开源，官方仓库 | CPU中 | video → poses + depth | 实测 | 待审 | 外部接入 |
| IMU 预积分（Forster） | 2013/2016 | 开源，GTSAM | CPU轻 | imu + poses → poses | 实测 | 待审（眼镜路线） | 外部接入 |
| 声呐占据栅格（逆传感器模型，确定性对数几率更新） | 1985 | 开源，Moravec/Elfes 法，本项目实现 | CPU轻 | sonar → occupancy | 实测 | 待审（远期，眼镜碰撞盒/体积扫描） | Muse写 |
| WebGL 点云查看器 | 2026 本项目 | 开源，本项目 viewer.html | CPU轻 | points → scene | 实测 | 主线在用（stage_present） | Muse写 |
| 特征物比例标定核心 estimate_scale | 2026 本项目 | 开源，本项目工单 WO-scale-core | CPU轻 | points + known_size → scale | 估计 | 已注册（DS 批一，台架已验） | DS写 |
| 回放路径构建 build_replay_path | 2026 本项目 | 开源，本项目工单 WO-replay-path | CPU轻 | poses → path | 实测 | 已注册（DS 批一，台架已验） | DS写 |
| 体素表面下标 surface_voxel_indices | 2026 本项目 | 开源，本项目工单 WO-surface-voxels | CPU轻 | points → points | 实测 | 已注册（DS 批一，台架已验） | DS写 |
| 路径等弧长重采样 resample_path_even | 经典 | 开源，本项目工单 WO-resample-path | CPU轻 | path → path | 实测 | 已注册（DS 批二，台架已验） | DS写 |
| 向心 Catmull-Rom 样条 catmull_rom_path | 1974/1988 | 开源，Barry–Goldman 法，本项目工单 WO-catmull-rom | CPU轻 | path → path | 实测 | 已注册（DS 批二，台架已验） | DS写 |
| 视锥选取 frustum_select_points | 经典 | 开源，本项目工单 WO-frustum-select | CPU轻 | points → points（只选取不动点） | 实测 | 已注册（DS 批二，台架已验） | DS写 |
| 平面拟合＋逐点有符号距离 plane_fit_lsq | 经典 | 开源，本项目工单 WO-plane-fit-lsq | CPU轻 | points → points（平面参数存 meta、距离随 data） | 实测 | 已注册（DS 批二，台架已验） | DS写 |
| 体素代表点选取 voxel_select_representative | 经典规则 | 开源，本项目工单 WO-voxel-representative | CPU轻 | points → points（每 cell 只留离中心最近的实测点，绝不平均） | 实测 | 已注册（DS 批二，台架已验；体素平均合并的确定性替代件之一） | DS写 |

| 特征物比例标定（已知尺寸物 → 全场缩放） | 2026 本项目 | 本项目，用户提出 | CPU轻 | frames + points + 已知尺寸 → scaled points | 估计 | 待建（用户点名方案，2026-10-31：画面里找到标准尺寸的特征物，如农夫山泉瓶身，按其真实尺寸反推整个场景的缩放比例；证据标「估计」） | Muse写 |

计数：共 47 条。主线在用 8（抽帧、清晰度挑选、SIFT、自定义配对、增量 SfM、单目深度、逐帧对齐、查看器）；已注册 11（orient_normals、平面拟合、泊松重建经 register_real_kernels 注册；estimate_scale、build_replay_path、surface_voxel_indices 为 DS 批一；resample_path_even、catmull_rom_path、frustum_select_points、plane_fit_lsq、voxel_select_representative 为 DS 批二——两批均 DS 写、Muse 逐个台架验过才注册）；待审 20；已验不采用／方法不适用／环境不可行 3（逐帧形变场、SGBM、PatchMatch）；待替换 1（体素平均合并）；排除 3（RANSAC、TSDF、Hough）；待建 1（特征物比例标定，其核心 estimate_scale 已注册）。

---

# 第二批（扩容令 · 全路线收录，2026-11-03）

扩容总则（用户 2026-11-03 22:54 令，覆盖第一批的收录口径）：
- 能组装到的技术路线**全部先收进目录**：成熟的、前沿的、甚至不成熟的，都作为超级集成的**可选路线**在列。先集成进工具箱，再看效果、记差距。
- 「目录收录」只代表进了可选清单，**不代表已集成**。状态升级只有一条路：写成内核 → 注册 → Muse 台架逐个验过。没验的永远停在「目录收录·待审／待议」，不许写成在用。
- 每条写明输入 → 输出、依赖、体积（权重/产出量级，约数）、许可、建议证据标签。标签口径同第一批：实测／估计／生成；学习类一律估计或生成，进主线只能作参考输入、挂牌。
- 第一批里 TSDF 的「排除」条目保留原样作历史记录；本批按扩容令以「待议」重新收录，是否启用、用什么实现，由后续逐个验收决定，不预先翻案也不预先否决。

## A · 高斯泼溅与神经表示

| 技术 | 输入 → 输出 | 依赖 | 体积 | 许可 | 建议标签 | 状态 |
|---|---|---|---|---|---|---|
| 3DGS 高斯泼溅训练（INRIA GraphDeco 原版） | frames + poses → gaussians（.ply） | CUDA GPU、PyTorch、COLMAP 位姿 | 产出 .ply 约 70 MB／百万高斯（17×f4/个），场景可达数 GB | INRIA 研究许可（非商业） | 生成 | 目录收录·待审 |
| gsplat（Nerfstudio 系 3DGS 实现） | frames + poses → gaussians（.ply） | CUDA GPU、PyTorch | 同上量级 | Apache-2.0 | 生成 | 目录收录·待审 |
| 2D Gaussian Splatting（2DGS） | frames + poses → gaussians（面元 .ply）/ mesh | CUDA GPU、PyTorch | 约数十 MB／场景 | 随原 3DGS 系研究许可 | 生成 | 目录收录·待审 |
| 前馈高斯 pixelSplat / MVSplat | frames（少量） → gaussians（免逐场景优化） | GPU、PyTorch 权重 | 权重约数百 MB | MIT | 生成 | 目录收录·待审 |
| Instant-NGP（NeRF 哈希网格） | frames + poses → 场（密度+颜色） → points / mesh 导出 | CUDA GPU、tiny-cuda-nn | 权重/快照约数十 MB | NVIDIA 代码许可（以仓库为准） | 生成 | 目录收录·待审 |
| **3DGS → 彩色点云适配器 gaussian_to_points（本项目）** | gaussians（.ply） → points（中心 + SH DC 色，uint8） | 本项目，numpy；读取器为 DS 工单 WO-gs-ply-reader | 0（无权重） | 本项目 | 生成 | **已注册·已验**（2026-11-03：合成夹具逐值验，输出直接进冻结的点云固化 build_voxel_mesh；见 tests/test_gaussian_adapter.py） |

## B · 前馈重建（一次前向出几何）

| 技术 | 输入 → 输出 | 依赖 | 体积 | 许可 | 建议标签 | 状态 |
|---|---|---|---|---|---|---|
| DUSt3R | frames（图对/组） → pointmaps + poses | GPU、PyTorch 权重 | 权重约 2 GB | CC BY-NC-SA 4.0（非商业） | 估计 | 目录收录·待审（估计类笼子） |
| MASt3R | frames → pointmaps + 匹配 | GPU、PyTorch 权重 | 权重约 GB 级 | CC BY-NC-SA 4.0（非商业） | 估计 | 目录收录·待审（估计类笼子） |
| VGGT（Meta） | frames → poses + depth + pointmaps | GPU、PyTorch 权重 | 权重约 5 GB（1B 版）；另有商用版 | 权重许可受限（非商业研究版） | 估计 | 目录收录·待审（估计类笼子） |
| Fast3R / CUT3R（流式前馈） | frames（长序列/流） → pointmaps + poses | GPU、PyTorch 权重 | 权重约 GB 级 | 研究许可（逐项核） | 估计 | 目录收录·待审（估计类笼子） |

## C · 单目深度（Depth Anything V2 之外的主流备选）

| 技术 | 输入 → 输出 | 依赖 | 体积 | 许可 | 建议标签 | 状态 |
|---|---|---|---|---|---|---|
| Depth Anything V1 | frames → depth（相对） | onnx/PyTorch 权重 | 权重约 0.1–1.4 GB（按骨干） | Apache-2.0 | 估计 | 目录收录·待审（与主线 V2 同族备选） |
| MiDaS / DPT | frames → depth（相对） | PyTorch 权重 | DPT-Large 约 1.3 GB | MIT | 估计 | 目录收录·待审 |
| ZoeDepth | frames → depth（度量） | PyTorch 权重 | 约 0.7 GB | MIT | 估计 | 目录收录·待审 |
| UniDepth V2 | frames → depth（度量） + 内参 | PyTorch 权重 | 约 GB 级 | Apache-2.0 | 估计 | 目录收录·待审 |
| Metric3D V2 | frames → depth（度量） + normals | PyTorch 权重 | 约 GB 级 | 代码开源；权重许可逐项核 | 估计 | 目录收录·待审 |
| Marigold（扩散式深度） | frames → depth | 扩散权重（SD 系） | 约 4 GB | Apache-2.0（代码）；基模型条款另计 | 生成 | 目录收录·待审（生成类） |
| MoGe（微软） | frames → pointmap + depth | PyTorch 权重 | 约 GB 级 | MIT | 估计 | 目录收录·待审 |

## D · 点云补全（缺失面补全，生成类）

| 技术 | 输入 → 输出 | 依赖 | 体积 | 许可 | 建议标签 | 状态 |
|---|---|---|---|---|---|---|
| PCN | points（残缺） → points（补全） | PyTorch 权重 | 权重约数十 MB | MIT | 生成 | 目录收录·待审 |
| PoinTr | points（残缺） → points（补全） | PyTorch 权重 | 权重约百 MB | MIT | 生成 | 目录收录·待审 |
| SnowflakeNet | points（残缺） → points（补全） | PyTorch 权重 | 权重约百 MB | MIT | 生成 | 目录收录·待审 |
| PointAttN | points（残缺） → points（补全） | PyTorch 权重 | 权重约百 MB | MIT | 生成 | 目录收录·待审 |

注：本族对应「生成补空」四步里的点云填充环节；补出的点永远挂「生成」，只填实测钉住的物体空缺（用户 2026-11-03 23:11 定的流程）。

## E · 网格化（第一批泊松/BPA/Alpha 之外的路线）

| 技术 | 输入 → 输出 | 依赖 | 体积 | 许可 | 建议标签 | 状态 |
|---|---|---|---|---|---|---|
| Screened Poisson（筛选泊松） | points + normals → mesh | Kazhdan 参考实现 | 0（无权重） | BSD | 实测 | 目录收录·待审（与主线泊松同族，二选一评估） |
| 广义绕数 GWN + Marching Cubes（libigl） | points → mesh | libigl、CGAL 系 | 0 | MPL-2.0（libigl） | 实测 | 目录收录·待审 |
| FlexiCubes / DMTet（nvdiffrast） | grid / SDF → mesh（可微） | CUDA GPU、PyTorch | 0 | NVIDIA 研究许可（非商业） | 估计 | 目录收录·待审（多与学习 SDF 搭配） |
| OpenVDB 等值面提取 | occupancy / SDF 栅格 → mesh | OpenVDB | 0 | MPL-2.0 | 实测 | 目录收录·待审 |

## F · TSDF 类体积融合（扩容令收录 · 待议）

| 技术 | 输入 → 输出 | 依赖 | 体积 | 许可 | 建议标签 | 状态 |
|---|---|---|---|---|---|---|
| TSDF 融合（Curless–Levoy 经典法） | depth + poses → TSDF 体积 → mesh | 多实现 | 0 | 依实现 | 估计（加权平均融合） | 目录收录·待议（第一批曾排除、理由存档；扩容令收录，启用前逐个验） |
| KinectFusion / InfiniTAM（体素哈希） | depth 流 → TSDF 体积 → mesh | GPU（InfiniTAM） | 0 | InfiniTAM 开源许可 | 估计 | 目录收录·待议 |
| Open3D ScalableTSDFVolume | depth + poses → mesh | Open3D（本机已在依赖内） | 0 | MIT | 估计 | 目录收录·待议（接入成本最低的一条） |
| nvblox（NVIDIA Isaac） | depth 流 → TSDF / ESDF + mesh | CUDA GPU | 0 | Apache-2.0 | 估计 | 目录收录·待议（与机器人/实时路线相关） |

## G · 体素化路线

| 技术 | 输入 → 输出 | 依赖 | 体积 | 许可 | 建议标签 | 状态 |
|---|---|---|---|---|---|---|
| 点中心体素固化（本项目，冻结特性） | points → voxel 体积文件（每点一立方、带原色） | 本项目 echo_scene/voxels.py | 0 | 本项目 | 实测 | 主线在用·冻结（行为锁定，tests/test_voxel_frozen.py；彩色体积导出见 tests/test_voxel_export.py） |
| OctoMap 占据八叉树 | depth / sonar → occupancy 八叉树 | OctoMap 库 | 0 | BSD | 实测 | 目录收录·待审（眼镜/实时体积路线候选） |
| 体素雕刻 + 体素下采样（Open3D） | frames + poses / points → voxel 栅格 | Open3D | 0 | MIT | 实测 | 目录收录·待审 |
| MinkowskiEngine 稀疏体素卷积 | voxel 栅格 → 特征 / 分割 | CUDA GPU、PyTorch | 权重依任务 | MIT | 估计 | 目录收录·待审（估计类笼子） |

## H · 物体发现（照片路线）

| 技术 | 输入 → 输出 | 依赖 | 体积 | 许可 | 建议标签 | 状态 |
|---|---|---|---|---|---|---|
| **照片发现 photo_discovery（本项目）** | job（帧+深度+位姿+点云）+ 目录词条表 → objects（物体清单：label／3D 包围盒／固实点数／检出帧数＋逐件点集） | OWLv2（google/owlv2-base-patch16-ensemble）+ SAM（facebook/sam-vit-base），权重 HF 缓存约 GB 级；深度对齐只读复用 echo_scene.fuse；词条表是内核包内数据文件 catalog.json（可增删） | 权重约 GB 级（HF 缓存） | 本项目（模型许可随 OWLv2／SAM 原许可） | 估计 | **已集成·已验**（2026-11-04：experiments/discovery_photo 三轮先锋收编为 echo_super/kernels/photo_discovery 并注册；20:41 真采集 job 上达标 4/6——书桌／衣柜／显示器／椅子，缓存注入契约测试下名字、固实点数、包围盒与 REPORT3 逐值复现，tests/test_photo_discovery.py）。已知边界：杂物架一词多物未解（需实例分离后逐件交付，基线输出是多物合并集，词条 pitfalls 已标注）；空调暗光小物不可靠（三轮均误报，需形状/位置先验或个人层人工录入）；掩膜层清理（深度裁切外加最大连通域）已证为空刀，实例隔离定版在印证集层（0.3 体素最大连通分量） |

注：本条是「程序发现」线的第一个已集成内核：目录词条在帧上认物（首次见面即可认，不靠多次出现当判据），云只做固实与实例隔离——与用户 2026-11-03 22:57 定的强发现原理一致（一致性只管固实）。绝对尺寸仍缺尺度锚（该 job 尺度未定，包围盒只作云内相对判断），归尺度线不归本条。

计数（第二批）：共 34 条（A 高斯/神经表示 6、B 前馈重建 4、C 单目深度备选 7、D 点云补全 4、E 网格化 4、F TSDF 族 4、G 体素化 4、H 物体发现 1）。已注册·已验 2（gaussian_to_points 适配器、photo_discovery 照片发现）；主线在用·冻结 1（点中心体素固化，第一批起就在用，本批补记）；目录收录·待审 27；目录收录·待议 4（TSDF 族）。全部「待审/待议」条目均未集成，只是可选路线清单。两批合计 81 条。
