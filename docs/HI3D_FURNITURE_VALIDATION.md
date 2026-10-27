# GPT D｜首件真实家具：TUM证据与冻结留出验证

> **审查结论（2026-10-26）：现有 ECHO TUM 数据可核查真实 RGB-D 配对，但截至本分支创建时，单件家具身份/独立尺寸/逐帧家具实例掩码均缺失，故第一件真实家具的 Hi3D 前后精度实验为 BLOCKED，而非 PASS。**
>
> 这不是运行 Hi3D GPU 的报告；不修改 main、#43、#44 和 GPT C 的分支。

## 1. 已确认的真实数据与不能夸大的地方

来源：TUM RGB-D 官方 Freiburg1 XYZ `rgbd_dataset_freiburg1_xyz.tgz`；
官方网址 https://cvg.cit.tum.de/rgbd/dataset/ ，约 448,204,271 字节，
SHA-256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`。
此 SHA 是 ECHO 以前实际下载后观察并冻结的值，**不是 TUM 厂商签发的校验值**。
TUM 原论文说明 CC BY 3.0；使用或分发时需保留来源归属，不能推断所有二次衍生内容可任意再授权。

ECHO 已在主干 `docs/evidence/tum-static-ab-v2/` 保存
`sensor-scale-metric-report.json`、`b-48train-12holdout.json`、
`b-estimated-pose-holdout.json`、`ab-run-report.json` 和 A/B 房间 PLY。
#44 对这两份场景网格的真实文件检查为：

| ECHO源 | 网格 SHA-256 | 顶点／三角 | 连通块 | 家具身份 |
| --- | --- | --- | --- | --- |
| A 已知位姿整房间 | `7c83972f794dc1ad90041d36475be9d1a46d9916c576f8c754efd132310170b4` | 2270 / 3773 | 24 | 未验证 |
| B 估计位姿整房间 | `a5273c0d38117d6f2063b6edc9c49adb9c71f8e38c2f2bc35cfb3d63a33d5897` | 2518 / 4064 | 34 | 未验证 |

连通块不是桌子/椅子的身份证。TUM XYZ 是对典型办公桌的基本平移相机测试；
它包含真实视频原帧、Kinect测距与 motion capture 相机轨迹，**没有提供这一桌子的独立尺量
长度、每帧实例 mask 或独立激光家具表面**。粗网格 A/B 仍是全场景，不能直接喂 #44 的单物体入口。

## 2. 不准重新挑样本的真实照片／深度对照表

永久审计清单：
[`docs/evidence/hi3d-furniture-d/xyz-frozen-source-ledger.json`](./evidence/hi3d-furniture-d/xyz-frozen-source-ledger.json)。

严格重用此前 #28 / #34 **在 Hi3D 结果出现之前**已固定的 AB-V2-final：
60 张原始来源 RGB + 深度记录，48 张训练 / 12 张完全保留用于验证。
留出序号 **[4, 5, 9, 10, 14, 19, 24, 26, 29, 34, 35, 55]**；
源序号25因原 A/B 参考位姿缺失在当初预注册时已排除，
这里不能因为家具形状或 Hi3D 成绩再改一次。

对每张照片记录原始 TUM `rgb/<时间>.png`、原始拍摄时间、历史原图 SHA256；
对每张深度记录真实 `depth/<时间>.png` 和原始 Kinect 单位。
留出的12张原始深度有逐字节 SHA256；训练深度的已转换版本也有 SHA，
但历史报告**没有提供训练原始 PNG 的逐张哈希**，清单中为 JSON `null`，不补造。
历史 ECHO 验证分辨率下的总场景误差与家具误差不等价。

- **实际测量**：真实 Kinect 原始深度 PNG 数值，除以5000成为米；原始 RGB/深度时间；
  TUM motion-capture 的轨迹采样（相机参考位姿需明确校准/同步）。
- **计算估计**：COLMAP 相机位姿、TSDF 三维表面、这些表面的包围盒、
  从深度计算的相机坐标点；本程序计算的投影覆盖和误差。
- **未知**：某张桌子/椅子的独立实际尺寸、房间网格与该家具的真实
  局部分割、家具背面真实性、每个可见像素是否属于它、Hi3D 输出质量。

注意：先前 ECHO 从 TUM 原图重新编码的 4fps MP4 是**测试生成的视频**，
其帧时间不得反过来当作原 TUM 相机采样时间。清单保存原始十位 UNIX 秒时间戳。
原始整数深度转换一次后，浮点米制深度**不再除5000或应用 COLMAP平移尺度**。

## 3. 可重复运行：检验真的有这些相片与深度

仅运行冻结元数据审查，不下载权重、不访问原图：

```bash
python tools/hi3d_furniture_evidence.py \
  --ledger docs/evidence/hi3d-furniture-d/xyz-frozen-source-ledger.json \
  --out output/furniture-ledger-report.json
```

在隔离目录拥有官方原始完整 `.tgz` 后：
```bash
python tools/hi3d_furniture_evidence.py \
  --ledger docs/evidence/hi3d-furniture-d/xyz-frozen-source-ledger.json \
  --archive /private/dataset/rgbd_dataset_freiburg1_xyz.tgz \
  --out output/furniture-source-audit.json
```

程序逐字节核实整个原始压缩包、**全部12张留出原图 + 全部12张原始测距 PNG**
与 TUM原始 `groundtruth.txt` 的 SHA，保存最邻近动捕位姿采样时间、
`translation_m` 和四元数。**最近邻动捕采样绝不能冒充插值后、
已经外参校正的相机姿态**；报告把时间差和此限制写出。
无法完整核对时机器返回错误并保存失败报告。
专用 GitHub Actions 在全新runner运行同样的真实下载与核查、归档结果；
不把受限制许可的数据文件重新提交到Git。

## 4. 公平的 Hi3D 前后比较代码

`tools/hi3d_furniture_compare.py` 复用 ECHO
`echo4d.holdout.rasterize_mesh_depth` 的三角形 z-buffer 与 #43 的已实现
`surface_sample`，不另造容易“看上去更好”的光栅器。需要 GPT C 提供：

1. 审过的**单件家具**粗网格与同物体照片/对象掩码证据、两份独立存储的
   世界米制 ECHO PLY，源 SHA；
2. 每个原始留出视角相同家具**审核的**二值 mask、原始 TUM RGB/深度字节校验、
   时间戳与校准后的 camera-to-world（而非只靠最近邻 raw 动捕值），12张缺1张就拒绝；
3. 在任何 Hi3D 结果与留出真值打开之前确定的统一米制世界安放参数。
   **不能根据留出深度来拟合或调整 refined 模型尺寸。**
4. 独立尺量的该家具三方向长度（米）、真实轴方向及外部取证说明。
   若只有模型包围盒的估计，必须留空，拒绝尺寸真值结论。

比较接口（下一阶段真实对象审核成功后使用）：

```bash
python tools/hi3d_furniture_compare.py \
  --ledger docs/evidence/hi3d-furniture-d/xyz-frozen-source-ledger.json \
  --proof /private/furniture-42/object-proof.json \
  --root /private/furniture-42 \
  --out /private/furniture-42/results/comparison.json
```

对同12张帧、同一 masks、同一位姿、同一实测 camera-Z：
按原先固定的门槛比较覆盖率（目标>=80%、恶化<=2百分点）、
重叠区 RMSE（恶化不得超过5mm）、实际尺寸相对误差（不得增加）、
留出测量点到网格的 3D RMSE 和反向表面采样误差。
完整报告显示每帧遗漏、模型在掩码外出现的额外预测像素；
这些额外像素可能是遮挡或错误分割，**不能直接认定为真实/幻觉补全**。
未拍摄面保持 **UNKNOWN**；不能据此改善伪造历史外观。
输出总是新的报告文件，永不覆盖 ECHO 的 `mesh.ply`。

这是公平评分所需要的数据格式门禁，不自动验证人工签字和实物量尺证书。
报告即使满足统计门槛，仍需 Hi3D 总负责人审查来源和历史留出隔离。

## 5. 公共数据替代调查，非已下载或已使用

| 数据集 | 强项 | 重要限制和门槛 |
| --- | --- | --- |
| **[ARKitScenes](https://github.com/apple-aiml-research/ARKitScenes)** | 真实室内 RGB、LiDAR 深度、带时间的相机轨迹、椅桌等家具3D box；部分有独立 Faro 高分辨率扫描 | box 是标注/估计，不等于卷尺量出的真实物体尺寸。先选择具体 video ID、家具实例 ID，确认 Faro 与照片对齐及授权，不能把 box 直接标「独立实物真值」 |
| **[ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/)** | 高分辨率真实图、iPhone RGBD/pose、激光扫描及物体实例标注，最适合建立独立几何表面参照 | 需申请批准，仅按协议作非商业研究/教育；不可把其大批素材直接发布于公开 GitHub。要先获得正式权限后选择一件有足够覆盖的家具 |
| **TUM Freiburg1 XYZ / desk** | 现有 ECHO 完整可复现标定/测深通道和先冻结拆分 | 没有已审计的物体实例分割、家具独立尺寸或独立扫描；不能称为完整单件家具实验 |

**推荐的下一条真实对象研究路线**：在允许取得数据和合法使用的情况下，
优先验证一个具有**独立激光表面**的 ScanNet++ 椅子/桌子实例，或者
具有 Faro 地面真值资产的 ARKitScenes 家具。
不得使用从 Hi3D 输出包围盒拟合出来的「实测长度」。

## GPT C 与 Hi3D 整合负责人交接

GPT C：请提交单件家具的实例标记/源原图对照、未混入整房间的
粗模型、确定的尺度/世界定位来源；该团队的仓库分支无需被 GPT D 更新。
Hi3D 整合：只有在 `object-proof.json` 来源材料、全部12张留出实例 mask、
独立尺寸与隔离规则被审查通过后才使用比较器。审查之前的研究结论为
**BLOCKED、家具真实已核验 0、Hi3D 真正运行 0**，不能通过 CPU 测试升格为实验成功。

机器结果、最终SHA、失败和GitHub Actions链接在本 Draft PR 与 Issue #42 留痕。
