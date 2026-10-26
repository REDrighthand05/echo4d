# ECHO × Hi3D：物体形状增强的独立实验

**当前只有 CPU 几何来源/尺寸审查程序，没有真正运行 Hi3D 模型。**

## 核实的上游边界
- 源码：https://github.com/hi3d-official/Hi3Dv1.0 （2026-10-23 发布）
- 权重：https://huggingface.co/hi3d-official/Hi3Dv1.0
- 输入：同一物体的参考图片 + 已有粗网格。输出：没有颜色/材质的几何形状，且重新居中与正规化，不自动保留原来以米计的真实尺寸和位置。
- 官方环境：Linux、Python 3.10、NVIDIA CUDA GPU。最精细 1536³ 模式只确认在 96GB H20 上运行；本 PR 不下载权重、不运行模型。
- 许可证不自动等于 ECHO 的 MIT；使用 Hi3D 自有材料需要独立审阅 Hi3D Open License v1.0、商业标识/规模条款和其他第三方授权。

## 为什么先做审查关口？
生成的物体看起来更精细，也可能已被移动、缩放或填上猜测的背面。正式 ECHO 场景必须保留原来的、有真实测量依据的三维模型。AI 形状是可以关闭的另一个显示层。

## CPU 审查程序
工具：tools/hi3d_geometry_gate.py。输入清单版本 echo-hi3d-audit-1，单位 meters，四个具有 SHA256 的文件：
- coarse_world_mesh：原 ECHO 单个物体粗模型，真实米制世界坐标，.npz 中只有 vertices（Nx3）和 faces（Mx3）；
- refined_local_mesh：拟接入的 Hi3D 输出经过外部转换后的本地几何 .npz；此工具不能直接解析 GLB；
- reference_image：参考照片，只校验其原始字节身份；
- heldout_reference_points：生成期间没有使用过的真实世界米制测量点，.npz 中只有 points_m（Nx3）。

清单还必须记录 hi3d_revision、alignment_origin=coarse_preprocessing_only_not_holdout、alignment_provenance，以及列优先16元素的 world_from_refined_local 矩阵。该矩阵只允许右手旋转 + 一个正的统一尺度 + 平移。**它必须由粗模型预处理提前确定，不能用留出的测量点事后拟合。**本工具只能检查声明与矩阵，不能独立证明声明真实性。

执行示例：
    python tools/hi3d_geometry_gate.py --root research_inputs --manifest manifest.json --output outputs/hi3d-report.json

每个资产对象在 JSON 中都有 path 与 sha256。工具会校验这四个源文件，然后从前/后模型按三角形面积抽取相同数量表面点，以相同留出测量点比较两个方向的近邻误差、覆盖比例。

初步保留门槛：相对于原粗模型，留出点至模型的近邻 RMSE 最多增加 0.005 米，覆盖率最多下降 0.02（百分点比例）。注意：这只是近似、有限样本的初筛，并不能证明真正三维表面精度，更不能证明 AI 补全的是历史事实。

## 研发任务
- [x] CPU-only 实验室安全关口，14项合成正确值/错误输入测试。
- [ ] GitHub 的 Windows/Linux 自动验证，保留实际运行证据。
- [ ] 单物体分离与原图片身份登记。
- [ ] Hi3D GPU 真正运行（建议先 512³），锁定模型版本/随机种子并记录时间、显存及失败。
- [ ] 验证与上游 Hi3D 正规化完全一致的逆变换，否则不得导入米制 ECHO。
- [ ] 对没有参与生成的真实测量做前后误差比较与许可审核。

用于后期重现模式的可选 AI 物体细化，眼镜实时模式只使用事先生成的轻量版本，未知区域维持与真实记录明显不同的显示。
研究工作台：https://github.com/REDrighthand05/echo4d/issues/42
