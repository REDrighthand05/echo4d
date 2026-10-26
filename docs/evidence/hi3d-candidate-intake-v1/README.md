# Hi3D 首轮真实 ECHO 资料检查（无 GPU 推理）

来源：[Issue #42](https://github.com/REDrighthand05/echo4d/issues/42)、[Draft PR #44](https://github.com/REDrighthand05/echo4d/pull/44)、[实际 GitHub Actions #40316488214](https://github.com/REDrighthand05/echo4d/actions/runs/40316488214)。

**机器判定：blocked_before_real_Hi3D_GPU_inference**。这意味着已实际检查已有 ECHO 场景数据，但是目前不能可靠认定一件可直接交给 Hi3D 的独立家具。没有运行 Hi3D，没有提供效果提升百分比。

## 读取了主干中的两份真实场景模型

| 场景 | 顶点数 | 三角面数 | 几何连通块 | SHA256 |
|---|---:|---:|---:|---|
| a-known-pose-mesh.ply | 2270 | 3773 | 24 | `7c83972f794dc1ad90041d36475be9d1a46d9916c576f8c754efd132310170b4` |
| b-estimated-pose-mesh.ply | 2518 | 4064 | 34 | `a5273c0d38117d6f2063b6edc9c49adb9c71f8e38c2f2bc35cfb3d63a33d5897` |

这里的 24/34 个连通块不是 24/34 件家具；可能是房间里的不完整表面，不能据此直接判定家具身份。

## 继续真正 Hi3D 对照需要

1. 从实测房间中选出并核对一件**独立家具粗模型**和同物体原始照片，保留分割依据；
2. 为它准备没有参与生成的独立物体深度/形状测量；
3. 在合适 NVIDIA GPU 上运行 Hi3D 512³，固定权重与运行参数，保留失败和 OOM；
4. 将前后模型做同条件真实精度检查，不能用评价数据事后拟合实际尺寸。

原始文件 `report.json` **直接来自真实 GitHub Actions 压缩包**，未经手工重写。第二轮后续 PR 代码修改只解决测试依赖，不影响本次两份实际 ECHO 网格输入与读取结果。
