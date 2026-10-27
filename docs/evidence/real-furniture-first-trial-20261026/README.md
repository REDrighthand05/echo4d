# GPT C — 第一次真实家具分离实验（原始机器结果保留）

- **输入**：#34 已合入 main 的两份真实 TUM RGB-D 静态 Mesh；验收对象为一张桌子或椅子，但不能只凭网格连通性指认。
- **原始执行**：[Actions #40321454964](https://github.com/REDrighthand05/echo4d/actions/runs/40321454964)，包含 [artifact 12649222222](https://github.com/REDrighthand05/echo4d/actions/runs/40321454964/artifacts/12649222222)。从这一 artifact 的 `report.json` 原始数据逐值保存在本目录 `report.json`；未人为修正实验数字。
- **修正后的回归运行**：[Actions #40321601645](https://github.com/REDrighthand05/echo4d/actions/runs/40321601645)，原对象提取工具新测与 #44 原有输入工具共 **26 passed**。同一代码树的 [ECHO 标准验证 #40321601569](https://github.com/REDrighthand05/echo4d/actions/runs/40321601569) **4/4 PASS**。基础 CI 的首次失败仅是新测试缺少 optional trimesh 的 skip 入口，现沿用 #44 原先的模块可选策略，不修改冻结真实输入。
- **真实 A**：SHA-256 `7c83972f794dc1ad90041d36475be9d1a46d9916c576f8c754efd132310170b4`，2,270 个顶点、3,773 个三角面、24 个连通块。
- **真实 B**：SHA-256 `a5273c0d38117d6f2063b6edc9c49adb9c71f8e38c2f2bc35cfb3d63a33d5897`，2,518 个顶点、4,064 个三角面、34 个连通块。
- **真实提取结果**：`blocked_missing_furniture_instance_evidence`，**0 件已证实独立家具**。场景没有经原始照片核对的同一件桌椅 mask 和逐帧深度对应表，不可取最大几何块来包装成家具。未输出家具 mesh，所以没有可证明的**单家具世界位置或尺寸**。
- **备选真实数据**：Stanford `single-chair` 原始 RGB-D 约130MB，CC BY-NC-SA 4.0；首次 Actions 下载尝试 `URLError: <urlopen error timed out>`。这表示下载失败，**不是已经得到或分割了椅子**；也不复用其网站提供的外观更完整的检索 ShapeNet 模型，因为那不是实拍扫描的形状。
- **模型输出**：仅存在**合成回归测试**生成的 original-face 子网格（不会归档或冒充真实家具）。源场景 PLY 不变；没有 Hi3D 推理、物理尺寸改写、AI 补洞或独立家具精度报告。

下一轮明确需要：至少两幅带被独立审核的“同一件家具” mask 的原始 RGB、相应配准实测 depth、真实相机内外参、实例审阅者与 SHA；或有合法使用权限的真实家具实例标注公开数据。只有确认这些后，才允许生成粗家具 mesh、交给 #44 打包和 #43 留出测量门禁。原始任务仍然开放，Draft PR 不自行合并。
