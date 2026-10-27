# ECHO Hi3D 试验｜在真实重建房间中选择第一件物体

关联 [Issue #42](https://github.com/REDrighthand05/echo4d/issues/42)、[Draft PR #43](https://github.com/REDrighthand05/echo4d/pull/43)。该程序是后续实验的**候选素材准备工作**，不是 Hi3D 模型，也没有运行显卡推理。

## 为什么不能直接用现有房间模型？

ECHO 主干现有两份来自实际 TUM RGB-D 资料的、已检查过的三维场景：

- `docs/evidence/tum-static-ab-v2/a-known-pose-mesh.ply` （拍摄位置使用数据提供的真实记录）
- `docs/evidence/tum-static-ab-v2/b-estimated-pose-mesh.ply` （拍摄位置由软件估计）

它们是**完整场景的部分重建网格**，并非已分离、已确认的单件家具。Hi3D 官方 `demo/refine_mesh.py` 需要的是**同一件物体**的参考图片与粗网格。因此，把整个场景当作一件椅子投入 Hi3D，不是有效实验。

这个仓库暂时没有已核对的「单件家具轮廓/相应原始照片/这件家具本身的独立深度测量」三件套。即使全场景由真实测深生成，也不表示我们知道哪些三角面属于同一件家具。CPU 工具会报告 `requires_furniture_mask_and_matching_real_image`，不会擅自从几何连通块推断家具类别。

## 可执行的第一阶段

```sh
python -m pip install trimesh==4.11.1
python tools/hi3d_candidate_intake.py inventory \
  --mesh docs/evidence/tum-static-ab-v2/b-estimated-pose-mesh.ply \
  --output /tmp/hi3d-b-inventory.json
```

[GitHub Actions](../.github/workflows/hi3d-object-intake.yml) 同时检查真正的 A/B 场景 PLY，打印每个真实源文件的 SHA256、三角形/顶点数量、模型范围、各几何连通块大小。**如果没有可信的单物体与图片证据，就登记阻塞，不进行 Hi3D 推理。**

## 未来准备输入包（不对整个房间运行）

确认一件椅子/桌子，并保留它的原始拍摄照片后，需要：

1. `scene.ply`：ECHO 未改动的带实测物理尺寸的场景网格。
2. `one-object.ply`：通过人工核对或能复查的独立物体分割方法取出的同一件家具（仍需和原场景坐标保持一致）。
3. `reference.png`：实际能拍到该物体的参考画面，必要时裁剪/去背景并保留处理记录。
4. `selection.json`：有来源描述和上述三份文件的 SHA256；标明单物体经复核、照片确为这件物体、尺度来自独立真实测距。**声明不是证明**，它的来源需要人工和外部测量审核。

示意 `selection.json`（其中 SHA、来源必须由真实文件产生，不可抄示例）：

```json
{
  "version": "echo-hi3d-selection-1",
  "source_scene_sha256": "sha256 from the real scene file",
  "object_mesh_sha256": "sha256 from the actual selected one-object file",
  "reference_image_sha256": "sha256 from the actual reference image",
  "selection_provenance": "Specify source recording, frame number, segmentation evidence, reviewer and selection method",
  "scale_evidence": "independent_sensor_depth_m",
  "reviewed_as_single_object": true,
  "reference_image_same_object": true
}
```

工具命令：

```sh
python tools/hi3d_candidate_intake.py prepare \
  --scene scene.ply --object one-object.ply \
  --image reference.png --selection selection.json \
  --out prepared-hi3d-candidate
```

成功时只生成 `coarse-normalized.glb`、`reference-image.png` 和 `candidate.json`，不修改原始 PLY。GLB 的粗模型归一化可以逆转回原来的米制坐标，软件把那个计算所需的大小和位置记下来；但 **Hi3D 以后生成的形状经过自己的居中、缩放，不能据此承诺每个细节保持真实物理尺寸**。必须让 #43 对生成前后与留出的实测数据分别评分。

## 后续真正 Hi3D 运行（尚未实现）

官方目前最小名义提取档位为 512³；需要 Linux/NVIDIA GPU/Python3.10/CUDA 与模型权重。先准备好刚才的输入包，再单独在GPU上执行 `demo/refine_mesh.py`，记录上游代码SHA、模型revision、GPU/CUDA/依赖、seed、时长及失败原因，生成 `refined.glb`，然后用 #43 工具对未参与细化的真实深度/点云做对照。**不能在留出的测量点上调整比例或挑好看的结果。**

本实验阶段的百分比：CPU 交接可行性预计 80%～90%；在尚无对应单物体资料与GPU时，**真实 Hi3D 精细化成功率不能凭空给出可信数字**。不等于 ECHO 整体实现进度。
