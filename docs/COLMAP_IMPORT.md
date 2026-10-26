# COLMAP + 已配准深度 → echo4d 输入适配

**能力边界**：只转换已经由其他工具计算好的相机位姿和已对齐、真实米制的 Z 深度；不具备从普通视频自动估计位姿、深度或尺度的功能。这是普通视频重建管线的一个独立、可测量的接口层。

## 输入要求

- COLMAP TEXT 模型文件 cameras.txt 与 images.txt。图像须去畸变，并且模型相机是 PINHOLE 或 SIMPLE_PINHOLE。当前仅接受一个相机 ID。
- 已去畸变 RGB 图像文件夹，与 images.txt 的名称和分辨率一致；支持 PNG/JPEG。
- 深度文件夹：相同相对路径但后缀为 .npy 的浮点数组，尺寸与相机一致，值代表相机 Z 轴的距离，单位为米，0 代表无数据。**不是整数毫米深度，也不是沿射线距离或未经尺度校准的 AI 相对深度。**
- timestamps.csv：表头严格为 name,t，每个文件名恰有一个时间戳（秒）；不按字典序推测事件发生时间。
- 独立获得的物理尺度：通过参照物、实际距离或传感器等取得 meters-per-colmap-unit，并用 scale-source 记录其真实依据。**仅填写 1 并不证明 COLMAP 已经有米制尺度。**

## 运行示例

下面的 0.82 仅用于展示参数结构，绝不能直接用于未知场景。先安装仓库 README 的 Python 依赖。

```bash
python -m echo4d.colmap --model datasets/model-text --images datasets/undistorted --depth datasets/depth-meters --timestamps datasets/timestamps.csv --meters-per-colmap-unit 0.82 --scale-source "independent measured reference" --output outputs/colmap-import
echo4d validate outputs/colmap-import/recording.json
```

结果是 echo-dataset-1 格式的 recording.json、import-provenance.json、RGB 和深度数组拷贝。可继续运行仓库现有的 reconstruct（需要设置合理的 bounds）。输出目录必须尚不存在，失败不会覆盖旧结果。

## 几何理由

COLMAP 的 qvec / tvec 为 world-to-camera 刚体；echo4d 需要 camera-to-world。若 COLMAP 为 x_cam = R_wc x_world + t_wc，则 R_cw = R_wcᵀ，相机中心 C = -R_wcᵀ t_wc；C 还需乘以独立确定的物理尺度。**直接把平移变成 -t_wc 是错误的。**

现有 echo-dataset-1 只允许每段记录一组内参；混用多个镜头/不同分辨率、未经校正的径向畸变会被拒绝。世界竖直方向未自动对齐重力。记录 metadata 中区分 COLMAP 估计位姿、独立尺度依据、外部深度来源；它们不是由本适配器推理获得。

## 验收与未完成

- tests/python/test_colmap.py 用可算出真实答案的 90° 旋转 + 平移 + 尺度测试，确认输出真值；还测试缺少尺度来源、深度单位错误、畸变相机、错误时间戳、路径逃逸、未知相机及不覆盖原有目录。
- 这只是单元测试；必须再用 TUM/ETH3D 等真实序列做独立端到端测试才能声称真实环境可靠。
- 本阶段不支持普通视频自动抽帧/时间读取、COLMAP 运行管理、AI 深度估计、真实世界尺度自动恢复、多镜头、重力对齐、动态物体遮罩。

总档案：https://github.com/REDrighthand05/echo4d/issues/1
上游开发任务：https://github.com/REDrighthand05/echo4d/issues/2


## Unified top-level ECHO command (staged for maintainer review)

The canonical adapter is the strict `src/echo4d/colmap.py` implementation
from PR #6 **with** PR #24's missing-POINTS2D-row and disguised-NPZ rejection
fixes. The ordinary `echo4d` command now exposes the **same adapter**:

```bash
python -m echo4d import-colmap \
  --model ./colmap-text --images ./undistorted-rgb \
  --depth ./registered-camera-z-npy \
  --timestamps ./timestamps.csv \
  --meters-per-colmap-unit 0.0509981667 \
  --scale-source "three independent measured camera-Z depth anchor frames" \
  --output ./echo-recording
```

The numerical value above is **only a demonstration from a specific documented
TUM experiment**, not a scale to copy into another video. A new recording
must have its own actual, independently measured scale.

The `--timestamps` file has the precise CSV header `name,t` and one
registered image name per row. The `--depth` directory must contain a
matching, already-registered float32/float64 `.npy` camera-Z depth map for
every COLMAP image, expressed in **metres**. For each camera transform
COLMAP world-to-camera is inverted and its translation multiplied by
`--meters-per-colmap-unit` exactly once; **rotation matrices and the
already-metric sensor Z values are not multiplied**.

Draft PR #19 uses a different two-column depth-index format, extra flags, and
a competing implementation of the same `colmap.py`. This staging PR does
NOT silently accept those incompatible arguments. Indexed-depth convenience
is future explicitly tested migration, not a second hidden input contract.

The separate measured-sensor bridge `src/echo4d/metric_rgbd.py` from PR #23
still exists for the real SfM→scale→48-frame scene experiment and produces
the same `echo-dataset-1` format. It must remain protected by its own
48/12 non-leakage and sensor-scale checks before any interface consolidation
into the above general-purpose importer. This is a Draft integration change,
not approval to merge it into stable main.
