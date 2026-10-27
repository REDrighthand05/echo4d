# COLMAP + 已配准深度 → echo4d 输入适配

> **棋盘模块标注（2026-10-27）：M02/M03。** 共享坐标/输入适配；保持单一入口，供所有满足接口的路线复用，不为每条路线复制导入器。 [路线策略](STRATEGY_CHESSBOARD.md) · [模块合同](STRATEGY_MODULES.md) · [逐格验证](STRATEGY_VALIDATION.md)。下文历史状态/数字不因该标注改变，最新证据以GitHub为准。

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

The indexed raw-depth convenience of PR #19 is now an optional, explicitly selected preprocessor. It invokes the same hardened `colmap.import_colmap` core as the standard CLI, instead of importing the conflicting PR #19 converter. See below.

The separate measured-sensor bridge `src/echo4d/metric_rgbd.py` from PR #23
still exists for the real SfM→scale→48-frame scene experiment and produces
the same `echo-dataset-1` format. It must remain protected by its own
48/12 non-leakage and sensor-scale checks before any interface consolidation
into the above general-purpose importer. This is a Draft integration change,
not approval to merge it into stable main.

## 可选的 indexed raw-depth 输入（GPT A）

**唯一标准命令仍是 `python -m echo4d import-colmap`。** 原有 `--depth` 浮点 `.npy` 米制深度方式不变；仅在指定 `--depth-index` 时，才允许显式转换原始深度。新模式同时要求 `--depth-scale`。

示例：

```bash
python -m echo4d import-colmap \
  --model ./colmap-text --images ./undistorted-rgb \
  --depth ./raw-depth --depth-index ./depth-index.txt \
  --depth-scale 1000 \
  --timestamps ./timestamps.csv \
  --meters-per-colmap-unit <INDEPENDENTLY_MEASURED_VALUE> \
  --scale-source "physical calibration measurement identifier" \
  --output ./echo-recording
```

`<INDEPENDENTLY_MEASURED_VALUE>` 是占位符，绝不能照抄测试数据；只有独立参照物或实测传感器标定的尺度才可填入，不能拿评价集 GT 的事后 Sim3 对齐尺度代替。

`depth-index.txt` 是 UTF-8 无表头的两列文本，左侧为 COLMAP 图像名称，右侧是相对于 `--depth` 目录的深度文件路径：

```text
a.png sensor/frame-0001.png
b.png sensor/frame-0003.npy
```

索引必须恰好覆盖 COLMAP 登记图像，不得重复/越界/软链接跳出目录。时间戳仍是既有 **`name,t` CSV**，不接受 PR #19 的另一个空格分隔时间戳协议。索引深度可为二维灰度 PNG 或数字 NPY，必须是已配准相机光轴 Z 深度。整数值除以 `--depth-scale` 一次；已经是米的浮点 NPY **只允许 `--depth-scale 1`**。若未提供索引，`--depth-scale` 一律报错，以避免误乘。

两个彼此独立的物理量：`--meters-per-colmap-unit` 只将 COLMAP **相机中心平移**从 SfM 单位转换为米，不改变旋转和深度；`--depth-scale` 只将索引文件中原始深度值转换为米，不改变任何位姿。

程序先预处理、验证深度并输出暂存的米制 float32 NPY，再调用 **同一 `src/echo4d/colmap.py::import_colmap`**；不会实现第二套位姿解析/坐标转换。`import-provenance.json` 额外保留索引 SHA-256、原始深度 SHA-256、输出深度 SHA-256、单位换算和源路径；整套输出采用原子目录发布，异常时不留下半成品或覆盖既有结果。

下游接口：视频定位模块需生成 COLMAP 去畸变 TEXT 位姿与逐帧准确时间 CSV；三维重建/TSDF 程序读取产出的 `recording.json`、`depth/*.npy`（米制 camera Z）、`color/*` 和 `cameraToWorld`。目前仍不自动估计视频位姿、真实尺度或密集深度；尺度来源的**真实性**仍需人工或独立实验核验。
