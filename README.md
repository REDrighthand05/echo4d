# 回声 ECHO / echo4d

**标定 RGB-D → 带观测证据的三维表面 → 时间回放。**

包含可独立安装的 Python 重建内核、命令行工具和浏览器检查台。原生内核从实际深度数据计算网格，不依赖网页、预制房间或远程 AI API。浏览器另有一个简化基线，用于检查稀疏采样、噪声和定位漂移的影响。

版本 **0.1.0 工程验证版**。已用 TUM 实拍数据跑通并复现。它不是经过设备集成、长期运行和独立精度验收的完整生产系统；测试通过不代表已经实现“人生重建”。

## 安装与运行

建议 Python 3.12；声明支持 Python 3.11–3.13。初次安装需要网络，重建本身不需要联网、显卡、Node 或浏览器。

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell：.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .

echo4d demo --output outputs/demo
echo4d validate outputs/demo/recording.json
echo4d reconstruct outputs/demo/recording.json --output outputs/result --bounds -1.8 -.2 -1.9 1.8 2.2 1.8 --voxel .04
```

结果目录必须不存在；重跑请选择新目录。程序不会清空已有结果。`demo` 是解析盒体生成的合成传感器数据，不是真实采集。

## 静态场景整合候选

本分支收敛了视频定位、传感器定尺度、实测深度融合和留出误差检查。它仍为工程验证版：**生成表面需要已配准的实测深度；普通 RGB 视频本身不提供完整米制表面。**

已有去畸变 COLMAP TEXT 相机模型、逐图时间戳 CSV、同名 RGB 图片和浮点米制 NPY 深度时，统一导入入口为：

```bash
echo4d import-colmap --model datasets/model --images datasets/rgb --depth datasets/depth --timestamps datasets/times.csv --meters-per-colmap-unit 0.05 --scale-source "独立实测尺度说明" --output outputs/imported
echo4d validate outputs/imported/recording.json
```

`0.05` 只是格式示例，必须换成实际测得的比例。这个比例只换算相机位置；已经是米的深度不会再次缩放。导入后按实际场景确定 `reconstruct --bounds`。输入格式见 [COLMAP 导入](docs/COLMAP_IMPORT.md)。#19 的 depth-index/整数 PNG 便利输入尚未迁入这个入口。

公开 TUM 端到端对照（会下载约0.42GB数据，另需安装 CPU COLMAP 和 FFmpeg）：

```bash
python tools/tum_static_ab.py --work outputs/tum-ab-work --report-dir outputs/tum-ab-report
```

程序分别用参考相机位置和软件估计位置重建，再用相同12张未融合深度检查。查看 `ab-comparison.json`、`ab-run-report.json` 和两组 `mesh.ply`；覆盖率与覆盖区域内误差需同时阅读。[复现协议](docs/STATIC_FAIR_AB.md)和[整合清单](docs/INTEGRATION_20261026.md)列出来源、固定数据划分和剩余限制。

打开检查台：

```bash
python -m http.server 8080 --bind 127.0.0.1 --directory dist
```

访问 `http://localhost:8080`，导入 `outputs/result/scene.echo.json`。不要双击 HTML：模块和 Worker 需要 HTTP。也可选择内置 TUM 实拍重建。默认演示骨架由程序生成，缺失段插值明确标记，可关闭。

## 目录

| 模块 | 实现 |
| --- | --- |
| `src/echo4d/fusion.py` | 有界 CPU TSDF、观测支持过滤、Marching Cubes |
| `src/echo4d/model.py` | 输入验证、逐帧读取、路径限制、输入 SHA-256 |
| `src/echo4d/tum.py` | TUM 时间关联、SLERP 旋转插值、标定保持降采样 |
| `src/echo4d/output.py` | 二进制 PLY、证据数组、场景导出、失败回滚 |
| `src/echo4d/evaluation.py` | 与独立、已对齐参考点集比较 |
| `src/echo4d/colmap.py` | 唯一 COLMAP 导入核心、数值尺度与严格输入校验 |
| `src/echo4d/video_sfm.py` | FFmpeg / CPU COLMAP 视频定位，输出未知尺度轨迹 |
| `src/echo4d/metric_anchor.py`、`metric_rgbd.py` | 实测深度定尺度与训练深度桥接 |
| `src/echo4d/metric_holdout.py` | 统一三角形投影与未融合深度评分 |
| `dist/` | 无第三方运行时依赖的浏览器检查台；WebGL / Canvas |
| `tests/` | 数值、边界、命令行端到端、跨语言数据契约测试 |
| `tools/verify.py` | 自动测试及 Python → JavaScript 场景导出验收 |
| `.github/workflows/ci.yml` | Linux / Windows 验证与 wheel 打包 |

## 实拍数据复现

下载并解压 [TUM Freiburg 1 / xyz](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)，原始文件约 0.42 GB。源码不包含完整原始数据。

```bash
echo4d import-tum datasets/rgbd_dataset_freiburg1_xyz --output outputs/tum-input --intrinsics 525 525 319.5 239.5 --stride 12 --limit 60 --downsample 4 --world-up z
echo4d reconstruct outputs/tum-input/recording.json --output outputs/tum-result --bounds -2.45 -.25 -2.25 1.15 1.45 1.65 --voxel .035 --max-depth 4 --min-observations 2
```

使用 TUM 提供的已知真值轨迹，不是 SLAM 定位评估；内参为官方格式页列出的 ROS 默认值。深度除以 5000 换算为米，RGB 已配准。`--world-up z` 将输入 Z-up 旋转为显示 Y-up，保持右手坐标。其他设备不要照抄标定和坐标参数。

本次结果：**60 帧 → 10,810 顶点 / 18,497 三角形**；零面积面 0、非流形边 0、边界边 3,007。存在孔洞，不是封闭可打印模型。PLY SHA-256：`8380f4cf2e3257ab58217cbd96322870794c94022c6293200a1750c33740bffa`。完整报告在 `docs/validation/tum-report.json`。

这里没有独立场景表面真值，所以不报告真实几何厘米级精度。若有独立、坐标对齐的 Nx3 数值 NPY 点集，使用 `--reference reference.npy --threshold .05` 可得到双向最近点 RMSE、precision、coverage 和 F-score；不会自动对齐或以输入深度冒充真值。

## 输出与输入

- `mesh.ply`：米制二进制网格，包含法线和 RGB。
- `vertices.npy`：顶点；`evidence.npz`：近邻体素观测次数及首末支持帧。
- `scene.echo.json`：完整网格和降采样深度预览；超过浏览器限制时用 `--no-scene`。
- `report.json`：配置、输入哈希、逐帧更新数、网格诊断、输出哈希。
- `--save-volume`：另存 `volume.npz` 计算体积；本版没有恢复运行接口。

原生输入是 `echo-dataset-1` 清单与 PNG/NPY 深度、可选 RGB/排除掩码；浏览器接受小型内联 `echo-1` 或原生导出的 `echo-scene-1`。必须提供内参和每帧 camera-to-world 位姿。详见 [数据契约](docs/FORMAT.md)。

## 验证与打包

测试还需要 Node.js 22+：

```bash
python tools/verify.py
python -m pip wheel --no-deps . --wheel-dir artifacts
python tools/package_source.py --output artifacts/echo4d-source.zip
```

测试包括尺度、旋转、平移、零深度、动态排除、噪声平均、故意位姿误差、未知区域边界、异常数据、资源限制、文件回滚和端到端哈希。详见 [验证记录](docs/VALIDATION.md)。

可选 `docker build -t echo4d .`；容器以 UID 10001 运行，将数据目录挂载至 `/data` 并确保可写。Dockerfile 已提供，交付环境未执行容器构建。

## 边界

重建内核需要逐帧位姿和米制深度；实验工具可从视频估计位姿，再借助实测深度定尺度。当前重建静态表面，动态物体需要外部排除掩码。没有实时 SLAM、自动人体识别、身份跟踪、可穿戴驱动或穿墙感知。人物关节是世界坐标输入，插值不构成历史事实。

默认最多 400 万体素，数组约 128 MB，提取网格还需要额外内存。观测次数不是校准置信概率。模型不补洞、不保证无自交或可打印。面向部署的限制与验收要求见 [运行说明](docs/OPERATIONS.md)。

代码采用 MIT；TUM 派生样例按 CC BY 3.0 署名，见 [第三方出处](THIRD_PARTY_NOTICES.md)。
