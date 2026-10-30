# Exe 打包准备笔记（2026-10-31，本地准备阶段）

## 模型文件与 git / CI 的分工

- `models/*.onnx`（small 95MB + base 371MB，约 466MB）**不进 git**，已在
  `.gitignore` 中排除。仓库只存代码、查看器和配置。
- CI 出 exe 时由 workflow 的 “Fetch depth model” 步骤从 Hugging Face
  现场下载，与 `echo-scene.spec` 的 `datas` 保持一致。
- **当前只下 small**：打包代码路径 `echo_scene/pipeline.py` 的
  `stage_depth` 写死加载 `models/depth_anything_v2_small.onnx`，
  spec 也只打包 small，两边一致。base 模型目前只被服务器端实验脚本
  用过，没有进主线。若以后主线换 base，必须同时改 spec 的 datas 和
  workflow 的下载步骤，且安装包体积约 +371MB。

## 依赖核对（已完成）

- `requirements-app.txt` 与 `app.py` + `echo_scene/*.py` 的实际 import
  逐项对过：pycolmap、onnxruntime、numpy、Pillow 齐全，其余全是标准库
  （tkinter 随 setup-python 的官方 Python 自带）。
- workflow 的 pip 安装行已补 `huggingface_hub`（模型下载步骤要用，
  原先缺失会让 CI 在下载模型那步直接失败）。

## 还差哪几步才能出 exe

| # | 步骤 | 谁做 | 预计耗时 |
|---|------|------|----------|
| 1 | 代码进 echo4d 的 product/echo-scene 分支（用户 2026-10-31 拍板用分支、不另建仓）；分支已建好 | 已完成 | — |
| 2 | 把本地文件传上分支（整包推送被应用权限挡住，改逐个文件传） | Muse | 逐个确认中 |
| 3 | 触发 Actions（push 到 main 自动触发，或手动 workflow_dispatch），等 Windows 构建：装依赖 + 下模型 + PyInstaller 打包 | Muse 触发并盯结果 | ~15–40 分钟 |
| 4 | 下载 Actions 产物 `EchoScene-windows.zip`（预估数百 MB），解压运行 `EchoScene.exe` 做真机冒烟 | 用户（只有用户手里有 Windows 真机） | ~10 分钟 |

## 已知风险点

- **pycolmap**：原生 DLL 较多，`collect_all` 一般能收全，但
  PyInstaller 漏收动态加载的 DLL 是经典故障，表现为启动或算位姿时
  “DLL load failed”。若首建中招，补法是把 pycolmap 包内 DLL 显式
  列进 spec 的 binaries。
- **onnxruntime**：风险更隐蔽——exe 能正常打开、界面无异常，但跑到
  第一次深度推理时 `InferenceSession` 才失败（原生库路径被打平）。
  冒烟必须真跑一段短视频走通全链，不能只看窗口能打开。
- **ffmpeg（额外发现，非上述两库）**：`stage_extract` 是以裸命令
  `ffmpeg` 调外部程序的，exe 目前**没有打包 ffmpeg**。用户机器上
  若没装 ffmpeg，第一步抽帧就会失败。出正式版前要么把 ffmpeg
  二进制作为 datas 打进包并改成相对路径调用，要么在说明里写明
  前置依赖——建议前者，发版前处理。

## 不进安装包的东西

`echo_super/`、`experiments/`、`tests/` 是开发侧资产，会进仓库，
但 spec 只分析 `app.py`，不会被打进 exe。
