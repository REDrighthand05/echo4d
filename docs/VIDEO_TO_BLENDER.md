# 普通视频到可编辑场景

同一入口：

```bash
echo4d video-scene --video room.mp4 --output result \
  --openmvs-bin /opt/openmvs/bin
```

依赖 FFmpeg/ffprobe、现有 CPU COLMAP 接口、OpenMVS、Blender。
Python 环境另安装项目依赖和 trimesh；OpenMVS 独立程序采用 AGPL-3.0，
Blender 导出脚本随 Python 包安装，无需用户手动另找脚本路径。
不将它的许可误写为本仓库许可。安装和分发时保留对应许可/源码义务。
实跑固定 OpenMVS v2.3.0 `ce03889e2055a3d6cbcae9dd5a93b5eeea94b909`。
首次尝试2.4开发提交58117204时，Ubuntu OpenCV4.6缺少JPEGXL枚举导致编译失败；
没有伪造构建成功，改用仍支持纹理GLB的正式2.3版本，无数值算法补丁。

## 执行内容

复用 video_sfm 的原始 PTS 抽帧和 COLMAP 定位，不另造导入器。
选择有效模型、核查注册比例后，COLMAP 去畸变，OpenMVS InterfaceCOLMAP →
DensifyPointCloud → ReconstructMesh → RefineMesh → TextureMesh。
使用输入 RGB 投影生成嵌入纹理的 GLB，再用 Blender 实际导入、保存和重开。
输出 scene_textured.glb、scene.blend、camera-timeline.json、scene-report.json，
以及带3张原图/渲染对照与2张自由视角的 scene-delivery.zip。
Blender 内含录制相机动画；时间帧=1+源相对秒×30，精确时间 JSON 同时嵌入工程。
去畸变相机内参与世界位姿用于回放，允许切换自由轨道观察同一网格。

## 数据边界与失败

单目尺度 unknown，Blender 不显示米制；几何为多视图估计，纹理来自观测。
没有传感器深度、GT 位姿或库存模型的隐式兜底。关闭 OpenMVS 默认小孔填充；
所有网格仍是几何估计，尚未逐面独立证明观测支持，不能把网格完整性当成已知表面。
本功能未实现动态人物/穿戴采集。
至少80%选中帧注册才继续，避免一个注册成功的局部片段冒充全部录像；
可显式更改阈值但报告保留真实注册数。每阶段保留完整日志和错误。
只接受新输出目录，不覆盖失败档案或历史成果。

## 当前验证状态

新增本地检查覆盖未纹理化/点云伪装、GLB损坏、实际外部程序失败日志、缺依赖与覆盖保护。
单元/合成验证不等于普通视频真实重建成功。native timeline 工作流用明确标记的合成方块检查
Blender真实保存、重开、浮点源时间相机位置/朝向与 unknown 单位；不把它展示为房间。
普通真实 RGB 视频的稠密模型、视觉质量、第二场景及独立验收仍须实际执行。
既有 RGB-D 完整系统和已交付真实 Lounge .blend 保留；#74 及旧 desk FAIL 不变。
