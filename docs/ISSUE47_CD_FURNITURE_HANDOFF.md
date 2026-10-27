# ECHO #47 C/D 单件家具交接：两个真实代码断点的修复候选

本独立 Draft PR 叠加 GPT C #45，同时引入 GPT D #46 的现有代码；**两个原分支未改动**。仅为验证接口而合并候选，未授权进入 main。

## 1. 两张不同照片必须来自真正不同的观测

原来的 #45 只要求名字不同；同一帧复制并改名字也可以为同一顶点累计两票。现在明确拒绝重复 RGB SHA、重复传感器深度 SHA 以及相同相机姿态（保守要求不同视角）；更新合成正例，原始两次不同投影视角及独立数据身份，不让纯粹文件名决定证据数量。未来需要保留独立设备时钟/采集ID，不能把此规则误说成真实性认证。

## 2. 对象几何与 ECHO TSDF 网格不得混为一谈

原 C 的 `observed-object-world.ply` 是 trimesh 输出的 xyz+faces 预览，并不是严格 ECHO 测量TSDF PLY；原 D 的 `read_echo_ply` 正确拒绝它。现在 C **额外**导出 `observed-object-world.npz`，精确包含数值型世界坐标 `vertices` 与原几何 `faces`，附独立哈希。D 新增 `read_furniture_geometry` 接受严格 ECHO PLY 或此几何-only NPZ，严格禁止 pickle、NaN、越界面索引、多余字段。原 TSDF 读取接口**不放宽**，无法把 AI 外形冒充原始实测网格。

## 3. 真实证据状态

测试夹具是人工平面，不是实际家具。#45/#46 已验证的原始场景、原TUM照片及深度哈希均未更改，仍未完成真正的家具身份审核、独立物体尺寸测量、生成视图/Hunyuan/Hi3D GPU 路线或物体准确放置。真实 C→D 的世界坐标系身份和外参标定还需要在后续资料验收中独立锁定；同名 world 也不能取代物理转换证据。

**成功条件：** 在专用 GitHub Actions 中跑 `test_real_furniture_extract.py`、`test_hi3d_furniture_evidence.py`、`test_hi3d_cd_handoff.py`，使 C 实际写出的 NPZ 能由 D 原样读回，且原 `read_echo_ply` 不误接受预览 PLY；检查改名/复制、无效 NPZ 均拒绝。失败或跳过必须如实保留。

## 2026-10-27 接收边界与小范围安全补修
唯一正式候选交换方式为 geometry-only NPZ（vertices/faces），预览PLY不注入占位颜色或法线，也不放松实测TSDF的严格PLY契约。D与Hi3D CPU工具共用有界数字NPZ读取器：所有ZIP成员、展开大小、NPY形状/dtype在任何数组分配前验证。

D的冻结单向观测指标、尺寸误差与阈值保留；它不能验收无支持表面，故输出 `OBSERVED_METRICS_PASS_GEOMETRY_INCONCLUSIVE` 或 `FAIL_OR_INCONCLUSIVE`，总体几何状态始终INCONCLUSIVE、CLI返回3。此修复没有追溯改变历史数值，也没有新增几何质量门槛。真实物体身份、场景世界系校准、独立尺寸/12份对象掩码以及GPU推理仍由#42/#47跟踪；现有真实提取结果仍为0件、BLOCKED。
