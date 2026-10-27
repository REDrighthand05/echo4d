# 数据契约

> **棋盘模块标注（2026-10-27）：M00/M01/M02/M12。** 跨路线数据底座，米制camera-Z与位姿合同保持；新策略先适配，不放宽现有读取器。 [路线策略](STRATEGY_CHESSBOARD.md) · [模块合同](STRATEGY_MODULES.md) · [逐格验证](STRATEGY_VALIDATION.md)。下文历史状态/数字不因该标注改变，最新证据以GitHub为准。

长度为米，时间为秒。深度是相机 **Z 轴方向距离**，不是射线长度。相机 X 向右、Y 向下、Z 向前。显示世界 Y 向上；其他坐标定义须显式转换。

## 原生 echo-dataset-1

```json
{
  "version": "echo-dataset-1", "name": "my-room", "source": "recorded",
  "units": "meters", "duration": 1, "depth_scale": 1000,
  "intrinsics": {"width": 640, "height": 480, "fx": 525, "fy": 525, "cx": 319.5, "cy": 239.5},
  "frames": [{
    "t": 0,
    "cameraToWorld": [1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,1.5,0,1],
    "depth": "depth/00000.png", "color": "color/00000.png", "mask": "mask/00000.png"
  }]
}
```

`color`、`mask` 可省略。路径必须指向清单目录内部；拒绝绝对路径、上级逃逸和指向外部的符号链接。数据目录运行时应保持不可变。

深度为单通道 PNG 或真实数值 NPY，除以 `depth_scale` 后为米，0 表示缺失，范围 [0,100] 米。默认融合范围 [0.15,8] 米，上限可用 `--max-depth` 调整。NPY 禁用 pickle。颜色必须为与深度配准、等尺寸的 uint8 RGB；不会自动配准。掩码非零表示排除。

`cameraToWorld` 为 **列优先** 16 数组，`P_world = R * P_camera + translation`。Python 用 `matrix.flatten(order="F").tolist()`。旋转须正交、行列式接近 +1；不允许尺度或镜像。平移和关节绝对值不超过 10 km。

时间非负、严格递增，duration 覆盖所有采样且不超过 24 小时。清单最大 40 MiB、100,000 帧。每帧宽高 2–4096、总计不超过 400 万像素。逐帧读取，不一次加载全序列。所有帧共用内参；变焦序列请分段。

## 浏览器 echo-1

`depth` 改为宽×高行优先数值数组，单位已经是米；可选 `colors` 为宽×高×3 RGB 整数，可选 `excluded` 为宽×高布尔数组。`depth_scale` 省略或为 1。最多 640×480、300 帧、合计 150 万深度像素，文件不超过 40 MiB。

`echo4d preview recording.json --output outputs/preview` 生成紧凑预览，保留抽取像素的标定。它不替代原始数据，不用于严肃精度验收。

## 人物轨迹

两个格式都可含 `entities`，原生融合只验证并转存，不识别人。示例元素：

```json
{"id":"P-01","jointNames":["hip","head"],"bones":[[0,1]],"samples":[
  {"t":0,"joints":[[0,1,0],[0,1.7,0]]},
  {"t":1,"joints":[[1,1,0],[1,1.7,0]]}
]}
```

关节须处于场景世界坐标，不是髋部相对坐标。最多 8 条轨迹、2–40 个唯一关节名、80 条连线、10,000 个采样。每个采样提供全部关节；不要用零向量代替缺失关节。

回放使用关节线性插值；间隔 >0.65 秒标记缺失，允许的缺失插值最长 4.1 秒，开关可关闭，端点外不外推。这是展示策略，不是动作识别。

## 原生结果 echo-scene-1

包含 algorithm=`projective-tsdf-v1`、recording 预览、mesh、provenance。网格含 vertices、faces、colors、observations、first、last。证据时间来自原始融合帧，不能按预览帧索引解释。上限 18 万顶点、30 万三角形、40 MiB。

完整 TSDF 网格不会随时间倒退；此模式禁用浏览器采集参数和“仅过去帧”重建。修改原生参数须重新运行 CLI。`source` 仅是作者声明，SHA-256 仅证明字节一致，不是设备身份或内容真实性签名。
