# COLMAP TEXT 导入（外部位姿/深度）

`echo4d import-colmap` 将 **COLMAP TEXT 位姿** + **外部米制 Z 深度** 转换为 `echo-dataset-1`。

## 输入要求

- `--colmap-text` 目录必须包含 `cameras.txt` 与 `images.txt`。
- 当前仅接受单相机 `PINHOLE` / `SIMPLE_PINHOLE`。含畸变模型需先去畸变并导出针孔模型。
- `--timestamps`：两列 `image_name timestamp_seconds`，时间非负。
- `--depth-index`：两列 `image_name depth_path`，深度路径必须位于 `dataset` 根目录内。
- `--meters-per-colmap-unit`：COLMAP 相机中心缩放到米制的数值因子（米/Colmap单位），必须为正数。
- `--scale-evidence` 必填：记录米制尺度来自何处（例如独立传感器深度锚点）。

## 位姿与深度约定

- COLMAP `images.txt` 的 `QW QX QY QZ TX TY TZ` 是 `world->camera`。
- 导出时转换为 echo4d 需要的 `cameraToWorld`：
  - `R_cw = R_wc.T`
  - `C_world = -R_wc.T @ t_wc`
- 导出相机中心会再乘 `--meters-per-colmap-unit`，用于把 SfM 单位映射到米制；该参数与深度换算分离。
- 深度必须是相机光轴 Z 深度（非 ray distance）。
- 深度值会先除以 `--depth-scale`，结果必须在 `[0,100]` 米，0 允许表示缺失。

## 命令示例

```bash
echo4d import-colmap datasets/my-seq \
  --colmap-text datasets/my-seq/colmap_text \
  --timestamps timestamps.txt \
  --depth-index depth-index.txt \
  --meters-per-colmap-unit 0.42 \
  --depth-scale 1000 \
  --scale-evidence "3 RGB-D sensor anchor frames" \
  --output outputs/colmap-import
```

输出目录包含：

- `recording.json`（`echo-dataset-1`）
- `import-provenance.json`（哈希、尺度依据、源文件说明）
- `depth/*.npy`、`color/*.png`
