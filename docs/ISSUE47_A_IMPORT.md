# Issue #47 / GPT A — 唯一 indexed 深度入口与真实来源往返

基线：main 已合入 #34；数据输入候选采用 #39（`8bb15b646ce6fadfb2bd54b85655901e26dd99f9`），#35 作为补充安全检查来源。**不同时将 #35 的 `import-colmap-indexed` 和 #39 的 `import-colmap --depth-index` 宣称为标准入口**，也不增写第二套 COLMAP 姿态解析器。

## 选型

| 项目 | #39 选择 | #35 参考 |
| --- | --- | --- |
| 公共命令 | 既有 `echo4d import-colmap` 的可选 `--depth-index` | 新 `import-colmap-indexed`（不采纳并行入口） |
| 深度索引 | 既有 `image_name relative_depth_path` 两列严格文本 | `name,depth_path` CSV（不同时接入） |
| 物理深度 | 原始 PNG / NPY camera-Z 除以显式 `--depth-scale` 一次；浮点米制 NPY 只允许 1 | 值得保留的 raw-unit 显式来源记录和安全负例 |
| COLMAP | 唯一 `src/echo4d/colmap.py::import_colmap` | 复用同一核心 |
| 源记录 | source RGB / 原始 depth / 转换后 metric NPY SHA-256、时间戳、单位比例 | 原始单位名称、回滚与同源错误 |
| 输出 | 原子事务，无覆盖；新增规范路径去重及可选单位名称 | 原子事务，无覆盖 |

一个原始测深文件不能通过不同别名路径假装两帧。该问题单独通过回归负例约束。新增 `--depth-unit` **仅**把用户声明的单位文字写入 `import-provenance.json.indexed_depth.raw_depth_unit_label`，并标记 `raw_unit_label_verified_independently=false`；它不是系统证明过的单位。数值换算仍完全依赖显式 `--depth-scale`。

相机由核心从 COLMAP world-to-camera 求逆为 camera-to-world，**只**将相机中心乘以 `--meters-per-colmap-unit`；旋转不得跟着缩放，深度不会受到这个参数影响。可选 raw-unit 只是转换传感器 camera-Z 数据，**不**提供单目视频自动米制尺度。

## 官方真实数据回归（与受控测试区分）

`tools/tum_indexed_depth_roundtrip.py` 基于已有 `tools/tum_public_roundtrip.py`：
1. 下载官方 TUM RGB-D Freiburg1 XYZ（448204271 字节，固定 SHA256 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`）；或接受严格核验 SHA 的离线归档。安全解压保留原始 RGB 与 16-bit PNG。
2. 采用固定 `stride=12, limit=12, downsample=1, depth=5000 raw units/m`，使用 **TUM 自带的 known pose** 生成 12 帧 ECHO 米制 baseline，再经 `known_pose_colmap` 将这些已知姿态格式化为 COLMAP TEXT。这里**没有执行 RGB SfM 或真值尺度恢复**。
3. 读取 baseline 每帧原始 `association.json`，把完全相同的原始 TUM depth PNG 按 `index.txt` 指向 #39 的唯一可选 indexed 入口，用原始颜色、时间戳、已知 COLMAP TEXT、`--depth-scale 5000`、`--meters-per-colmap-unit 1` 导入。
4. 与 baseline 验证像素尺寸/相机内参、12 帧相机矩阵、时间、RGB 原像素、米制深度偏差（≤`1e-6 m` 浮点量化界）及 **转换后数字乘5000四舍五入必须逐像素恢复原始整数 PNG**。每帧写原始 PNG SHA、度量 NPY SHA、有效传感器像素数和最大差值。
5. 在同一真实来源上给错误 `--depth-scale 1` 和缺少一行 depth-index 分别运行**失败路径**，均要求没有 published 目录，且不能意外向 stdout 写成功 JSON。
6. 始终保留 `report.json`、`SUMMARY.md` 和失败阶段日志。真实数据不提交 Git，以保留其来源和许可证边界。

复现：

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest tests/python/test_colmap_indexed.py tests/python/test_colmap.py tests/python/test_colmap_cli.py -q
python tools/tum_indexed_depth_roundtrip.py \
  --work /tmp/issue47-a-tum-work \
  --report-dir artifacts/issue47-a-indexed
```

`--work` 须为不存在的新目录。有已核验的官方 TGZ 时可加 `--archive /path/to/source.tgz`。官方数据大小约 448MB 压缩包，需合法公开网络、磁盘资源与下载许可；GitHub Actions 工作流 `.github/workflows/issue47-a-indexed-real-tum.yml` 在 Ubuntu 跑真实数据，而默认 `verify` 工作流独立检查 Linux/Windows × Python 3.12/3.13 的四平台软件契约。

## 当前验收与不宣称

- 既有 #39 [verify 40241639346](https://github.com/REDrighthand05/echo4d/actions/runs/40241639346) 四平台通过；[原 TUM roundtrip 40241639294](https://github.com/REDrighthand05/echo4d/actions/runs/40241639294) 通过，但当时**未**对新增 indexed 原始 PNG 分支跑官方实际源。
- 本分支的真实 indexed 工作流与增强 4 平台测试必须**另行通过**后才能记录为已验收。CI 未出结果时仅属候选。
- 测试的 GT 位姿专门用于格式转换参照，而非运行 COLMAP 成像定位。我们没有在本任务自动测量独立尺、估计 RGB-only 深度、证明房间表面绝对精度或验证真实设备深度与 RGB 对齐质量。
- 不修改 #37/#40 的视频时间、#45/#46 的生成/测量工作；不写 main，需主干看护审查后方可合流。参考 Issue #47、#34、#35、#39。

**合流条件：** 唯一入口、四平台软件测试 + 真实官方 raw-PNG indexed 数值报告、输入身份与失败记录、公开数据许可证、下游 `recording.json` 兼容审核全部确认，再由维护者决定是否合并 #39 和本候选、保留 #35 历史出处。