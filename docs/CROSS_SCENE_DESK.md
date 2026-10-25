# ECHO 在第二个真实场景上的定位测试

## 为什么要换一组画面？

前一轮验证用的是 TUM 的 `freiburg1_xyz`：相机面对办公桌，移动方向相对简单。现在固定选择官方 `freiburg1_desk`：围绕四张办公桌多次转动/走动的不同录制。它不是同一段录像重采样。

- TUM 官方数据目录：https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- 实际下载地址：https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
- 代码评审与实验 Issue：[Issue #20](https://github.com/REDrighthand05/echo4d/issues/20)

## 怎样检测？

仍然从真实彩色图像每隔8张取一张，固定前75张（本序列约613帧，凑足90张需要至少713帧，因此在第一次运行**之前**确认75张）；用 FFmpeg 打包为4fps的**测试视频**。视频编码发生在真实实拍画面之后，**不是原生手机MP4，也不保留原TUM拍摄时间间隔**。运行与之前完全相同的 RGB-only COLMAP 程序，只告诉它真实的相机内参。相机的参考路径仅在计算结束后读取，用于检查。

GitHub 工作流 `.github/workflows/tum-cross-scene-desk.yml` 真实运行 CPU COLMAP；机器报告 `report.json`、原始相机轨迹文件和失败日志上传为实验产物。必须记录成功注册的**所有**图像数，不能只挑成功画面计算注册率。

### 数据校验

首次 GitHub Actions #40203647194 真正观察到 **344,011,403 字节**和 SHA-256 `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`。此值已在下一次运行之前写入程序常量；**第二次运行必须从官方HTTPS重新下载且字节与SHA同时匹配**。这是自主记录的可复现性校验，不是TUM颁发的签名，也不能把第一次下载称作提前验证。

### 这些数字能说明什么？

75张画面的成功定位数，说明计算机有多大程度能从这段视频推断自己怎样移动；和数据集提供的实际拍摄路线事后对比的误差，说明轨迹形状是否接近。这是同一组相机轨迹上同时匹配与评分的“最佳对齐”，**不能代表视频自己恢复了真实米制尺度**，更不是三维房间模型或人物动作精度。

### 重现命令

```bash
python tools/tum_desk_sfm.py --work /tmp/desk-work --report-dir /tmp/desk-result
```

仓库安装要求见 `docs/VIDEO_SFM.md`。必须先有 Python、COLMAP、FFmpeg；输出工作目录需不存在，避免覆盖旧结果。若失败，先打开 `report.json`、`nested-sfm-failure.json` 和 GitHub Actions 日志，不改变既定场景/内参挑更好看的数字。

## 首次实测（未提前固定源哈希）

[Run #40203647194](https://github.com/REDrighthand05/echo4d/actions/runs/40203647194)：**75/75** 张真实RGB成功注册，COLMAP生成 **7,015** 个稀疏点；与TUM真实相机路线事后最佳Sim(3)对齐后，位置ATE RMSE **2.600厘米**，朝向误差中位数 **3.103°**。这些是相机运动评价，不是整个房间的真实几何精度。
