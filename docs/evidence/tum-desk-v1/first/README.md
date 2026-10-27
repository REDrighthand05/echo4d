# ECHO4D 跨第二场景测试：第一次实测（未事先固定来源 SHA）

来自 TUM `freiburg1_desk` 办公室公开实拍 RGB，取75张，COLMAP 单目恢复相机相对轨迹。

- [研究 Issue #20](https://github.com/REDrighthand05/echo4d/issues/20)
- [Draft PR #21](https://github.com/REDrighthand05/echo4d/pull/21)
- [第一次真机 Actions #40203647194](https://github.com/REDrighthand05/echo4d/actions/runs/40203647194)
- 官方数据首次观测 SHA256：`e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`，字节数 344011403。**首次不是事前钉死**；只有下一轮重新下载得到同 SHA 才是二次一致性检验。
- 75/75 帧注册；自动稀疏三维点 7015 个；事后使用同一组75帧 TUM GT 最优对齐后的相机位置 RMSE 0.02599613 米，朝向中位数 3.10313 度。**仅比较路径形状，不能宣布实际米制尺度或完整场景精度。**
- 公开原始真实图像先重编码为测试 MP4；不是原生手机视频。COLMAP 只使用 RGB 与已知标定相机内参；TUM GT 在SfM之后读取。

## 长期机器文件
- `report.json`：真实下载、所有源图哈希、外部命令、成功/失败、75帧评价及限制
- `trajectory-scale-unknown.json`：COLMAP 真正估计的75个相机位置，单位未定
- `registered_timestamps.csv` / `cameras.txt`：注册画面顺序和实际相机内参
- `SUMMARY.md`：机器原始执行摘要
- `zip-file-checksums.json`：Actions ZIP 中所有实际文件的 SHA-256；体积较大的 `images.txt` 与 `points3D.txt` 暂留在 Actions Artifact #12603250858，可能过期

以下Git证据是源 ZIP 按**原始字节**写入，不经过文件解析重排。完整成绩和审计结论也写入 [Issue #20 首次评论](https://github.com/REDrighthand05/echo4d/issues/20#issuecomment-6399091144)。
