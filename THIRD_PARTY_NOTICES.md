# 第三方出处

`dist/samples/tum-xyz.echo.json` 和 `docs/validation/tum-report.json` 派生自 **TUM RGB-D Benchmark / Freiburg 1 xyz**。

作者：Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, Daniel Cremers。

论文：*A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012。

- https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
- 原始序列：https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
- 论文及许可说明：https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf
- 许可：[Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/)

修改：选 60 帧，各轴降采样 4 倍，已知位姿按时间插值，Z-up 转 Y-up，TSDF 融合；网格附 20 帧 80×60 预览。派生内容保留上述 CC BY 3.0 署名，不暗示原作者认可本项目。完整原始数据未打包，没有用这组结果宣称独立表面精度。

融合方法参考 Curless and Levoy, *A Volumetric Method for Building Complex Models from Range Images*, SIGGRAPH 1996：https://graphics.stanford.edu/papers/volrange/ 。融合代码为新写实现；网格提取调用 scikit-image 的 marching_cubes。

NumPy、SciPy、Pillow、scikit-image 及传递依赖遵循各自安装包附带许可证，固定版本在 requirements.lock。Open3D、MediaPipe 仅为文档参考，没有集成或打包。界面不加载第三方脚本、字体或模型权重。
