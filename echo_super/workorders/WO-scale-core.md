# WO-scale-core: estimate_scale
纯函数，numpy。输入 object_points (N,3) float64 = 画面中某个已知尺寸物体的三维点（场景单位）；real_size_m = 该物体真实最长边（米）。
用 PCA 求主轴，点投影到主轴后取 1%~99% 分位跨度为 measured_size，scale = real_size_m / measured_size。
返回 (scale: float, measured_size: float)。N<2 或跨度<=0 抛 ValueError。确定性，无随机。
