# WO-plane-fit-lsq: plane_fit_lsq
纯函数，numpy。输入 points (N,3) float64，N>=3。
总最小二乘平面拟合（显式目标：最小化各点到平面有符号距离的平方和）：质心 + 协方差最小奇异向量作法向。
平面约定：normal·x + offset = 0，normal 为单位向量；有符号距离 distances[i] = normal·points[i] + offset。
符号约定（确定性）：normal 中绝对值最大的分量为正（并列取靠前分量）。
返回 (normal (3,) float64, offset float, distances (N,) float64)。N<3 或形状非法抛 ValueError。确定性。
台架：无噪声已知平面点 → 法向/offset 精确、距离≈0；固定种子噪声 σ=0.01 → 法向夹角 <1°。
注：与既有 fit_plane（只回法向+d+rms）的区别在本单输出每点有符号距离，供下游残差门限直接消费。
