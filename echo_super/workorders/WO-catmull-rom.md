# WO-catmull-rom: catmull_rom_path
纯函数，numpy。输入 knots (N,3) float64 控制点序列，samples_per_segment int>=1。
向心 Catmull-Rom 样条（Barry–Goldman 递推，alpha=0.5）穿过全部 knots，每段均匀采 samples_per_segment 个点（段起点含、终点不含），末点追加最后一个 knot。
返回 (M,3) float64，M=(N-1)*samples_per_segment+1。N<2、samples_per_segment<1、相邻 knots 重合（距离<1e-12，向心参数化分母为零）抛 ValueError。确定性。
台架：输出穿过每个 knot（距离<1e-9）；共线 knot 集输出仍为直线。
