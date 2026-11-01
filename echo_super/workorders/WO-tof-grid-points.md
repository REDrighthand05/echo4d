# WO-tof-grid-points: tof_grid_points
纯函数，numpy。把 8×8 多区 ToF 的一帧测距变成传感器坐标系点云（传感器系：x 右、y 上、z 前）。
输入：dist_mm (8,8) 整数毫米（<=0 或 >max_range_mm 视为无效），hfov_deg、vfov_deg 为整幅水平/垂直视场角（度），max_range_mm 上限。
格子方向按本项目记录格式约定（tan 空间均匀划分）：u = (2*(c+0.5)/8 - 1) * tan(hfov/2)，v = (1 - 2*(r+0.5)/8) * tan(vfov/2)，r=0 为最上方一行（v 为正）。
方向向量 d = (u, v, 1) 归一化，点 = d * (dist_mm/1000.0)。
返回 (points (K,3) float64 米, zones (K,2) int (r,c))，仅含有效格，按行优先顺序。形状不符抛 ValueError。确定性。
台架：全格 1000mm、hfov=vfov=45 时逐点核对（正前方中心四格的 |x|,|y| 相等且小于边角格）；含 -1 格时 K 相应减少且 zones 对位。
