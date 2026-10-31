# WO-frustum-select: frustum_select_points
纯函数，numpy。输入 points (N,3) float64 世界坐标点云；cam_position (3,) float64 相机中心（世界系）；cam_rotation (3,3) float64 世界→相机旋转（p_cam = R @ (p_world - cam_position)，相机朝 +Z 看）；fx, fy, cx, cy float 针孔内参；image_width, image_height int 像面尺寸；near, far float 深度界。
选取规则（只选取，不动点）：near <= z <= far 且 0 <= fx*x/z+cx < image_width 且 0 <= fy*y/z+cy < image_height 的点，返回其原始下标 (K,) int64，升序。
输入形状非法抛 ValueError。确定性。
台架：手摆合成点云（内/外/近远界外/像面外/相机背后），期望下标集合精确一致；含平移相机一例。
