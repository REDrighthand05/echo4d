# WO-surface-voxels: surface_voxel_indices
纯函数，numpy。输入 points (N,3) float64（点云固化的立方体中心），cube float 立方体边长。
把中心按 round(p/cube) 量化到整数网格；一个立方体若 6 个面邻居中至少一个缺失，则是表面立方体。
返回表面立方体的原始下标数组（升序）。确定性、无随机。用字典/集合查邻居，N=30万时秒级。
