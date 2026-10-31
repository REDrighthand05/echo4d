# WO-voxel-representative: voxel_select_representative
纯函数，numpy。输入 points (N,3) float64，cell_size float>0。
体素格：cell 下标 = floor(points / cell_size)，cell 中心 = (idx + 0.5) * cell_size。每个被占 cell 只保留离 cell 中心最近的那个原始点（只选取，绝不平均/造点）；距离并列时保留原始下标较小者。
返回保留点原始下标 (K,) int64，升序。空点云返回空 int64 数组；cell_size<=0 或形状非法抛 ValueError。确定性。
台架：手摆 fixture 逐 cell 核对幸存者（含并列取小下标一例）；幂等性——对自身输出再跑一遍，点集不变。
背景：本项目的体素平均合并在 CATALOG 中标「待替换：平均式合并违反原则」，本内核即确定性选取替代件之一。
