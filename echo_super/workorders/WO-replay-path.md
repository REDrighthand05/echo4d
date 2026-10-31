# WO-replay-path: build_replay_path
纯函数，numpy。输入 centers (N,3) float64 相机中心序列，look_targets (N,3) float64 每帧注视点，samples_per_segment int=10。
Catmull-Rom 样条穿过 centers 均匀采样得 path (M,3)；look_dirs (M,3) 为对应注视方向（对 look_targets 同样插值后与 path 作差并归一化）。
N>=4 走样条，N==2/3 退化线性插值，N<2 抛 ValueError。确定性。
