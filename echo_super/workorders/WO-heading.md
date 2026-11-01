# WO-heading: tilt_compensated_heading
纯函数，numpy。输入 mag (3,) 磁力计原始读数（任意单位）、accel (3,) 加速度读数（任意单位，同机体坐标系，静止时指向重力反方向即"上"）。
做倾斜补偿：先由 accel 求 roll/pitch，把 mag 旋转回水平面，再 heading = atan2(-my_h, mx_h) 换算到 [0,360) 度（0=磁北，顺时针为正，按 x 前、y 左、z 上的机体约定——在代码注释里写明所用约定）。
accel 模长为 0 或 mag 模长为 0 抛 ValueError。返回 float。确定性。
台架：水平静止、accel=(0,0,1)：mag=(1,0,0.2)→0 度；mag=(0,-1,0.2)→90 度（容差 1e-6）；整机绕竖轴转 45 度后数值随转角平移；前倾 30 度时同一磁场读数应还原同一 heading（构造合成数据核对，容差 1e-4）。
