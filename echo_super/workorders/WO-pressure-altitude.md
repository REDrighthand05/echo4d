# WO-pressure-altitude: pressure_altitude_m
纯函数（可用 math）。输入 pressure_hpa、reference_hpa（均为 float，>0，否则 ValueError）。
用国际标准气压公式返回相对高度（米）：h = 44330.0 * (1 - (p/p0)**0.1903)。
返回 float。确定性。
台架：p==p0 → 0（精确）；p0=1013.25、p=900 → 约 988.7m（容差 ±1.5）；单调性：p 越小 h 越大。
