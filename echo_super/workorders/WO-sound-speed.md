# WO-sound-speed: speed_of_sound_mps
纯函数（可用 math）。输入 temp_c float（摄氏度，须在 -60..80，否则 ValueError）。
返回干燥空气声速（米/秒）：c = 331.3 * sqrt(1 + temp_c/273.15)。
台架：0°C → 331.3（精确）；20°C → 约 343.2（容差 ±0.3）；温度越高声速越大。
