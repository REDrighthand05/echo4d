# WO-sample-track（Studio 预开发，2026-11-01）

单函数工单，DeepSeek Flash 实现，Muse 验收。

## 函数 sample_track(keys, t)

- keys：(time: float, value: float, easing: str) 元组列表，按 time 排序，至少 1 个。
  easing 表示从该关键帧离开、走向下一帧的曲线，取值：
  'linear' | 'ease_in' | 'ease_out' | 'ease_in_out' | 'hold'。
- 行为：
  - t 在首帧之前 → 首帧值；t 在末帧之后 → 末帧值；单帧 → 该值。
  - 在第 i 与 i+1 帧之间：u = (t - t_i)/(t_{i+1} - t_i)，对 u 施加 easing：
    linear = u；hold = 0.0；
    ease_in = cubic-bezier(0.42, 0, 1, 1)；
    ease_out = cubic-bezier(0, 0, 0.58, 1)；
    ease_in_out = cubic-bezier(0.42, 0, 0.58, 1)。
    cubic-bezier(x1,y1,x2,y2)：用 40 次二分在 [0,1] 上解参数 s 使 x(s)=u，再取 y(s)。
  - 结果 = v_i + (v_{i+1} - v_i) * eased_u。
- 纯 Python，无第三方导入，无 I/O，确定性。

## 验收（echo_super/test_ds_batch4.py，Muse 写）

钳制、单帧、linear 中点、hold 保持、ease_in/out 在 0.5 处的已知值域、
ease_in_out 对称中点、二者镜像关系、多段各自缓动、非均匀时间、确定性。
