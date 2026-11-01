# WO-sync-stats: stream_sync_stats
纯函数，numpy。输入 streams：list of (N_i,) float64 时间戳数组（秒，各自单调递增，至少 1 个元素；不满足抛 ValueError）。
以 streams[0] 为基准，返回 dict：
- "counts": list[int] 每条流样本数
- "spans_s": list[float] 每条流 (末-首)
- "median_interval_s": list[float] 每条流相邻间隔中位数（N=1 时为 0.0）
- "max_nearest_offset_s": list[float] 每条流（基准流自身为 0.0）每个时间戳到基准流最近时间戳的绝对差的最大值
- "p95_nearest_offset_s": list[float] 同上的 95 分位（numpy.percentile 线性插值；N=1 时等于该值）
确定性，无随机。
台架：基准 0,0.1,...,1.0；第二流整体偏移 +0.03 且间隔 0.2 → max/p95 均为 0.03（容差 1e-9）；单元素流不崩。
