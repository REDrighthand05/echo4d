# WO-resample-path: resample_path_even
纯函数，numpy。输入 points (N,3) float64 折线路径，spacing float>0 目标弧长间距。
沿折线弧长在 0, s, 2s, … 处线性插值采样，首尾端点必含（末端点若未被等距点命中则追加，不重复）。
返回 (M,3) float64。零长线段跳过；N<2 或 spacing<=0 抛 ValueError。确定性，无随机。
台架：L 形折线 (0,0,0)->(3,0,0)->(3,4,0)，s=1 期望采样逐点精确、总长保持 7（误差<1e-9）；s=2 期望位置逐点核对。
