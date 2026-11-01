# WO-spectral-ratios: spectral_ratios
纯函数，numpy。输入 counts (11,) float64：AS7341 顺序 [F1,F2,F3,F4,F5,F6,F7,F8,Clear,NIR,Flicker]，须非负、Clear>0，否则 ValueError。
返回 dict：
- "normalized": (8,) float64 = F1..F8 / Clear
- "nir_ratio": float = NIR / Clear
- "visible_sum_ratio": float = sum(F1..F8) / Clear
确定性。
台架：Clear=100、F1..F8=10..80、NIR=25 → normalized 逐项核对、nir_ratio=0.25、visible_sum_ratio=3.6（容差 1e-12）。
