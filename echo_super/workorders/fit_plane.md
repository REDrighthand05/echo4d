# 工单：fit_plane（总最小二乘平面拟合）— 已交付已验收

规格要点：centroid+SVD 总最小二乘；符号规则=法向最大绝对值分量为正；
退化（共线，第二小奇异值≈0）、N<3、非有限值 → ValueError。numpy only，文件自含 import。

## 审核记录（Muse，2026-10-31）
- DS Flash 产出存 fit_plane_ds.py。全部通过：z=5 平面精确、倾斜平面
  (2,3,6)/7 精确、符号规则（含翻面情形）、噪声平面 RMS 与 σ 一致、
  三种非法输入均正确 ValueError。本次规格写明"文件自含 import"后无缺失。
  结论：验收，待注册（points → plane）。
