# 正经技术内核链：点云 → 泊松网格（按 SPEC_TEMPLATE 拆单）

## 内核 1（派 DS Flash）
函数名:        orient_normals
consumes:      [points, normals, cam_centers]（均为 EchoObject，data 为 numpy 数组）
produces:      normals（已定向）
签名:          def orient_normals(points, normals, cam_centers) -> np.ndarray
说明:          三个入参 data 均为 (N,3) float 数组。对每个点 i：若
               dot(normals[i], cam_centers[i]-points[i]) < 0 则 normals[i] 取反。
               返回新数组，不修改入参。
性能要求:      全向量化，禁逐点 Python 循环；N=1e6 时 < 1s。
禁止事项:      只许 import numpy；不读全局状态；不写文件；不吞异常。
失败方式:      形状不一致、非有限值、N=0 时 raise ValueError 并说明哪项不对。
自测素材:      points=[[0,0,0]], normals=[[-1,0,0]], cam_centers=[[5,0,0]]
               → 期望 [[1,0,0]]（法向被翻向相机一侧）。

## 内核 2（Muse 亲手，先验方法再拆单）
函数名:        poisson_reconstruct
consumes:      [points, normals]
produces:      mesh
说明:          Kazhdan 泊松重建：解 ∇²χ = ∇·V（V 为法向场），取等值面。
               Open3D create_from_point_cloud_poisson(depth=9, linear_fit=True)，
               裁剪到点云 1–99 百分位包围盒。方法已在 experiments/poisson_exp.py
               中由 Muse 实现待验；验通后按同一规格拆单给 DS 复写并注册。

## 审核记录（Muse，2026-10-30）
- orient_normals（DS Flash 产出）：功能全对——工单自测、入参不变性、
  三种非法输入 ValueError 均通过。两个小问题已在集成层处理：
  1) 漏了 `import numpy as np` 一行（以后工单明确写"文件须自含 import"）；
  2) 1e6 点实测 1.3s（规格 <1s），当时机器在并行装库，集成时空闲复测再定。
  结论：验收，待注册进 echo_super（points+normals+cam_centers → normals）。
