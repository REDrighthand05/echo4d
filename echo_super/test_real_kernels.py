"""Tests for the real kernels registered by echo_super.adapters.

One test per kernel:
  fit_plane            recovers a known plane's normal / d / rms
  orient_normals       flips a normal toward its source camera
  poisson_reconstruct  routing only: points + normals -> mesh is
                       reachable (Open3D itself is NOT run here)

Run directly:  .venv/bin/python echo_super/test_real_kernels.py
or via pytest: .venv/bin/python -m pytest echo_super/test_real_kernels.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echo_super import wrap
from echo_super.adapters import KERNEL_META, register_real_kernels


class RealKernelTests(unittest.TestCase):
    def setUp(self):
        self.reg = register_real_kernels()

    def test_fit_plane_kernel_recovers_known_plane(self):
        # plane 2x + 3y + 6z = 42  ->  unit normal (2,3,6)/7, d = -6, rms = 0
        xs, ys = np.meshgrid(np.linspace(-5, 5, 9), np.linspace(-5, 5, 9))
        pts = np.column_stack([xs.ravel(), ys.ravel(),
                               (42 - 2 * xs.ravel() - 3 * ys.ravel()) / 6])
        out = self.reg.kernels["fit_plane"].fn(wrap(pts, kind="points"))
        self.assertEqual(out.kind, "points")
        self.assertTrue(np.allclose(out.meta["plane_normal"],
                                    np.array([2, 3, 6]) / 7, atol=1e-9))
        self.assertAlmostEqual(out.meta["plane_d"], -6.0, places=9)
        self.assertAlmostEqual(out.meta["plane_rms"], 0.0, places=9)
        self.assertEqual(out.meta["evidence"], KERNEL_META["fit_plane"]["evidence"])
        self.assertEqual(out.meta["status"], KERNEL_META["fit_plane"]["status"])

    def test_orient_normals_kernel_flips_toward_camera(self):
        # workorder self-test: point at origin, normal pointing -x,
        # source camera at +x -> the normal must come back as +x.
        points_obj = wrap(
            {"points": np.array([[0.0, 0.0, 0.0]]),
             "cam_centers": np.array([[5.0, 0.0, 0.0]])},
            kind="points")
        normals_obj = wrap(np.array([[-1.0, 0.0, 0.0]]), kind="normals")
        out = self.reg.kernels["orient_normals"].fn(points_obj, normals_obj)
        self.assertEqual(out.kind, "normals")
        self.assertTrue(np.allclose(out.data, [[1.0, 0.0, 0.0]]))
        self.assertEqual(out.meta["status"],
                         KERNEL_META["orient_normals"]["status"])

    def test_poisson_route_exists(self):
        kernel = self.reg.kernels["poisson_reconstruct"]
        self.assertEqual(kernel.consumes, ("points", "normals"))
        self.assertEqual(kernel.produces, "mesh")
        self.assertEqual(KERNEL_META["poisson_reconstruct"]["status"], "已验候选")
        path = self.reg.find_path(["points", "normals"], "mesh")
        self.assertEqual([k.name for k in path], ["poisson_reconstruct"])


if __name__ == "__main__":
    unittest.main()
