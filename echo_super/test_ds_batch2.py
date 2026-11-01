"""Bench tests for DS batch 2 (2026-10-31): five kernels written by
DeepSeek Flash from echo_super/workorders/WO-*.md, verified here one by
one before registration (adapters.register_ds_batch2).

Every fixture's expected output is hand-computed in the test comments —
nothing is compared against the implementation's own logic except the
voxel cross-check, which uses an independent brute-force reimplementation.

Run directly:  .venv/bin/python echo_super/test_ds_batch2.py
or via pytest: .venv/bin/python -m pytest echo_super/test_ds_batch2.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echo_super import wrap
from echo_super.adapters import KERNEL_META, register_ds_batch2
from echo_super.ds_batch2 import (
    catmull_rom_path,
    frustum_select_points,
    plane_fit_lsq,
    resample_path_even,
    voxel_select_representative,
)


class ResamplePathEvenTests(unittest.TestCase):
    L = np.array([[0.0, 0, 0], [3, 0, 0], [3, 4, 0]])  # total length 7

    def test_unit_spacing_exact_samples_and_length(self):
        out = resample_path_even(self.L, 1.0)
        expected = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
                             [3, 1, 0], [3, 2, 0], [3, 3, 0], [3, 4, 0]],
                            dtype=float)
        self.assertTrue(np.allclose(out, expected, atol=1e-9))
        seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
        self.assertAlmostEqual(float(seg.sum()), 7.0, places=9)

    def test_spacing_two_hand_positions(self):
        # arc 0,2,4,6 plus endpoint at 7: (0,0,0),(2,0,0),(3,1,0),(3,3,0),(3,4,0)
        out = resample_path_even(self.L, 2.0)
        expected = np.array([[0, 0, 0], [2, 0, 0], [3, 1, 0],
                             [3, 3, 0], [3, 4, 0]], dtype=float)
        self.assertTrue(np.allclose(out, expected, atol=1e-9))

    def test_spacing_longer_than_path_gives_endpoints_only(self):
        out = resample_path_even(self.L, 10.0)
        self.assertTrue(np.allclose(out, self.L[[0, -1]], atol=1e-12))

    def test_zero_length_segment_skipped(self):
        path = np.array([[0.0, 0, 0], [0, 0, 0], [2, 0, 0]])
        out = resample_path_even(path, 1.0)
        self.assertTrue(np.allclose(
            out, [[0, 0, 0], [1, 0, 0], [2, 0, 0]], atol=1e-9))

    def test_errors(self):
        with self.assertRaises(ValueError):
            resample_path_even(self.L, 0.0)
        with self.assertRaises(ValueError):
            resample_path_even(np.array([[0.0, 0, 0]]), 1.0)


class CatmullRomPathTests(unittest.TestCase):
    def test_passes_through_every_knot(self):
        knots = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0],
                          [3, 1, 0], [3, 3, 0]])
        out = catmull_rom_path(knots, 6)
        self.assertEqual(out.shape, ((len(knots) - 1) * 6 + 1, 3))
        for k in knots:
            d = np.linalg.norm(out - k, axis=1).min()
            self.assertLess(d, 1e-9)
        # segment starts are exactly the knots, in order
        for i, k in enumerate(knots):
            self.assertTrue(np.allclose(out[i * 6], k, atol=1e-9))

    def test_straight_knots_stay_on_the_line(self):
        knots = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        out = catmull_rom_path(knots, 4)
        self.assertEqual(out.shape, (13, 3))
        self.assertTrue(np.allclose(out[:, 1:], 0.0, atol=1e-9))
        self.assertTrue(np.all(np.diff(out[:, 0]) >= -1e-12))

    def test_duplicate_consecutive_knots_rejected(self):
        knots = np.array([[0.0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]])
        with self.assertRaises(ValueError):
            catmull_rom_path(knots, 4)


class FrustumSelectPointsTests(unittest.TestCase):
    KW = dict(fx=100.0, fy=100.0, cx=50.0, cy=50.0,
              image_width=100, image_height=100, near=1.0, far=10.0)

    def test_hand_placed_points_exact_index_set(self):
        pts = np.array([
            [3.0, 0, 5],        # 0  u=110 -> out
            [0.0, 0, 5],        # 1  centre -> in
            [0.0, 0, 0.5],      # 2  inside near -> out
            [-0.5, -0.5, 5],    # 3  u=v=40 -> in
            [0.0, 0, 20],       # 4  beyond far -> out
            [2.45, 0, 5],       # 5  u=99 -> in
            [0.0, 0, -5],       # 6  behind camera -> out
            [0.0, -3, 5],       # 7  v=-10 -> out
            [0.0, 0, 1],        # 8  z == near -> in
            [2.55, 0, 5],       # 9  u=101 -> out
            [0.0, 0, 10],       # 10 z == far -> in
            [-2.5, 2.5, 5],     # 11 v=100 (not < h) -> out
            [-2.5, -2.5, 5],    # 12 u=v=0 -> in
        ])
        idx = frustum_select_points(pts, np.zeros(3), np.eye(3), **self.KW)
        self.assertEqual(idx.dtype, np.int64)
        self.assertEqual(idx.tolist(), [1, 3, 5, 8, 10, 12])

    def test_translated_camera(self):
        pts = np.array([[0.0, 0, 0], [0, 0, -20], [0, 0, -4]])
        idx = frustum_select_points(pts, np.array([0.0, 0, -5]),
                                    np.eye(3), **self.KW)
        self.assertEqual(idx.tolist(), [0, 2])

    def test_rotated_camera_looks_down_world_minus_x(self):
        # world->camera rotation: camera +Z is world -X
        R = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        pts = np.array([[-5.0, 0, 0], [5, 0, 0], [0, 0, 5]])
        idx = frustum_select_points(pts, np.zeros(3), R, **self.KW)
        self.assertEqual(idx.tolist(), [0])


class PlaneFitLsqTests(unittest.TestCase):
    @staticmethod
    def plane_points():
        # plane 2x - y + 2z = 6, points exact in binary floating point
        xs, ys = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
        xs, ys = xs.ravel().astype(float), ys.ravel().astype(float)
        return np.column_stack([xs, ys, (6 - 2 * xs + ys) / 2])

    def test_noiseless_plane_exact(self):
        normal, offset, distances = plane_fit_lsq(self.plane_points())
        self.assertTrue(np.allclose(normal, [2 / 3, -1 / 3, 2 / 3],
                                    atol=1e-9))
        self.assertAlmostEqual(offset, -2.0, places=9)
        self.assertLess(float(np.abs(distances).max()), 1e-9)

    def test_vertical_plane_sign_convention(self):
        ys, zs = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
        pts = np.column_stack([np.full(25, 3.0),
                               ys.ravel().astype(float),
                               zs.ravel().astype(float)])
        normal, offset, distances = plane_fit_lsq(pts)
        self.assertTrue(np.allclose(normal, [1, 0, 0], atol=1e-9))
        self.assertAlmostEqual(offset, -3.0, places=9)
        self.assertLess(float(np.abs(distances).max()), 1e-9)

    def test_noisy_plane_normal_within_one_degree(self):
        rng = np.random.default_rng(0)
        base = self.plane_points()
        true_n = np.array([2 / 3, -1 / 3, 2 / 3])
        noisy = base + rng.normal(0, 0.01, size=(len(base), 1)) * true_n
        normal, offset, _ = plane_fit_lsq(noisy)
        cosang = abs(float(normal @ true_n))
        angle_deg = np.degrees(np.arccos(min(cosang, 1.0)))
        self.assertLess(angle_deg, 1.0)
        self.assertAlmostEqual(offset, -2.0, delta=0.05)

    def test_too_few_points_rejected(self):
        with self.assertRaises(ValueError):
            plane_fit_lsq(np.zeros((2, 3)))


def _brute_force_representatives(points, cell_size):
    """Independent reimplementation: plain dict + loops, no shared code."""
    best = {}
    for i, p in enumerate(points):
        cell = tuple(np.floor(p / cell_size).astype(np.int64))
        centre = (np.array(cell) + 0.5) * cell_size
        d2 = float(((p - centre) ** 2).sum())
        if cell not in best or d2 < best[cell][0]:
            best[cell] = (d2, i)
    return sorted(i for _d2, i in best.values())


class VoxelSelectRepresentativeTests(unittest.TestCase):
    PTS = np.array([
        [0.10, 0.10, 0.10],   # 0 cell (0,0,0), d2=0.48
        [0.40, 0.50, 0.50],   # 1 cell (0,0,0), d2=0.01
        [0.45, 0.55, 0.50],   # 2 cell (0,0,0), d2=0.005 -> survivor
        [1.20, 0.20, 0.20],   # 3 cell (1,0,0), alone -> survivor
        [-0.20, 0.0, 0.0],    # 4 cell (-1,0,0), alone -> survivor
        [2.25, 0.50, 0.50],   # 5 cell (2,0,0), d2=0.0625, tie, smaller idx
        [2.75, 0.50, 0.50],   # 6 cell (2,0,0), d2=0.0625, tie -> dropped
    ])

    def test_hand_fixture_survivors(self):
        idx = voxel_select_representative(self.PTS, 1.0)
        self.assertEqual(idx.dtype, np.int64)
        self.assertEqual(idx.tolist(), [2, 3, 4, 5])

    def test_idempotent(self):
        idx = voxel_select_representative(self.PTS, 1.0)
        again = voxel_select_representative(self.PTS[idx], 1.0)
        self.assertEqual(again.tolist(), list(range(len(idx))))

    def test_matches_brute_force_on_random_cloud(self):
        rng = np.random.default_rng(7)
        pts = rng.uniform(-3, 3, size=(500, 3))
        for cell in (0.5, 1.0, 2.3):
            self.assertEqual(
                voxel_select_representative(pts, cell).tolist(),
                _brute_force_representatives(pts, cell))

    def test_empty_and_errors(self):
        out = voxel_select_representative(np.empty((0, 3)), 1.0)
        self.assertEqual(out.dtype, np.int64)
        self.assertEqual(len(out), 0)
        with self.assertRaises(ValueError):
            voxel_select_representative(self.PTS, 0.0)


class RegistryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.reg = register_ds_batch2()

    def test_all_five_registered_with_declared_kinds(self):
        expected = {
            "resample_path_even": (("path",), "path"),
            "catmull_rom_path": (("path",), "path"),
            "frustum_select_points": (("points",), "points"),
            "plane_fit_lsq": (("points",), "points"),
            "voxel_select_representative": (("points",), "points"),
        }
        for name, (consumes, produces) in expected.items():
            k = self.reg.kernels[name]
            self.assertEqual(k.consumes, consumes, name)
            self.assertEqual(k.produces, produces, name)
            self.assertEqual(KERNEL_META[name],
                             {"evidence": "实测", "status": "已验"}, name)

    def test_kernels_run_through_echo_objects(self):
        path_obj = wrap({"path": np.array([[0.0, 0, 0], [3, 0, 0],
                                           [3, 4, 0]])},
                        kind="path", meta={"spacing": 1.0})
        out = self.reg.kernels["resample_path_even"].fn(path_obj)
        self.assertEqual(out.kind, "path")
        self.assertEqual(len(out.data["path"]), 8)
        self.assertEqual(out.meta["status"], "已验")

        pts_obj = wrap(PlaneFitLsqTests.plane_points(), kind="points")
        out = self.reg.kernels["plane_fit_lsq"].fn(pts_obj)
        self.assertEqual(out.kind, "points")
        self.assertTrue(np.allclose(out.meta["plane_normal"],
                                    [2 / 3, -1 / 3, 2 / 3], atol=1e-9))
        self.assertLess(
            float(np.abs(out.data["signed_distances"]).max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
