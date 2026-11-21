"""Unit tests for conservative spatial structure completion."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
import sys
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from echo_scene.geometry import (
    ConservativeExtrapolator,
    EvidenceSource,
    MultiViewConsistencyChecker,
    PlaneModel,
    SceneGeometry,
    StructuralCategory,
    detect_structural_planes,
    execute_structural_completion,
    fit_plane_ransac_huber,
    project_points_to_camera,
)


class TestGeometryCompletion(unittest.TestCase):
    def test_plane_fitting(self):
        # Generate horizontal floor plane: y = 0 + small noise
        rng = np.random.default_rng(123)
        x = rng.uniform(-2, 2, 300)
        z = rng.uniform(-2, 2, 300)
        y = rng.normal(0.0, 0.005, 300)
        pts = np.column_stack([x, y, z])

        # Add outliers
        outliers = rng.uniform(-3, 3, (50, 3))
        all_pts = np.vstack([pts, outliers])

        res = fit_plane_ransac_huber(all_pts, distance_threshold=0.03)
        self.assertIsNotNone(res)
        normal, d, inliers = res

        # Normal should be close to [0, 1, 0] or [0, -1, 0]
        self.assertAlmostEqual(abs(normal[1]), 1.0, places=1)
        self.assertGreaterEqual(len(inliers), 250)

    def test_structural_plane_detection(self):
        # Create a floor and a wall
        rng = np.random.default_rng(456)
        # Floor: y = -0.5
        floor_x = rng.uniform(-1, 1, 200)
        floor_z = rng.uniform(-1, 1, 200)
        floor_y = np.full(200, -0.5) + rng.normal(0, 0.002, 200)
        floor_pts = np.column_stack([floor_x, floor_y, floor_z])

        # Wall: z = 1.0
        wall_x = rng.uniform(-1, 1, 200)
        wall_y = rng.uniform(-0.5, 1.5, 200)
        wall_z = np.full(200, 1.0) + rng.normal(0, 0.002, 200)
        wall_pts = np.column_stack([wall_x, wall_y, wall_z])

        combined = np.vstack([floor_pts, wall_pts])
        planes = detect_structural_planes(combined, min_support_points=100)
        self.assertGreaterEqual(len(planes), 2)
        categories = {p.category for p in planes}
        self.assertIn(StructuralCategory.WALL, categories)

    def test_multiview_projection(self):
        pts = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
        r = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        pixels, z, valid = project_points_to_camera(
            pts, r, t, fx=500.0, fy=500.0, cx=320.0, cy=240.0
        )
        self.assertTrue(valid[0])
        self.assertAlmostEqual(z[0], 2.0)
        self.assertAlmostEqual(pixels[0, 0], 320.0)
        self.assertAlmostEqual(pixels[0, 1], 240.0)

    def test_extrapolator_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            # Create synthetic poses.json
            poses_data = {
                "frames": {
                    "f1": {
                        "center": [0.0, 1.0, -2.0],
                        "rotation": np.eye(3).tolist(),
                        "width": 640,
                        "height": 480,
                    },
                    "f2": {
                        "center": [0.5, 1.2, -2.2],
                        "rotation": np.eye(3).tolist(),
                        "width": 640,
                        "height": 480,
                    },
                }
            }
            (tmp_path / "poses.json").write_text(json.dumps(poses_data))

            # Create synthetic points.bin (floor points)
            rng = np.random.default_rng(789)
            x = rng.uniform(-1, 1, 200).astype("<f4")
            z = rng.uniform(-1, 1, 200).astype("<f4")
            y = rng.normal(0.0, 0.005, 200).astype("<f4")
            rgb = np.full((200, 3), 180, dtype=np.uint8)

            raw = np.empty((200, 15), dtype=np.uint8)
            coords = np.column_stack([x, y, z]).astype("<f4")
            raw[:, 0:12] = coords.view(np.uint8).reshape(200, 12)
            raw[:, 12:15] = rgb
            raw.tofile(tmp_path / "points.bin")

            res = execute_structural_completion(tmp_path)
            self.assertTrue(res["ok"])
            self.assertTrue((tmp_path / "structural_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
