"""Unit tests for shared scene manifest and dual rendering engine."""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
import sys
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from echo_scene.rendering import (
    CameraSpecification,
    LightSpecification,
    OfflineRenderConfig,
    SceneManifest,
    build_scene_manifest,
    execute_offline_render,
    generate_blender_cycles_script,
)


class TestRendering(unittest.TestCase):
    def test_manifest_serialization(self):
        manifest = SceneManifest(
            job_id="test_job_1",
            geometry={"observed_mesh": "mesh.ply"},
            cameras=[
                CameraSpecification(
                    name="cam1",
                    fx=500.0,
                    fy=500.0,
                    cx=320.0,
                    cy=240.0,
                    width=640,
                    height=480,
                    center=[0.0, 1.0, 2.0],
                    rotation=np.eye(3).tolist(),
                    is_free_view=False,
                )
            ],
            lighting=[
                LightSpecification(type="ambient", color=[1.0, 1.0, 1.0], intensity=1.0)
            ],
        )

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scene_manifest.json"
            manifest.save(p)
            self.assertTrue(p.exists())

            loaded = SceneManifest.load(p)
            self.assertEqual(loaded.job_id, "test_job_1")
            self.assertEqual(len(loaded.cameras), 1)
            self.assertEqual(loaded.cameras[0].name, "cam1")
            self.assertEqual(len(loaded.lighting), 1)

    def test_build_manifest_from_job(self):
        with tempfile.TemporaryDirectory() as td:
            job_dir = Path(td)
            (job_dir / "mesh.ply").write_text("ply\nend_header\n")
            (job_dir / "poses.json").write_text(
                json.dumps({
                    "f1": {"center": [0, 1, 0], "rotation": np.eye(3).tolist()},
                    "f2": {"center": [1, 1, 0], "rotation": np.eye(3).tolist()},
                    "f3": {"center": [2, 1, 0], "rotation": np.eye(3).tolist()},
                })
            )

            manifest = build_scene_manifest(job_dir)
            self.assertEqual(manifest.job_id, job_dir.name)
            # 3 source views + 2 free views = 5 cameras
            self.assertEqual(len(manifest.cameras), 5)
            self.assertTrue(any(c.is_free_view for c in manifest.cameras))

    def test_generate_cycles_script(self):
        manifest = SceneManifest(
            job_id="job_cycles_test",
            geometry={"observed_mesh": "C:/data/mesh.ply"},
            cameras=[
                CameraSpecification(
                    name="cam0",
                    fx=600.0, fy=600.0, cx=320.0, cy=240.0,
                    width=640, height=480, center=[0, 0, 1],
                    rotation=np.eye(3).tolist(),
                )
            ],
        )
        config = OfflineRenderConfig.from_preset("draft")
        with tempfile.TemporaryDirectory() as td:
            script_p = Path(td) / "render_cycles.py"
            generate_blender_cycles_script(manifest, config, script_p)
            self.assertTrue(script_p.exists())
            content = script_p.read_text(encoding="utf-8")
            self.assertIn("CYCLES", content)
            self.assertIn("samples = 32", content)


if __name__ == "__main__":
    unittest.main()
