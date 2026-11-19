"""Check the standalone scene payload and atomic replacement on write failure."""
import base64
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo_scene import pipeline


class SceneExport(unittest.TestCase):
    def test_buffers_and_failed_reexport(self):
        template = (Path(pipeline.__file__).resolve().parents[1] / "viewer.html").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            points = bytes(range(253)) * 4001
            (root / "points.bin").write_bytes(points)
            expected = template.replace("<script>\n", '<script>window.POINTS_B64 = "'
                                        + base64.b64encode(points).decode()
                                        + '";</script>\n<script>\n', 1)
            for name in [None, "mesh_vertices.bin", "mesh_indices.bin"]:
                if name:
                    (root / name).write_bytes(b"xyz1")
                    expected = expected.replace(f"fetch('{name}')",
                                                "fetch('data:application/octet-stream;base64,eHl6MQ==')")
                self.assertTrue(pipeline.stage_present(root)["ok"])
                self.assertEqual((root / "scene.html").read_text(encoding="utf-8"), expected)
            names = set(root.iterdir())
            with patch.object(pipeline, "_write_embedded_buffer", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    pipeline.stage_present(root)
            self.assertEqual(set(root.iterdir()), names)
            self.assertEqual((root / "scene.html").read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()
