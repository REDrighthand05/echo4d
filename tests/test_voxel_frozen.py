"""Frozen-feature regression test: the voxel solidification contract.

User directive (2026-10-31): the point-cloud solidification feature is
foundational. It must remain available in the software and unaltered.
This test locks its contract:
  * every input point becomes exactly one cube,
  * each cube is centred on its point (nothing moved),
  * each cube carries its point's own colour (nothing recoloured),
  * the pipeline exposes the `voxel` stage and the app runs it as a step,
  * stage_voxel_model writes a re-readable voxel_model.ply.
If this test fails, someone changed the frozen feature -- stop and ask
the user before proceeding.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echo_scene.voxels import build_voxel_mesh, load_points_bin, stage_voxel_model


def _sample_cloud(n=500):
    rng = np.random.default_rng(42)
    pts = rng.random((n, 3)) * np.array([6.0, 3.0, 5.0])
    cols = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    return pts, cols


def test_every_point_becomes_one_cube_unmoved_uncoloured():
    pts, cols = _sample_cloud()
    verts, vcols, tris, meta = build_voxel_mesh(pts, cols)
    assert meta["cubes"] == len(pts), "a point was dropped or merged"
    assert len(tris) == 12 * len(pts)
    assert len(verts) == 8 * len(pts)
    centres = verts.reshape(len(pts), 8, 3).mean(axis=1)
    assert np.allclose(centres, pts, atol=1e-5), "a cube is not centred on its point"
    assert (vcols.reshape(len(pts), 8, 3) == cols[:, None, :]).all(), "a colour changed"
    # every face normal points away from its cube centre (solid, outward)
    vv = verts.astype(np.float64)
    normals = np.cross(vv[tris[:, 1]] - vv[tris[:, 0]], vv[tris[:, 2]] - vv[tris[:, 0]])
    face_centres = vv[tris].mean(axis=1)
    cube_of_face = np.repeat(np.arange(len(pts)), 12)
    assert (np.einsum("ij,ij->i", normals, face_centres - pts[cube_of_face]) > 0).all()


def test_stage_writes_readable_model():
    pts, cols = _sample_cloud(50)
    with tempfile.TemporaryDirectory() as td:
        job = Path(td)
        buf = np.empty((len(pts), 15), dtype=np.uint8)
        buf[:, :12] = pts.astype("<f4").view(np.uint8).reshape(len(pts), 12)
        buf[:, 12:15] = cols
        (job / "points.bin").write_bytes(buf.tobytes())
        info = stage_voxel_model(job)
        assert info["ok"] and info["cubes"] == len(pts)
        ply = (job / "voxel_model.ply").read_bytes()
        assert ply.startswith(b"ply\nformat binary_little_endian")
        assert b"element vertex 400" in ply and b"element face 600" in ply
        back_pts, _ = load_points_bin(job / "points.bin")
        assert np.allclose(back_pts, pts, atol=1e-6)


def test_feature_stays_wired_into_the_software():
    pipeline_src = (ROOT / "echo_scene" / "pipeline.py").read_text()
    assert "def stage_voxel" in pipeline_src and '"voxel"' in pipeline_src
    app_src = (ROOT / "app.py").read_text()
    assert "正在生成像素模型" in app_src, "the app lost the voxel step"
    out = subprocess.run(
        [sys.executable, "-m", "echo_scene.pipeline"],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode in (0, 2)  # usage path must not crash the import


if __name__ == "__main__":
    test_every_point_becomes_one_cube_unmoved_uncoloured()
    test_stage_writes_readable_model()
    test_feature_stays_wired_into_the_software()
    print("voxel frozen-feature contract: ALL PASS")
