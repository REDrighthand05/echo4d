"""Real frozen TUM MVS: observed-edge Mesh with independent inferred-Z support.

Runs the exact existing 48-view TSDF replay once, then filters ONLY its extra
observed-edge triangles. Measured sensor Z is never opened until posthoc check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from echo4d.fusion import Mesh
from echo4d.known_pose_edge_crossview import filter_crossview_faces
from echo4d.model import Camera, file_sha256
from echo4d.output import mesh_summary, write_ply
from tum_mvs_observed_edges import replay


def run(old: Path, source_root: Path, target: Path) -> dict:
    if target.exists():
        raise FileExistsError("Choose a fresh output directory")
    base = target / "unfiltered"
    # This recomputes the old control from frozen RGB-inferred maps and verifies
    # its exact PLY SHA; no external Mesh is used as a geometric prior.
    replay(old, source_root, base)
    manifest = json.loads((old / "inputs/recording.json").read_text())
    camera = Camera.from_dict(manifest["intrinsics"])
    entries = manifest["frames"]
    if len(entries) != 48 or manifest["units"] != "meters":
        raise ValueError("Expected the pinned 48-frame metric MVS manifest")
    depths = [np.load(old / "inputs" / e["depth"], allow_pickle=False)
              for e in entries]
    poses = [np.asarray(e["cameraToWorld"], dtype=np.float64).reshape(4, 4, order="F")
             for e in entries]
    original = trimesh.load_mesh(base / "control-replayed.ply",
                                 force="mesh", process=False)
    candidate = trimesh.load_mesh(base / "reconstruction/mesh.ply",
                                  force="mesh", process=False)
    mask, stats = filter_crossview_faces(
        original.vertices, original.faces, candidate.vertices, candidate.faces,
        depths, poses, camera)
    selected = candidate.faces[mask]
    used, remap = np.unique(selected, return_inverse=True)
    colors = candidate.visual.vertex_colors[used, :3].astype(np.uint8)
    normals = candidate.vertex_normals[used].astype(np.float32)
    # PLY exports normals/color/geometry only. TSDF observation counts are not
    # serialized in PLY, so these placeholders must NOT be interpreted as evidence.
    result_mesh = Mesh(
        candidate.vertices[used].astype(np.float32),
        remap.reshape(-1, 3).astype(np.int32),
        normals, colors, np.ones(len(used), dtype=np.uint32),
        np.zeros(len(used), dtype=np.int32), np.zeros(len(used), dtype=np.int32))
    result_mesh.validate()
    reconstruction = target / "reconstruction"
    reconstruction.mkdir(exist_ok=True)
    write_ply(reconstruction / "mesh.ply", result_mesh)
    report = {
        "experiment": "TUM 48-view frozen RGB-inferred-Z independent face support",
        "source": "TUM fr1_xyz original RGB 640x480, fusion 320x240",
        "retained": ["48 training RGB", "measured metric camera poses",
                     "camera intrinsics", "48 frozen RGB-inferred depth maps"],
        "removed": ["sensor depth pixels from reconstruction",
                    "external prior Mesh from reconstruction"],
        "sensor_depth_files_opened": 0,
        "prior_mesh_used_for_reconstruction": False,
        "baseline_face_label": "recomputed TSDF control from same frozen RGB inferred depth",
        "unfiltered_mesh_sha256": file_sha256(base / "reconstruction/mesh.ply"),
        "control_replay": json.loads((base / "report.json").read_text())["control_replay"],
        "new_mesh_sha256": file_sha256(reconstruction / "mesh.ply"),
        "new_mesh_summary": mesh_summary(result_mesh),
        "face_support": stats,
        "mesh_observation_counts": "unavailable in source PLY; 1 is export placeholder, not real count",
        "visual_acceptance": "FAIL until direct five-view scene recognition",
        "historical_fail": "Observed-edge predecessor 14.60% >30cm wrong new-only sensor-valid pixels",
    }
    (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"mesh_sha256": report["new_mesh_sha256"],
            "faces": len(result_mesh.faces), "face_support": stats}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.old, a.source_root, a.target)))
