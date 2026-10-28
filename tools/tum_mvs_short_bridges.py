"""Real TUM 48-view RGB-MVS A/B: short topology bridges vs centroid filter.

Reuses PR90's exact same-volume TSDF replay. The measured sensor depth is
never opened; it may be used by a separate, strictly posthoc diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh

from echo4d.fusion import Mesh
from echo4d.known_pose_edge_bridges import filter_bridge_faces
from echo4d.model import Camera, file_sha256
from echo4d.output import mesh_summary, write_ply
from tum_mvs_edge_crossview import run as previous_run


def run(old: Path, source_root: Path, target: Path) -> dict:
    if target.exists():
        raise FileExistsError("Choose a fresh output directory")
    # Recompute old and candidate Mesh from exactly the frozen RGB depth maps.
    prior = target / "prior"
    previous_run(old, source_root, prior)
    manifest = json.loads((old / "inputs/recording.json").read_text())
    if manifest["units"] != "meters" or len(manifest["frames"]) != 48:
        raise ValueError("Expected 48 fixed metric-pose RGB inferred depth maps")
    camera = Camera.from_dict(manifest["intrinsics"])
    depths = [np.load(old / "inputs" / f["depth"], allow_pickle=False)
              for f in manifest["frames"]]
    poses = [np.asarray(f["cameraToWorld"], dtype=np.float64).reshape(4, 4, order="F")
             for f in manifest["frames"]]
    original = trimesh.load_mesh(prior / "unfiltered/control-replayed.ply",
                                 force="mesh", process=False)
    candidate = trimesh.load_mesh(prior / "unfiltered/reconstruction/mesh.ply",
                                  force="mesh", process=False)
    mask, support = filter_bridge_faces(
        original.vertices, original.faces, candidate.vertices, candidate.faces,
        depths, poses, camera)
    selected = candidate.faces[mask]
    used, remap = np.unique(selected, return_inverse=True)
    result = Mesh(
        candidate.vertices[used].astype(np.float32),
        remap.reshape(-1, 3).astype(np.int32),
        candidate.vertex_normals[used].astype(np.float32),
        candidate.visual.vertex_colors[used, :3].astype(np.uint8),
        np.ones(len(used), dtype=np.uint32),
        np.zeros(len(used), dtype=np.int32),
        np.zeros(len(used), dtype=np.int32),
    )
    result.validate()
    reconstruction = target / "reconstruction"
    reconstruction.mkdir(parents=True)
    write_ply(reconstruction / "mesh.ply", result)
    old_mesh = prior / "reconstruction/mesh.ply"
    # Frozen five cameras are inherited, never an old render hash or result.
    old_report = json.loads((old / "report.json").read_text())
    control = target / "prior-comparison"
    (control / "reconstruction").mkdir(parents=True)
    shutil.copyfile(old_mesh, control / "reconstruction/mesh.ply")
    (control / "report.json").write_text(json.dumps({
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "mesh_sha256": file_sha256(old_mesh),
        "views": [{"name": v["name"], "pose": v["pose"]} for v in old_report["views"]],
        "view_pose_source": "frozen original RGB-only experiment; no copied render hashes",
    }, indent=2) + "\n")
    summary = mesh_summary(result)
    summary["min_nearest_voxel_observations"] = None
    summary["max_nearest_voxel_observations"] = None
    report = {
        "checkout_sha": os.environ.get("GITHUB_SHA"),
        "source": "TUM fr1_xyz 640x480 RGB, 320x240 fusion; CC BY 4.0",
        "retained": ["48 original RGB frames", "48 frozen RGB-inferred Z maps",
                     "known metric camera poses", "calibrated intrinsics"],
        "removed": ["measured sensor depth pixels", "external prior Mesh"],
        "prior_same_volume_centroid_mesh_sha256": file_sha256(old_mesh),
        "new_mesh_sha256": file_sha256(reconstruction / "mesh.ply"),
        "unfiltered_same_volume_mesh_sha256": file_sha256(
            prior / "unfiltered/reconstruction/mesh.ply"),
        "summary": summary,
        "support": support,
        "sensor_depth_files_opened": 0,
        "baseline_mesh_used_only_for_same_volume_face_label": True,
        "mesh_observation_counts": "unavailable in source PLY; exporter placeholder is not evidence",
        "visual_acceptance": "FAIL until independent five-view scene review",
        "historical_fail": "PR90 removed erroneous surfaces but broke valid bridges; PR88 new-only sensor-valid pixels 14.60% >30cm wrong",
    }
    (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"mesh_sha256": report["new_mesh_sha256"],
            "faces": len(result.faces), "support": support}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.old, a.source_root, a.target)))
