"""Posthoc real 3+2 Mesh comparison: frozen DA2 baseline versus DA2 RGB consensus.

This script runs *after* candidate reconstruction. Baseline Mesh never enters
inference or fusion. It refuses a changed source archive or training split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh

import tum_room_knownpose_multiview as base
from echo4d.model import DataError, file_sha256
from echo4d.holdout import read_echo_ply
from tum_graph_visual import read_vertex_rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--current", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    args = p.parse_args()
    previous = json.loads((args.previous / "report.json").read_text())
    current_path = args.current / "report.json"
    current = json.loads(current_path.read_text())
    old_mesh = args.previous / "multiview-mesh" / "mesh.ply"
    new_mesh = args.current / "multiview-mesh" / "mesh.ply"
    if (file_sha256(old_mesh) != previous["new_mesh"]["sha256"]
            or file_sha256(new_mesh) != current["new_mesh"]["sha256"]
            or previous["source_archive_sha256"] != current["source_archive_sha256"]
            or previous["selection"]["expanded_indices"]
               != current["selection"]["expanded_indices"]):
        raise DataError("Frozen DA2 A/B source, split or mesh checksum mismatch")
    high = json.loads((args.baseline / "room-sparse-quality-report.json").read_text())
    current["visual_da2_vs_consensus"] = base.compare_multiview_five(
        args.work / "room-import", old_mesh, new_mesh, high,
        args.current / "visual-da2-vs-consensus",
        "previous DA2 Small known-pose RGB / no sensor Z")
    vertices, faces = read_echo_ply(old_mesh)
    colors = read_vertex_rgb(old_mesh, len(vertices))
    glb = args.current / "old-da2-rgb.glb"
    trimesh.Trimesh(vertices=vertices, faces=faces,
                    vertex_colors=colors, process=False).export(glb)
    current["previous_da2_control"] = {
        "mesh_sha256": file_sha256(old_mesh),
        "glb_sha256": file_sha256(glb),
        "role": "posthoc 3+2 A/B only, not reconstruction input",
        "tested_checkout": previous.get("checkout_sha")}
    current["issue_74"] = "FAIL / NOT ACCEPTED; 3+2 human review required"
    current_path.write_text(json.dumps(current, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"old_faces": previous["new_mesh"]["faces"],
                      "new_faces": current["new_mesh"]["faces"],
                      "views": current["visual_da2_vs_consensus"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
