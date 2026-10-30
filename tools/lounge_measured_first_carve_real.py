"""Real Lounge measured-first fusion: remove duplicate PRIOR surfaces near measured geometry.

The supplied lounge.ply is pre-existing scan geometry (PRIOR), never independently
recovered. Compare 144 measured RGB-D frames against the old prior-first overlay
and two measured-first scan-carving distances on identical 3+2 views.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from PIL import Image

from echo4d.model import Camera, DataError
from lounge_spatial_prior_real import (VOXEL_M, prior_assisted, read_frames,
                                       select_views)
from tum_xyz_open3d_320 import export_glb, free_view, fuse, panel, render


def carve_scan_around_measurements(o3d, measured, source_scan, radius_m):
    """Keep all measured triangles; use existing scan ONLY where RGB-D is absent.

    Scan triangles touching a measured vertex within radius_m are removed.
    This deliberately prioritizes measured surfaces and avoids the old overlay
    that kept the entire scan and appended outlying measured fragments.
    """
    from scipy.spatial import cKDTree

    scan = o3d.geometry.TriangleMesh(source_scan)
    if len(scan.triangles) > 350_000:
        scan = scan.simplify_quadric_decimation(350_000)
    scan.remove_unreferenced_vertices()
    p = np.asarray(scan.vertices)
    m = np.asarray(measured.vertices)
    if not len(p) or not len(m):
        raise DataError("Empty measured or supplied scan mesh")
    m_to_p, _ = cKDTree(p).query(m, workers=2)
    alignment = float(np.median(m_to_p))
    if not np.isfinite(alignment) or alignment > .40:
        raise DataError(f"Scan/measurements misaligned: {alignment:.3f}m")

    # Scan color is explicitly inherited from nearby measured RGB-D vertices.
    # Unseen scan regions stay neutral, rather than hallucinating RGB colors.
    p_to_m, nearest = cKDTree(m).query(p, workers=2)
    colors = np.tile([.58, .59, .60], (len(p), 1))
    if measured.has_vertex_colors():
        near = p_to_m < .12
        colors[near] = np.asarray(measured.vertex_colors)[nearest[near]]
    scan.vertex_colors = o3d.utility.Vector3dVector(colors)

    triangles = np.asarray(scan.triangles)
    # Drop triangles touching the measured surface. The 2cm TSDF surface
    # replaces them; the remaining scan fills areas with no measured support.
    keep = np.all(p_to_m[triangles] > radius_m, axis=1)
    # Avoid long scan triangles spanning a hole boundary after carving.
    xyz = p[triangles]
    e01 = np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1)
    e12 = np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1)
    e20 = np.linalg.norm(xyz[:, 2] - xyz[:, 0], axis=1)
    keep &= np.maximum(np.maximum(e01, e12), e20) < .20

    # Preserve per-vertex colors and exact origin of each triangle.
    scan.triangles = o3d.utility.Vector3iVector(triangles[keep])
    scan.remove_unreferenced_vertices()
    hybrid = o3d.geometry.TriangleMesh(measured) + scan
    origin = o3d.geometry.TriangleMesh(hybrid)
    provenance = np.empty((len(hybrid.vertices), 3))
    provenance[:len(measured.vertices)] = [.10, .85, .35]  # measured RGB-D
    provenance[len(measured.vertices):] = [.95, .43, .13]  # existing scan
    origin.vertex_colors = o3d.utility.Vector3dVector(provenance)
    return hybrid, origin, {
        "radius_m": radius_m,
        "measured_to_scan_median_m": alignment,
        "measured_triangles": len(measured.triangles),
        "retained_prior_triangles": len(scan.triangles),
        "hybrid_triangles": len(hybrid.triangles),
        "source": "MEASURED + existing scan PRIOR; no independent recovery claim",
    }


def experiment(work, output, receipt):
    import open3d as o3d

    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    data = o3d.data.LoungeRGBDImages(data_root=str(work))
    params = o3d.io.read_pinhole_camera_trajectory(
        data.trajectory_log_path).parameters
    _, selected, poses, pose_indices = select_views(params, data.color_paths)
    intr = params[pose_indices[0]].intrinsic
    k = intr.intrinsic_matrix
    full = Camera(intr.width, intr.height, float(k[0, 0]), float(k[1, 1]),
                  float(k[0, 2]), float(k[1, 2]))
    if full.width % 2 or full.height % 2:
        raise DataError("Even RGB-D image dimensions required")
    half = full.scaled(2)
    for i in selected:
        q = params[pose_indices[i]].intrinsic
        if (q.width != full.width or q.height != full.height
                or not np.allclose(q.intrinsic_matrix, k, atol=1e-3)):
            raise DataError("Variable camera intrinsics unsupported")

    frames, originals, photos = read_frames(data, selected, poses, full, half)
    measured, _ = fuse(o3d, [frames[i] for i in selected], half,
                       pixel_stride=1, voxel=VOXEL_M)
    scan = o3d.io.read_triangle_mesh(data.reconstruction_path)
    _, old, _, old_counts = prior_assisted(
        o3d, measured, o3d.geometry.TriangleMesh(scan))
    carved06, origin06, counts06 = carve_scan_around_measurements(
        o3d, measured, scan, .06)
    carved12, origin12, counts12 = carve_scan_around_measurements(
        o3d, measured, scan, .12)

    for filename, mesh in (
        ("MEASURED-144-only.glb", measured),
        ("OLD-PRIOR-first-overlay.glb", old),
        ("NEW-MEASURED-first-6cm.glb", carved06),
        ("NEW-MEASURED-first-12cm.glb", carved12),
        ("ORIGIN-6cm-green-MEASURED-orange-PRIOR.glb", origin06),
        ("ORIGIN-12cm-green-MEASURED-orange-PRIOR.glb", origin12),
    ):
        export_glb(mesh, output / filename)

    boards, views = [], []
    labels = ["REAL RGB / FREE", "MEASURED 144", "OLD PRIOR-FIRST",
              "NEW CARVE 6cm", "NEW CARVE 12cm"]
    for i in photos:
        Image.fromarray(originals[i]).save(output / f"REAL-lounge-{i:04d}.png")
        photo = np.asarray(Image.fromarray(originals[i]).resize(
            (half.width, half.height), Image.Resampling.LANCZOS))
        imgs = [photo] + [render(o3d, mesh, half, poses[i])
                          for mesh in (measured, old, carved06, carved12)]
        name = f"recorded-{i:04d}.png"
        boards.append(panel(output / name, imgs, labels))
        views.append({"kind": "recorded", "frame": i, "file": name})

    for i, side in ((photos[0], -1.), (photos[-1], 1.)):
        pose = free_view(poses[i], side)
        imgs = [render(o3d, mesh, half, pose)
                for mesh in (measured, old, carved06, carved12)]
        name = f"free-{i:04d}.png"
        boards.append(panel(output / name,
                            [np.full_like(imgs[0], 20)] + imgs, labels))
        views.append({"kind": "free", "frame": i, "file": name})

    board = Image.new("RGB", (half.width * 5, (half.height + 27) * 5))
    for j, view in enumerate(boards):
        board.paste(view, (0, j * (half.height + 27)))
    board.save(output / "lounge-five-view-measured-first-carve.png")
    receipt.update(
        status="real_five_view_exported_review_pending",
        original_resolution=[full.width, full.height],
        fusion_resolution=[half.width, half.height],
        measured_frames=len(selected), photos=photos, views=views,
        measured_triangles=len(measured.triangles),
        old_prior_first=old_counts, new_6cm=counts06, new_12cm=counts12,
        visual_verdict="PENDING HUMAN REVIEW; triangle counts are not quality",
        free_orbit="FAIL until human-reviewed",
        issue_74="FAIL unchanged",
        old_desk="FAIL 16.7779cm > 15cm unchanged",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    receipt = {
        "scene": "Stanford Lounge real RGB-D",
        "source": "Open3D LoungeRGBDImages: real depth, poses, supplied PRIOR scan",
        "status": "running",
        "run_url": (
            f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
            f"{os.getenv('GITHUB_RUN_ID')}"
            if os.getenv("GITHUB_RUN_ID") else None),
    }
    try:
        experiment(args.work, args.output, receipt)
        code = 0
    except Exception as exc:
        traceback.print_exc()
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": receipt["status"],
                          "error": receipt.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
