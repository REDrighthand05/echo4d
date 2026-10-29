"""Fixed-budget, measured-depth coverage selection for real Redwood Bedroom.

Compare 600 frames chosen from actual sensor-surface coverage against the
same 600 uniformly spaced frames at 4.5m.  Both meshes are observed-only.
The frozen 600-frame Mesh is used ONLY for A/B visualization, not selection,
fusion, geometry inference, or pose alignment.
"""
import argparse
from collections import defaultdict
import heapq
import json
from pathlib import Path

import numpy as np
from PIL import Image

from open3d_official_room_baseline import read_poses, spaced_real_frames
from redwood_dense_native_review import export_full
from redwood_gap_guided_fusion import compare, integrate

BASELINE_FACES = 553208
FRAME_BUDGET = 600
DEPTH_LIMIT_M = 4.5
GRID_M = 0.10
SAMPLE_STEP = 16
ANCHORS = 60


def measured_surface_cells(depth, world_to_camera, k, step=SAMPLE_STEP, cell_m=GRID_M):
    """Coarse real depth samples, transformed into metric world-space cells."""
    w, h, fx, fy, cx, cy = k
    zraw = np.asarray(depth)
    pose = np.asarray(world_to_camera, dtype=np.float64)
    if (zraw.shape != (h, w) or zraw.dtype != np.uint16 or
            pose.shape != (4, 4) or not np.isfinite(pose).all() or
            not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5) or
            not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-3) or
            not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0 or
            step < 1 or not np.isfinite(cell_m) or cell_m <= 0):
        raise ValueError("Invalid original measured depth, calibration or pose")
    v, u = np.mgrid[step//2:h:step, step//2:w:step]
    z = zraw[v, u].astype(np.float64) / 1000.
    valid = (z > .25) & (z < DEPTH_LIMIT_M)
    if not np.any(valid):
        return set()
    zz, uu, vv = z[valid], u[valid], v[valid]
    cam = np.column_stack(((uu - cx)*zz/fx, (vv - cy)*zz/fy, zz))
    camera_to_world = np.linalg.inv(pose)
    xyz = cam @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    cells = np.floor(xyz/cell_m).astype(np.int64)
    if np.any(np.abs(cells) >= 32000):
        raise ValueError("World-space depth samples exceed local metric room grid")
    packed = cells + 32768
    keys = (packed[:, 0] << 32) | (packed[:, 1] << 16) | packed[:, 2]
    return set(np.unique(keys).tolist())


def select_coverage_frames(cells_by_frame, budget=FRAME_BUDGET, anchors=ANCHORS):
    """Deterministic lazy greedy 2-view surface coverage, fixed total frame count.

    An unseen 10cm cell contributes 1; a cell seen by only one selected frame
    contributes 0.35; already multiply supported cells contribute 0.
    This selects measured surfaces, not the prior Mesh or a visual score.
    """
    n = len(cells_by_frame)
    if not (3 <= budget <= min(n, 600) and 3 <= anchors <= budget):
        raise ValueError("Invalid fixed frame budget or uniform anchor count")
    seeds = spaced_real_frames(n, anchors)
    chosen = set(seeds)
    counts = defaultdict(int)

    def add(i):
        for cell in cells_by_frame[i]:
            counts[cell] = min(2, counts[cell] + 1)

    for i in seeds:
        add(i)

    def gain(i):
        return sum(1. if counts[c] == 0 else .35 if counts[c] == 1 else 0.
                   for c in cells_by_frame[i])

    # Lazy bounds are valid because the objective is monotone submodular.
    queue = [(-float(len(cells)), i) for i, cells in enumerate(cells_by_frame)
             if i not in chosen]
    heapq.heapify(queue)
    selected_greedy, gains = [], []
    while queue and len(chosen) < budget:
        upper, idx = heapq.heappop(queue)
        actual = gain(idx)
        if actual <= 0:
            # This candidate has no gain; another stale upper bound may.
            continue
        if queue and actual + 1e-8 < -queue[0][0]:
            heapq.heappush(queue, (-actual, idx))
            continue
        chosen.add(idx)
        selected_greedy.append(idx)
        gains.append(actual)
        add(idx)

    # Keep exactly the same frame count as the uniform baseline, even if
    # coarse sensor cells saturate; no invented or duplicate frames.
    before_fallback = len(chosen)
    for idx in spaced_real_frames(n, budget) + list(range(n)):
        if len(chosen) == budget:
            break
        chosen.add(idx)
    if len(chosen) != budget:
        raise RuntimeError("Could not select distinct real recorded frames")
    return sorted(chosen), {
        "uniform_anchor_count": len(seeds),
        "coverage_greedy_count": len(selected_greedy),
        "uniform_fallback_count": budget - before_fallback,
        "greedy_total_marginal_gain": round(sum(gains), 2),
        "selected_indices": sorted(chosen),
    }


def run(source, cache, out, budget):
    import open3d as o3d
    data = o3d.data.BedroomRGBDImages(data_root=str(cache))
    colors, depths = list(data.color_paths), list(data.depth_paths)
    if len(colors) != len(depths) or len(colors) < budget:
        raise ValueError("Need matching full original Redwood RGB-D sequence")
    if any(Path(c).stem != Path(d).stem for c, d in zip(colors, depths)):
        raise ValueError("Real RGB and depth frame identifiers disagree")
    trajectory = o3d.io.read_pinhole_camera_trajectory(data.trajectory_log_path)
    k, poses = read_poses(trajectory, colors, data.trajectory_log_path)
    if k[:2] != (640, 480):
        raise ValueError("Expected native 640x480 measured Redwood calibration")
    baseline = o3d.io.read_triangle_mesh(str(source))
    if not baseline.has_vertex_colors() or len(baseline.triangles) != BASELINE_FACES:
        raise ValueError("Incorrect frozen observed-only 4.5m baseline")

    print(f"Reading measured depth coverage from {len(depths)} original frames", flush=True)
    frame_cells = []
    for idx, path in enumerate(depths):
        frame_cells.append(measured_surface_cells(
            np.asarray(o3d.io.read_image(path)), poses[idx], k))
        if (idx + 1) % 300 == 0:
            print(f"Measured coverage: {idx+1}/{len(depths)}", flush=True)
    selected, selection = select_coverage_frames(frame_cells, budget,
                                                 anchors=min(ANCHORS, budget))
    baseline_indices = spaced_real_frames(len(colors), budget)
    print(f"Coverage-selected {len(selected)} real frames; "
          f"{len(set(selected)-set(baseline_indices))} differ from uniform", flush=True)
    del frame_cells

    out.mkdir(parents=True, exist_ok=False)
    candidate = integrate(o3d, data, selected, poses, k)
    if not o3d.io.write_triangle_mesh(str(out/"coverage-selected-observed.ply"), candidate):
        raise RuntimeError("Could not export native measured-depth Mesh")
    exported = export_full(candidate, out)
    (out/exported["glb"]).rename(out/"coverage-selected-observed.glb")
    views = compare(o3d, data, poses, k, baseline, candidate, out)
    board = Image.new("RGB", (960, 268*len(views)), (26, 26, 26))
    for j, entry in enumerate(views):
        with Image.open(out/entry["file"]) as picture:
            board.paste(picture.convert("RGB"), (0, 268*j))
    board.save(out/"five-view-coverage-ab.png")
    report = {
        "scene": "REAL Redwood Bedroom",
        "input": "original 640x480 RGB-D, metric measured depth and logged camera poses",
        "dataset": "Open3D BedroomRGBDImages / Redwood",
        "selection": "measured-depth 10cm world-cell coverage with 60 uniform anchors",
        "same_frame_budget": budget,
        "baseline_uniform_frames": len(baseline_indices),
        "different_frames_from_uniform": len(set(selected)-set(baseline_indices)),
        "baseline_4p5m_faces": len(baseline.triangles),
        "coverage_selected_faces": len(candidate.triangles),
        "baseline_4p5m_run": "https://github.com/REDrighthand05/echo4d/actions/runs/40778686280",
        "prior_mesh_used_for_selection_or_fusion": False,
        "old_mesh_used_only_for_AB_visuals": True,
        "preexisting_reference_scan_used": False,
        "voxel_m": .025, "depth_limit_m": DEPTH_LIMIT_M,
        "selection_detail": selection,
        "views": views,
        "colored_glb": "coverage-selected-observed.glb",
        "human_visual_acceptance": "PENDING",
        "issue_74": "FAIL unchanged",
        "old_desk": "FAIL 16.7779cm > 15cm unchanged",
    }
    (out/"result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="selection_detail"}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=FRAME_BUDGET)
    args = parser.parse_args()
    try:
        run(args.source_ply, args.cache, args.output, args.budget)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"FAILURE.txt").write_text(f"{type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
