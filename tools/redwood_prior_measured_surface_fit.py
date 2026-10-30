"""Fit aligned existing Redwood scan PRIOR to independently measured RGB-D surfaces.

Moves only prior vertices supported by consistent nearby measured observations.
Unknown areas stay prior. Not an independent video-only reconstruction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def fit_prior_to_measured(prior, observed, source_mask, *, radius=.045,
                          max_displacement=.025, blend=.75):
    import trimesh
    p = np.asarray(prior.vertices, dtype=np.float64)
    q = np.asarray(observed.vertices, dtype=np.float64)
    mask = np.asarray(source_mask, dtype=bool)
    if (p.ndim != 2 or p.shape[1] != 3 or q.ndim != 2 or q.shape[1] != 3
            or len(q) < 6 or mask.shape != (len(p),)
            or not np.isfinite(p).all() or not np.isfinite(q).all()
            or not .02 <= radius <= .15 or not .001 <= max_displacement <= .05
            or not 0 < blend <= 1):
        raise ValueError("Invalid prior, observed geometry or fit settings")
    normals = np.asarray(prior.vertex_normals, dtype=np.float64)
    measured_normals = np.asarray(observed.vertex_normals, dtype=np.float64)
    if normals.shape != p.shape or measured_normals.shape != q.shape:
        raise ValueError("Missing vertex normals")
    tree = cKDTree(q)
    delta = np.zeros(len(p), dtype=np.float64)
    valid = np.zeros(len(p), dtype=bool)
    ids_to_fit = np.flatnonzero(mask)
    for start in range(0, len(ids_to_fit), 40000):
        chunk = ids_to_fit[start:start + 40000]
        distances, ids = tree.query(p[chunk], k=6, workers=2)
        neighbors = q[ids]
        norm = normals[chunk]
        normal_agreement = np.abs(np.einsum(
            "ij,ikj->ik", norm, measured_normals[ids]))
        good = (distances <= radius) & (normal_agreement >= .65)
        count = good.sum(axis=1)
        mean = np.sum(neighbors * good[:, :, None], axis=1) / np.maximum(count[:, None], 1)
        signed = np.einsum("ij,ij->i", mean - p[chunk], norm)
        plane_spread = np.max(np.where(
            good, np.abs(np.einsum(
                "ikj,ij->ik", neighbors - mean[:, None, :], norm)), 0), axis=1)
        ok = ((count >= 4) & (plane_spread <= .012)
              & (np.abs(signed) <= max_displacement)
              & (np.abs(signed) >= .001))
        valid[chunk] = ok
        delta[chunk] = np.where(ok, signed * blend, 0)
    fitted = trimesh.Trimesh(
        vertices=p + normals * delta[:, None],
        faces=prior.faces.copy(),
        vertex_colors=prior.visual.vertex_colors, process=False)
    return fitted, {
        "eligible_prior_vertices": int(mask.sum()),
        "moved_prior_vertices": int(valid.sum()),
        "max_displacement_m": float(np.max(np.abs(delta))),
        "median_displacement_m": float(np.median(np.abs(delta[valid]))) if valid.any() else 0.,
        "prior_geometry_source": "PRE-EXISTING aligned Redwood scan",
        "surface_fit_source": "600-frame measured RGB-D-only reconstruction",
        "independent_video_reconstruction": False,
    }


def run(prior_glb, provenance_glb, measured_glb, output):
    import trimesh
    prior = trimesh.load(prior_glb, force="mesh", process=False)
    provenance = trimesh.load(provenance_glb, force="mesh", process=False)
    observed = trimesh.load(measured_glb, force="mesh", process=False)
    if (len(prior.vertices) != len(provenance.vertices)
            or not np.allclose(prior.vertices, provenance.vertices)
            or not np.array_equal(prior.faces, provenance.faces)):
        raise ValueError("Provenance does not match aligned prior")
    rgb = provenance.visual.vertex_colors[:, :3]
    # Green is measured-only geometry; cyan and orange are scan PRIOR.
    prior_mask = ~np.all(rgb == [76, 242, 89], axis=1)
    if prior_mask.sum() < 1000 or (~prior_mask).sum() < 100:
        raise ValueError("Missing explicit source provenance")
    fitted, report = fit_prior_to_measured(prior, observed, prior_mask)
    output.mkdir(parents=True, exist_ok=False)
    fitted.export(output / "PRIOR-surface-fitted-to-measured.glb")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-glb", type=Path, required=True)
    parser.add_argument("--provenance-glb", type=Path, required=True)
    parser.add_argument("--measured-glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.prior_glb, args.provenance_glb, args.measured_glb, args.output)
