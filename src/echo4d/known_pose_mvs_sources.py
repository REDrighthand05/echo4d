"""Geometry-only source selection for known-pose RGB plane-sweep MVS.

Unlike rectified stereo, MVS does not require horizontal epipolar geometry.
Only supplied TRAIN camera poses and intrinsics determine eligibility; no RGB,
measured Z, holdout observations, or reconstruction scores enter selection.
"""
from __future__ import annotations

import math
import numpy as np
from echo4d.model import Camera, DataError, rigid_matrix


def select_plane_sweep_views(frames: list[dict], camera: Camera,
                             training_indices: list[int], *,
                             max_anchors: int = 20,
                             max_sources: int = 4) -> tuple[list[tuple[int, list[int]]], dict]:
    """Choose diverse, overlapping source cameras for each bounded train anchor.

    The source ranking uses a 3x3 reference-image grid projected at fixed 2m
    camera-Z (not a depth estimate). Opposite viewing directions are preferred
    when available, not required. Sources need 3.5--30cm metric baselines and
    <=35 degree relative rotation.
    """
    if (type(max_anchors) is not int or not 1 <= max_anchors <= 48
            or type(max_sources) is not int or not 2 <= max_sources <= 4):
        raise DataError('MVS source/anchor budget outside bounded range')
    if (not training_indices or len(set(training_indices)) != len(training_indices)
            or any(type(i) is not int or i < 0 or i >= len(frames)
                   for i in training_indices)):
        raise DataError('MVS requires distinct valid training indices')
    training = sorted(training_indices)
    poses = {i: rigid_matrix(np.asarray(frames[i]['pose']).flatten(order='F'))
             for i in training}
    k = np.array([[camera.fx, 0., camera.cx],
                  [0., camera.fy, camera.cy], [0., 0., 1.]], np.float64)
    # Nine rays are a deterministic geometry-only overlap proxy, not scene Z.
    xs = np.linspace(.15 * (camera.width - 1), .85 * (camera.width - 1), 3)
    ys = np.linspace(.15 * (camera.height - 1), .85 * (camera.height - 1), 3)
    xx, yy = np.meshgrid(xs, ys)
    uv = np.stack((xx.ravel(), yy.ravel(), np.ones(9)), axis=0)
    rays = np.linalg.solve(k, uv) * 2.0
    choices = {}
    eligible_counts = {}
    for i in training:
        a = poses[i]
        options = []
        for j in training:
            if j == i:
                continue
            b = poses[j]
            r = b[:3, :3].T @ a[:3, :3]
            t = b[:3, :3].T @ (a[:3, 3] - b[:3, 3])
            baseline = float(np.linalg.norm(t))
            angle = math.acos(float(np.clip((np.trace(r) - 1.) / 2., -1., 1.)))
            if not (0.035 <= baseline <= 0.30 and angle <= math.radians(35)):
                continue
            projected = r @ rays + t[:, None]
            z = projected[2]
            safe_z = np.maximum(z, 1e-6)
            pu = camera.fx * projected[0] / safe_z + camera.cx
            pv = camera.fy * projected[1] / safe_z + camera.cy
            inside = ((z > 0) & (pu >= 2) & (pu < camera.width - 3)
                      & (pv >= 2) & (pv < camera.height - 3))
            overlap = float(np.mean(inside))
            if overlap < 5 / 9:
                continue
            flow = np.array([pu[4] - uv[0, 4], pv[4] - uv[1, 4]])
            length = float(np.linalg.norm(flow))
            if not np.isfinite(length) or length < 1:
                continue
            direction = flow / length
            # Fixed pose-only ranking; no heldout or source pixel scores.
            score = abs(baseline - .12) + .10 * (1. - overlap) + .015 * angle
            options.append((score, j, direction))
        eligible_counts[i] = len(options)
        if len(options) < 2:
            continue
        selected = []
        while options and len(selected) < max_sources:
            best = min(options, key=lambda c: (
                c[0] + (.055 * max(0., max(float(np.dot(c[2], d))
                                                for _, d in selected))
                        if selected else 0.), c[1]))
            selected.append((best[1], best[2]))
            options = [candidate for candidate in options if candidate[1] != best[1]]
        choices[i] = [j for j, _ in selected]
    anchors = sorted(choices)
    if len(anchors) > max_anchors:
        positions = np.linspace(0, len(anchors) - 1, max_anchors, dtype=int)
        anchors = [anchors[int(p)] for p in positions]
    selected = [(i, choices[i]) for i in anchors]
    return selected, {
        'selector': 'known-pose geometry-only 2m grid overlap and diverse flow',
        'eligible_training_anchors': len(choices),
        'selected_training_anchors': len(selected),
        'selected_source_counts': [len(js) for _, js in selected],
        'eligible_source_counts_for_selected': [eligible_counts[i] for i in anchors],
        'max_anchors': max_anchors,
        'max_sources': max_sources,
        'train_only': True,
    }
