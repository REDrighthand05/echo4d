"""Global RGB appearance retrieval + verified RGB-D loop edges for TUM fr1_room.

A narrow adapter over the existing measured-RGB-D pose ladder. No unknown GT
poses or prior Mesh are used for retrieval, edge estimation or candidate fusion.
Known GT poses outside declared anchors are used only by the inherited posthoc
diagnostic. Scale comes from measured depth, NOT RGB-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import tum_room_rgbd_pose_ladder as base
from echo4d.model import file_sha256


def global_candidates(features, frame_count, exclude_gap=6, per_frame=10, clusters=48):
    """Deterministic OpenCV SIFT bag-of-visual-words with TF-IDF cosine retrieval."""
    descriptors = [des for _, des in features]
    pool = [d[np.linspace(0, len(d) - 1, min(110, len(d)), dtype=int)]
            for d in descriptors if d is not None and len(d) >= 12]
    if len(pool) < 2:
        return [], {"vocabulary_words": 0, "proposed_nonlocal_pairs": 0}
    samples = np.ascontiguousarray(np.vstack(pool), dtype=np.float32)
    words = min(clusters, len(samples) // 10)
    if words < 4:
        return [], {"vocabulary_words": 0, "proposed_nonlocal_pairs": 0}
    cv2.setRNGSeed(20261029)
    _, _, centers = cv2.kmeans(
        samples, words, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 12, .1),
        1, cv2.KMEANS_PP_CENTERS)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    hist = np.zeros((frame_count, words), np.float32)
    for i, des in enumerate(descriptors):
        if des is None or len(des) == 0:
            continue
        assigned = matcher.match(np.ascontiguousarray(des, np.float32), centers)
        hist[i] = np.bincount([m.trainIdx for m in assigned],
                              minlength=words).astype(np.float32)
    # Images with common, uninformative texture should not dominate retrieval.
    document_frequency = (hist > 0).sum(axis=0)
    hist *= (np.log((frame_count + 1) / (document_frequency + 1)) + 1)[None, :]
    hist /= np.maximum(np.linalg.norm(hist, axis=1, keepdims=True), 1e-9)
    similarity = hist @ hist.T
    pairs = set()
    for i in range(frame_count):
        order = np.argsort(-similarity[i], kind="stable")
        near = [int(j) for j in order if abs(int(j) - i) > exclude_gap]
        far = [j for j in near if abs(j - i) >= 20]
        for j in near[:per_frame] + far[:3]:
            if similarity[i, j] > 0:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs), {"vocabulary_words": int(words),
                           "proposed_nonlocal_pairs": len(pairs),
                           "retrieval": "OpenCV SIFT visual vocabulary, TF-IDF cosine, RGB only"}


def connected_sizes(graph):
    seen = set()
    sizes = []
    for root in range(len(graph)):
        if root in seen:
            continue
        todo = [root]
        seen.add(root)
        size = 0
        while todo:
            i = todo.pop()
            size += 1
            for j, _, _ in graph[i]:
                if j not in seen:
                    seen.add(j)
                    todo.append(j)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def checked_loop_edge(features, depths, K, i, j, matcher):
    f = base.pair_pnp(*features[i], *features[j], depths[i], K, matcher)
    r = base.pair_pnp(*features[j], *features[i], depths[j], K, matcher)
    if f is None and r is None:
        return None
    if f is not None and r is not None:
        cycle = r[0] @ f[0]
        degrees = np.degrees(np.arccos(np.clip((np.trace(cycle[:3, :3]) - 1) / 2, -1, 1)))
        if np.linalg.norm(cycle[:3, 3]) > .35 or degrees > 18:
            return None
    else:
        one = f if f is not None else r
        if one[1]["inliers"] < 25 or one[1]["median_reprojection_px"] > 1.5:
            return None
    if r is not None and (f is None or r[1]["inliers"] > f[1]["inliers"]):
        return np.linalg.inv(r[0]), r[1]
    return f


LAST_GRAPH_REPORT = {}


def retrieval_pose_graph(colors, depths, K, max_gap=6):
    # Keep the already-working local RGB-D edges intact.
    graph, local = base_graph_builder(colors, depths, K, max_gap=max_gap)
    before = connected_sizes(graph)
    sift = cv2.SIFT_create(nfeatures=3000)
    features = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None)
                for im in colors]
    proposals, meta = global_candidates(features, len(colors), exclude_gap=max_gap)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    accepted = 0
    for i, j in proposals:
        edge = checked_loop_edge(features, depths, K, i, j, matcher)
        if edge is None:
            continue
        T, quality = edge
        cost = 1.0 + 25.0 / quality["inliers"] + quality["median_reprojection_px"] / 4.0
        graph[i].append((j, T, cost))
        graph[j].append((i, np.linalg.inv(T), cost))
        accepted += 1
    after = connected_sizes(graph)
    result = {**local, **meta, "accepted_nonlocal_pairs": accepted,
              "connected_sizes_before": before, "connected_sizes_after": after,
              "nonlocal_pair_verification": "metric RGB-D SIFT-PnP with bidirectional cycle gate"}
    LAST_GRAPH_REPORT.clear()
    LAST_GRAPH_REPORT.update(result)
    return graph, result


base_graph_builder = base.build_pose_graph


def relabel_visual_board(path, left, right):
    """Correct inherited generic DA2 captions for the actual pose experiment."""
    with Image.open(path) as original:
        board = original.convert("RGB")
    draw = ImageDraw.Draw(board)
    for row in range(5):
        y = row * 480
        draw.rectangle((640, y, 1919, y + 28), fill=(12, 12, 12))
        draw.text((648, y + 7), left, fill="white")
        draw.text((1288, y + 7), right, fill="white")
    board.save(path)


def run(source, room96, old, output, work, archive=None):
    # Scope this replacement strictly to one call; the original algorithm
    # and its independent fair TSDF/GLB/visual implementation remain unchanged.
    base.build_pose_graph = retrieval_pose_graph
    try:
        report = base.run(source, room96, output, work, archive_override=archive,
                          levels=(5, 1), max_gap=6)
    finally:
        base.build_pose_graph = base_graph_builder
    frozen, replay, _ = base.checked_replay(Path(source))
    previous = Path(old)
    for count in (5, 1):
        old_mesh = previous / f"mesh-poses-{count}/mesh.ply"
        new_mesh = Path(output) / f"mesh-poses-{count}/mesh.ply"
        if not old_mesh.is_file():
            raise FileNotFoundError(f"Prior real pose ablation Mesh missing: {old_mesh}")
        visual = Path(output) / f"visual-old-new-poses-{count}"
        visual.mkdir()
        report["variants"][str(count)]["old_new_five_views"] = base.five_view_comparison(
            Path(source), replay, old_mesh, new_mesh, visual, frozen)
        relabel_visual_board(visual / "source-old-new-3plus2.png",
                             f"OLD: {count} known poses / local RGB-D PnP",
                             f"NEW: {count} known poses / global RGB-D loop")
        relabel_visual_board(Path(output) / f"visual-poses-{count}/source-old-new-3plus2.png",
                             "REFERENCE: 96 known poses / measured depth",
                             f"NEW: {count} known poses / global RGB-D loop")
        report["variants"][str(count)]["old_mesh_sha256"] = file_sha256(old_mesh)
        report["variants"][str(count)]["old_new_board_sha256"] = file_sha256(
            visual / "source-old-new-3plus2.png")
    report["prior_run"] = "https://github.com/REDrighthand05/echo4d/actions/runs/40847424620"
    report["visual_acceptance"] = "UNREVIEWED; real 3+2 images generated, #74 not PASS"
    report["no_prior_mesh_in_candidate"] = True
    report["ground_truth_pose_only_posthoc_except_declared_anchors"] = True
    (Path(output) / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--room96", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive")
    args = parser.parse_args()
    report = run(args.source, args.room96, args.old, args.output, args.work, args.archive)
    print(json.dumps({"graph": report["pose_graph"],
                      "variants": {k: {"estimated": v["estimated_pose_frames"],
                                       "unresolved": v["unresolved_pose_frames"],
                                       "faces": v["faces"],
                                       "median_camera_error_m": v["withheld_gt_translation_error_median_m"]}
                                   for k, v in report["variants"].items()},
                      "visual": report["visual_acceptance"]}, indent=2))


if __name__ == "__main__":
    main()
