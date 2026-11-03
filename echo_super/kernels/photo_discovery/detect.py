"""Stage 1 — open-vocabulary detection (OWLv2) + box convergence.

Ported from experiments/discovery_photo/detect2.py (round 2), which is
the converged form rounds 2-3 ran with:

  * dark frames are gamma-lifted before detection (box coordinates are
    on the original frame and do not move);
  * BOX CONVERGENCE — per class per frame, every raw box from every
    query of that class is clustered greedily by score (a box joins the
    seed's cluster when IoU >= converge_iou OR containment >=
    converge_containment — that is what multi-scale duplicates of one
    object look like); each cluster contributes only its best box and
    at most max_clusters_per_class clusters survive. Round 1 kept up
    to 8 boxes/frame of one class and SAM merged screen+desk+wall into
    a wall-sized blob; convergence is what fixed that.

The heavy imports (torch / transformers) are lazy: importing this
module is free, and callers that inject cached detections never pay
for the model. Detection records are plain dicts:
{frame, class, query, score, box, cluster_n, cluster_queries}.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import DiscoveryConfig


def gamma_lift(img: Image.Image, gamma: float = 0.45) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.clip(arr ** gamma, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_iou(a, b):
    inter = _inter(a, b)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_containment(a, b):
    inter = _inter(a, b)
    amin = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / amin if amin > 0 else 0.0


def converge_boxes(dets, cfg: DiscoveryConfig):
    """Cluster one class's raw boxes of one frame; return at most
    cfg.max_clusters_per_class representatives (best box per cluster),
    in descending seed-score order — that order is the cluster rank
    the later stages rely on."""
    pool = sorted(dets, key=lambda d: -d["score"])
    clusters = []
    while pool:
        seed = pool.pop(0)
        members = [seed]
        rest = []
        for d in pool:
            if (box_iou(seed["box"], d["box"]) >= cfg.converge_iou
                    or box_containment(seed["box"], d["box"])
                    >= cfg.converge_containment):
                members.append(d)
            else:
                rest.append(d)
        pool = rest
        rep = dict(seed)
        rep["cluster_n"] = len(members)
        rep["cluster_queries"] = sorted({m["query"] for m in members})
        clusters.append(rep)
        if len(clusters) == cfg.max_clusters_per_class:
            break
    return clusters


def owlv2_detect(frames_dir, stems, catalog, cfg: DiscoveryConfig,
                 log=print):
    """Run OWLv2 over the given frame stems and return the converged
    detection records. Heavy: loads the model on first call."""
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    torch.set_num_threads(4)
    qmap = []
    for cls, spec in catalog.items():
        for q in spec["queries"]:
            qmap.append((cls, q))
    queries = [q for _, q in qmap]

    proc = Owlv2Processor.from_pretrained(cfg.owl_model)
    model = Owlv2ForObjectDetection.from_pretrained(cfg.owl_model)
    model.eval()

    results = []
    for stem in stems:
        img = Image.open(frames_dir / f"{stem}.jpg").convert("RGB")
        lifted = gamma_lift(img, cfg.gamma)
        inputs = proc(text=[queries], images=lifted, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        det = proc.post_process_grounded_object_detection(
            out, target_sizes=torch.tensor([img.size[::-1]]),
            threshold=cfg.detect_threshold)[0]
        raw = []
        for score, label, box in zip(det["scores"], det["labels"],
                                     det["boxes"]):
            cls, q = qmap[int(label)]
            raw.append({"frame": stem, "class": cls, "query": q,
                        "score": float(score),
                        "box": [float(v) for v in box.tolist()]})
        kept = []
        for cls in catalog:
            sub = [d for d in raw if d["class"] == cls]
            if sub:
                kept.extend(converge_boxes(sub, cfg))
        kept.sort(key=lambda d: -d["score"])
        results.extend(kept)
        log(f"detect {stem}: raw={len(raw)} kept={len(kept)}")
    return results
