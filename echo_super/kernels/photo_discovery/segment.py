"""Stage 2 — segmentation (SAM), one best mask per converged box.

Ported from experiments/discovery_photo/segment2.py: per frame, boxes
are picked round-robin across classes (each class's boxes in score
order, classes ordered by their best remaining score) under a budget
of cfg.sam_max_boxes_per_frame; SAM returns several candidate masks
per box and the one with the best predicted IoU is kept.

Difference from the experiment: masks stay in memory. The experiment
wrote PNGs because its runs were restarted mid-flight; the kernel
returns (index_records, {mask_file: bool ndarray}) and the lift stage
consumes either that or an injected on-disk cache of the same shape.
Index records: {frame, class, query, score, box, cluster_n,
cluster_queries, mask_file, sam_iou, mask_px}.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import DiscoveryConfig


def pick_frame_boxes(dets_of_frame, cfg: DiscoveryConfig):
    """Round-robin pick of at most cfg.sam_max_boxes_per_frame boxes
    from one frame's converged detections (segment2's rule, verbatim)."""
    by_cls = {}
    for d in dets_of_frame:
        by_cls.setdefault(d["class"], []).append(d)
    for v in by_cls.values():
        v.sort(key=lambda d: -d["score"])
    picked = []
    while len(picked) < cfg.sam_max_boxes_per_frame and any(by_cls.values()):
        for cls in sorted(
                by_cls,
                key=lambda c: -(by_cls[c][0]["score"] if by_cls[c] else 0)):
            if by_cls[cls] and len(picked) < cfg.sam_max_boxes_per_frame:
                picked.append(by_cls[cls].pop(0))
    return picked


def sam_segment(frames_dir, detections, cfg: DiscoveryConfig, log=print):
    """Run SAM over the detections; returns (index, arrays). Heavy:
    loads the model on first call."""
    import torch
    from transformers import SamModel, SamProcessor

    torch.set_num_threads(4)
    proc = SamProcessor.from_pretrained(cfg.sam_model)
    model = SamModel.from_pretrained(cfg.sam_model)
    model.eval()

    by_frame = {}
    for d in detections:
        by_frame.setdefault(d["frame"], []).append(d)

    index = []
    arrays = {}
    for stem, ds in sorted(by_frame.items()):
        picked = pick_frame_boxes(ds, cfg)
        if not picked:
            continue
        img = Image.open(frames_dir / f"{stem}.jpg").convert("RGB")
        boxes = [d["box"] for d in picked]
        inputs = proc(images=img, input_boxes=[boxes], return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        masks = proc.post_process_masks(
            out.pred_masks, inputs["original_sizes"],
            inputs["reshaped_input_sizes"])[0]
        ious = out.iou_scores[0]
        for i, d in enumerate(picked):
            j = int(torch.argmax(ious[i]))
            m = masks[i, j].numpy()
            fn = f"{stem}__{d['class']}__{i}.png"
            arrays[fn] = m
            index.append({**d, "mask_file": fn,
                          "sam_iou": float(ious[i, j]),
                          "mask_px": int(m.sum())})
        log(f"segment {stem}: {len(picked)} masks")
    return index, arrays
