"""Programmatic discovery: nomination -> measurement verdict -> inventory.

Architecture (project rule, user 2026-11-01/02):
  * NOMINATION rights belong to a learned open-vocabulary detector
    (OWLv2, fully offline). It may only say "something like X might
    be here" (a labelled 2D box in one frame).
  * VERDICT rights belong to deterministic measurement on the job's
    own fused point cloud: project the cloud into the nominating
    frame, keep the points inside the box, take their densest depth
    layer, and measure centre + extents. A nomination that cannot be
    measured is discarded (or listed as unverified, never as fact).
  * An inventory entry needs corroboration: the same object measured
    from >= 2 distinct frames at a consistent position ("已证"), or
    it is labelled "单见" (single sighting) and kept separate.

Everything is an estimate (evidence label ``估计``): box positions
come from a model, centres/sizes from an estimated-depth cloud.

CLI:
    python -m echo_scene.discovery nominate <job_dir> --frames a.jpg,b.jpg \
        --out nominations.json
    python -m echo_scene.discovery run <job_dir> --nominations nominations.json \
        [--scale 0.2761] [--out objects.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_QUERIES = [
    "electrical outlet panel", "power socket", "door", "laptop screen",
    "laptop", "bed", "desk", "bookshelf", "book", "bottle", "chair",
    "sofa", "table", "cabinet", "wardrobe", "television", "curtain",
    "window", "air conditioner", "lamp", "pillow", "backpack",
    "shoes", "cardboard box", "quilt",
]

# synonyms proven to be the same physical thing (identical boxes in
# pioneer G); everything else keeps its own label.
CANONICAL = {"power socket": "electrical outlet panel"}

LABELS_CN = {
    "electrical outlet panel": "开关/插座面板",
    "door": "房门",
    "laptop screen": "笔记本屏幕",
    "laptop": "笔记本电脑",
    "bed": "床",
    "desk": "书桌",
    "bookshelf": "书架",
    "book": "书",
    "bottle": "瓶子",
    "chair": "椅子",
    "sofa": "沙发",
    "table": "桌子",
    "cabinet": "柜子",
    "wardrobe": "衣柜",
    "television": "电视",
    "curtain": "窗帘",
    "window": "窗",
    "air conditioner": "空调",
    "lamp": "灯",
    "pillow": "枕头",
    "backpack": "背包",
    "shoes": "鞋",
    "cardboard box": "纸箱",
    "quilt": "被子",
}

MIN_SCORE = 0.15          # nominations below this never reach measurement
MIN_POINTS = 120          # measured points required inside the box layer
CLUSTER_DIST = 0.55       # scene units; same-object centre tolerance
LAYER_WIDTH = 0.9         # scene units; depth-layer thickness for one object


# ---------------------------------------------------------------- cloud io

def read_ply_points(path) -> np.ndarray:
    """Minimal binary PLY reader: vertex x/y/z -> (N,3) float64."""
    data = Path(path).read_bytes()
    head_end = data.index(b"end_header") + len(b"end_header")
    if data[head_end:head_end + 1] == b"\n":
        head_end += 1
    header = data[:head_end].decode("ascii", "replace")
    count, props = 0, []
    in_vertex = False
    for line in header.splitlines():
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count, in_vertex = int(parts[2]), True
            continue
        if parts and parts[0] == "element":
            in_vertex = False
        if in_vertex and parts and parts[0] == "property":
            props.append((parts[1], parts[2]))
    types = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
             "uchar": "u1", "uint8": "u1", "int": "i4", "int32": "i4",
             "uint": "u4", "short": "i2", "ushort": "u2"}
    dtype = np.dtype([(name, "<" + types[t]) for t, name in props])
    arr = np.frombuffer(data[head_end:head_end + count * dtype.itemsize],
                        dtype=dtype, count=count)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_job(job_dir):
    job_dir = Path(job_dir)
    poses = json.loads((job_dir / "poses.json").read_text())
    points = read_ply_points(job_dir / "scene_points.ply")
    return poses, points


# ---------------------------------------------------------------- measure

def locate(points, pose, box):
    """Measure one nominated box against the fused cloud.

    Returns dict(centre, size, n_points) in scene units, or None when
    the box carries too little measured geometry to be believed.
    """
    R = quat_to_matrix(pose["cam_from_world_quat_xyzw"])
    t = np.asarray(pose["cam_from_world_t"], dtype=np.float64)
    cam = points @ R.T + t                      # world -> camera
    z = cam[:, 2]
    front = z > 0.05
    if front.sum() < MIN_POINTS:
        return None
    cam_f, pts_f = cam[front], points[front]
    f, cx, cy = pose["camera"]["params"][:3]
    u = f * cam_f[:, 0] / cam_f[:, 2] + cx
    v = f * cam_f[:, 1] / cam_f[:, 2] + cy
    inside = ((u >= box["xmin"]) & (u <= box["xmax"])
              & (v >= box["ymin"]) & (v <= box["ymax"]))
    if inside.sum() < MIN_POINTS:
        return None
    sel_cam, sel_pts = cam_f[inside], pts_f[inside]
    # densest depth layer: the object's own surface, not the wall behind
    hist, edges = np.histogram(sel_cam[:, 2], bins=24)
    peak = int(np.argmax(hist))
    lo, hi = edges[peak], edges[peak] + LAYER_WIDTH
    layer = (sel_cam[:, 2] >= lo) & (sel_cam[:, 2] <= hi)
    if layer.sum() < MIN_POINTS:
        layer = np.ones(len(sel_cam), dtype=bool)
    obj = sel_pts[layer]
    centre = np.median(obj, axis=0)
    size = np.percentile(obj, 95, axis=0) - np.percentile(obj, 5, axis=0)
    return {"centre": centre, "size": size, "n_points": int(len(obj))}


def adjudicate(poses, points, nominations, scale_m=None):
    """Nominations -> measured instances -> corroborated inventory."""
    instances, unverified = [], []
    for frame, entry in sorted(nominations.items()):
        pose = poses.get(frame)
        if pose is None or "hits" not in entry:
            continue
        for hit in entry["hits"]:
            if hit["score"] < MIN_SCORE:
                continue
            label = CANONICAL.get(hit["label"], hit["label"])
            m = locate(points, pose, hit["box"])
            if m is None:
                unverified.append({"label": label, "frame": frame,
                                    "score": hit["score"],
                                    "reason": "框内测不到足够的点"})
                continue
            instances.append({"label": label, "frame": frame,
                               "score": hit["score"], **m})
    # corroboration: cluster same-label instances by centre proximity.
    # Large furniture is seen part-by-part from different frames, so the
    # centre tolerance grows with the measured size of the object.
    def same_object(a, b):
        big = max(float(np.max(a["size"])), float(np.max(b["size"])))
        return np.linalg.norm(a["centre"] - b["centre"]) <= max(
            0.5, 0.30 + 0.35 * big)

    entries = []
    used = [False] * len(instances)
    for i, inst in enumerate(instances):
        if used[i]:
            continue
        group = [inst]
        used[i] = True
        for j in range(i + 1, len(instances)):
            if used[j] or instances[j]["label"] != inst["label"]:
                continue
            if any(same_object(instances[j], g) for g in group):
                group.append(instances[j])
                used[j] = True
        frames = sorted({g["frame"] for g in group})
        centre = np.median([g["centre"] for g in group], axis=0)
        size = np.median([g["size"] for g in group], axis=0)
        e = {"label": inst["label"],
             "label_cn": LABELS_CN.get(inst["label"], inst["label"]),
             "tier": "已证" if len(frames) >= 2 else "单见",
             "frames": frames,
             "centre_units": [round(float(v), 3) for v in centre],
             "size_units": [round(float(v), 3) for v in size],
             "points": int(sum(g["n_points"] for g in group)),
             "evidence": "估计"}
        if scale_m:
            e["size_m"] = [round(float(v) * scale_m, 2) for v in size]
            e["centre_m"] = [round(float(v) * scale_m, 2) for v in centre]
        entries.append(e)
    entries.sort(key=lambda e: (e["tier"] != "已证",
                                -len(e["frames"]), e["label"]))
    return {"objects": entries, "unverified": unverified,
            "instances": len(instances)}


# ---------------------------------------------------------------- outputs

def write_outputs(job_dir, result, scale_m=None):
    job_dir = Path(job_dir)
    (job_dir / "objects.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2))
    lines = ["# 物品清单（程序发现：模型提名 → 测量裁决）", ""]
    if scale_m:
        lines.append(f"尺寸已按真尺度换算（1 单位 = {scale_m} 米，估计）。")
    else:
        lines.append("未定真尺度，尺寸为场景单位。")
    lines.append("")
    for tier, head in (("已证", "## 已证（≥2 帧量到同一位置）"),
                       ("单见", "## 单见（只量到一帧，先记下）")):
        group = [e for e in result["objects"] if e["tier"] == tier]
        if not group:
            continue
        lines += [head, ""]
        by_label = {}
        for e in group:
            by_label.setdefault(e["label_cn"], []).append(e)
        for cn, es in by_label.items():
            big = max(es, key=lambda e: float(np.prod(
                e.get("size_m", e["size_units"]))))
            dims = "×".join(
                f"{v:.2f}" for v in big.get("size_m", big["size_units"]))
            unit = "米" if "size_m" in big else "单位"
            frames = sorted({f for e in es for f in e["frames"]})
            where = f"（{len(es)} 处）" if len(es) > 1 else ""
            support = (f"{len(frames)} 帧佐证" if tier == "已证"
                       else "每处仅 1 帧")
            lines.append(f"- **{cn}**{where}：最大一处约 {dims} {unit}"
                         f"（{support}，估计）")
        lines.append("")
    if result["unverified"]:
        lines.append(f"提名未验（框内测不到）：{len(result['unverified'])} 条，"
                     "已丢弃，不算数。")
    (job_dir / "objects.md").write_text("\n".join(lines))
    _write_top_png(job_dir, result)


def _write_top_png(job_dir, result, size_px=760):
    """Top-down map: cloud grey, found objects as labelled markers."""
    try:
        from PIL import Image, ImageDraw
        _, points = load_job(job_dir)
    except Exception:
        return
    pts = points[::7]
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)

    def px(p):
        x = (p[0] - lo[0]) / span[0] * (size_px - 40) + 20
        y = (p[2] - lo[2]) / span[2] * (size_px - 40) + 20
        return float(x), float(y)

    img = Image.new("RGB", (size_px, size_px), (8, 10, 16))
    d = ImageDraw.Draw(img)
    for p in pts:
        x, y = px(p)
        d.point((x, y), fill=(90, 100, 115))
    for e in result["objects"]:
        x, y = px(np.asarray(e["centre_units"]))
        colour = (255, 190, 60) if e["tier"] == "已证" else (140, 140, 140)
        d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=colour, width=2)
        d.text((x + 9, y - 6), e["label"], fill=colour)
    img.save(job_dir / "objects_top.png")


# ---------------------------------------------------------------- nominate

def nominate(frame_paths, queries=None, threshold=0.10):
    """Run the open-vocabulary nominator over the given frames.

    Heavy (CPU minutes per frame) and fully offline. Returns the same
    dict shape pioneer G produced: {frame_name: {seconds, hits}}.
    """
    import time
    from PIL import Image
    from transformers import pipeline

    pipe = pipeline("zero-shot-object-detection",
                    model="google/owlv2-base-patch16-ensemble")
    out = {}
    for path in frame_paths:
        path = Path(path)
        img = Image.open(path).convert("RGB")
        t0 = time.time()
        preds = pipe(img, candidate_labels=queries or DEFAULT_QUERIES,
                     threshold=threshold)
        hits = [{"label": p["label"], "score": round(float(p["score"]), 3),
                 "box": {k: int(v) for k, v in p["box"].items()}}
                for p in preds]
        hits.sort(key=lambda h: -h["score"])
        out[path.name] = {"seconds": round(time.time() - t0, 1),
                          "hits": hits[:16]}
        print(f"{path.name}: {time.time() - t0:.0f}s, {len(hits)} hits",
              flush=True)
    return out


# ---------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(prog="echo_scene.discovery")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nominate")
    n.add_argument("job_dir")
    n.add_argument("--frames", required=True,
                   help="comma-separated frame file names")
    n.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("job_dir")
    r.add_argument("--nominations", required=True)
    r.add_argument("--scale", type=float, default=None,
                   help="metres per scene unit (estimate)")
    args = ap.parse_args(argv)

    job_dir = Path(args.job_dir)
    if args.cmd == "nominate":
        names = [s.strip() for s in args.frames.split(",") if s.strip()]
        paths = [job_dir / "frames" / name for name in names]
        result = nominate(paths)
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2))
        print(f"nominations -> {args.out}")
        return 0
    nominations = json.loads(Path(args.nominations).read_text())
    scale = args.scale
    if scale is None and (job_dir / "scale.json").exists():
        scale = json.loads((job_dir / "scale.json").read_text()).get(
            "scale_m_per_unit")
    poses, points = load_job(job_dir)
    result = adjudicate(poses, points, nominations, scale_m=scale)
    write_outputs(job_dir, result, scale_m=scale)
    print(f"objects: {len(result['objects'])} "
          f"(已证 {sum(1 for e in result['objects'] if e['tier'] == '已证')}, "
          f"单见 {sum(1 for e in result['objects'] if e['tier'] == '单见')}), "
          f"unverified nominations: {len(result['unverified'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
