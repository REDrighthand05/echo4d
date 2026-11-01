"""Pioneer G — route-3 (learned) discovery on the user's real frames.

Question: can an off-the-shelf open-vocabulary detector, running
locally on CPU, NOMINATE the anchor objects the measurement chain
already proved (86 panel / door / laptop screen) plus the big
furniture — well enough to feed the nomination -> measurement-verdict
architecture?

Pre-set verdict bars (written before the run):
  PASS  panel nominated in frame_00310 or frame_00016 with a box on
        the actual panel cluster, door in frame_00180, laptop screen
        in frame_00194.
  WEAK  furniture found but the small anchors missed (nominator only
        useful for the inventory layer, not for scale anchors).
  FAIL  anchors and furniture both missed or boxes unusable.

Model: google/owlv2-base-patch16-ensemble via transformers pipeline,
CPU, threshold 0.10. Outputs: results.json + annotated copies.
"""
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
FRAMES = HERE.parent.parent.parent / "jobs" / "room2" / "frames"
TARGETS = ["frame_00310.jpg", "frame_00180.jpg", "frame_00194.jpg",
           "frame_00016.jpg", "frame_00040.jpg", "frame_00057.jpg"]
QUERIES = ["electrical outlet panel", "power socket", "door",
           "laptop screen", "laptop", "bed", "desk", "bookshelf",
           "book", "bottle", "chair"]


def main():
    from transformers import pipeline

    t0 = time.time()
    pipe = pipeline("zero-shot-object-detection",
                    model="google/owlv2-base-patch16-ensemble")
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    results = {}
    for name in TARGETS:
        path = FRAMES / name
        if not path.exists():
            results[name] = {"missing": True}
            continue
        img = Image.open(path).convert("RGB")
        t1 = time.time()
        preds = pipe(img, candidate_labels=QUERIES, threshold=0.10)
        dt = time.time() - t1
        hits = [{"label": p["label"], "score": round(float(p["score"]), 3),
                 "box": {k: int(v) for k, v in p["box"].items()}}
                for p in preds]
        hits.sort(key=lambda h: -h["score"])
        results[name] = {"seconds": round(dt, 1), "hits": hits[:12]}
        draw = ImageDraw.Draw(img)
        for h in hits[:8]:
            b = h["box"]
            draw.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]],
                           outline=(255, 60, 60), width=6)
            draw.text((b["xmin"] + 4, b["ymin"] + 4),
                      f'{h["label"]} {h["score"]}', fill=(255, 60, 60))
        img.save(HERE / f"annotated_{name}")
        print(f"{name}: {dt:.0f}s, {len(hits)} hits", flush=True)
    (HERE / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
