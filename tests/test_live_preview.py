"""Offline simulation bench for echo_app.live_preview: replay the
room2 job's products into a fresh folder in pipeline order and check
the watcher + sequencer + renderer track them honestly."""
import json, shutil, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_app.live_preview import (
    JobWatcher, LiveSequencer, WatchEvent, render_snapshot)

SRC = Path(__file__).resolve().parent.parent / "jobs" / "room2"
TMP = Path("/tmp/live_preview_sim")
fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

if TMP.exists(): shutil.rmtree(TMP)
(TMP / "frames").mkdir(parents=True)
(TMP / "depth").mkdir()
w = JobWatcher(TMP, stable_polls=1)
seq = LiveSequencer(TMP)

# 1) extract: 30 frames land
frames = sorted((SRC / "frames").glob("frame_*.jpg"))[:30]
for p in frames: shutil.copy(p, TMP / "frames" / p.name)
ev = w.poll(); seq.consume(ev)
check("extract: 30 frame events", sum(e.kind == "frame" for e in ev) == 30
      and seq.state.frames_seen == 30)

# 2) select lands
shutil.copy(SRC / "selected_frames.json", TMP / "selected_frames.json")
ev = w.poll(); seq.consume(ev)
check("select: event + set", any(e.kind == "selected" for e in ev)
      and len(seq.state.selected) > 0)

# 3) poses land -> route
shutil.copy(SRC / "poses.json", TMP / "poses.json")
ev = w.poll(); seq.consume(ev)
n_poses = len(json.loads((SRC / "poses.json").read_text()))
check("poses: route complete", len(seq.state.route) == n_poses)

# 4) depth frames land one by one (first 8 posed frames)
names = sorted(json.loads((SRC / "poses.json").read_text()))[:8]
for n in names:
    shutil.copy(SRC / "depth" / (Path(n).stem + ".npy"), TMP / "depth" / (Path(n).stem + ".npy"))
    ev = w.poll(); seq.consume(ev)
check("depth: reliefs for posed frames", len(seq.state.reliefs) >= 6)
check("depth: relief points > 10k", seq.state.relief_point_count > 10000)
img, dt = render_snapshot(seq, t_s=10.0, width=640, height=360)
nonbg = float(np.mean(np.any(img > 12, axis=2)))
print(f"   depth-phase render: {dt:.2f}s, non-bg {nonbg:.3f}")
check("depth-phase render shows scene", img.shape == (360, 640, 3) and nonbg > 0.01)
check("depth-phase render < 2s", dt < 2.0)

# 5) fuse lands -> settled cloud replaces reliefs
shutil.copy(SRC / "scene_points.ply", TMP / "scene_points.ply")
ev = w.poll(); seq.consume(ev)
check("fused: settled present, reliefs cleared",
      seq.state.settled is not None and len(seq.state.reliefs) == 0)
check("fused: settled points > 100k", len(seq.state.settled[0]) > 100000)
img, dt = render_snapshot(seq, t_s=20.0, width=640, height=360)
nonbg = float(np.mean(np.any(img > 12, axis=2)))
print(f"   fused-phase render: {dt:.2f}s, non-bg {nonbg:.3f}")
check("fused-phase render shows scene", nonbg > 0.01)
check("fused-phase render < 2s", dt < 2.0)

# idempotent re-poll: no duplicate events
ev = w.poll()
check("re-poll emits nothing new", len(ev) == 0)

# 6) shared view state: manipulation moves the camera, renders differ
from echo_app.live_preview import ViewState
view = ViewState()
seq2 = LiveSequencer(TMP, view=view)
seq2.consume([WatchEvent("poses"), WatchEvent("fused")])
img_a, _ = render_snapshot(seq2, width=480, height=270)
az0, el0, dist0 = view.az, view.el, view.dist
view.drag(120, -40)
check("view: drag turns + raises", view.az != az0 and view.el != el0
      and view.version > 0)
view.zoom(3)
check("view: zoom-in shrinks radius", view.dist < dist0)
pan0 = view.pan.copy()
view.pan_drag(50, 20)
check("view: pan moves look-at", not np.allclose(view.pan, pan0))
img_b, _ = render_snapshot(seq2, width=480, height=270)
diff = float(np.mean(np.abs(img_a.astype(int) - img_b.astype(int))))
print(f"   manipulated-view render diff: {diff:.2f}")
check("view: manipulated camera renders a different picture", diff > 1.0)
view.reset()
check("view: reset restores defaults",
      view.az == 0.6 and view.dist == 4.0 and not view.pan.any())

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
