"""Live process preview driver — the app-window feature the user asked
for on 2026-10-31: while a job runs, the middle of the window shows the
scene actually growing from the job's own intermediate products.

Productisation of pioneer C (experiments/pioneers/C_live_preview.md),
which proved the frozen render path (Echo Render operator) keeps up
easily at full data scale (worst visible update ~1.2 s on this weak VM;
the pipeline itself produces data slower than the renderer consumes
it). What was missing, and what this module is: the DRIVER that watches
a real job folder and sequences what has actually landed into a live
visual state. No fake animation: every element on screen is a product
the pipeline has already written to disk.

Honest phase model (dictated by what the data can support when):

  extract   frames/*.jpg land            -> frame cards arrive (count)
  select    selected_frames.json lands   -> selected cards light up
  poses     poses.json lands             -> camera route draws
  depth     depth/<frame>.npy lands      -> that frame's photo becomes
            a LOCAL RELIEF of true-colour points at its camera: the
            depth maps are still relative at this stage (the affine
            alignment to scene scale only exists inside fuse), so a
            relief is placed along the frame's own rays at scene
            distance -- labelled estimation by construction, never
            passed off as the final geometry.
  fuse      scene_points.ply lands       -> reliefs dissolve; the true
            settled cloud streams in (provenance order), points swell
            toward solid for the finale (presentation step, app side).

This module is the watcher + sequencer + snapshot renderer. The app
owns the Tk surface and the swell choreography; snapshots here are
plain operator renders of the current live state.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Watcher: diff a job folder, emit ordered events for stable new files.
# ---------------------------------------------------------------------------

#: event kinds, in pipeline order
EVENT_ORDER = ("frame", "selected", "poses", "depth", "fused", "stage")


@dataclass
class WatchEvent:
    kind: str                 # one of EVENT_ORDER
    name: str = ""            # frame file name for frame/depth events
    stage: str = ""           # pipeline stage name for stage events


class JobWatcher:
    """Poll a job directory; a file counts as landed once its size is
    unchanged across `stable_polls` consecutive polls (writers here
    write files in one pass, but a half-written .npy must never be
    consumed)."""

    def __init__(self, job_dir: Path, stable_polls: int = 2):
        self.job_dir = Path(job_dir)
        self.stable_polls = max(1, stable_polls)
        self._seen: dict[str, tuple[int, int]] = {}   # key -> (size, polls)
        self._emitted: set[str] = set()
        self._stage_seen: str = ""

    def _track(self, key: str, path: Path, events: list, kind: str,
               name: str = ""):
        if key in self._emitted:
            return
        try:
            size = path.stat().st_size
        except OSError:
            return
        prev = self._seen.get(key)
        if prev is not None and prev[0] == size:
            count = prev[1] + 1
        else:
            count = 1
        self._seen[key] = (size, count)
        if count >= self.stable_polls:
            self._emitted.add(key)
            events.append(WatchEvent(kind, name=name))

    def poll(self) -> list[WatchEvent]:
        events: list[WatchEvent] = []
        jd = self.job_dir
        frames_dir = jd / "frames"
        if frames_dir.is_dir():
            for p in sorted(frames_dir.glob("frame_*.jpg")):
                self._track(f"frame:{p.name}", p, events, "frame", p.name)
        self._track("selected", jd / "selected_frames.json", events,
                    "selected")
        self._track("poses", jd / "poses.json", events, "poses")
        depth_dir = jd / "depth"
        if depth_dir.is_dir():
            for p in sorted(depth_dir.glob("frame_*.npy")):
                self._track(f"depth:{p.name}", p, events, "depth",
                            p.name.replace(".npy", ".jpg"))
        self._track("fused", jd / "scene_points.ply", events, "fused")
        job_json = jd / "job.json"
        if job_json.exists():
            try:
                stage = json.loads(job_json.read_text()).get("stage", "")
            except Exception:
                stage = ""
            if stage and stage != self._stage_seen:
                self._stage_seen = stage
                events.append(WatchEvent("stage", stage=stage))
        events.sort(key=lambda e: EVENT_ORDER.index(e.kind))
        return events


# ---------------------------------------------------------------------------
# Geometry helpers (poses.json is the pipeline's own product).
# ---------------------------------------------------------------------------

def _quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_centre(pose: dict) -> np.ndarray:
    R = _quat_xyzw_to_matrix(pose["cam_from_world_quat_xyzw"])
    t = np.asarray(pose["cam_from_world_t"], dtype=np.float64)
    return -R.T @ t


def relief_points(depth: np.ndarray, pose: dict, photo_rgb: np.ndarray,
                  scene_distance: float, stride: int = 24,
                  max_points: int = 6000):
    """One frame's local relief: true-colour points along the frame's
    own rays. depth is the pipeline's relative map (affine in 1/Z), so
    z_local = 1/rel is monotone-correct in depth; the frame's median
    z is scaled to `scene_distance` so the relief sits at scene range
    from its camera. Returns (points_world (K,3) float64, rgb (K,3)
    uint8). Estimation by construction (see module docstring)."""
    h, w = depth.shape
    cam = pose["camera"]
    f, cx, cy = (float(cam["params"][0]), float(cam["params"][1]),
                 float(cam["params"][2]))
    rows = np.arange(0, h, stride)
    cols = np.arange(0, w, stride)
    sub = depth[np.ix_(rows, cols)].astype(np.float64)
    ok = np.isfinite(sub) & (sub > 1e-6)
    if not np.any(ok):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    z = np.zeros_like(sub)
    z[ok] = 1.0 / sub[ok]
    med = np.median(z[ok])
    if med <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    z *= scene_distance / med
    uu, vv = np.meshgrid(cols, rows)
    x = (uu - cx) / f * z
    y = (vv - cy) / f * z
    pts_cam = np.stack([x, y, z], axis=-1)[ok]
    rgb = photo_rgb[np.ix_(rows, cols)].reshape(-1, 3)[ok.reshape(-1)]
    if len(pts_cam) > max_points:
        pick = np.linspace(0, len(pts_cam) - 1, max_points).astype(int)
        pts_cam, rgb = pts_cam[pick], rgb[pick]
    R = _quat_xyzw_to_matrix(pose["cam_from_world_quat_xyzw"])
    centre = camera_centre(pose)
    pts_world = pts_cam @ R + centre     # rows are R^T-applied: X = R^T Xc + C
    return pts_world, rgb


def read_points_ply(path: Path):
    """Minimal reader for the fuse stage's binary PLY (float xyz +
    uchar rgb) — the preview path deliberately avoids the open3d
    dependency. Returns (points (N,3) float64, rgb (N,3) uint8)."""
    raw = Path(path).read_bytes()
    head_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:head_end].decode("ascii", errors="replace")
    count, props = 0, []
    for line in header.splitlines():
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
        elif parts and parts[0] == "property":
            props.append((parts[1], parts[2]))
    types = {"float": "f4", "float32": "f4", "double": "f8",
             "uchar": "u1", "uint8": "u1", "int": "i4", "int32": "i4"}
    dtype = np.dtype([(name, types[typ]) for typ, name in props])
    arr = np.frombuffer(raw[head_end:head_end + count * dtype.itemsize],
                        dtype=dtype, count=count)
    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    if {"red", "green", "blue"} <= {name for _, name in props}:
        rgb = np.stack([arr["red"], arr["green"], arr["blue"]],
                       axis=1).astype(np.uint8)
    else:
        rgb = np.full((count, 3), 200, dtype=np.uint8)
    return pts, rgb


# ---------------------------------------------------------------------------
# ViewState: the one camera the user can grab. Written by the UI thread
# (mouse drags / wheel on the fullscreen view), read by the render
# thread. It OUTLIVES any single view: the fullscreen manipulates it,
# exiting fullscreen hands the same state back to the card, and the
# next job inherits it too. When nobody has touched it for a few
# seconds the camera resumes its slow auto-orbit from wherever it is.
# ---------------------------------------------------------------------------

class ViewState:
    def __init__(self):
        self._lock = threading.Lock()
        self.az = 0.6            # orbit angle around the scene centre
        self.el = 0.35           # eye height as a fraction of the radius
        self.dist = 4.0          # radius in scene-distance units
        self.pan = np.zeros(3)   # world-space look-at offset
        self.version = 0         # bumps on every manipulation
        self.last_touch = 0.0    # monotonic time of last manipulation
        # filled in by the sequencer at every camera build, used to
        # turn right-drag pixel motion into world-space panning
        self.frame_right = np.array([1.0, 0.0, 0.0])
        self.frame_up = np.array([0.0, -1.0, 0.0])
        self.pan_scale = 0.002   # world units per pixel

    def _touched(self):
        self.version += 1
        self.last_touch = time.monotonic()

    def drag(self, dx_px: float, dy_px: float):
        """Left-drag: orbit. dx turns around, dy raises/lowers the eye."""
        with self._lock:
            self.az -= dx_px * 0.005
            self.el = float(np.clip(self.el + dy_px * 0.0025, -0.05, 1.4))
            self._touched()

    def zoom(self, steps: float):
        """Wheel: steps > 0 zooms in."""
        with self._lock:
            self.dist = float(np.clip(self.dist * (0.88 ** steps),
                                      0.7, 14.0))
            self._touched()

    def pan_drag(self, dx_px: float, dy_px: float):
        """Right-drag: slide the look-at point in the view plane."""
        with self._lock:
            self.pan = self.pan + (-dx_px * self.frame_right
                                   + dy_px * self.frame_up) * self.pan_scale
            self._touched()

    def reset(self):
        with self._lock:
            self.az, self.el, self.dist = 0.6, 0.35, 4.0
            self.pan = np.zeros(3)
            self._touched()

    def snapshot(self):
        with self._lock:
            return (self.az, self.el, self.dist, self.pan.copy(),
                    self.version, self.last_touch)


# ---------------------------------------------------------------------------
# Sequencer: events in, live visual state out.
# ---------------------------------------------------------------------------

@dataclass
class LiveState:
    frames_seen: int = 0
    selected: set = field(default_factory=set)
    route: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3)))  # camera centres, ordered
    reliefs: dict = field(default_factory=dict)   # name -> (pts, rgb)
    settled: tuple | None = None                  # (pts, rgb) from fuse
    stage: str = ""
    phase: str = "extract"

    @property
    def relief_point_count(self) -> int:
        return int(sum(len(p) for p, _ in self.reliefs.values()))


class LiveSequencer:
    """Consumes WatchEvents against the job folder's real products and
    maintains the LiveState. Rendering is a pure function of state."""

    def __init__(self, job_dir: Path, view: ViewState | None = None):
        self.job_dir = Path(job_dir)
        self.view = view if view is not None else ViewState()
        self._cam_t: float | None = None
        self.state = LiveState()
        self._poses: dict | None = None
        self._scene_centre: np.ndarray | None = None
        self._scene_distance: float = 1.0

    # -- product loaders -------------------------------------------------
    def _load_poses(self):
        self._poses = json.loads((self.job_dir / "poses.json").read_text())
        names = sorted(self._poses)
        centres = np.stack([camera_centre(self._poses[n]) for n in names])
        self.state.route = centres
        self._scene_centre = centres.mean(axis=0)
        spread = float(np.median(np.linalg.norm(
            centres - self._scene_centre, axis=1))) if len(centres) else 1.0
        self._scene_distance = max(spread, 1e-3)
        self.state.phase = "poses"

    def _load_selected(self):
        raw = json.loads((self.job_dir / "selected_frames.json").read_text())
        self.state.selected = {Path(p).name for p in raw}
        self.state.phase = "select"

    def _load_depth(self, name: str):
        if self._poses is None or name not in self._poses:
            return
        from PIL import Image
        stem = Path(name).stem
        depth = np.load(self.job_dir / "depth" / f"{stem}.npy")
        photo = np.asarray(Image.open(
            self.job_dir / "frames" / name).convert("RGB"))
        if photo.shape[:2] != depth.shape:
            photo = np.asarray(Image.open(
                self.job_dir / "frames" / name).convert("RGB").resize(
                    (depth.shape[1], depth.shape[0])))
        pts, rgb = relief_points(depth, self._poses[name], photo,
                                 self._scene_distance)
        if len(pts):
            self.state.reliefs[name] = (pts, rgb)
        self.state.phase = "depth"

    def _load_fused(self):
        pts, rgb = read_points_ply(self.job_dir / "scene_points.ply")
        self.state.settled = (pts, rgb)
        self.state.reliefs = {}
        self.state.phase = "fused"

    # -- event intake ------------------------------------------------------
    def consume(self, events: list[WatchEvent]) -> LiveState:
        for ev in events:
            if ev.kind == "frame":
                self.state.frames_seen += 1
            elif ev.kind == "selected":
                self._load_selected()
            elif ev.kind == "poses":
                self._load_poses()
            elif ev.kind == "depth":
                self._load_depth(ev.name)
            elif ev.kind == "fused":
                self._load_fused()
            elif ev.kind == "stage":
                self.state.stage = ev.stage
        return self.state

    # -- snapshot rendering --------------------------------------------------
    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray]:
        """(points, rgb, line_segments, line_rgb) for the operator."""
        st = self.state
        pts_chunks, rgb_chunks = [], []
        if st.settled is not None:
            pts_chunks.append(st.settled[0])
            rgb_chunks.append(st.settled[1])
        else:
            for name in sorted(st.reliefs):
                p, c = st.reliefs[name]
                pts_chunks.append(p)
                rgb_chunks.append(c)
        if pts_chunks:
            points = np.concatenate(pts_chunks).astype(np.float64)
            colors = np.concatenate(rgb_chunks).astype(np.uint8)
        else:
            points = np.zeros((0, 3))
            colors = np.zeros((0, 3), dtype=np.uint8)
        lines = np.zeros((0, 2, 3))
        line_rgb = np.zeros((0, 3), dtype=np.uint8)
        if len(st.route) >= 2:
            lines = np.stack([st.route[:-1], st.route[1:]], axis=1)
            line_rgb = np.tile(np.array([[90, 160, 255]], dtype=np.uint8),
                               (len(lines), 1))
        return points, colors, lines, line_rgb

    def camera(self, width: int = 1280, height: int = 720) -> dict:
        """The shared ViewState as a render camera. Untouched for a
        few seconds, the orbit slowly resumes from wherever the user
        left it — the state is never reset by view changes."""
        centre = self._scene_centre if self._scene_centre is not None \
            else np.zeros(3)
        base4 = max(self._scene_distance * 4.0, 1.0)   # radius at dist=4
        if self.state.settled is not None:
            pts = self.state.settled[0]
            centre = pts.mean(axis=0)
            base4 = max(float(np.linalg.norm(
                pts.max(axis=0) - pts.min(axis=0))) * 0.9, base4)
        view = self.view
        now = time.monotonic()
        if self._cam_t is not None:
            with view._lock:
                if now - view.last_touch > 5.0:
                    view.az += 0.05 * (now - self._cam_t)
        self._cam_t = now
        az, el, dist, pan, _ver, _touch = view.snapshot()
        radius = base4 * dist / 4.0
        target = centre + pan
        eye = target + np.array([radius * np.cos(az),
                                 -radius * el,
                                 radius * np.sin(az)])
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        up_world = np.array([0.0, -1.0, 0.0])   # recon frame: y is down-ish
        right = np.cross(fwd, up_world)
        right /= (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, fwd)
        with view._lock:
            view.frame_right = right
            view.frame_up = up
            view.pan_scale = radius * 1.2 / max(height, 1)
        R = np.stack([right, -up, fwd])         # world -> camera (OpenCV)
        t = -R @ eye
        f_px = 0.9 * width
        K = np.array([[f_px, 0, width / 2.0],
                      [0, f_px, height / 2.0],
                      [0, 0, 1.0]])
        return {"K": K, "R": R, "t": t}

    # Backwards-compatible alias (the pioneer bench used this name).
    def orbit_camera(self, t_s: float = 0.0, width: int = 1280,
                     height: int = 720) -> dict:
        return self.camera(width, height)


def render_snapshot(seq: LiveSequencer, t_s: float | None = None,
                    width: int = 1280, height: int = 720) -> tuple:
    """Render the sequencer's current state with the Echo Render
    operator. Returns (image uint8 (H,W,3), seconds_taken)."""
    from echo_super.kernels.echo_render import render_frame
    points, colors, lines, line_rgb = seq.snapshot()
    cam = seq.camera(width, height)
    t0 = time.perf_counter()
    img = render_frame(points.astype(np.float32), colors,
                       lines.astype(np.float32), line_rgb,
                       camera=cam, width=width, height=height,
                       point_radius_px=2.0, glow_strength=0.5,
                       glow_radius_factor=2.0, background_rgb=(3, 6, 11))
    return img, time.perf_counter() - t0
