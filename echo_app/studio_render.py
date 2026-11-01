"""Render Studio — renderer.

Replays one finished job's build process on the Studio's source
timeline and renders any moment of it with the Echo Render operator.
The visual state machine is the live preview's (`LiveSequencer`) —
the Studio adds direction on top of it:

  * the event prefix is a function of source time (scrubbing
    backwards rebuilds the sequencer; forward play just consumes),
  * the camera obeys the project's attachment mode at that moment
    (route = ride the recovered path, photo = watch the newest
    photo's points, orbit = the live preview's slow环视),
  * the element switches filter what reaches the operator
    (route line, photo reliefs, plus a global Detroit-style grid
    computed from the scene bounds — the grid belongs to the space,
    never to the room's surfaces),
  * the crush: the newest photo's points start scattered by the
    project's crush amplitude and settle onto their wall within
    ~1.5 source seconds — deterministic per point, so previews and
    exports agree frame for frame.

Exports follow the house encode rule (crf 23, maxrate 5M,
bufsize 2.5M, yuv420p, faststart).
"""
from __future__ import annotations

import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np

from .live_preview import LiveSequencer, WatchEvent
from .studio import StudioProject

CRUSH_WINDOW = 1.5     # source seconds a fresh photo takes to settle
GRID_COLOR = np.array([52, 72, 102], dtype=np.uint8)


def build_events(job_dir: Path) -> list[WatchEvent]:
    """The job's build process as an ordered event list."""
    import json
    job_dir = Path(job_dir)
    events = [WatchEvent("selected"), WatchEvent("poses")]
    selected = json.loads((job_dir / "selected_frames.json").read_text())
    names = sorted((Path(p).name for p in selected),
                   key=lambda n: int(n.split("_")[1].split(".")[0]))
    depth_dir = job_dir / "depth"
    for name in names:
        if (depth_dir / f"{Path(name).stem}.npy").exists():
            events.append(WatchEvent("depth", name=name))
    if (job_dir / "scene_points.ply").exists():
        events.append(WatchEvent("fused"))
    return events


def _camera_dict(eye, target, width, height):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    fwd = target - eye
    fwd /= (np.linalg.norm(fwd) + 1e-12)
    up_world = np.array([0.0, -1.0, 0.0])   # recon frame: y is down-ish
    right = np.cross(fwd, up_world)
    right /= (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, fwd)
    R = np.stack([right, -up, fwd])         # world -> camera (OpenCV)
    t = -R @ eye
    f_px = 0.9 * width
    K = np.array([[f_px, 0, width / 2.0],
                  [0, f_px, height / 2.0],
                  [0, 0, 1.0]])
    return {"K": K, "R": R, "t": t}


class StudioRenderer:
    def __init__(self, job_dir: Path, project: StudioProject):
        self.job_dir = Path(job_dir)
        self.project = project
        self.events = build_events(self.job_dir)
        self._seq: LiveSequencer | None = None
        self._applied = 0
        self._relief_t: dict[str, float] = {}
        self._dirs_cache: dict[str, np.ndarray] = {}

    # -- timeline -------------------------------------------------------
    def event_time(self, i: int) -> float:
        return self.project.duration * (i + 1) / max(len(self.events), 1)

    def _target_events(self, source_t: float) -> int:
        n = 0
        for i in range(len(self.events)):
            if self.event_time(i) <= source_t:
                n = i + 1
            else:
                break
        return n

    def _advance(self, source_t: float) -> LiveSequencer:
        target = self._target_events(source_t)
        if self._seq is None or target < self._applied:
            self._seq = LiveSequencer(self.job_dir)
            self._applied = 0
            self._relief_t = {}
        while self._applied < target:
            ev = self.events[self._applied]
            if ev.kind == "depth":
                self._relief_t[ev.name] = self.event_time(self._applied)
            self._seq.consume([ev])
            self._applied += 1
        return self._seq

    # -- scene assembly ---------------------------------------------------
    def _points(self, seq: LiveSequencer, source_t: float):
        st = seq.state
        if st.settled is not None:
            return st.settled[0], st.settled[1]
        if not self.project.elements.get("photos", True):
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        chunks, colors = [], []
        newest = max(self._relief_t, key=self._relief_t.get) \
            if self._relief_t else None
        for name in sorted(st.reliefs):
            p, c = st.reliefs[name]
            if name == newest:
                age = source_t - self._relief_t.get(name, source_t)
                amp = self.project.crush_at(source_t)
                if 0.0 <= age < CRUSH_WINDOW and amp > 0:
                    p = p + self._dirs(name, len(p)) * (
                        amp * seq._scene_distance * 0.18
                        * (1.0 - age / CRUSH_WINDOW))
            chunks.append(p)
            colors.append(c)
        if not chunks:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        return (np.concatenate(chunks).astype(np.float64),
                np.concatenate(colors).astype(np.uint8))

    def _dirs(self, name: str, n: int) -> np.ndarray:
        if name not in self._dirs_cache or len(self._dirs_cache[name]) < n:
            rng = np.random.default_rng(zlib.crc32(name.encode()))
            d = rng.normal(size=(n, 3))
            d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
            self._dirs_cache[name] = d
        return self._dirs_cache[name][:n]

    def _grid_lines(self, seq: LiveSequencer, points: np.ndarray):
        if len(points) < 10 or seq._scene_centre is None:
            return np.zeros((0, 2, 3)), np.zeros((0, 3), dtype=np.uint8)
        # Anchor on the camera route, not the raw cloud: stray points
        # stretch the cloud's percentiles far past the room. The floor
        # is the low plane of the points near the route; the grid then
        # extends past that footprint — it belongs to the space.
        centre, dist = seq._scene_centre, max(seq._scene_distance, 1e-3)
        horiz = np.linalg.norm(points[:, [0, 2]] - centre[[0, 2]],
                               axis=1)
        near = points[horiz < 1.5 * dist]
        if len(near) < 50:
            near = points
        y = float(np.percentile(near[:, 1], 98))   # y is down-ish
        lo = np.percentile(near, 2, axis=0)
        hi = np.percentile(near, 98, axis=0)
        ex = max((hi[0] - lo[0]) * 0.8, dist * 1.2)
        ez = max((hi[2] - lo[2]) * 0.8, dist * 1.2)
        x0, x1 = centre[0] - ex, centre[0] + ex
        z0, z1 = centre[2] - ez, centre[2] + ez
        segs = []
        for gx in np.linspace(x0, x1, 13):
            segs.append([(gx, y, z0), (gx, y, z1)])
        for gz in np.linspace(z0, z1, 13):
            segs.append([(x0, y, gz), (x1, y, gz)])
        lines = np.array(segs, dtype=np.float64)
        rgb = np.tile(GRID_COLOR, (len(lines), 1))
        return lines, rgb

    # -- cameras ------------------------------------------------------------
    def _camera(self, seq: LiveSequencer, source_t: float, width, height):
        mode = self.project.attach_at(source_t)
        route = seq.state.route
        dist = max(seq._scene_distance, 1e-3)
        if mode == "route" and len(route) >= 2:
            p = (source_t / max(self.project.duration, 1e-6))
            fidx = p * (len(route) - 1)
            i = int(min(max(fidx, 0.0), len(route) - 2))
            pos = route[i]
            ahead = route[min(i + 8, len(route) - 1)]
            back = pos - route[max(i - 3, 0)]
            n = np.linalg.norm(back)
            back = back / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
            eye = pos - back * dist * 0.45 + np.array(
                [0.0, -dist * 0.22, 0.0])
            return _camera_dict(eye, ahead, width, height)
        if mode == "photo":
            target = None
            if seq.state.reliefs:
                newest = max(self._relief_t, key=self._relief_t.get,
                             default=None)
                if newest in seq.state.reliefs:
                    target = seq.state.reliefs[newest][0].mean(axis=0)
            if target is None and len(route):
                target = route[-1]
            if target is not None:
                az = 0.6 + 0.05 * source_t
                r = dist * 1.15
                eye = target + np.array([r * np.cos(az), -r * 0.35,
                                         r * np.sin(az)])
                return _camera_dict(eye, target, width, height)
        view = seq.view
        view.az = 0.6 + 0.07 * source_t
        view.el = 0.35
        view.dist = 4.0
        view._touched()
        seq._cam_t = None
        return seq.camera(width, height)

    # -- rendering ------------------------------------------------------------
    def render(self, source_t: float, width: int = 768,
               height: int = 432) -> np.ndarray:
        from echo_super.kernels.echo_render import render_frame
        seq = self._advance(source_t)
        points, colors = self._points(seq, source_t)
        _p, _c, lines, line_rgb = seq.snapshot()
        chunks_l, chunks_c = [], []
        if self.project.elements.get("route", True) and len(lines):
            chunks_l.append(lines)
            chunks_c.append(line_rgb)
        if self.project.elements.get("grid", True):
            gl, gc = self._grid_lines(seq, points)
            if len(gl):
                chunks_l.append(gl)
                chunks_c.append(gc)
        if chunks_l:
            lines = np.concatenate(chunks_l).astype(np.float64)
            line_rgb = np.concatenate(chunks_c).astype(np.uint8)
        else:
            lines = np.zeros((0, 2, 3))
            line_rgb = np.zeros((0, 3), dtype=np.uint8)
        cam = self._camera(seq, source_t, width, height)
        img = render_frame(points.astype(np.float32), colors,
                           lines.astype(np.float32), line_rgb,
                           camera=cam, width=width, height=height,
                           point_radius_px=2.0, glow_strength=0.5,
                           glow_radius_factor=2.0,
                           background_rgb=(3, 6, 11))
        return img

    def render_output(self, t_out: float, width: int = 768,
                      height: int = 432) -> np.ndarray:
        return self.render(self.project.to_source(t_out), width, height)


def export_mp4(job_dir: Path, project: StudioProject, out_path: Path,
               fps: int = 12, width: int = 960, height: int = 540,
               progress=None) -> Path:
    """Render the whole output timeline to an mp4 (house encode rule)."""
    from PIL import Image
    renderer = StudioRenderer(job_dir, project)
    total_out = project.output_duration()
    n_frames = max(int(round(total_out * fps)), 1)
    out_path = Path(out_path)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(n_frames):
            img = renderer.render_output(i / fps, width, height)
            Image.fromarray(img).save(td / f"f_{i:05d}.png")
            if progress:
                progress(i + 1, n_frames)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
             "-i", str(td / "f_%05d.png"),
             "-c:v", "libx264", "-profile:v", "high", "-crf", "23",
             "-maxrate", "5M", "-bufsize", "2.5M", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out_path)], check=True)
    return out_path
