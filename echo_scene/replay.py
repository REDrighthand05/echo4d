#!/usr/bin/env python3
"""ECHO route replay: watch the walk that produced the scene.

Reads a finished job directory and writes one standalone HTML page
(replay.html) that shows the finished scene with a simple stand-in
figure gliding the recovered camera route -- the photographer's own
walk, in the original direction, at walking pace, looping:

  * the scene is the job's final points.bin (post-cleanup cloud);
  * the route comes from poses.json: camera centres are carried into
    the exported (upright) frame with the job's alignment.json
    rotation, then interpolated by the registered echo_super kernel
    build_replay_path (Catmull-Rom, passes through every pose knot);
  * the figure is a gold point-built humanoid, feet on the floor the
    reconstruction found under the route, eyes riding the route
    itself (its height is calibrated from the median camera height);
  * the walked part of the route traces bright on the floor, the
    part still ahead stays dim;
  * the camera follows the figure gently (a slow drift around it);
    grabbing the view overrides the follow for a few seconds.

The figure is a STAND-IN v1: it glides without limb motion, exactly
the accepted spec until skeleton data exists. Everything else shown
is the job's real data (poses, fused points) -- a presentation layer,
not a fake animation. The page is one self-contained file: all data
embedded, no server, no external references (the v0.2 outputs-as-
files rule).

Usage:
  python3 -m echo_scene.replay <job_dir> <out.html>
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

try:
    from echo_super.ds_batch1 import build_replay_path
except ImportError:  # running outside the repo root: add it and retry
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from echo_super.ds_batch1 import build_replay_path

from .preview import _floor_level, _quat_rot, _read_points_bin

SAMPLES_PER_SEGMENT = 8          # kernel sampling of the route spline
MAX_PAGE_POINTS = 300_000        # deterministic stride cap for the page
LOOP_DUR = 20.0                  # seconds per replay loop
WALK_T0, WALK_T1 = 1.2, 17.6     # the glide lives between these times
WALK_ACC = 1.5                   # trapezoid ease-in/out at both ends
WARP_STEPS = 800                 # time -> arc-fraction table resolution
FIG_SEED = 20261031
FIG_RGB = (255, 166, 66)         # gold accent (same as the sample render)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _camera_route(job_dir: Path, poses: dict):
    """Camera centres + look targets in the exported frame.

    Same convention as preview.build_preview_data: points.bin already
    lives in the exported frame, so the COLMAP camera centres are
    carried there with the alignment rotation (row-vector form).
    Walking order is the sorted frame-name order (frames are extracted
    sequentially), exactly as the preview's card order.
    """
    rot = None
    align_path = job_dir / "alignment.json"
    if align_path.exists():
        rot = np.array(json.loads(align_path.read_text())["rotation"],
                       dtype=np.float64)

    def to_export(v):
        return v @ rot.T if rot is not None else v

    stems = sorted(Path(k).stem for k in poses)
    centres, targets = [], []
    for stem in stems:
        p = poses[stem + ".jpg"]
        R = _quat_rot(p["cam_from_world_quat_xyzw"])     # cam_from_world
        t = np.asarray(p["cam_from_world_t"], dtype=np.float64)
        centre = to_export(-R.T @ t)
        centres.append(centre)
        targets.append(centre + to_export(R[2]))         # camera forward
    return stems, np.array(centres), np.array(targets)


def _route_floor(pts: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Floor height under the route: robust low percentile of the
    cloud near each route checkpoint, smoothed, interpolated to every
    path sample. Where the reconstruction is thin the floor falls back
    to a fixed drop below the camera (the sample render's recipe)."""
    chk_idx = np.arange(0, len(path), 10)
    floors = np.empty(len(chk_idx), dtype=np.float64)
    for i, k in enumerate(chk_idx):
        c = path[k]
        d2 = (pts[:, 0] - c[0]) ** 2 + (pts[:, 2] - c[2]) ** 2
        near = pts[d2 < 2.5 ** 2]
        floors[i] = (float(np.percentile(near[:, 1], 5))
                     if len(near) > 50 else float(c[1] - 5.5))
    k = 9
    smooth = np.convolve(np.pad(floors, k // 2, mode="edge"),
                         np.ones(k) / k, mode="valid")[:len(floors)]
    return np.interp(np.arange(len(path)), chk_idx, smooth)


def _walk_warp() -> np.ndarray:
    """Loop time -> travelled arc fraction, trapezoid speed profile
    (gentle start, steady walk, gentle stop), as in the sample."""
    tt = np.linspace(0.0, LOOP_DUR, 4001)
    vel = np.zeros_like(tt)
    vpeak = 1.0 / ((WALK_T1 - WALK_T0) - WALK_ACC)
    m1 = (tt >= WALK_T0) & (tt < WALK_T0 + WALK_ACC)
    m2 = (tt >= WALK_T0 + WALK_ACC) & (tt <= WALK_T1 - WALK_ACC)
    m3 = (tt > WALK_T1 - WALK_ACC) & (tt <= WALK_T1)
    vel[m1] = vpeak * (tt[m1] - WALK_T0) / WALK_ACC
    vel[m2] = vpeak
    vel[m3] = vpeak * (WALK_T1 - tt[m3]) / WALK_ACC
    dist = np.concatenate(
        [[0.0], np.cumsum(0.5 * (vel[1:] + vel[:-1]) * np.diff(tt))])
    dist /= dist[-1]
    times = np.linspace(0.0, LOOP_DUR, WARP_STEPS)
    return np.interp(times, tt, dist).astype(np.float32)


def _figure_points(height: float) -> np.ndarray:
    """Point-built stand-in figure, local frame: feet at origin,
    +Y up, faces +Z. Deterministic (fixed seed), proportions follow
    the sample render the user approved."""
    rng = np.random.default_rng(FIG_SEED)
    h = float(height)

    def sphere(c, r, n):
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return np.array(c, dtype=np.float64) + r * v

    def capsule(a, b, r, n):
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        u = rng.random(n)
        base = a[None, :] + u[:, None] * (b - a)[None, :]
        v = rng.normal(size=(n, 3))
        axis = (b - a) / np.linalg.norm(b - a)
        v -= (v @ axis)[:, None] * axis[None, :]
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        out = base + r * v
        for m, c in ((u < 0.12, a), (u > 0.88, b)):
            if m.any():
                w = rng.normal(size=(int(m.sum()), 3))
                w /= np.linalg.norm(w, axis=1, keepdims=True)
                out[m] = np.array(c, dtype=np.float64) + r * w
        return out

    fig = np.vstack([
        sphere((0, 0.935 * h, 0), 0.075 * h, 500),                       # head
        capsule((0, 0.55 * h, 0), (0, 0.845 * h, 0), 0.100 * h, 1100),   # torso
        capsule((-0.02 * h, 0.47 * h, 0), (0.02 * h, 0.56 * h, 0),
                0.105 * h, 300),                                         # hips
        capsule((0.115 * h, 0.80 * h, 0), (0.175 * h, 0.50 * h, 0),
                0.040 * h, 350),                                         # arm L
        capsule((-0.115 * h, 0.80 * h, 0), (-0.175 * h, 0.50 * h, 0),
                0.040 * h, 350),                                         # arm R
        capsule((0.058 * h, 0.50 * h, 0), (0.070 * h, 0.03 * h, 0),
                0.055 * h, 550),                                         # leg L
        capsule((-0.058 * h, 0.50 * h, 0), (-0.070 * h, 0.03 * h, 0),
                0.055 * h, 550),                                         # leg R
    ])
    return fig.astype(np.float64)


def build_replay_data(job_dir: Path):
    """Collect every array the replay page needs. (manifest, blob)."""
    job_dir = Path(job_dir)
    points_path = job_dir / "points.bin"
    if not points_path.exists():
        raise FileNotFoundError(f"points.bin missing in {job_dir}")
    poses_path = job_dir / "poses.json"
    if not poses_path.exists():
        raise FileNotFoundError(f"poses.json missing in {job_dir}")

    pts_full, rgb_full = _read_points_bin(points_path)
    poses = json.loads(poses_path.read_text())
    _stems, centres, targets = _camera_route(job_dir, poses)
    if len(centres) < 2:
        raise ValueError("need at least 2 posed frames for a route")

    path, _look_dirs = build_replay_path(
        centres, targets, SAMPLES_PER_SEGMENT)
    floor = _route_floor(pts_full, path)

    cam_h = path[:, 1] - floor
    median_cam_h = float(np.median(cam_h))
    fig_h = median_cam_h / 0.93 if median_cam_h > 0 else 0.0
    if not np.isfinite(fig_h) or fig_h <= 0:
        fig_h = max(float(np.percentile(pts_full[:, 1], 95)
                          - np.percentile(pts_full[:, 1], 5)) * 0.6, 1e-6)
    fig = _figure_points(fig_h)

    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    route_length = float(arc[-1])
    arc_frac = (arc / max(route_length, 1e-9)).astype(np.float32)

    # page cloud: deterministic stride cap, colours ride along
    pts, rgb = pts_full, rgb_full
    if len(pts) > MAX_PAGE_POINTS:
        stride = int(np.ceil(len(pts) / MAX_PAGE_POINTS))
        pts, rgb = pts[::stride], rgb[::stride]

    warp = _walk_warp()
    parts = [
        pts.astype(np.float32).tobytes(),        # scene points
        rgb.astype(np.uint8).tobytes(),          # scene colours
        path.astype(np.float32).tobytes(),       # route samples
        floor.astype(np.float32).tobytes(),      # floor under the route
        arc_frac.tobytes(),                      # arc fraction per sample
        fig.astype(np.float32).tobytes(),        # figure, local frame
        warp.tobytes(),                          # time -> arc fraction
    ]
    # pad every part to a 4-byte boundary (typed-array views need it)
    blob = b"".join(p + b"\x00" * (-len(p) % 4) for p in parts)

    lo = pts_full.min(axis=0)
    hi = pts_full.max(axis=0)
    manifest = {
        "duration": LOOP_DUR,
        "walk_t0": WALK_T0,
        "walk_t1": WALK_T1,
        "n_points": int(len(pts)),
        "n_path": int(len(path)),
        "n_knots": int(len(centres)),
        "n_fig": int(len(fig)),
        "n_warp": int(len(warp)),
        "fig_height": float(fig_h),
        "fig_rgb": [int(c) for c in FIG_RGB],
        "route_length": route_length,
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
        "floor_y": _floor_level(pts_full[:, 1]),
        "knots": [[float(v) for v in c] for c in centres],
        "note": ("route from poses.json via build_replay_path; "
                 "figure is a stand-in placeholder"),
    }
    return manifest, blob


# ---------------------------------------------------------------------------
# page template (same family as echo_scene.preview's page: raw WebGL2,
# everything inline, data embedded as base64 -- opens by double-click)
# ---------------------------------------------------------------------------
_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>ECHO 路线回放</title>
<style>
  html, body { margin: 0; height: 100%; background: #04070d; overflow: hidden; }
  #gl { display: block; width: 100%; height: 100%; touch-action: none; cursor: grab; }
  #gl:active { cursor: grabbing; }
  #cap {
    position: fixed; left: 14px; top: 12px; color: #bcd7ff;
    font: 14px/1.5 ui-monospace, Menlo, Consolas, monospace;
    text-shadow: 0 1px 4px #000; pointer-events: none; white-space: pre-line;
  }
  #cap b { color: #6fb2ff; font-size: 15px; }
  #bar {
    position: fixed; left: 50%; bottom: 12px; transform: translateX(-50%);
    display: flex; align-items: center; gap: 10px;
    max-width: calc(100vw - 16px); box-sizing: border-box;
    background: rgba(6, 12, 22, 0.82); border: 1px solid rgba(111, 178, 255, 0.35);
    border-radius: 12px; padding: 8px 12px;
    font: 13px/1.4 ui-monospace, Menlo, Consolas, monospace; color: #cfe4ff;
    user-select: none; -webkit-user-select: none; touch-action: manipulation;
  }
  #bar button {
    font: inherit; color: #dceeff; background: rgba(111, 178, 255, 0.12);
    border: 1px solid rgba(111, 178, 255, 0.45); border-radius: 8px;
    padding: 6px 12px; cursor: pointer; min-height: 32px; white-space: nowrap;
  }
  #bar button:active { background: rgba(111, 178, 255, 0.3); }
  #segs {
    display: flex; gap: 2px; align-items: center; cursor: pointer;
    width: min(46vw, 520px); padding: 8px 2px; touch-action: none;
  }
  .seg {
    flex: 1 1 0; height: 12px; background: rgba(140, 175, 225, 0.30);
    transform: scaleY(0.42); pointer-events: none;
  }
  .seg.on { background: #e9f3ff; transform: none; }
  .seg.head { background: #6fb2ff; transform: none; }
  #clock { min-width: 86px; text-align: center; color: #8fc2ff; }
  #hud {
    position: fixed; right: 14px; top: 12px; text-align: right;
    color: rgba(168, 204, 255, 0.60); pointer-events: none;
    font: 11px/1.9 ui-monospace, Menlo, Consolas, monospace;
    letter-spacing: 0.08em; white-space: pre-line;
    text-shadow: 0 1px 4px #000;
  }
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="cap">加载中…</div>
<div id="hud"></div>
<div id="bar">
  <button id="btnPlay">⏸ 暂停</button>
  <div id="segs"></div>
  <span id="clock">0.0s</span>
</div>
<script>
'use strict';
const MANIFEST = /*__MANIFEST__*/{};
const DATA_B64 = "/*__DATA_B64__*/";
const cap = document.getElementById('cap');
const hud = document.getElementById('hud');
const bar = document.getElementById('bar');
const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl2', { antialias: true });
if (!gl) { cap.textContent = '此设备不支持 WebGL2，无法播放回放。'; throw new Error('no webgl2'); }

function b64ToBytes(s) {
  const bin = atob(s); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
const raw = b64ToBytes(DATA_B64);
let off = 0;
function takeF32(n) { const a = new Float32Array(raw.buffer, raw.byteOffset + off, n); off += 4 * n; return a; }
function takeU8(n) { const a = new Uint8Array(raw.buffer, raw.byteOffset + off, n); off += n; off = (off + 3) & ~3; return a; }
const NP = MANIFEST.n_points, MP = MANIFEST.n_path, NF = MANIFEST.n_fig, NW = MANIFEST.n_warp;
const ptsPos = takeF32(3 * NP), ptsCol = takeU8(3 * NP);
const pathPos = takeF32(3 * MP), pathFloor = takeF32(MP), pathArc = takeF32(MP);
const figLocal = takeF32(3 * NF);
const warpTab = takeF32(NW);
const DUR = MANIFEST.duration, FIG_H = MANIFEST.fig_height;
const FIG_RGB = MANIFEST.fig_rgb.map(v => v / 255);

/* ---------------- mat4 (column-major) ---------------- */
function mMul(a, b) { const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; } return o; }
function mPersp(fovy, asp, n, f) { const t = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
  o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o; }
function mLook(eye, ctr, up) { let zx = eye[0] - ctr[0], zy = eye[1] - ctr[1], zz = eye[2] - ctr[2];
  let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  const o = new Float32Array(16);
  o[0] = xx; o[1] = yx; o[2] = zx; o[4] = xy; o[5] = yy; o[6] = zy; o[8] = xz; o[9] = yz; o[10] = zz;
  o[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  o[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  o[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]); o[15] = 1; return o; }

/* ---------------- shaders ---------------- */
function prog(vs, fs) {
  function sh(t, s) { const h = gl.createShader(t); gl.shaderSource(h, s); gl.compileShader(h);
    if (!gl.getShaderParameter(h, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(h) + '\n' + s); return h; }
  const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p;
}
const VS_HEAD = '#version 300 es\nprecision highp float;\nuniform mat4 uMVP;\n';
const gridP = prog(VS_HEAD + `
  layout(location=0) in vec3 aPos; layout(location=1) in float aShade;
  out float vS;
  void main(){ vS = aShade; gl_Position = uMVP * vec4(aPos, 1.0); }`,
  `#version 300 es
  precision highp float; uniform float uAlpha; in float vS; out vec4 o;
  void main(){ o = vec4(vec3(0.24, 0.5, 1.0) * (0.35 + 0.65 * vS), uAlpha * (0.25 + 0.75 * vS)); }`);
const ptsP = prog(VS_HEAD + `
  layout(location=0) in vec3 aPos; layout(location=1) in vec3 aCol;
  uniform float uScale; out vec3 vC;
  void main(){
    vec4 mv = uMVP * vec4(aPos, 1.0);
    gl_PointSize = clamp(3.1 * uScale * (10.0 / -mv.z), 1.2, 8.0);
    vC = aCol; gl_Position = mv; }`,
  `#version 300 es
  precision highp float; in vec3 vC; out vec4 o;
  void main(){ float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.30, d); if (a < 0.02) discard;
    o = vec4(vC, 0.96 * a); }`);
const figP = prog(VS_HEAD + `
  layout(location=0) in vec3 aPos;
  uniform float uScale;
  void main(){
    vec4 mv = uMVP * vec4(aPos, 1.0);
    gl_PointSize = clamp(4.8 * uScale * (10.0 / -mv.z), 2.0, 14.0);
    gl_Position = mv; }`,
  `#version 300 es
  precision highp float; uniform vec3 uCol; out vec4 o;
  void main(){ float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.22, d); if (a < 0.02) discard;
    vec3 c = mix(uCol, vec3(1.0, 0.92, 0.75), smoothstep(0.30, 0.0, d) * 0.35);
    o = vec4(c, 0.97 * a); }`);
const lineP = prog(VS_HEAD + `
  layout(location=0) in vec3 aPos; layout(location=1) in float aS;
  out float vS;
  void main(){ vS = aS; gl_Position = uMVP * vec4(aPos, 1.0); }`,
  `#version 300 es
  precision highp float; uniform float uWalked; in float vS; out vec4 o;
  void main(){
    float done = step(vS, uWalked);
    float head = exp(-abs(vS - uWalked) * 60.0);
    vec3 c = mix(vec3(0.10, 0.17, 0.30), vec3(0.43, 0.75, 1.0), done);
    c += vec3(0.30, 0.42, 0.55) * head;
    o = vec4(c, 0.88); }`);

/* ---------------- buffers ---------------- */
function vao(build) { const v = gl.createVertexArray(); gl.bindVertexArray(v); build(); gl.bindVertexArray(null); return v; }
function attrBuf(loc, data, size, type, norm, usage) {
  const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, data, usage || gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, type || gl.FLOAT, !!norm, 0, 0);
  return b;
}
function uni(p, name) { return gl.getUniformLocation(p, name); }

/* grid: floor grid + grid-filled walls + scattered shards (Detroit
   android-vision dressing, same construction as the process page) */
const bmin = MANIFEST.bbox_min, bmax = MANIFEST.bbox_max, floorY = MANIFEST.floor_y;
let gridVAO, gridCount = 0;
{
  const cx = (bmin[0] + bmax[0]) / 2, cz = (bmin[2] + bmax[2]) / 2;
  const hx = (bmax[0] - bmin[0]) / 2 + 1.2, hz = (bmax[2] - bmin[2]) / 2 + 1.2;
  const topY = bmax[1];
  const pos = [], shade = [];
  function line(a, b, s) { pos.push(a[0], a[1], a[2], b[0], b[1], b[2]); shade.push(s, s); }
  const step = 0.75;
  for (let x = cx - hx; x <= cx + hx + 1e-4; x += step)
    line([x, floorY, cz - hz], [x, floorY, cz + hz], Math.abs(x - cx) < step / 2 ? 0.9 : 0.35);
  for (let z = cz - hz; z <= cz + hz + 1e-4; z += step)
    line([cx - hx, floorY, z], [cx + hx, floorY, z], Math.abs(z - cz) < step / 2 ? 0.9 : 0.35);
  const cs = [[cx - hx, cz - hz], [cx + hx, cz - hz], [cx + hx, cz + hz], [cx - hx, cz + hz]];
  for (let i = 0; i < 4; i++) {
    const a = cs[i], b = cs[(i + 1) % 4];
    line([a[0], floorY, a[1]], [b[0], floorY, b[1]], 1.0);
    line([a[0], topY, a[1]], [b[0], topY, b[1]], 0.85);
    line([a[0], floorY, a[1]], [a[0], topY, a[1]], 0.85);
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    for (let s = step; s < len - 1e-4; s += step) {
      const u = s / len, x = a[0] + (b[0] - a[0]) * u, z = a[1] + (b[1] - a[1]) * u;
      line([x, floorY, z], [x, topY, z], 0.30);
    }
    for (let y = floorY + step; y < topY - 1e-4; y += step)
      line([a[0], y, a[1]], [b[0], y, b[1]], 0.22);
  }
  {
    let seed = 987654321;
    const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const extL = Math.max(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]);
    for (let i = 0; i < 150; i++) {
      const push = 0.45 + 0.75 * rnd();
      const px = cx + (rnd() * 2 - 1) * hx * push;
      const pz = cz + (rnd() * 2 - 1) * hz * push;
      const py = floorY + rnd() * (topY - floorY);
      let dx = rnd() * 2 - 1, dy = (rnd() * 2 - 1) * 0.55, dz = rnd() * 2 - 1;
      const dl = Math.hypot(dx, dy, dz) || 1; dx /= dl; dy /= dl; dz /= dl;
      const L = extL * (0.04 + 0.20 * Math.pow(rnd(), 1.5));
      const s = 0.25 + 0.75 * rnd() * rnd();
      line([px - dx * L / 2, py - dy * L / 2, pz - dz * L / 2],
           [px + dx * L / 2, py + dy * L / 2, pz + dz * L / 2], s);
    }
  }
  gridCount = pos.length / 3;
  gridVAO = vao(() => { attrBuf(0, new Float32Array(pos), 3); attrBuf(1, new Float32Array(shade), 1); });
}

/* scene points */
const ptsVAO = vao(() => {
  attrBuf(0, ptsPos, 3); attrBuf(1, ptsCol, 3, gl.UNSIGNED_BYTE, true);
});

/* route line on the floor (walked part bright, ahead dim) */
let routeVAO;
{
  const pos = new Float32Array(3 * MP);
  for (let i = 0; i < MP; i++) {
    pos[3 * i] = pathPos[3 * i];
    pos[3 * i + 1] = pathFloor[i] + 0.025 * FIG_H;
    pos[3 * i + 2] = pathPos[3 * i + 2];
  }
  routeVAO = vao(() => { attrBuf(0, pos, 3); attrBuf(1, pathArc, 1); });
}

/* figure (positions rebuilt per frame on the CPU) */
let figVAO, figBuf;
const figWorld = new Float32Array(3 * NF);
figVAO = vao(() => { figBuf = attrBuf(0, figWorld, 3, gl.FLOAT, false, gl.DYNAMIC_DRAW); });

/* ---------------- walk state ---------------- */
function warpAt(t) {
  const x = Math.max(0, Math.min(1, t / DUR)) * (NW - 1);
  const i = Math.floor(x), f = x - i;
  return warpTab[i] * (1 - f) + warpTab[Math.min(i + 1, NW - 1)] * f;
}
function sampleAt(u) {
  /* binary search the arc-fraction table, then interpolate */
  let lo = 0, hi = MP - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pathArc[mid] <= u) lo = mid; else hi = mid; }
  const span = Math.max(pathArc[hi] - pathArc[lo], 1e-9);
  const f = Math.max(0, Math.min(1, (u - pathArc[lo]) / span));
  const pos = [0, 1, 2].map(k => pathPos[3 * lo + k] * (1 - f) + pathPos[3 * hi + k] * f);
  const flr = pathFloor[lo] * (1 - f) + pathFloor[hi] * f;
  const j0 = Math.max(lo - 25, 0), j1 = Math.min(hi + 25, MP - 1);
  let tx = pathPos[3 * j1] - pathPos[3 * j0], tz = pathPos[3 * j1 + 2] - pathPos[3 * j0 + 2];
  const tl = Math.hypot(tx, tz);
  if (tl > 1e-9) { tx /= tl; tz /= tl; } else { tx = 0; tz = 1; }
  return { pos, flr, tan: [tx, tz] };
}
function placeFigure(u) {
  const s = sampleAt(u);
  const yaw = Math.atan2(s.tan[0], s.tan[1]);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  for (let i = 0; i < NF; i++) {
    const x = figLocal[3 * i], y = figLocal[3 * i + 1], z = figLocal[3 * i + 2];
    figWorld[3 * i] = s.pos[0] + x * cy + z * sy;
    figWorld[3 * i + 1] = s.flr + y;
    figWorld[3 * i + 2] = s.pos[2] - x * sy + z * cy;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, figBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, figWorld);
  return [s.pos[0], s.flr + 0.55 * FIG_H, s.pos[2]];
}

/* ---------------- camera: gentle follow + manual override --------- */
const centre = [(bmin[0] + bmax[0]) / 2, floorY + 0.34 * (bmax[1] - floorY), (bmin[2] + bmax[2]) / 2];
const extent = Math.max(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]);
let az = 0.75, el = 0.36, radius = extent * 1.28, target = centre.slice();
let lastInteract = -10, perfT = 0;
{
  let drag = null;
  canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, b: e.button }; canvas.setPointerCapture(e.pointerId); lastInteract = perfT; });
  canvas.addEventListener('pointermove', e => {
    if (!drag) return; lastInteract = perfT;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
    if (drag.b === 2 || e.shiftKey) {
      const s = radius * 0.0016, ce = Math.cos(el), se = Math.sin(el), ca = Math.cos(az), sa = Math.sin(az);
      const right = [ca, 0, -sa], upv = [-se * sa, ce, -se * ca];
      for (let k = 0; k < 3; k++) target[k] += (-dx * right[k] + dy * upv[k]) * s;
    } else {
      az -= dx * 0.005; el = Math.max(-1.4, Math.min(1.5, el + dy * 0.005));
    }
  });
  const end = () => { drag = null; };
  canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('wheel', e => { e.preventDefault(); lastInteract = perfT;
    radius = Math.max(extent * 0.15, Math.min(extent * 4, radius * (e.deltaY > 0 ? 1.12 : 0.9))); }, { passive: false });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
}

/* ---------------- timeline ---------------- */
const hash = new URLSearchParams(location.hash.slice(1));
let time = 0, playing = true;
if (hash.has('t')) time = Math.max(0, Math.min(1, parseFloat(hash.get('t')))) * DUR;
const still = hash.get('still') === '1';
if (still) { playing = false; bar.style.display = 'none'; }
const btnPlay = document.getElementById('btnPlay'), clock = document.getElementById('clock');
function setPlayUI() { btnPlay.textContent = playing ? '⏸ 暂停' : '▶ 播放'; }
btnPlay.addEventListener('click', () => { playing = !playing; setPlayUI(); });
setPlayUI();
const NSEG = 56, segsEl = document.getElementById('segs'), segDivs = [];
for (let i = 0; i < NSEG; i++) { const d = document.createElement('div'); d.className = 'seg'; segsEl.appendChild(d); segDivs.push(d); }
let segIdx = -1;
function updateSegs() {
  const idx = Math.min(NSEG, Math.floor(time / DUR * NSEG));
  if (idx === segIdx) return;
  for (let i = 0; i < NSEG; i++)
    segDivs[i].className = 'seg' + (i < idx ? ' on' : i === idx ? ' head' : '');
  segIdx = idx;
}
function seekEvt(e) {
  const r = segsEl.getBoundingClientRect();
  time = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * DUR;
}
segsEl.addEventListener('pointerdown', e => { segsEl.setPointerCapture(e.pointerId); seekEvt(e); });
segsEl.addEventListener('pointermove', e => { if (e.buttons & 1) seekEvt(e); });

/* ---------------- frame ---------------- */
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
}
let lastNow = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - lastNow) / 1000); lastNow = now; perfT = now / 1000;
  if (playing) { time += dt; if (time >= DUR) time -= DUR; }
  resize();
  const u = warpAt(time);
  const anchor = placeFigure(u);
  /* follow: unless the user grabbed the view in the last 4 s, ease
     the orbit onto the walker -- target onto the figure, radius to a
     comfortable follow distance, azimuth drifting slowly onward */
  const manual = perfT - lastInteract < 4;
  if (!manual) {
    const k = 1 - Math.exp(-dt * 2.0);
    for (let i = 0; i < 3; i++) target[i] += (anchor[i] - target[i]) * k;
    radius += (FIG_H * 2.7 - radius) * (1 - Math.exp(-dt * 0.8));
    el += (0.30 - el) * (1 - Math.exp(-dt * 0.8));
    az += dt * 0.045;
  }
  const eye = [target[0] + radius * Math.cos(el) * Math.sin(az),
               target[1] + radius * Math.sin(el),
               target[2] + radius * Math.cos(el) * Math.cos(az)];
  const mvp = mMul(mPersp(0.96, canvas.width / canvas.height, 0.1, 300), mLook(eye, target, [0, 1, 0]));
  const uScale = canvas.height / 850;
  gl.clearColor(0.012, 0.022, 0.042, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  gl.depthMask(false);
  gl.useProgram(gridP); gl.uniformMatrix4fv(uni(gridP, 'uMVP'), false, mvp);
  gl.uniform1f(uni(gridP, 'uAlpha'), 0.30);
  gl.bindVertexArray(gridVAO); gl.drawArrays(gl.LINES, 0, gridCount);

  gl.depthMask(true);
  gl.useProgram(ptsP); gl.uniformMatrix4fv(uni(ptsP, 'uMVP'), false, mvp);
  gl.uniform1f(uni(ptsP, 'uScale'), uScale);
  gl.bindVertexArray(ptsVAO); gl.drawArrays(gl.POINTS, 0, NP);

  gl.depthMask(false);
  gl.useProgram(lineP); gl.uniformMatrix4fv(uni(lineP, 'uMVP'), false, mvp);
  gl.uniform1f(uni(lineP, 'uWalked'), u);
  gl.bindVertexArray(routeVAO); gl.drawArrays(gl.LINE_STRIP, 0, MP);

  gl.depthMask(true);
  gl.useProgram(figP); gl.uniformMatrix4fv(uni(figP, 'uMVP'), false, mvp);
  gl.uniform1f(uni(figP, 'uScale'), uScale);
  gl.uniform3f(uni(figP, 'uCol'), FIG_RGB[0], FIG_RGB[1], FIG_RGB[2]);
  gl.bindVertexArray(figVAO); gl.drawArrays(gl.POINTS, 0, NF);
  gl.depthMask(true);

  const pct = Math.round(u * 100);
  const phase = time < MANIFEST.walk_t0 ? '准备出发' : time > MANIFEST.walk_t1 ? '走完全程' : '正在沿路线行走';
  cap.innerHTML = '<b>ECHO · 路线回放</b>\n' + phase + ' · 已走 ' + pct + '%';
  hud.textContent = 'ROUTE REPLAY\n机位 ' + MANIFEST.n_knots + ' 个\n金色小人 · 示意占位';
  clock.textContent = time.toFixed(1) + 's / ' + DUR.toFixed(0) + 's';
  updateSegs();
}
requestAnimationFrame(frame);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def write_replay(job_dir, out_path) -> dict:
    manifest, blob = build_replay_data(Path(job_dir))
    html = (_PAGE
            .replace("/*__MANIFEST__*/{}",
                     json.dumps(manifest, ensure_ascii=False))
            .replace("/*__DATA_B64__*/", base64.b64encode(blob).decode()))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return {
        "ok": True,
        "out": str(out_path),
        "bytes": out_path.stat().st_size,
        "points": manifest["n_points"],
        "path_samples": manifest["n_path"],
        "knots": manifest["n_knots"],
        "figure_points": manifest["n_fig"],
        "route_length": manifest["route_length"],
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__)
        return 2
    info = write_replay(argv[0], argv[1])
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
