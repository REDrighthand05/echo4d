# WO-MUS-001 — MUS 高性能渲染算子 (native CPU point/line renderer)

Worker: DeepSeek Flash (implementation) · Architect/reviewer: Muse
Status: DONE 2026-10-31 — with a provenance note: DS Flash was called
twice (full spec, then one tightened re-spec, both deepseek-flash /
16384 tokens / effort low) and BOTH times returned empty content with
finish_reason=length (all 16384 completion tokens consumed as
reasoning; responses kept in experiments/render_op/ds_mus_response*.json).
Per the delegation fallback rule, Muse then wrote the kernel itself
to this exact contract. Verification below is Muse's, as specced.

Delivered: echo_super/kernels/mus_render/ (core.py + __init__.py,
public API render_frame / render_sequence), registry wrapper in
echo_super/kernels/__init__.py (register_mus_kernels, kind points ->
frame, evidence 实测 · 已验）, tests in tests/test_mus_render.py
(9/9 pass: hand-computed projection, cube corners, z-order both input
orders, culling, Wu lines incl. culled-endpoint skip, glow falloff
exact, bit determinism, sequence driver, registry wrapper).
Real still (orbit view, room recognisable — desk, blue monitor, brown
chair): experiments/render_op/room_still_orbit.png.
Bench on this VM (4-core, shared with other agents — treat as a band,
not a point): full 208,666-point cloud, glow on, median frame time
1280x720 ≈ 112–278 ms, 1920x1080 ≈ 178–316 ms across three runs;
cost scales with points drawn. numba was developed/benched in an
isolated venv (experiments/render_op/.venv-mus, numba 0.68.1 +
numpy 2.5.4 — pairing verified compatible) so the shared app .venv
was not mutated while another agent renders from it; adding
numba>=0.68 to the app venv is the remaining integration step.

## Why

Promo/preview frames are currently produced by screenshotting a WebGL
page in headless Chromium (software GL) — slow on the 4-core, GPU-less
VM and not bit-deterministic. Our scenes are only coloured points,
line segments and glow on a dark background: a purpose-built CPU
rasterizer wins by a wide margin and is reproducible.

## Product constraints (hard)

- Name: **MUS**. Package: `mus_render`. Fully generic: plain numpy
  arrays + camera dict + scalar params in, uint8 frame out. NO
  echo-specific file formats, NO hardcoded paths inside the package —
  the identical code must lift into a standalone public repo unchanged.
- Deterministic: projection is parallel (numba prange); splatting runs
  in a fixed depth-sorted (far→near, stable) order; compositing is a
  pure function of the accumulation buffer. Same inputs → same bytes.
- Principled only: it draws exactly the geometry it is given. No
  voting, no averaging of geometry, no estimation.
- numpy + numba only (PIL allowed, lazily, inside the sequence driver
  for PNG output).

## Contract (mus_render/core.py)

```
render_frame(points_xyz, points_rgb, lines_xyz=None, lines_rgb=None,
             camera=None, width=1280, height=720,
             point_radius_px=2.0, glow_strength=0.6,
             glow_radius_factor=2.5, background_rgb=(5, 7, 14),
             near=1e-4) -> uint8[H, W, 3]

camera = {"K": (3,3) intrinsics, "R": (3,3) world->cam, "t": (3,)}
points_xyz float32[N,3]; points_rgb uint8[N,3]
lines_xyz  float32[M,2,3] (may be None/empty); lines_rgb uint8[M,3]
```

Behaviour:
- Pinhole projection Xc = R·X + t, u = fx·Xc/Zc + cx, v = fy·Yc/Zc + cy.
  Points with Zc <= near are culled.
- Draw order: background → lines (1px Xiaolin-Wu AA, alpha-blended,
  segment skipped if either endpoint is behind the near plane) →
  points far→near.
- Each point: additive glow disc (radius = point_radius_px ×
  glow_radius_factor, weight glow_strength·(1 − d/r_glow)²) plus an
  opaque core disc (radius = max(point_radius_px, 0.5)) that
  overwrites — painter order makes the nearest point win each pixel
  of its core disc.
- Final composite: clamp [0,255], round, uint8. N=0 → background+lines.

```
render_sequence(spec_json_path, out_dir) -> list[Path]
```
JSON: global width/height/radius/glow/background + frames[] each with
png name, points_npy (+optional colors_npy), optional lines_npy /
lines_rgb, camera, optional point_count (draw first K — progressive
reveal) or point_indices. Paths resolve relative to the spec file.

## Acceptance (Muse verifies, not the worker)

1. Synthetic cube + lines: projection lands within 1px of hand-computed
   pixels; nearer colour wins an overlap; behind-camera point absent.
2. Real still: 208,666-point room cloud renders recognisably.
3. Benchmark on the VM at 720p/1080p, reported as measured.
