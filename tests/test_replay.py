#!/usr/bin/env python3
"""Route-replay regression locks.

Layers:
  1. fixture job (synthetic poses + floor cloud): the embedded path
     has the kernel's sample count, passes through every pose knot,
     and is byte-identical to calling build_replay_path directly --
     the stage must reuse the registered kernel, not a parallel stack;
  2. the written page is standalone: data embedded, no external
     references of any kind;
  3. pipeline contract: stage_replay writes replay.html + job.json,
     and a degenerate job (a single pose) degrades to
     replay_applied=False instead of failing the job;
  4. determinism: two builds of the same job are byte-identical.

Run: .venv/bin/python tests/test_replay.py
"""
import base64
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echo_scene.fuse import _write_points_bin            # noqa: E402
from echo_scene.pipeline import stage_replay             # noqa: E402
from echo_scene.replay import (SAMPLES_PER_SEGMENT,      # noqa: E402
                               build_replay_data, write_replay)
from echo_super.ds_batch1 import build_replay_path       # noqa: E402

FAILURES = []


def _check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILURES.append(name)


# fixture route: six camera centres along a gentle S, identity
# rotations (cam_from_world = I -> centre = -t), eyes at y=1.6.
CENTRES = np.array([
    [0.0, 1.6, 0.0],
    [1.0, 1.6, 0.3],
    [2.0, 1.6, 0.5],
    [3.0, 1.6, 0.3],
    [4.0, 1.6, 0.0],
    [5.0, 1.6, -0.2],
])


def _fixture_job(job: Path) -> Path:
    job.mkdir(parents=True, exist_ok=True)
    poses = {}
    for k, c in enumerate(CENTRES):
        poses[f"frame_{k + 1:05d}.jpg"] = {
            "cam_from_world_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "cam_from_world_t": [float(-v) for v in c],
            "camera": {"model": "PINHOLE", "width": 1920, "height": 1080,
                       "params": [1600.0, 960.0, 540.0]},
        }
    (job / "poses.json").write_text(json.dumps(poses))
    (job / "alignment.json").write_text(json.dumps(
        {"rotation": np.eye(3).tolist(), "note": "fixture identity"}))
    rng = np.random.default_rng(3)
    gx, gz = np.meshgrid(np.arange(-1.0, 7.01, 0.25),
                         np.arange(-2.0, 2.01, 0.25))
    floor = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    inner = rng.uniform(low=[-0.5, 0.1, -1.5], high=[5.5, 2.2, 1.5],
                        size=(1500, 3))
    pts = np.concatenate([floor, inner])
    rgb = rng.integers(40, 230, size=(len(pts), 3)).astype(np.uint8)
    _write_points_bin(pts, rgb, job / "points.bin")
    return job


def _decode_blob(html: str, manifest: dict):
    b64 = re.search(r'const DATA_B64 = "([A-Za-z0-9+/=]+)";', html).group(1)
    raw = base64.b64decode(b64)
    off = 0

    def take(nbytes):
        nonlocal off
        part = raw[off:off + nbytes]
        off += nbytes + (-nbytes % 4)
        return part

    n, m, f = manifest["n_points"], manifest["n_path"], manifest["n_fig"]
    pts = np.frombuffer(take(12 * n), dtype="<f4").reshape(-1, 3)
    take(3 * n)                                            # colours
    path = np.frombuffer(take(12 * m), dtype="<f4").reshape(-1, 3)
    floor = np.frombuffer(take(4 * m), dtype="<f4")
    arc = np.frombuffer(take(4 * m), dtype="<f4")
    fig = np.frombuffer(take(12 * f), dtype="<f4").reshape(-1, 3)
    warp = np.frombuffer(take(4 * manifest["n_warp"]), dtype="<f4")
    return pts, path, floor, arc, fig, warp


def fixture_checks():
    print("[fixture job: embedded data]")
    with tempfile.TemporaryDirectory() as td:
        job = _fixture_job(Path(td) / "job")
        manifest, blob = build_replay_data(job)
        out = Path(td) / "replay.html"
        info = write_replay(job, out)
        html = out.read_text(encoding="utf-8")

        n_knots = len(CENTRES)
        want_path = (n_knots - 1) * SAMPLES_PER_SEGMENT + 1
        _check("path sample count", manifest["n_path"] == want_path,
               f"{manifest['n_path']} == {want_path}")
        _check("knot count", manifest["n_knots"] == n_knots)

        pts, path, floor, arc, fig, warp = _decode_blob(html, manifest)
        # every knot is hit exactly at its segment start (kernel rule)
        worst = max(
            float(np.abs(path[k * SAMPLES_PER_SEGMENT]
                         - CENTRES[k]).max())
            for k in range(n_knots))
        _check("path passes through every pose knot", worst < 1e-6,
               f"worst deviation {worst:.2e}")
        # and the embedded path IS the registered kernel's output
        targets = CENTRES + np.array([0.0, 0.0, 1.0])
        want, _ = build_replay_path(CENTRES, targets, SAMPLES_PER_SEGMENT)
        _check("path identical to build_replay_path output",
               bool(np.allclose(path, want.astype(np.float32), atol=1e-6)),
               f"max diff {float(np.abs(path - want).max()):.2e}")
        _check("figure present", manifest["n_fig"] > 3000,
               f"{manifest['n_fig']} points")
        _check("figure spans its calibrated height",
               abs(float(fig[:, 1].max()) - manifest["fig_height"]) < 0.25,
               f"fig y max {float(fig[:, 1].max()):.3f} vs "
               f"{manifest['fig_height']:.3f}")
        _check("figure height from camera height (1.6/0.93)",
               abs(manifest["fig_height"] - 1.6 / 0.93) < 0.05,
               f"{manifest['fig_height']:.3f}")
        _check("floor finite everywhere", bool(np.isfinite(floor).all()))
        _check("arc fraction spans 0..1",
               abs(float(arc[0])) < 1e-6 and abs(float(arc[-1]) - 1) < 1e-6)
        _check("warp table monotonic 0..1",
               bool((np.diff(warp) >= -1e-6).all())
               and abs(float(warp[0])) < 1e-6
               and abs(float(warp[-1]) - 1) < 1e-6)
        _check("route length positive", manifest["route_length"] > 5.0,
               f"{manifest['route_length']:.2f}")
        _check("write info agrees", info["path_samples"] == want_path
               and info["knots"] == n_knots)

        print("[standalone page]")
        for bad in ("http://", "https://", "<script src", "fetch(",
                    "@import", "url("):
            _check(f"no external reference {bad!r}", bad not in html)
        _check("page carries the replay title", "路线回放" in html)
        _check("page carries the embedded blob", len(html) > 100_000,
               f"{len(html)} bytes")

        print("[determinism]")
        out2 = Path(td) / "replay2.html"
        write_replay(job, out2)
        _check("two builds byte-identical",
               out.read_bytes() == out2.read_bytes())


def stage_checks():
    print("[pipeline stage contract]")
    with tempfile.TemporaryDirectory() as td:
        job = _fixture_job(Path(td) / "job")
        info = stage_replay(job)
        _check("stage ok", info.get("ok") is True)
        _check("replay applied", info.get("replay_applied") is True)
        _check("replay.html written", (job / "replay.html").exists())
        saved = json.loads((job / "job.json").read_text())
        _check("job.json records the stage", saved.get("stage") == "replay")

    print("[degenerate job: one pose]")
    with tempfile.TemporaryDirectory() as td:
        job = Path(td) / "job"
        job.mkdir()
        poses = {"frame_00001.jpg": {
            "cam_from_world_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "cam_from_world_t": [0.0, -1.6, 0.0],
            "camera": {"model": "PINHOLE", "width": 1920, "height": 1080,
                       "params": [1600.0, 960.0, 540.0]}}}
        (job / "poses.json").write_text(json.dumps(poses))
        _write_points_bin(np.zeros((10, 3)),
                          np.full((10, 3), 120, dtype=np.uint8),
                          job / "points.bin")
        try:
            build_replay_data(job)
            raised = False
        except ValueError:
            raised = True
        _check("builder refuses a one-pose route", raised)
        info = stage_replay(job)
        _check("stage stays non-fatal", info.get("ok") is True
               and info.get("replay_applied") is False)


def main():
    fixture_checks()
    stage_checks()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"FAILURES: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
