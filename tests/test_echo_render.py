"""Echo Render operator — correctness contract tests.

Verifies echo_super/kernels/echo_render against hand-computed ground
truth (plain Python arithmetic below, independent of the kernel):
  * perspective projection lands on the hand-computed pixels,
  * the nearer point wins an overlap (z-order), in either input order,
  * behind-camera and off-frame points are culled,
  * lines draw where expected; a segment with an endpoint behind the
    camera is skipped whole,
  * glow falloff matches the specified (1 - d/r)^2 weight,
  * empty input renders pure background,
  * output is bit-deterministic across repeated renders,
  * render_sequence writes real PNGs,
  * the registry wrapper produces a "frame" EchoObject.

Run:  .venv/bin/python tests/test_echo_render.py
(needs numba; prints SKIP and exits 0 when numba is unavailable)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import numba  # noqa: F401
except ImportError:
    print("SKIP: numba not installed in this interpreter")
    raise SystemExit(0)

from echo_super.kernels.echo_render import render_frame, render_sequence

W, H = 100, 80
CAM = {"K": [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
       "R": np.eye(3), "t": [0.0, 0.0, 0.0]}
BG = (0, 0, 0)


def px(img, row, col):
    return tuple(int(v) for v in img[row, col])


def hand_uv(p):
    """The projection, computed by hand from the camera definition."""
    x, y, z = p
    return 100.0 * x / z + 50.0, 100.0 * y / z + 40.0


def test_projection():
    pts = np.array([[0.1, 0.2, 2.0], [-0.3, -0.1, 1.0]], dtype=np.float32)
    rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    img = render_frame(pts, rgb, camera=CAM, width=W, height=H,
                       point_radius_px=1.0, glow_strength=0.0,
                       background_rgb=BG)
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    # A -> (u, v) = (55, 50); B -> (20, 30)
    assert px(img, 50, 55) == (255, 0, 0), px(img, 50, 55)
    assert px(img, 49, 54) == (255, 0, 0)
    assert px(img, 30, 20) == (0, 255, 0), px(img, 30, 20)
    assert px(img, 10, 10) == BG
    print("PASS projection")


def test_cube_corners():
    corners, colours = [], []
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
               (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)]
    k = 0
    for sx in (-0.2, 0.2):
        for sy in (-0.2, 0.2):
            for sz in (1.8, 2.2):
                corners.append([sx, sy, sz])
                colours.append(palette[k])
                k += 1
    pts = np.array(corners, dtype=np.float32)
    rgb = np.array(colours, dtype=np.uint8)
    img = render_frame(pts, rgb, camera=CAM, width=W, height=H,
                       point_radius_px=1.0, glow_strength=0.0,
                       background_rgb=BG)
    for p, c in zip(corners, colours):
        u, v = hand_uv(p)
        cu, cv = int(round(u)), int(round(v))
        found = any(px(img, r, col) == c
                    for r in range(cv - 1, cv + 2)
                    for col in range(cu - 1, cu + 2))
        assert found, f"corner {p} colour {c} not at hand-computed ({u:.1f},{v:.1f})"
    print("PASS cube corners (8/8 at hand-computed pixels)")


def test_z_order():
    near = [0.0, 0.0, 1.0]
    far = [0.0, 0.0, 3.0]
    yellow, blue = (255, 255, 0), (0, 0, 255)
    for pts_list, cols in (([far, near], [blue, yellow]),
                           ([near, far], [yellow, blue])):
        pts = np.array(pts_list, dtype=np.float32)
        rgb = np.array(cols, dtype=np.uint8)
        img = render_frame(pts, rgb, camera=CAM, width=W, height=H,
                           point_radius_px=1.5, glow_strength=0.0,
                           background_rgb=BG)
        assert px(img, 40, 50) == yellow, px(img, 40, 50)
    print("PASS z-order (nearer wins, both input orders)")


def test_culling():
    pts = np.array([[0.0, 0.0, -1.0], [10.0, 0.0, 1.0]], dtype=np.float32)
    rgb = np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)
    img = render_frame(pts, rgb, camera=CAM, width=W, height=H,
                       point_radius_px=2.0, glow_strength=0.6,
                       background_rgb=BG)
    assert (img == 0).all(), "culled points leaked into the frame"
    print("PASS culling (behind camera + off frame)")


def test_lines():
    seg = np.array([[[-0.4, 0.0, 2.0], [0.4, 0.0, 2.0]]], dtype=np.float32)
    white = np.array([[255, 255, 255]], dtype=np.uint8)
    empty = np.zeros((0, 3), dtype=np.float32)
    ec = np.zeros((0, 3), dtype=np.uint8)
    img = render_frame(empty, ec, seg, white, camera=CAM, width=W, height=H,
                       background_rgb=BG)
    # u runs 30 -> 70 at v = 40; interior pixel is fully blended.
    assert px(img, 40, 50)[0] >= 250, px(img, 40, 50)
    assert px(img, 40, 20) == BG
    assert px(img, 20, 50) == BG
    # Vertical (steep) segment: u = 50, v runs 25 -> 55.
    seg_v = np.array([[[0.0, -0.3, 2.0], [0.0, 0.3, 2.0]]], dtype=np.float32)
    img_v = render_frame(empty, ec, seg_v, white, camera=CAM, width=W,
                         height=H, background_rgb=BG)
    assert px(img_v, 40, 50)[0] >= 250, px(img_v, 40, 50)
    assert px(img_v, 40, 55) == BG
    # One endpoint behind the camera -> whole segment skipped.
    seg_b = np.array([[[-0.4, 0.0, 2.0], [0.4, 0.0, -1.0]]], dtype=np.float32)
    img_b = render_frame(empty, ec, seg_b, white, camera=CAM, width=W,
                         height=H, background_rgb=BG)
    assert (img_b == 0).all(), "segment with culled endpoint was drawn"
    print("PASS lines (horizontal, steep, culled-endpoint skip)")


def test_glow_falloff():
    # Point projects exactly onto pixel centre (50.5, 40.5).
    pts = np.array([[0.005, 0.005, 1.0]], dtype=np.float32)
    rgb = np.array([[255, 255, 255]], dtype=np.uint8)
    img = render_frame(pts, rgb, camera=CAM, width=W, height=H,
                       point_radius_px=1.0, glow_strength=1.0,
                       glow_radius_factor=2.5, background_rgb=BG)
    assert px(img, 40, 50) == (255, 255, 255)          # core
    # Pixel centre (52.5, 40.5): d = 2.0, w = (1 - 2/2.5)^2 = 0.04 -> 10.
    got = px(img, 40, 52)
    assert abs(got[0] - 10) <= 1, got
    assert px(img, 40, 53) == BG                        # d = 3 > rg = 2.5
    print("PASS glow falloff ((1-d/r)^2 weight)")


def test_empty_and_determinism():
    img = render_frame(np.zeros((0, 3), np.float32),
                       np.zeros((0, 3), np.uint8), camera=CAM,
                       width=W, height=H, background_rgb=(7, 8, 9))
    assert img.shape == (H, W, 3) and (img == np.array([7, 8, 9])).all()

    rng = np.random.default_rng(42)
    pts = rng.uniform(-1.0, 1.0, (5000, 3)).astype(np.float32)
    pts[:, 2] = rng.uniform(0.5, 4.0, 5000).astype(np.float32)
    rgb = rng.integers(0, 256, (5000, 3)).astype(np.uint8)
    lines = rng.uniform(-0.5, 0.5, (20, 2, 3)).astype(np.float32)
    lines[:, :, 2] = 2.0
    a = render_frame(pts, rgb, lines, camera=CAM, width=W, height=H)
    b = render_frame(pts, rgb, lines, camera=CAM, width=W, height=H)
    assert np.array_equal(a, b), "same inputs, different bytes"
    print("PASS empty input + bit determinism")


def test_sequence_driver():
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=np.float32)
        rgb = np.array([[255, 255, 0], [0, 0, 255]], dtype=np.uint8)
        np.save(td / "pts.npy", pts)
        np.save(td / "rgb.npy", rgb)
        spec = {"width": W, "height": H, "point_radius_px": 1.5,
                "glow_strength": 0.0, "background_rgb": [0, 0, 0],
                "frames": [{"png": "f0.png", "points_npy": "pts.npy",
                            "colors_npy": "rgb.npy",
                            "camera": {"K": CAM["K"], "R": np.eye(3).tolist(),
                                       "t": [0, 0, 0]}}]}
        (td / "spec.json").write_text(json.dumps(spec))
        out = render_sequence(td / "spec.json", td / "out")
        assert len(out) == 1 and out[0].exists()
        with Image.open(out[0]) as im:
            assert im.size == (W, H)
            arr = np.asarray(im)
        assert px(arr, 40, 50) == (255, 255, 0)
    print("PASS render_sequence driver")


def test_registry_wrapper():
    from echo_super.core import EchoObject
    from echo_super.kernels import KERNEL_META, register_echo_render_kernels

    reg = register_echo_render_kernels()
    assert "echo_render_frame" in reg.kernels
    kernel = reg.kernels["echo_render_frame"]
    assert kernel.consumes == ("points",) and kernel.produces == "frame"
    pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    rgb = np.array([[255, 0, 0]], dtype=np.uint8)
    obj = EchoObject("points", {"points": pts, "colors": rgb, "camera": CAM},
                     meta={"width": W, "height": H, "glow_strength": 0.0})
    out = kernel.fn(obj)
    assert out.kind == "frame" and out.data.shape == (H, W, 3)
    assert out.meta["evidence"] == KERNEL_META["echo_render_frame"]["evidence"]
    assert px(out.data, 40, 50) == (255, 0, 0)
    print("PASS registry wrapper (echo_render_frame: points -> frame)")


if __name__ == "__main__":
    test_projection()
    test_cube_corners()
    test_z_order()
    test_culling()
    test_lines()
    test_glow_falloff()
    test_empty_and_determinism()
    test_sequence_driver()
    test_registry_wrapper()
    print("ALL ECHO RENDER TESTS PASSED")
