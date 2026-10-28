"""Synthetic geometry oracle and failure counterexamples, NOT real TUM evidence."""
import numpy as np
import pytest
pytest.importorskip('cv2')
from echo4d.model import Camera, DataError
from echo4d.known_pose_plane_sweep import infer_plane_sweep


def sample():
    c = Camera(160, 120, 160., 160., 79.5, 59.5)
    ref = np.random.default_rng(42).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    sources = []
    for tx in (-.1, .1):
        pose = np.eye(4)
        pose[0, 3] = tx
        shift = round(c.fx * tx / 2.)
        src = np.zeros_like(ref)
        if shift > 0:
            src[:, :-shift] = ref[:, shift:]
        else:
            src[:, -shift:] = ref[:, :shift]
        sources.append((src, pose))
    return c, ref, sources


def test_two_source_metric_plane_oracle():
    c, ref, sources = sample()
    depth, stats = infer_plane_sweep(ref, sources, c, np.eye(4))
    assert stats['measured_depth_used'] is False
    assert stats['source_view_count'] == 2
    assert stats['valid_depth_pixels'] > 8000
    assert abs(float(np.median(depth[depth > 0])) - 2.) < .08


def test_textureless_scene_remains_unknown():
    c, ref, sources = sample()
    ref[:] = 127
    sources = [(np.full_like(ref, 127), pose) for _, pose in sources]
    depth, _ = infer_plane_sweep(ref, sources, c, np.eye(4), planes=16)
    assert not np.any(depth)


def test_invalid_sources_and_unsafe_ranges_rejected():
    c, ref, sources = sample()
    with pytest.raises(DataError):
        infer_plane_sweep(ref, sources[:1], c, np.eye(4))
    with pytest.raises(DataError):
        infer_plane_sweep(ref, sources, c, np.eye(4), near=0.1)
    with pytest.raises(DataError):
        infer_plane_sweep(ref, sources, c, np.eye(4), planes=8)
    with pytest.raises(DataError):
        infer_plane_sweep(ref, [(src[:80], pose) for src, pose in sources], c, np.eye(4))
    with pytest.raises(DataError):
        infer_plane_sweep(ref, [(src, np.eye(4)) for src, _ in sources], c, np.eye(4))
