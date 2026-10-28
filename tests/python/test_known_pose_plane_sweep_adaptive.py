"""Synthetic weak-texture oracle; never counts as a real-scene quality PASS."""
import numpy as np
import pytest

pytest.importorskip("cv2")
from echo4d.model import Camera, DataError
from echo4d.known_pose_plane_sweep import infer_plane_sweep


def _weak_texture_plane(*, flat=False):
    camera = Camera(160, 120, 160., 160., 79.5, 59.5)
    rng = np.random.default_rng(42)
    if flat:
        gray = np.full((120, 160), 127, dtype=np.uint8)
    else:
        # Weak real image-like texture: 5x5 ZNCC has insufficient variance,
        # while a wider observed neighbourhood contains matching evidence.
        gray = np.clip(127 + rng.standard_normal((120, 160)) * 7,
                       0, 255).astype(np.uint8)
    reference = np.repeat(gray[:, :, None], 3, axis=2)
    sources = []
    for tx in (-.1, .1):
        pose = np.eye(4)
        pose[0, 3] = tx
        shift = round(camera.fx * tx / 2.)
        source = np.zeros_like(reference)
        if shift > 0:
            source[:, :-shift] = reference[:, shift:]
        else:
            source[:, -shift:] = reference[:, :shift]
        sources.append((source, pose))
    return camera, reference, sources


def test_adaptive_recovers_real_correspondence_in_weak_texture():
    camera, reference, sources = _weak_texture_plane()
    fixed, _ = infer_plane_sweep(reference, sources, camera, np.eye(4),
                                 window_mode="fixed")
    adaptive, stats = infer_plane_sweep(reference, sources, camera, np.eye(4),
                                        window_mode="adaptive")
    assert stats["adaptive_reference_pixels"] > 3000
    assert stats["adaptive_valid_depth_pixels"] > 2000
    assert np.count_nonzero(adaptive) > 3 * np.count_nonzero(fixed)
    supported = adaptive[adaptive > 0]
    assert np.mean(np.abs(supported - 2.) < .15) > .95
    assert stats["measured_depth_used"] is False


def test_adaptive_does_not_invent_a_textureless_plane():
    camera, reference, sources = _weak_texture_plane(flat=True)
    depth, stats = infer_plane_sweep(reference, sources, camera, np.eye(4),
                                     window_mode="adaptive", planes=16)
    assert not np.any(depth)
    assert stats["adaptive_reference_pixels"] == 0


def test_invalid_window_mode_rejected():
    camera, reference, sources = _weak_texture_plane()
    with pytest.raises(DataError, match="window mode"):
        infer_plane_sweep(reference, sources, camera, np.eye(4),
                          window_mode="hallucinate")


@pytest.mark.parametrize("window_mode", ["fixed", "adaptive"])
@pytest.mark.parametrize("rotated", [False, True])
def test_duplicate_source_center_cannot_vote_twice(window_mode, rotated):
    camera, reference, sources = _weak_texture_plane()
    image, pose = sources[0]
    duplicate_pose = pose.copy()
    if rotated:
        theta = .1
        duplicate_pose[:3, :3] = [[np.cos(theta), -np.sin(theta), 0],
                                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    # An otherwise valid opposite-side source must not hide the duplicate.
    repeated = [sources[0], sources[1], (image.copy(), duplicate_pose)]
    with pytest.raises(DataError, match="distinct source camera centers"):
        infer_plane_sweep(reference, repeated, camera, np.eye(4),
                          window_mode=window_mode)
