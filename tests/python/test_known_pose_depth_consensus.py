"""Synthetic checks for observation-preserving RGB depth regularization."""
import numpy as np
import pytest
from echo4d.known_pose_depth_consensus import regularize_observed_depth


def test_flat_noisy_observed_surface_improves_without_filling():
    rng = np.random.default_rng(123)
    z = (2 + rng.normal(0, .02, (60, 80))).astype('float32')
    z[10:20, 10:20] = 0
    rgb = np.full((60, 80, 3), 128, dtype='uint8')
    result, stats = regularize_observed_depth(z, rgb)
    assert np.all(result[10:20, 10:20] == 0)
    assert np.mean(np.abs(result[25:45, 25:55] - 2)) < np.mean(np.abs(z[25:45, 25:55] - 2))
    assert stats['newly_filled_pixels'] == 0


def test_depth_and_color_edge_preserved():
    z = np.full((40, 60), 1.5, dtype='float32')
    z[:, 30:] = 3
    rgb = np.full((40, 60, 3), 40, dtype='uint8')
    rgb[:, 30:] = 200
    result, _ = regularize_observed_depth(z, rgb)
    assert np.array_equal(result, z)


def test_single_speckle_removed_and_unknown_stays_unknown():
    z = np.zeros((30, 30), dtype='float32')
    z[15, 15] = 2
    rgb = np.zeros((30, 30, 3), dtype='uint8')
    result, stats = regularize_observed_depth(z, rgb)
    assert not result.any() and stats['removed_isolated_pixels'] == 1


def test_invalid_depth_rejected():
    z = np.ones((20, 20), dtype='float32')
    z[2, 2] = np.nan
    with pytest.raises(ValueError):
        regularize_observed_depth(z, np.zeros((20, 20, 3), dtype='uint8'))


def test_recording_receipt_matches_regularized_depth(monkeypatch, tmp_path):
    import sys
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'tools'))
    import tum_plane_sweep as runner
    depth = np.zeros((240, 320), np.float32)
    depth[20:30, 20:30] = 2
    depth[100, 100] = 2
    pixels = depth.size
    stats = {'valid_depth_pixels': 101, 'total_pixels': pixels,
             'valid_fraction': 101 / pixels, 'unknown_pixels': pixels - 101,
             'adaptive_reference_pixels': 30, 'adaptive_valid_depth_pixels': 20,
             'penalized_photo_samples': 0, 'occluded_photo_samples': 0}
    selected = [(i, [8, 11]) for i in [0, 1, 2, 3, 6, 7]]
    monkeypatch.setattr(runner, 'select_plane_sweep_views', lambda *a, **k: (selected, {}))
    monkeypatch.setattr(runner, 'load_stereo_rgb', lambda *a: np.zeros((240, 320, 3), np.uint8))
    monkeypatch.setattr(runner, 'infer_plane_sweep', lambda *a, **k: (depth.copy(), stats.copy()))
    frames = [{'rgb': Path('unused'), 'pose': np.eye(4), 't_s': float(i)} for i in range(60)]
    report = {}
    runner.build_plane_sweep_recording(frames, tmp_path / 'recording', report,
                                      depth_regularization='rgb-consensus')
    actual = np.load(tmp_path / 'recording/depth/00000.npy', allow_pickle=False)
    valid = int(np.count_nonzero(actual))
    receipt = report['plane_sweep']['per_anchor'][0]
    assert valid == 100
    assert receipt['valid_depth_pixels'] == valid
    assert receipt['unknown_pixels'] == pixels - valid
    assert receipt['valid_fraction'] == valid / pixels
    assert receipt['inference_before_regularization']['valid_depth_pixels'] == 101
