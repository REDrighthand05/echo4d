"""Positive and counterexample tests for bounded stereo backup extension."""
import numpy as np
import pytest
from echo4d.stereo_surface_extension import extend_contiguous_stereo_depth


def test_backup_extends_surface_without_inpainting_unknown():
    rgb = np.full((12, 12, 3), 100, np.uint8)
    base = np.zeros((12, 12), np.float32); base[2:9, 2:8] = 2.
    backup = np.zeros_like(base); backup[3:8, 7:10] = 2.03
    result, stats = extend_contiguous_stereo_depth([base, backup], rgb)
    assert stats['selected_candidate_index'] == 0
    assert stats['added_pixels'] == 10
    assert result[4, 9] == pytest.approx(2.03)
    assert result[0, 0] == 0
    assert np.array_equal(result[2:9, 2:7], base[2:9, 2:7])


def test_color_and_depth_discontinuities_stay_unknown():
    rgb = np.full((12, 12, 3), 100, np.uint8); rgb[:, 8:] = 240
    base = np.zeros((12, 12), np.float32); base[2:10, 2:8] = 2.
    backup = np.zeros_like(base); backup[3:9, 7:11] = 2.02
    out, _ = extend_contiguous_stereo_depth([base, backup], rgb)
    assert out[5, 8] == 0 and out[5, 7] == 2.
    bad = np.zeros_like(base); bad[3:9, 7:11] = 3.
    out, stats = extend_contiguous_stereo_depth([base, bad], np.full_like(rgb, 100))
    assert stats['added_pixels'] == 0 and out[5, 9] == 0


def test_conflicting_backups_stay_unknown():
    rgb = np.full((12, 12, 3), 100, np.uint8)
    base = np.zeros((12, 12), np.float32); base[2:10, 2:8] = 2.
    a = np.zeros_like(base); a[3:9, 7:10] = 1.94
    b = np.zeros_like(base); b[3:9, 7:10] = 2.10
    out, stats = extend_contiguous_stereo_depth([base, a, b], rgb)
    assert stats['conflict_pixels'] > 0 and out[5, 8] == 0


def test_invalid_inputs_fail_closed():
    rgb = np.zeros((8, 8, 3), np.uint8); d = np.zeros((8, 8), np.float32)
    for depths, color in (([d], rgb), ([d]*4, rgb), ([d, d[:3]], rgb),
                          ([d, np.full_like(d, np.nan)], rgb), ([d, d], rgb[:3])):
        with pytest.raises(ValueError):
            extend_contiguous_stereo_depth(depths, color)
