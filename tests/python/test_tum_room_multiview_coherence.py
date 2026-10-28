"""CPU checks for RGB-gated depth stabilization (not a real room Mesh test)."""
import numpy as np
import pytest

pytest.importorskip("cv2")
from echo4d.model import DataError
from tum_room_multiview_coherence import stabilize_depth


def test_repairs_isolated_wrong_depth_without_inventing_missing_z():
    rgb = np.full((24, 24, 3), 100, dtype=np.uint8)
    z = np.full((24, 24), 2.0, dtype=np.float32)
    z[12, 12] = 3.0
    z[6, 6] = 0.
    corrected, receipt = stabilize_depth(rgb, z)
    assert abs(float(corrected[12, 12]) - 2.) < 1e-5
    assert corrected[6, 6] == 0.
    assert receipt["corrected_spikes"] >= 1
    assert receipt["unknown_pixels_filled"] == 0
    assert np.array_equal(corrected == 0, z == 0)


def test_rgb_edge_protects_real_two_plane_depth_discontinuity():
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[:, :12] = [220, 20, 20]
    rgb[:, 12:] = [20, 20, 220]
    z = np.ones((24, 24), dtype=np.float32)
    z[:, 12:] = 2.
    corrected, _ = stabilize_depth(rgb, z)
    assert np.max(np.abs(corrected - z)) < 1e-5


def test_all_unknown_stays_unknown():
    rgb = np.full((12, 12, 3), 80, dtype=np.uint8)
    z = np.zeros((12, 12), dtype=np.float32)
    corrected, receipt = stabilize_depth(rgb, z)
    assert not np.any(corrected)
    assert receipt["corrected_spikes"] == 0


def test_invalid_inferred_depth_is_rejected():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    z = np.zeros((8, 8), dtype=np.float32)
    z[3, 3] = np.nan
    with pytest.raises(DataError):
        stabilize_depth(rgb, z)
