"""Guards for real Stanford 15mm sparse TSDF A/B (no Open3D required)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from stanford_sparse_15mm import assert_native_intrinsics, frozen_indices


def test_exact_480_unique_recorded_frame_set():
    baseline = list(range(0, 720, 2))  # 360 measured frames
    extra = list(range(1, 240, 2))  # 120 different measured frames
    actual = frozen_indices(3000, baseline, extra)
    assert len(actual) == len(set(actual)) == 480
    assert set(actual) == set(baseline + extra)


def test_refuse_reused_frame_and_missing_frame():
    baseline = list(range(360))
    extra = list(range(360, 480))
    with pytest.raises(ValueError, match="Repeated"):
        frozen_indices(3000, baseline, extra[:-1] + [10])
    with pytest.raises(ValueError, match="360"):
        frozen_indices(3000, baseline[:-1], extra)
    with pytest.raises(ValueError, match="out of range"):
        frozen_indices(500, baseline, extra[:-1] + [501])


def test_native_calibration_rejects_hidden_downsampling():
    k = (640, 480, 525., 525., 319.5, 239.5)
    assert_native_intrinsics(k, (480, 640, 3), (480, 640))
    with pytest.raises(ValueError, match="native calibrated"):
        assert_native_intrinsics(k, (240, 320, 3), (240, 320))
    with pytest.raises(ValueError, match="native calibrated"):
        assert_native_intrinsics((320, 240, 262.5, 262.5, 159.75, 119.75),
                                 (480, 640, 3), (480, 640))
    with pytest.raises(ValueError, match="native calibrated"):
        assert_native_intrinsics((640, 480, 525., 525., float("nan"), 239.5),
                                 (480, 640, 3), (480, 640))
