"""Frozen E3 B-path selection: one failed match must not quietly alter split."""
from __future__ import annotations

import pytest

from echo4d.model import DataError
from tum_sensor_metric_mesh import split_names, HOLDOUT_INDICES, ANCHOR_INDICES


def _frames():
    return [{"name":f"frame_{i:05d}.png","t_tum_rgb_s":float(i)/30} for i in range(90)]


def test_pre_registered_48_12_split():
    train, holdout = split_names(_frames())
    assert len(train) == 48 and len(holdout) == 12
    assert set(train).isdisjoint(holdout)
    assert holdout == [f"frame_{i:05d}.png" for i in HOLDOUT_INDICES]
    assert all(f"frame_{i:05d}.png" not in holdout for i in ANCHOR_INDICES)
    assert all(name.startswith("frame_000") for name in train+holdout)


def test_wrong_source_count_fails_closed():
    with pytest.raises(DataError, match="90"):
        split_names(_frames()[:89])


def test_duplicate_registered_name_fails():
    src = _frames()
    src[1]["name"] = src[0]["name"]
    with pytest.raises(DataError, match="Duplicate"):
        split_names(src)
