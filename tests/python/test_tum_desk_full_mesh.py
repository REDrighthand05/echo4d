"""Frozen desk cohort tests; no network/COLMAP required."""
from __future__ import annotations
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from echo4d.model import DataError
from tum_desk_full_mesh import (
    DEPTH_UNITS_PER_M, FRAME_COUNT, GEOMETRY_HELDOUT, SCALE_ANCHORS,
    SCALE_DIAGNOSTICS, frozen_protocol, match_sensor_inputs,
    sensor_pairing_diagnostics,
)

def fake_source():
    return [{'name': f'frame_{i:05d}.png', 't_tum_rgb_s': float(i)}
            for i in range(FRAME_COUNT)]

def test_frozen_roles_and_parameters():
    p = frozen_protocol(fake_source())
    assert len(p['train_names']) == 48
    assert len(p['holdout_names']) == 12
    assert p['holdout_names'] == [f'frame_{i:05d}.png' for i in GEOMETRY_HELDOUT]
    assert set(p['train_names']).isdisjoint(p['holdout_names'])
    assert set(GEOMETRY_HELDOUT).isdisjoint(SCALE_ANCHORS)
    assert set(GEOMETRY_HELDOUT).isdisjoint(SCALE_DIAGNOSTICS)
    assert set(SCALE_DIAGNOSTICS).isdisjoint(range(60))
    assert p['depth_numeric_units_per_meter'] == DEPTH_UNITS_PER_M
    assert p['max_rgb_depth_gap_s'] == 0.03
    assert p['voxel_m'] == 0.07 and p['max_voxels'] == 1_500_000

@pytest.mark.parametrize('mode', ['missing', 'extra', 'swapped', 'duplicated'])
def test_source_cohort_cannot_change(mode):
    source = fake_source()
    if mode == 'missing': source.pop()
    elif mode == 'extra': source.append(source[-1].copy())
    elif mode == 'swapped': source[4], source[5] = source[5], source[4]
    else: source[12] = source[11].copy()
    with pytest.raises(DataError, match='frozen 75'):
        frozen_protocol(source)

def make_index(tum, *, offset=0.0, duplicate=False):
    (tum / 'depth').mkdir(parents=True)
    rows = ['# time relative_path']
    for i in range(FRAME_COUNT):
        name = f'depth/{i:05d}.png'
        if duplicate and i == 19: name = 'depth/00018.png'
        (tum / name).touch(exist_ok=True)
        rows.append(f'{i+offset:.6f} {name}')
    (tum / 'depth.txt').write_text('\n'.join(rows) + '\n')

def test_measured_pairing_frozen_names_and_time(tmp_path):
    make_index(tmp_path, offset=0.01)
    source = fake_source()
    mapping = match_sensor_inputs(tmp_path, source, frozen_protocol(source))
    assert len(mapping) == 68
    assert mapping['frame_00004.png']['depth'] == 'depth/00004.png'
    assert mapping['frame_00065.png']['depth'] == 'depth/00065.png'
    assert mapping['frame_00004.png']['timestamp_delta_seconds'] == pytest.approx(.01)

def test_large_depth_gap_fails_not_reselected(tmp_path):
    make_index(tmp_path, offset=.04)
    source = fake_source()
    with pytest.raises(DataError, match='within 30ms'):
        match_sensor_inputs(tmp_path, source, frozen_protocol(source))

def test_duplicate_depth_fails_before_mesh(tmp_path):
    make_index(tmp_path, duplicate=True)
    source = fake_source()
    with pytest.raises(DataError, match='Duplicate or absent depth source'):
        match_sensor_inputs(tmp_path, source, frozen_protocol(source))

def test_missing_depth_fails_before_mesh(tmp_path):
    make_index(tmp_path)
    (tmp_path / 'depth/00004.png').unlink()
    source = fake_source()
    with pytest.raises(DataError, match='Missing frame file or path outside dataset'):
        match_sensor_inputs(tmp_path, source, frozen_protocol(source))


def test_pairing_diagnostics_reports_all_frozen_roles_without_reselection(tmp_path):
    make_index(tmp_path, offset=.04)
    source = fake_source()
    frozen = frozen_protocol(source)
    audited = sensor_pairing_diagnostics(tmp_path, source, frozen)
    assert audited['frozen_rgb_roles'] == 68
    assert audited['outside_gap_count'] == 68
    assert audited['within_gap_count'] == 0
    assert audited['max_allowed_gap_s'] == .03
    assert audited['first_outside_gap']['name'] == 'frame_00000.png'
    assert audited['first_outside_gap']['absolute_gap_s'] == pytest.approx(.04)
    assert audited['first_outside_gap']['nearest_depth_path'] == 'depth/00000.png'
    assert len(audited['pairs']) == 68
    assert all(not row['within_frozen_30ms'] for row in audited['pairs'])
    # Merely observing the nearest out-of-window depth MUST NEVER authorize
    # feeding it to scale calibration or geometric training.
    with pytest.raises(DataError, match='within 30ms'):
        match_sensor_inputs(tmp_path, source, frozen)


def test_pairing_diagnostics_matches_successful_frozen_gate(tmp_path):
    make_index(tmp_path, offset=.01)
    source = fake_source()
    frozen = frozen_protocol(source)
    audited = sensor_pairing_diagnostics(tmp_path, source, frozen)
    assert audited['all_pairs_within_gap_and_unique'] is True
    assert audited['within_gap_count'] == 68
    assert audited['outside_gap_count'] == 0
    assert audited['median_abs_gap_s'] == pytest.approx(.01)
    assert audited['first_outside_gap'] is None
    assert len(match_sensor_inputs(tmp_path, source, frozen)) == 68


def test_pairing_diagnostics_exposes_duplicate_nearest_depth(tmp_path):
    make_index(tmp_path, duplicate=True)
    source = fake_source()
    frozen = frozen_protocol(source)
    audited = sensor_pairing_diagnostics(tmp_path, source, frozen)
    assert audited['duplicate_nearest_depth_count'] >= 1
    assert audited['all_pairs_within_gap_and_unique'] is False
    assert any(row['duplicate_nearest_depth'] for row in audited['pairs'])
    with pytest.raises(DataError, match='Duplicate or absent depth source'):
        match_sensor_inputs(tmp_path, source, frozen)

