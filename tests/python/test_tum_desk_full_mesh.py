"""Frozen desk cohort tests; no network/COLMAP required."""
from __future__ import annotations
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from echo4d.model import DataError
from tum_desk_full_mesh import (
    DEPTH_UNITS_PER_M, FRAME_COUNT, GEOMETRY_HELDOUT, SCALE_ANCHORS,
    SCALE_DIAGNOSTICS, frozen_protocol, match_sensor_inputs, audit_sensor_pairing,
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


def test_sensor_time_audit_reports_all_68_predeclared_roles(tmp_path):
    make_index(tmp_path, offset=.01)
    source = fake_source()
    audit = audit_sensor_pairing(tmp_path, source, frozen_protocol(source))
    assert audit['frozen_roles_count'] == 68
    assert audit['frozen_protocol_eligible'] is True
    assert audit['matched_within_30ms'] == 68
    assert audit['distinct_and_within_30ms'] == 68
    assert len(audit['per_frame']) == 68
    assert audit['per_frame'][0]['signed_delta_s'] == pytest.approx(.01)
    assert audit['max_gap_s_frozen'] == .03


def test_sensor_time_audit_never_rescues_out_of_window_frames(tmp_path):
    make_index(tmp_path, offset=.04)
    source = fake_source()
    audit = audit_sensor_pairing(tmp_path, source, frozen_protocol(source))
    assert audit['frozen_protocol_eligible'] is False
    assert audit['matched_within_30ms'] == 0
    assert len(audit['first_invalid_frozen_frames']) == 20
    assert len(audit['per_frame']) == 68
    assert audit['per_frame'][0]['closest_sensor_time_s'] == pytest.approx(.04)
    assert audit['per_frame'][0]['eligible_without_reselection'] is False
    # Original strict rejection remains intact, never silently taking +40ms.
    with pytest.raises(DataError, match='within 30ms'):
        match_sensor_inputs(tmp_path, source, frozen_protocol(source))


def test_sensor_time_audit_flags_repeated_depth_row(tmp_path):
    make_index(tmp_path)
    source = fake_source()
    # Two distinct RGB stamps can fall inside the same measured depth timestamp.
    source[1]['t_tum_rgb_s'] = .004
    audit = audit_sensor_pairing(tmp_path, source, frozen_protocol(source))
    a, b = audit['per_frame'][0], audit['per_frame'][1]
    assert a['within_frozen_30ms'] and b['within_frozen_30ms']
    assert a['distinct_sensor_row'] is True
    assert b['distinct_sensor_row'] is False
    assert audit['frozen_protocol_eligible'] is False
