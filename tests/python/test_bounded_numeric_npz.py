import io
import zipfile

import numpy as np
import pytest

from bounded_numeric_npz import read_numeric_npz, MAX_BYTES

SCHEMA = {'vertices': (3, 2000000, 3, 'fiu'), 'faces': (1, 3000000, 3, 'iu')}


def npy(value):
    stream = io.BytesIO()
    np.save(stream, value)
    return stream.getvalue()


@pytest.mark.parametrize('fault', ['shape', 'expanded', 'duplicate', 'extra', 'object'])
def test_bad_archive_rejected_before_any_array_allocation(tmp_path, monkeypatch, fault):
    p = tmp_path / 'bad.npz'
    vertices = npy(np.zeros((3, 3)))
    if fault == 'shape':
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {
            'shape': (2000001, 3), 'fortran_order': False, 'descr': '<f8'})
        vertices = stream.getvalue()
    if fault == 'expanded':
        vertices = b'\0' * (MAX_BYTES + 1)
    if fault == 'object':
        vertices = npy(np.full((3, 3), object(), dtype=object))
    with zipfile.ZipFile(p, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('vertices.npy', vertices)
        z.writestr('faces.npy', npy(np.array([[0, 1, 2]])))
        if fault == 'duplicate':
            with pytest.warns(UserWarning):
                z.writestr('vertices.npy', vertices)
        if fault == 'extra':
            z.writestr('unused.npy', b'ignored')
    def forbidden(*args, **kwargs):
        pytest.fail('array materialized before resource preflight')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError):
        read_numeric_npz(p, SCHEMA)


def test_valid_compressed_geometry_roundtrips(tmp_path):
    p = tmp_path / 'mesh.npz'
    v, f = np.zeros((3, 3)), np.array([[0, 1, 2]])
    np.savez_compressed(p, vertices=v, faces=f)
    data = read_numeric_npz(p, SCHEMA)
    assert np.array_equal(data['vertices'], v)
    assert np.array_equal(data['faces'], f)
