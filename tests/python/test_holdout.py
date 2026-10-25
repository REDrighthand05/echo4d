"""Independent exact-answer tests for the held-out depth z-buffer and scoring."""
from pathlib import Path
import numpy as np
import pytest
from echo4d.holdout import (rasterize_mesh_depth, compare_holdout_depth,
                            read_echo_ply)
from echo4d.model import Camera, DataError

CAM = Camera(32, 24, 24, 24, 15.5, 11.5)


def panel(z=2.0):
    # Large plane, enough to cover whole image, diagonally split as two triangles.
    vertices = np.array([[-2, -2, z], [2, -2, z], [2, 2, z], [-2, 2, z]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    return vertices, faces


def render(z=2):
    verts, faces = panel(z)
    return rasterize_mesh_depth(verts, faces, CAM, np.eye(4))[0]


def test_parallel_plane_exact_and_full_coverage():
    out = render()
    assert np.count_nonzero(out) == 32 * 24
    assert np.allclose(out, 2, atol=1e-6)
    result = compare_holdout_depth(out, np.full_like(out, 2))
    assert result['observed_depth_coverage'] == 1
    assert result['rmse_m'] < 1e-6
    assert result['within_3cm'] == 1


def test_holdout_metric_rejects_shifted_surface():
    actual = np.full((24, 32), 2, np.float32)
    base = compare_holdout_depth(render(2), actual)
    shifted = compare_holdout_depth(render(2.1), actual)
    assert shifted['mae_m'] > base['mae_m'] + .09
    assert shifted['within_5cm'] == 0
    assert shifted['rmse_m'] == pytest.approx(.1, abs=1e-5)


def test_occlusion_nearest_triangle_wins_independent_of_face_order():
    a, f = panel(2)
    b, g = panel(1.5)
    v = np.concatenate([a, b])
    front_last = np.concatenate([f, g + 4])
    front_first = np.concatenate([g + 4, f])
    d1 = rasterize_mesh_depth(v, front_last, CAM, np.eye(4))[0]
    d2 = rasterize_mesh_depth(v, front_first, CAM, np.eye(4))[0]
    assert np.array_equal(d1, d2)
    assert np.allclose(d1, 1.5)


def test_perspective_correct_depth_of_tilted_plane():
    # sloping 3-D triangle: Z at projected pixels must come from reciprocal depth,
    # NOT linear screen interpolation of world-space Z.
    a = np.array([-2, -2, 1.], float)
    b = np.array([2, -2, 3.], float)
    c = np.array([0, 2, 2.], float)
    vertices = np.stack([a,b,c])
    depth, _ = rasterize_mesh_depth(vertices, np.array([[0,1,2]], np.int32), CAM, np.eye(4))
    valid = np.argwhere(depth > 0)
    assert len(valid) > 3
    # Plane normal; intersect each camera ray with plane.
    n = np.cross(b-a, c-a)
    for iy, ix in valid[::max(1, len(valid)//12)]:
        ray = np.array([(ix-CAM.cx)/CAM.fx, (iy-CAM.cy)/CAM.fy, 1.])
        z = np.dot(n, a)/np.dot(n, ray)
        assert depth[iy,ix] == pytest.approx(z, abs=2e-6)


def test_unknown_sensor_depth_not_claimed_as_correct_or_wrong():
    predicted = np.full((4,4), 2., np.float32)
    truth = np.full((4,4), 2., np.float32)
    truth[:, :2] = 0
    predicted[:, 2:] = 0
    with pytest.raises(DataError,match='No overlapping'):
        compare_holdout_depth(predicted, truth)
    predicted[0,3] = 2.
    report = compare_holdout_depth(predicted, truth)
    assert report['sensor_valid_pixels'] == 8
    assert report['overlap_pixels'] == 1
    assert report['observed_depth_coverage'] == .125
    assert report['missing_surface_pixels'] == 7


def test_bad_input_and_near_crossing_reported():
    v,f = panel(.1)
    d, stats = rasterize_mesh_depth(v,f,CAM,np.eye(4))
    assert not np.any(d)
    assert stats['skip_near_crossing'] == 2
    with pytest.raises(DataError):
        compare_holdout_depth(np.full((4,4),np.nan), np.ones((4,4)))
    with pytest.raises(DataError):
        rasterize_mesh_depth(v, np.array([[0,-1,2]],np.int32), CAM, np.eye(4))


def test_read_echo_ply_roundtrip(tmp_path):
    vertices = np.array([[0,0,1],[1,0,1],[0,1,1]], dtype=np.float32)
    fields = np.dtype([(k, '<f4') for k in ('x','y','z','nx','ny','nz')]
                      +[(k,'u1') for k in ('red','green','blue')])
    v = np.zeros(3, dtype=fields)
    for i,k in enumerate(('x','y','z')): v[k]=vertices[:,i]
    face = np.zeros(1, dtype=[('count','u1'),('indices','<i4',(3,))])
    face['count']=3;face['indices']=[[0,1,2]]
    header = ('ply\nformat binary_little_endian 1.0\ncomment ECHO TSDF; metres; known camera poses\n'
              'element vertex 3\nproperty float x\nproperty float y\nproperty float z\n'
              'property float nx\nproperty float ny\nproperty float nz\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'element face 1\nproperty list uchar int vertex_indices\nend_header\n')
    p=tmp_path/'mesh.ply';p.write_bytes(header.encode()+v.tobytes()+face.tobytes())
    out,f=read_echo_ply(p)
    assert np.array_equal(out,vertices)
    assert np.array_equal(f,[[0,1,2]])
    p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(DataError,match='truncated or contains'):
        read_echo_ply(p)
