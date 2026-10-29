"""CPU regression guards for inferred room-shell geometry."""
import numpy as np
import pytest
from scipy.spatial import cKDTree
from stanford_room_envelope import fit_room_planes, extend_plane, depth_conflict_mask


def grid_points(n=100):
    a = np.arange(n, dtype=float) * .05
    x, y = np.meshgrid(a, a, indexing="ij")
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(n*n)))
    normals = np.tile([0., 0., 1.], (len(vertices), 1))
    return vertices, normals


def test_large_measured_plane_detected_without_prior():
    vertices, normals = grid_points()
    planes = fit_room_planes(vertices, normals, max_planes=2)
    assert len(planes) == 1
    assert len(planes[0][3]) > 4000
    assert abs(planes[0][1][2]) > .99


def test_extension_reaches_outside_measured_outline_but_is_bounded():
    vertices, normals = grid_points()
    rgb = np.tile([.5, .6, .7], (len(vertices), 1))
    center = vertices.mean(axis=0)
    plane = (center, np.array([0., 0., 1.]),
             np.array([[1., 0., 0.], [0., 1., 0.]]),
             np.arange(len(vertices)), np.array([4.95, 4.95]))
    out, faces, colors, cells = extend_plane(
        vertices, rgb, plane, cKDTree(vertices))
    assert cells > 0 and len(faces) == 2 * cells
    assert colors.shape == (len(out), 3)
    assert out[:, 0].min() < vertices[:, 0].min()
    assert out[:, 0].min() > vertices[:, 0].min() - .7
    assert out[:, 0].max() < vertices[:, 0].max() + .7
    assert np.max(np.abs(out[:, 2])) < 1e-6


def test_invalid_measured_geometry_rejected():
    with pytest.raises(ValueError):
        fit_room_planes(np.ones((5, 3)), np.ones((5, 3)))
    vertices, normals = grid_points()
    vertices[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_room_planes(vertices, normals)


def test_recorded_sensor_depth_veto_rejects_false_foreground_wall():
    vertices = np.array([
        [0, 0, 1], [.1, 0, 1], [0, .1, 1],
        [0, 0, 2], [.1, 0, 2], [0, .1, 2]], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    depth_mm = np.full((480, 640), 2000, dtype=np.uint16)
    k = (640, 480, 500, 500, 320, 240)
    poses = [np.eye(4) for _ in range(3)]
    poses[1][0, 3] = .02
    poses[2][0, 3] = -.02
    reject, record = depth_conflict_mask(
        vertices, faces, [(i, p, depth_mm) for i, p in enumerate(poses)], k)
    assert reject.tolist() == [True, False]
    assert record["vetoed_faces"] == 1


def test_repeated_recorded_observation_cannot_create_two_votes():
    vertices = np.array([[0, 0, 1], [.1, 0, 1], [0, .1, 1]], dtype=float)
    faces = np.array([[0, 1, 2]])
    depth = np.full((480, 640), 2000, dtype=np.uint16)
    for ids in ([7, 7, 7], [7, 8, 9]):
        reject, receipt = depth_conflict_mask(
            vertices, faces, [(i, np.eye(4), depth) for i in ids],
            (640, 480, 500, 500, 320, 240))
        assert reject.tolist() == [False]
        assert receipt["tested_recorded_frames"] == 1
        assert receipt["duplicate_recorded_frames"] == 2


def test_same_source_identity_with_different_depth_is_rejected():
    vertices = np.array([[0, 0, 1], [.1, 0, 1], [0, .1, 1]], dtype=float)
    faces = np.array([[0, 1, 2]])
    depth = np.full((480, 640), 2000, dtype=np.uint16)
    with pytest.raises(ValueError, match="conflicting data"):
        depth_conflict_mask(
            vertices, faces, [(7, np.eye(4), depth), (7, np.eye(4), depth + 1)],
            (640, 480, 500, 500, 320, 240))



def test_offcenter_false_wall_is_rejected_by_spatial_depth_samples():
    # The old centroid-only veto misses this false surface in front of a
    # measured chair: only one triangle corner projects into the 2m patch.
    from stanford_room_envelope import depth_conflict_mask
    vertices = np.array([[-.3, -.3, 1.], [.3, -.3, 1.],
                         [-.3, .3, 1.]], dtype=float)
    faces = np.array([[0, 1, 2]])
    depth = np.full((240, 320), 1000, dtype=np.uint16)
    depth[92:104, 132:144] = 2000
    poses = [np.eye(4), np.eye(4)]
    poses[0][0, 3] = -.01
    poses[1][0, 3] = .01
    rejected, receipt = depth_conflict_mask(
        vertices, faces, [(i, p, depth) for i, p in enumerate(poses)],
        (320, 240, 100, 100, 160, 120))
    assert rejected.tolist() == [True]
    assert receipt["triangle_depth_samples"] == 7
    assert receipt["vetoed_conflicting_faces"] == 1


def test_inferred_surface_needs_distinct_recorded_visibility():
    from stanford_room_envelope import depth_conflict_mask
    vertices = np.array([[0., 0., 1.], [.1, 0., 1.],
                         [0., .1, 1.]], dtype=float)
    faces = np.array([[0, 1, 2]])
    depth = np.full((240, 320), 1000, dtype=np.uint16)
    k = (320, 240, 100, 100, 160, 120)
    pose = np.eye(4)
    rejected, report = depth_conflict_mask(
        vertices, faces, [(7, pose, depth), (8, pose, depth)],
        k, min_valid_views=2)
    assert rejected.tolist() == [True]
    assert report["tested_recorded_frames"] == 1
    assert report["vetoed_unobserved_faces"] == 1
    pose2 = np.eye(4)
    pose2[0, 3] = .02
    rejected, report = depth_conflict_mask(
        vertices, faces, [(7, pose, depth), (8, pose2, depth)],
        k, min_valid_views=2)
    assert rejected.tolist() == [False]
    assert report["tested_recorded_frames"] == 2


def test_invalid_depth_cannot_validate_inferred_surface():
    from stanford_room_envelope import depth_conflict_mask
    vertices = np.array([[0., 0., 1.], [.1, 0., 1.],
                         [0., .1, 1.]], dtype=float)
    faces = np.array([[0, 1, 2]])
    depth = np.zeros((240, 320), dtype=np.uint16)
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = .02
    rejected, report = depth_conflict_mask(
        vertices, faces, [(i, p, depth) for i, p in enumerate(poses)],
        (320, 240, 100, 100, 160, 120), min_valid_views=2)
    assert rejected.tolist() == [True]
    assert report["vetoed_unobserved_faces"] == 1
