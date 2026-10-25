import numpy as np
import pytest
from echo4d.fusion import FusionConfig, TSDFVolume
from echo4d.model import Camera, Frame, DataError
from echo4d.evaluation import compare_points

CAMERA = Camera(64, 48, 50., 50., 31.5, 23.5)


def plane_frame(z=1.23, t=0, offset=None, noise=0, seed=0, mask=None):
    pose = np.eye(4)
    if offset is not None:
        pose[:3,3] = offset
    depth = np.full((48,64), z, dtype=np.float32)
    if noise:
        depth += np.random.default_rng(seed).normal(0, noise, depth.shape)
    return Frame(t, pose, depth, excluded=mask)


def volume(min_observations=1):
    return TSDFVolume(FusionConfig((-.6,-.45,.7,.6,.45,1.6), .03, .09, min_observations=min_observations))


def test_plane_is_at_measured_depth_not_voxel_boundary():
    v = volume(); v.integrate(plane_frame(), CAMERA, 0); m = v.extract_mesh()
    assert len(m.faces) > 100
    assert np.max(np.abs(m.vertices[:,2]-1.23)) < 1e-5
    assert np.all(m.observations == 1) and np.all(m.first_frame == 0)


def test_camera_translation_changes_world_surface():
    v = volume(); v.integrate(plane_frame(z=1.03, offset=[0,0,.2]), CAMERA, 0)
    assert np.max(np.abs(v.extract_mesh().vertices[:,2]-1.23)) < 1e-5


def test_camera_rotation_changes_surface_axis():
    v = TSDFVolume(FusionConfig((.7,-.4,-.4,1.6,.4,.4), .03, .09))
    pose = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]], dtype=float)
    v.integrate(Frame(0,pose,np.full((48,64),1.23,dtype=np.float32)), CAMERA, 0)
    assert np.max(np.abs(v.extract_mesh().vertices[:,0]-1.23)) < 1e-5


def test_unknown_pixels_do_not_turn_into_surfaces():
    v = volume(); f = plane_frame(); f.depth[:,:32] = 0
    v.integrate(f, CAMERA, 0); m = v.extract_mesh()
    assert m.vertices[:,0].min() >= -.001
    assert np.max(np.abs(m.vertices[:,2]-1.23)) < 1e-5


def test_dynamic_mask_excludes_observations():
    v = volume(); mask = np.ones((48,64), dtype=bool)
    assert v.integrate(plane_frame(mask=mask), CAMERA, 0) == 0
    assert np.count_nonzero(v.weight) == 0
    with pytest.raises(DataError, match="zero crossing"): v.extract_mesh()


def test_unobserved_back_boundary_has_no_false_sheet():
    v = volume(); v.integrate(plane_frame(), CAMERA, 0); m = v.extract_mesh()
    assert np.ptp(m.vertices[:,2]) < 1e-5
    assert m.vertices[:,2].max() < 1.24


def test_multiple_noisy_frames_reduce_error():
    first = volume(); first.integrate(plane_frame(noise=.015), CAMERA, 0)
    first_error = np.sqrt(np.mean((first.extract_mesh().vertices[:,2]-1.23)**2))
    many = volume()
    for i in range(24): many.integrate(plane_frame(noise=.015,seed=i,t=i*.1), CAMERA, i)
    error = np.sqrt(np.mean((many.extract_mesh().vertices[:,2]-1.23)**2))
    assert error < first_error*.55


def test_wrong_pose_is_visible_as_geometry_error():
    v = volume()
    for i in range(8): v.integrate(plane_frame(t=i,offset=[0,0,i*.02]), CAMERA, i)
    error = np.sqrt(np.mean((v.extract_mesh().vertices[:,2]-1.23)**2))
    assert error > .04


def test_min_observation_gate():
    v = volume(2); v.integrate(plane_frame(), CAMERA, 0)
    with pytest.raises(DataError): v.extract_mesh()
    v.integrate(plane_frame(t=1), CAMERA, 1)
    assert len(v.extract_mesh().faces) > 100


def test_repeated_frame_and_wrong_dimensions_fail_before_mutation():
    v = volume(); v.integrate(plane_frame(), CAMERA, 0); saved = v.weight.copy()
    with pytest.raises(DataError): v.integrate(plane_frame(), CAMERA, 0)
    assert np.array_equal(saved,v.weight)
    with pytest.raises(DataError): v.integrate(Frame(1,np.eye(4),np.ones((10,10))), CAMERA, 1)
    assert np.array_equal(saved,v.weight)


def test_resource_budget_checked_before_allocation():
    with pytest.raises(DataError,match="voxels"): FusionConfig((-100,-100,-100,100,100,100),.001)
    with pytest.raises(DataError): FusionConfig((0,0,0,1,1,1),float('nan'))


def test_cancellation_is_explicit():
    v=volume()
    with pytest.raises(InterruptedError): v.integrate(plane_frame(),CAMERA,0,cancelled=lambda:True)
    assert not np.count_nonzero(v.weight)


def test_reference_metrics_distinguish_precision_and_coverage():
    full=np.column_stack([np.arange(100)*.1,np.zeros(100),np.zeros(100)])
    report=compare_points(full[::10],full,.01)
    assert report['precision']==1 and report['coverage']==.1
    shifted=compare_points(full+[0,0,.2],full,.05)
    assert shifted['precision']==0 and shifted['point_rmse_m']==pytest.approx(.2)
