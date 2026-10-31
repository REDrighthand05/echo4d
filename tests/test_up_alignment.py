"""Up-alignment contract (added 2026-10-31, user report: the HTML
preview showed the room upside down).

The COLMAP world frame is arbitrary; the fuse stage now rotates every
export so world up lands on +Y (the viewer's up, and every other tool's
default). This test locks the behaviour of the estimator pieces:
  * world up is recovered from synthetic camera poses whose image-down
    axes cluster around a known gravity direction, even with a minority
    of hard-tilted outlier cameras mixed in,
  * the mapping rotation is a proper rigid rotation taking the
    estimated up onto +Y,
  * the mesh stage's camera centres follow points.bin into the aligned
    frame via alignment.json,
  * no poses -> no estimate (the stage then leaves the cloud alone).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echo_scene.fuse import (  # noqa: E402
    _estimate_world_up,
    _rotation_mapping,
)
from echo_scene.mesh import _camera_centers  # noqa: E402


def _matrix_to_quat_xyzw(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s,
                      (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
                      0.25 * s, (R[1, 0] - R[0, 1]) / s])
    return q / np.linalg.norm(q)


def _axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


def _camera_pose(rng, down_true, tilt_deg):
    """A cam_from_world rotation whose image-down axis is down_true
    tilted by tilt_deg around a random horizontal axis."""
    # level basis: forward = random direction perpendicular to down
    fwd = rng.normal(size=3)
    fwd = fwd - (fwd @ down_true) * down_true
    fwd /= np.linalg.norm(fwd)
    axis = np.cross(down_true, fwd)  # horizontal tilt axis
    down = _axis_angle(axis, np.radians(tilt_deg)) @ down_true
    right = np.cross(down, fwd)
    right /= np.linalg.norm(right)
    fwd2 = np.cross(right, down)
    rot = np.stack([right, down, fwd2])  # rows: cam x, y(down), z in world
    return {"cam_from_world_quat_xyzw": _matrix_to_quat_xyzw(rot).tolist(),
            "cam_from_world_t": [0.0, 0.0, 0.0]}


def test_up_recovered_despite_outliers():
    rng = np.random.default_rng(7)
    up_true = np.array([0.3, -0.9, 0.31622777])
    up_true /= np.linalg.norm(up_true)
    down_true = -up_true
    poses = {}
    for i in range(120):  # honest level-ish cameras, tilt up to 12 deg
        poses[f"f{i:03d}.jpg"] = _camera_pose(rng, down_true, rng.uniform(-12, 12))
    for i in range(25):  # outliers: phone swung up/down hard
        poses[f"x{i:03d}.jpg"] = _camera_pose(rng, down_true, rng.uniform(55, 85))
    est = _estimate_world_up(poses)
    assert est is not None
    ang = np.degrees(np.arccos(np.clip(est @ up_true, -1, 1)))
    assert ang < 2.0, f"estimated up off by {ang:.2f} deg"


def test_rotation_mapping_is_rigid_and_lands_on_y():
    a = np.array([0.0472, -0.9885, 0.1436])
    a /= np.linalg.norm(a)
    R = _rotation_mapping(a, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(R @ a, [0, 1, 0], atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    # antiparallel and parallel degenerate cases stay rigid
    for v in ([0.0, -1.0, 0.0], [0.0, 1.0, 0.0]):
        R = _rotation_mapping(np.array(v), np.array([0.0, 1.0, 0.0]))
        assert np.allclose(R @ np.array(v), [0, 1, 0], atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_no_poses_no_estimate():
    assert _estimate_world_up({}) is None
    assert _estimate_world_up({"a.jpg": {}}) is None


def test_mesh_centers_follow_alignment_json():
    rng = np.random.default_rng(3)
    up_true = np.array([0.1, -0.95, 0.29])
    up_true /= np.linalg.norm(up_true)
    with tempfile.TemporaryDirectory() as td:
        job = Path(td)
        pose = _camera_pose(rng, -up_true, 3.0)
        pose["cam_from_world_t"] = [1.0, -2.0, 0.5]
        (job / "poses.json").write_text(json.dumps({"a.jpg": pose}))
        plain = _camera_centers(job)
        R = _rotation_mapping(np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        (job / "alignment.json").write_text(json.dumps({"rotation": R.tolist()}))
        aligned = _camera_centers(job)
        assert np.allclose(aligned, plain @ R.T, atol=1e-9)


if __name__ == "__main__":
    test_up_recovered_despite_outliers()
    test_rotation_mapping_is_rigid_and_lands_on_y()
    test_no_poses_no_estimate()
    test_mesh_centers_follow_alignment_json()
    print("up-alignment contract: ALL PASS")
