"""Geometry regression only; real Lounge visual acceptance needs actual RGB-D run."""
from types import SimpleNamespace

import numpy as np

from lounge_visibility_prior_real import classify_scan_faces


def test_camera_ray_agreement_freespace_occlusion_and_missing_depth():
    camera = SimpleNamespace(width=64, height=64, fx=30., fy=30.,
                             cx=31.5, cy=31.5)
    # Four non-overlapping faces at depths 1, 2, 3, 2 meters.
    p = np.array([
        [-.03, -.03, 1.], [.03, -.03, 1.], [0, .03, 1.],
        [-.03, -.03, 2.], [.03, -.03, 2.], [0, .03, 2.],
        [-.03, -.03, 3.], [.03, -.03, 3.], [0, .03, 3.],
        [1.6, -.03, 2.], [1.66, -.03, 2.], [1.63, .03, 2.],
    ])
    faces = np.arange(12).reshape(4, 3)
    depth = np.full((64, 64), 2., np.float32)
    measured = np.array([[0., 0., 2.]])
    frames = [SimpleNamespace(depth=depth, camera_to_world=np.eye(4))
              for _ in range(3)]
    keep, stats = classify_scan_faces(p, faces, measured, frames, camera)
    assert keep.tolist() == [False, False, True, True]
    assert stats["contradicted_free_space"] == 1
    assert stats["redundant_near_measured"] == 1
    assert stats["preserved_occluded"] >= 1


def test_one_depth_observation_cannot_remove_existing_scan():
    camera = SimpleNamespace(width=64, height=64, fx=30., fy=30.,
                             cx=31.5, cy=31.5)
    p = np.array([[-.03, -.03, 1.], [.03, -.03, 1.], [0, .03, 1.]])
    depth = np.full((64, 64), 2., np.float32)
    frame = SimpleNamespace(depth=depth, camera_to_world=np.eye(4))
    keep, _ = classify_scan_faces(p, np.array([[0, 1, 2]]),
                                  np.array([[0., 0., 2.]]),
                                  [frame], camera)
    assert keep.tolist() == [True]


def test_unstable_depth_edge_does_not_erase_prior():
    camera = SimpleNamespace(width=64, height=64, fx=30., fy=30.,
                             cx=31.5, cy=31.5)
    p = np.array([[-.03, -.03, 1.], [.03, -.03, 1.], [0, .03, 1.]])
    depth = np.full((64, 64), 2., np.float32)
    depth[:, 32:] = 3.
    frame = SimpleNamespace(depth=depth, camera_to_world=np.eye(4))
    keep, stats = classify_scan_faces(p, np.array([[0, 1, 2]]),
                                      np.array([[0., 0., 2.]]),
                                      [frame, frame, frame], camera)
    assert keep.tolist() == [True]
    assert stats["preserved_unseen"] == 1
