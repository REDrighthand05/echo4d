"""Regression: 36-frame real-depth ladder uses only camera poses to select frames."""
import numpy as np
from tum_room_pose_depth_ladder import select_pose_spread


def test_depth36_is_subset_of_reference_and_keeps_temporal_coverage():
    poses = {}
    for i in range(145):
        angle = 2 * np.pi * i / 144
        pose = np.eye(4)
        pose[:3, 3] = [np.cos(angle), np.sin(angle), .2 * np.sin(2 * angle)]
        pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]]
        poses[2*i] = pose
    ids = tuple(poses)
    pose72 = select_pose_spread(ids, poses, 72)
    pose36 = select_pose_spread(ids, poses, 36)
    assert len(set(pose72)) == 72
    assert len(set(pose36)) == 36
    assert set(pose36).issubset(set(pose72))
    assert pose36[0] == ids[0] and pose36[-1] == ids[-1]
    assert all(any(abs(j - i) <= 18 for j in pose36) for i in ids)


def test_depth36_runner_uses_original_renderer_and_no_ai_or_prior():
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / 'tools/tum_room_pose_depth36_ladder.py'
    script = source.read_text()
    assert 'from tum_xyz_open3d_320 import fuse, export_glb, render' in script
    assert '"prior_mesh": False' in script
    assert '"predicted_depth": False' in script
    assert '"pose36": list(pose36)' in script
