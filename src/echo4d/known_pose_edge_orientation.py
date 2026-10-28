"""Reject TSDF bridge faces whose tangent plane disagrees with inferred RGB depth.

Opt-in post-filter for the existing short-bridge MVS route. Sensor Z is never
opened; depth votes come only from the frozen known-pose RGB plane sweep.
This does not infer unseen surfaces or change the original TSDF faces.
"""
from __future__ import annotations

import numpy as np

from .known_pose_edge_bridges import filter_bridge_faces
from .known_pose_edge_crossview import _face_keys
from .known_pose_edge_extraction import _attached_faces


def _orientation_votes(vertices, faces, depths, poses, camera,
                       *, normal_cosine=.57, baseline_m=.04):
    """At each projected centroid compare face normal to local inferred-Z plane.

    A local plane uses four independent depth pixels at +/-2 px; all must
    exist. Paired agreeing camera centres must be separated by >= baseline_m.
    The face normal sign is irrelevant because TSDF winding can flip.
    """
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid_face = lengths > 1e-9
    normals = normals / np.maximum(lengths[:, None], 1e-9)
    centers = triangles.mean(axis=1)
    votes = []
    for depth, pose in zip(depths, poses):
        xyz = (centers - pose[:3, 3]) @ pose[:3, :3]
        z = xyz[:, 2]
        safe_z = np.maximum(z, 1e-8)
        u = np.rint(camera.fx * xyz[:, 0] / safe_z + camera.cx).astype(np.int64)
        v = np.rint(camera.fy * xyz[:, 1] / safe_z + camera.cy).astype(np.int64)
        inside = (valid_face & (z >= .15) & (z <= 4.)
                  & (u >= 2) & (u < camera.width - 2)
                  & (v >= 2) & (v < camera.height - 2))
        supported = np.zeros(len(faces), dtype=bool)
        ids = np.flatnonzero(inside)
        if len(ids):
            x, y = u[ids], v[ids]
            dc = depth[y, x]
            dl, dr = depth[y, x-2], depth[y, x+2]
            dt, db = depth[y-2, x], depth[y+2, x]
            observed = ((dc > 0) & (dl > 0) & (dr > 0) & (dt > 0) & (db > 0)
                        & (np.abs(dc-z[ids]) <= np.maximum(.10, .04*z[ids])))
            # Reproject neighbouring pixels to 3D rather than treating image
            # gradients as world-space normals.
            left = np.stack(((x-2-camera.cx)*dl/camera.fx,
                             (y-camera.cy)*dl/camera.fy, dl), axis=1)
            right = np.stack(((x+2-camera.cx)*dr/camera.fx,
                              (y-camera.cy)*dr/camera.fy, dr), axis=1)
            top = np.stack(((x-camera.cx)*dt/camera.fx,
                            (y-2-camera.cy)*dt/camera.fy, dt), axis=1)
            bottom = np.stack(((x-camera.cx)*db/camera.fx,
                               (y+2-camera.cy)*db/camera.fy, db), axis=1)
            local = np.cross(right-left, bottom-top)
            local_norm = np.linalg.norm(local, axis=1)
            local_world = local @ pose[:3, :3].T
            cosine = np.abs(np.sum(local_world * normals[ids], axis=1)) / np.maximum(local_norm, 1e-9)
            supported[ids] = observed & (local_norm > 1e-8) & (cosine >= normal_cosine)
        votes.append(supported)
    independent = np.zeros(len(faces), dtype=bool)
    for j in range(len(poses)):
        for k in range(j):
            if np.linalg.norm(poses[j][:3, 3] - poses[k][:3, 3]) >= baseline_m:
                independent |= votes[j] & votes[k]
    return independent, np.sum(votes, axis=0, dtype=np.int16)


def filter_oriented_bridge_faces(original_vertices, original_faces,
                                 candidate_vertices, candidate_faces,
                                 inferred_depths, camera_to_world, camera,
                                 *, normal_cosine=.57):
    """Apply frozen PR91 bridge support, then independently check local normals.

    All baseline triangles remain, even if their inferred Z is poor. New faces
    need both the old support and two independent depth-normal agreements.
    """
    if not np.isfinite(normal_cosine) or not .3 <= normal_cosine <= .95:
        raise ValueError("Invalid normal agreement threshold")
    previous_mask, previous = filter_bridge_faces(
        original_vertices, original_faces, candidate_vertices, candidate_faces,
        inferred_depths, camera_to_world, camera)
    vertices = np.asarray(candidate_vertices)
    faces = np.asarray(candidate_faces)
    old_keys = set(_face_keys(original_vertices, original_faces))
    baseline = np.asarray([key in old_keys for key in _face_keys(vertices, faces)], dtype=bool)
    extras = np.flatnonzero(previous_mask & ~baseline)
    oriented = np.zeros(len(faces), dtype=bool)
    raw_votes = np.zeros(len(faces), dtype=np.int16)
    if len(extras):
        oriented_extra, votes_extra = _orientation_votes(
            vertices, faces[extras], inferred_depths, camera_to_world, camera,
            normal_cosine=normal_cosine)
        oriented[extras] = oriented_extra
        raw_votes[extras] = votes_extra
    keep = baseline | (previous_mask & oriented)
    attached = _attached_faces(faces[keep], baseline[keep])
    keep[np.flatnonzero(keep)[~attached]] = False
    if not np.all(keep[baseline]):
        raise ValueError("Baseline TSDF faces must survive")
    return keep, {
        "previous_extra_faces": int(np.count_nonzero(previous_mask & ~baseline)),
        "extra_faces_with_independent_local_normal": int(np.count_nonzero(oriented & ~baseline)),
        "retained_extra_faces": int(np.count_nonzero(keep & ~baseline)),
        "rejected_by_orientation_or_attachment": int(np.count_nonzero(previous_mask & ~keep)),
        "max_raw_normal_votes": int(raw_votes.max(initial=0)),
        "normal_cosine_threshold": normal_cosine,
        "normal_patch_offset_px": 2,
        "independent_camera_baseline_m": .04,
        "sensor_depth_used": False,
        "external_prior_mesh_used": False,
        "baseline_used_only_for_same_volume_face_labels": True,
        "prior_bridge_support": previous,
    }
