#!/usr/bin/env python3
"""Synthetic end-to-end check for echo_scene.markers.

Builds two "photos" of AprilTag markers with fully known geometry
(marker edge = 2.0 world units => true scale 0.16/2 = 0.08 m/unit;
floor marker tilted so its true normal is known), writes a fake job
dir (frames/ + poses.json in the exact stage_poses format), runs
stage_markers, and asserts the recovered scale is within 2% and the
floor normal within 2 degrees.

Run: .venv/bin/python tests/test_markers_synth.py
"""
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_scene.markers import stage_markers  # noqa: E402

EDGE_WORLD = 2.0
TRUE_SCALE = 0.16 / EDGE_WORLD
W, H = 1280, 720
K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
JOB = Path(__file__).resolve().parent.parent / "jobs" / "_markers_synth"

UNIT = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]


def orthonormal_frame(normal):
    """(u, v, n): u = marker local +x, v = local +y (down on the printed
    sheet), n = outward normal of the printed face. A viewer on the
    printed side sees x right / y down, so u x v = -n, i.e. v = u x n.
    (v = n x u instead would render the tag mirrored and undetectable.)
    """
    n = normal / np.linalg.norm(normal)
    u = np.cross(n, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, np.array([1.0, 0.0, 0.0]))
    u /= np.linalg.norm(u)
    v = np.cross(u, n)
    return u, v, n


def marker_corners_world(center, u, v):
    return np.array(
        [center + (x * u + y * v) * EDGE_WORLD for x, y in UNIT], dtype=np.float64
    )


def look_at(center, target):
    zc = target - center
    zc /= np.linalg.norm(zc)
    xc = np.cross(np.array([0.0, 1.0, 0.0]), zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    rot = np.stack([xc, yc, zc])  # world -> cam rows
    assert abs(np.linalg.det(rot) - 1.0) < 1e-9
    return rot, -rot @ center


def mat_to_quat_xyzw(rot):
    tr = np.trace(rot)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(rot)))
        if i == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def project(points_world, rot, t):
    cam = (rot @ points_world.T).T + t
    uv = (K @ cam.T).T
    return uv[:, :2] / uv[:, 2:3], cam[:, 2]


def render_photo(markers, rot, t, rng):
    """markers: [(texture_bgr, world_corners (4,3))]"""
    bg = np.zeros((H, W, 3), dtype=np.float64)
    gradient = np.linspace(70, 130, H, dtype=np.float64)[:, None]
    bg[:] = gradient[:, :, None] * np.array([1.0, 0.95, 0.9])
    bg += rng.normal(0, 2.5, bg.shape)
    img = np.clip(bg, 0, 255).astype(np.uint8)
    for texture, corners_world in markers:
        dst, z = project(corners_world, rot, t)
        assert np.all(z > 0), "marker behind camera"
        side = texture.shape[0]
        border = 80
        inner = side - 2 * border
        src = np.array(
            [[border, border], [border + inner, border],
             [border + inner, border + inner], [border, border + inner]],
            dtype=np.float32,
        )
        hom = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        warped = cv2.warpPerspective(texture, hom, (W, H))
        mask = cv2.warpPerspective(
            np.full(texture.shape[:2], 255, np.uint8), hom, (W, H)
        )
        sel = mask > 127
        img[sel] = warped[sel]
    return img


def marker_texture(marker_id):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    raw = cv2.aruco.generateImageMarker(dictionary, marker_id, 480)
    bordered = cv2.copyMakeBorder(raw, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)


def main():
    rng = np.random.default_rng(7)
    floor_normal_true = np.array([0.15, 1.0, -0.10])
    fu, fv, fn = orthonormal_frame(floor_normal_true)
    floor_corners = marker_corners_world(np.array([0.0, 0.0, 0.0]), fu, fv)
    wu, wv, _wn = orthonormal_frame(np.array([-0.20, 0.10, 1.0]))
    wall_corners = marker_corners_world(np.array([1.6, 0.9, -0.9]), wu, wv)
    markers = [(marker_texture(0), floor_corners), (marker_texture(1), wall_corners)]

    cameras = [
        (np.array([2.4, 3.6, 4.2]), np.array([0.4, 0.3, -0.3])),
        (np.array([-2.4, 3.2, 3.8]), np.array([0.4, 0.3, -0.3])),
        (np.array([0.2, 4.4, 2.6]), np.array([0.4, 0.2, -0.3])),
    ]
    if JOB.exists():
        shutil.rmtree(JOB)
    (JOB / "frames").mkdir(parents=True)
    poses = {}
    for i, (center, target) in enumerate(cameras):
        rot, t = look_at(center, target)
        photo = render_photo(markers, rot, t, rng)
        name = f"frame_{i + 1:05d}.jpg"
        cv2.imwrite(str(JOB / "frames" / name), photo,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        poses[name] = {
            "cam_from_world_quat_xyzw": mat_to_quat_xyzw(rot),
            "cam_from_world_t": [float(v) for v in t],
            "camera": {
                "model": "SIMPLE_PINHOLE",
                "width": W,
                "height": H,
                "params": [900.0, 640.0, 360.0],
            },
        }
    (JOB / "poses.json").write_text(json.dumps(poses, indent=2))

    info = stage_markers(JOB)
    result = json.loads((JOB / "markers.json").read_text())
    print("markers.json:", json.dumps(result))
    assert info["ok"] is True

    failures = []
    if result["detections"] < 3:
        failures.append(f"only {result['detections']} detections (< 3)")
    scale = result["scale_to_meters"]
    if scale is None:
        failures.append("scale is null")
        scale_err = float("nan")
    else:
        scale_err = abs(scale - TRUE_SCALE) / TRUE_SCALE
        if scale_err >= 0.02:
            failures.append(f"scale error {scale_err:.2%} >= 2%")
    normal = result["floor_normal_world"]
    if normal is None:
        failures.append("floor normal is null")
        angle_deg = float("nan")
    else:
        n = np.asarray(normal)
        dot = float(n @ (fn / np.linalg.norm(fn)))
        angle_deg = math.degrees(math.acos(min(1.0, abs(dot))))
        if dot <= 0:
            failures.append("floor normal points away from cameras")
        if angle_deg >= 2.0:
            failures.append(f"floor normal angle {angle_deg:.2f} deg >= 2")

    print(f"scale: got {scale}, true {TRUE_SCALE}, error {scale_err:.3%}")
    print(f"floor normal angle error: {angle_deg:.3f} deg")
    if failures:
        print("FAIL:", "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
