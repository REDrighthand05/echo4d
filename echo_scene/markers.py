"""Printed-marker metric scaling and floor leveling.

Coordinate convention (also noted in fuse.py): the scene viewer
(viewer.html) treats +Y as world "up" -- its lookAt uses up=[0, 1, 0]
and elevation moves the eye along +Y. So when a floor marker is seen,
fuse.py rotates the fused cloud so the fitted floor normal lands
exactly on +Y, and scales it by `scale_to_meters` from markers.json.

How it works, all deterministic least squares (no voting):
  * Every frame listed in poses.json is scanned for AprilTag 36h11
    markers (printed sheet: assets/markers_sheet.pdf, side 160 mm).
  * Each detected corner is triangulated from all views that saw it
    (linear DLT, solved by SVD -- a least-squares ray intersection in
    COLMAP world units). A single view cannot fix a corner's world
    position, so corners seen only once do not contribute.
  * Scale: one closed-form least-squares fit over every triangulated
    edge length L_i:  minimise sum (s * L_i - MARKER_SIZE_M)^2
    => s = MARKER_SIZE_M * sum(L_i) / sum(L_i^2).
  * Floor normal: SVD plane fit (total least squares) over the
    triangulated corners of marker FLOOR_MARKER_ID; sign chosen so
    the normal points toward the mean of the camera centers that
    observed it.

cv2 is imported lazily inside functions: if OpenCV is missing, this
stage quietly produces a null markers.json and the rest of the
pipeline is completely unaffected.
"""
import json
from pathlib import Path

import numpy as np

MARKER_SIZE_M = 0.16
FLOOR_MARKER_ID = 0

_detector_cache = None


def _get_detector():
    global _detector_cache
    if _detector_cache is None:
        import cv2

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        _detector_cache = cv2.aruco.ArucoDetector(dictionary, params)
    return _detector_cache


def detect_markers(bgr_image):
    """Return [(marker_id, corners_px (4, 2) float64)] for one BGR image.

    Corner order is OpenCV's: top-left, top-right, bottom-right,
    bottom-left of the marker as printed. The same physical corner
    keeps the same index across views, which triangulation relies on.
    """
    corners, ids, _ = _get_detector().detectMarkers(bgr_image)
    found = []
    if ids is not None:
        for marker_id, quad in zip(ids.flatten(), corners):
            found.append((int(marker_id), quad.reshape(4, 2).astype(np.float64)))
    return found


def _camera_intrinsics(camera: dict):
    """(K, dist) from a poses.json camera entry (COLMAP param order)."""
    params = [float(v) for v in camera["params"]]
    model = str(camera.get("model", ""))
    if "SIMPLE" in model or len(params) < 4:
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    else:  # PINHOLE / OPENCV*: fx, fy, cx, cy, ...
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    if model.startswith("OPENCV") and len(params) >= 8:
        dist[:4] = params[4:8]
    return K, dist


def _fit_plane_normal(points: np.ndarray) -> np.ndarray:
    """Total-least-squares plane normal via the reviewed DS kernel,
    falling back to the identical SVD computation when the kernel file
    is not on disk (e.g. inside the packaged app)."""
    kernel = (
        Path(__file__).resolve().parent.parent
        / "echo_super" / "workorders" / "fit_plane_ds.py"
    )
    if kernel.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("fit_plane_ds", str(kernel))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        normal, _d, _rms = module.fit_plane(points)
        return np.asarray(normal, dtype=np.float64)
    centered = points - points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    return normal / np.linalg.norm(normal)


def _triangulate(observations):
    """Least-squares 3D point from [(pixel_xy, K, rot, t)] views.

    Linear DLT: each view contributes x cross (P X) = 0 rows with
    P = K [rot | t]; the solution is the smallest right-singular
    vector. Returns None with < 2 views or when the point lands
    behind any observing camera.
    """
    if len(observations) < 2:
        return None
    rows = []
    for (px, py), K, rot, t in observations:
        P = K @ np.hstack([rot, t.reshape(3, 1)])
        rows.append(px * P[2] - P[0])
        rows.append(py * P[2] - P[1])
    _u, _s, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
    X = vh[-1]
    if abs(X[3]) < 1e-12:
        return None
    point = X[:3] / X[3]
    for _pix, _K, rot, t in observations:
        if float((rot @ point + t)[2]) <= 0:
            return None
    return point


def stage_markers(job_dir: Path) -> dict:
    """Scan posed frames for printed markers; write markers.json.

    Never raises: with no markers (or any failure) it writes a null
    result and returns normally so the main pipeline is unaffected.
    """
    result = {
        "detections": 0,
        "per_id": {},
        "scale_to_meters": None,
        "floor_normal_world": None,
    }
    info = {"stage": "markers", "ok": True, **result}
    try:
        info = _stage_markers_impl(job_dir, result)
    except Exception as exc:  # stage must never break the pipeline
        info = {"stage": "markers", "ok": True, **result, "error": str(exc)}
    (job_dir / "markers.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def _stage_markers_impl(job_dir: Path, result: dict) -> dict:
    import cv2

    from .fuse import _quat_xyzw_to_matrix

    poses_file = job_dir / "poses.json"
    if not poses_file.exists():
        return {"stage": "markers", "ok": True, **result}
    poses = json.loads(poses_file.read_text())

    # (marker_id, corner_index) -> [(pixel, K, rot, t)] across views
    corner_obs: dict = {}
    floor_cam_centers = []
    per_id: dict = {}
    detections = 0

    for name, pose in sorted(poses.items()):
        img_path = job_dir / "frames" / name
        if not img_path.exists():
            img_path = job_dir / "frames_selected" / name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        found = detect_markers(img)
        if not found:
            continue
        K, dist = _camera_intrinsics(pose["camera"])
        rot = _quat_xyzw_to_matrix(pose["cam_from_world_quat_xyzw"])
        t = np.array(pose["cam_from_world_t"], dtype=np.float64)
        for marker_id, corners_px in found:
            detections += 1
            per_id[str(marker_id)] = per_id.get(str(marker_id), 0) + 1
            if dist.any():
                corners_px = cv2.undistortPoints(
                    corners_px.reshape(1, -1, 2), K, dist, P=K
                ).reshape(4, 2)
            for j in range(4):
                corner_obs.setdefault((marker_id, j), []).append(
                    (corners_px[j], K, rot, t)
                )
            if marker_id == FLOOR_MARKER_ID:
                floor_cam_centers.append(-rot.T @ t)

    # Triangulate every corner seen in at least two views.
    world_corners: dict = {}
    for (marker_id, j), obs in corner_obs.items():
        point = _triangulate(obs)
        if point is not None:
            world_corners.setdefault(marker_id, {})[j] = point

    edge_lengths = []
    for corners in world_corners.values():
        if len(corners) == 4:
            quad = np.stack([corners[j] for j in range(4)])
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                edge_lengths.append(float(np.linalg.norm(quad[b] - quad[a])))

    result["detections"] = detections
    result["per_id"] = per_id
    if edge_lengths:
        lengths = np.asarray(edge_lengths, dtype=np.float64)
        # Closed-form least squares: s = M * sum(L) / sum(L^2).
        result["scale_to_meters"] = float(
            MARKER_SIZE_M * lengths.sum() / (lengths * lengths).sum()
        )
    floor = world_corners.get(FLOOR_MARKER_ID)
    if floor and len(floor) == 4 and floor_cam_centers:
        pts = np.stack([floor[j] for j in range(4)])
        try:
            normal = _fit_plane_normal(pts)
        except ValueError:
            normal = None
        if normal is not None:
            toward_cameras = np.mean(floor_cam_centers, axis=0) - pts.mean(axis=0)
            if float(normal @ toward_cameras) < 0:
                normal = -normal
            result["floor_normal_world"] = [
                float(v) for v in normal / np.linalg.norm(normal)
            ]
    return {"stage": "markers", "ok": True, **result}
