"""Reused, pinned CPU Depth Anything V2 and metric anchor projection from Fountain #182.

These routines have run on actual Fountain RGB-D. Lounge uses only retained
sensor depths for scale; target depths remain unread until both reduced GLBs.
"""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REV = "32d03942121d29edb49de4e2cc15831558af3f36"
WEIGHTS_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
DEPTH_MIN, DEPTH_MAX = .4, 3.8

def project_sensor_anchors(measured, camera, target_pose, stride=3):
    """Only retained sensor rays, reprojected into target RGB camera."""
    h, w = camera.height, camera.width
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    xx, yy = xx.ravel(), yy.ravel()
    zbuffer = np.full(h*w, np.inf, np.float32)
    target_inv = np.linalg.inv(target_pose)
    for frame in measured:
        z = frame.depth[::stride, ::stride].ravel()
        valid = np.isfinite(z) & (z > DEPTH_MIN) & (z < DEPTH_MAX)
        if not np.any(valid):
            continue
        depth = z[valid]
        xyz = np.column_stack(((xx[valid]-camera.cx)*depth/camera.fx,
                               (yy[valid]-camera.cy)*depth/camera.fy, depth))
        t = target_inv @ frame.camera_to_world
        xyz = xyz @ t[:3, :3].T + t[:3, 3]
        zz = xyz[:, 2]
        good = (zz > DEPTH_MIN) & (zz < DEPTH_MAX)
        xyz, zz = xyz[good], zz[good]
        u = np.rint(xyz[:, 0]*camera.fx/zz + camera.cx).astype(np.int32)
        v = np.rint(xyz[:, 1]*camera.fy/zz + camera.cy).astype(np.int32)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        np.minimum.at(zbuffer, v[inside]*w + u[inside], zz[inside])
    return zbuffer.reshape(h, w)


def metric_from_relative(relative, anchors, radius=70):
    """Robust inverse-depth scale fit, only from retained measured rays."""
    if relative.shape != anchors.shape or relative.ndim != 2:
        raise ValueError("Depth image size mismatch")
    support = np.isfinite(anchors) & np.isfinite(relative) & (relative > 0)
    y, x = np.where(support)
    if len(x) < 180:
        raise RuntimeError(f"Too few sensor reprojections: {len(x)}")
    x, y = x[::2], y[::2]
    z = anchors[y, x].astype(np.float64)
    p = relative[y, x].astype(np.float64)
    A = np.column_stack((p, np.ones_like(p)))
    target = 1. / z
    weights = np.ones(len(z))
    for _ in range(7):
        coef = np.linalg.lstsq(A*weights[:, None], target*weights, rcond=None)[0]
        residual = A @ coef - target
        sigma = max(float(np.median(np.abs(residual)))*1.4826, .02)
        weights = np.clip(1.5*sigma / np.maximum(np.abs(residual), 1e-8), .05, 1.)
    if not np.isfinite(coef).all() or coef[0] <= 0:
        raise RuntimeError("Invalid RGB relative-depth metric scale")
    estimated = 1. / np.maximum(A @ coef, 1e-6)
    inlier = np.abs(estimated-z) < np.maximum(.28, .22*z)
    if inlier.sum() < 120 or inlier.mean() < .40:
        raise RuntimeError(f"Insufficient sensor-consistent support: {inlier.sum()}")
    coef = np.linalg.lstsq(A[inlier], target[inlier], rcond=None)[0]
    if coef[0] <= 0:
        raise RuntimeError("Negative inverse-depth scale")
    inv = relative.astype(np.float64)*coef[0] + coef[1]
    valid = ((distance_transform_edt(~support) <= radius) &
             (inv > 1./DEPTH_MAX) & (inv < 1./DEPTH_MIN))
    metric = np.zeros(relative.shape, np.float32)
    metric[valid] = (1./inv[valid]).astype(np.float32)
    # Reject observed sensor conflicts rather than inventing a second surface.
    conflicts = support & (metric > 0) & (
        np.abs(metric-anchors) > np.maximum(.18, .15*anchors))
    metric[conflicts] = 0
    if np.count_nonzero(metric) < metric.size*.02:
        raise RuntimeError("Less than 2% trustworthy predicted depth")
    return metric, {"sensor_projection_pixels": int(support.sum()),
                    "calibration_inliers": int(inlier.sum()),
                    "predicted_valid_fraction": float(np.mean(metric > 0)),
                    "rejected_sensor_conflicts": int(conflicts.sum())}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CPUDepthAnything:
    """Previously executed Apache-2.0 checkpoint, pinned and hash-checked."""
    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        torch.set_num_threads(2)
        root = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REV,
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"]))
        if file_sha256(root / "model.safetensors") != WEIGHTS_SHA256:
            raise RuntimeError("Depth Anything checkpoint SHA256 mismatch")
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(str(root))
        self.model = AutoModelForDepthEstimation.from_pretrained(
            str(root), use_safetensors=True).to("cpu").eval()

    def infer(self, rgb):
        import torch.nn.functional as F
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        with self.torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth.unsqueeze(1)
            prediction = F.interpolate(prediction, size=rgb.shape[:2],
                                       mode="bicubic", align_corners=False)
        output = prediction[0, 0].cpu().numpy().astype(np.float32)
        if not np.isfinite(output).all():
            raise RuntimeError("Nonfinite CPU predicted depth")
        return output


