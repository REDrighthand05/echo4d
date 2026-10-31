"""Measured-depth probe: SGBM stereo on rectified neighbor pairs from the
undistorted COLMAP workspace (jobs/room1_mvs). One best pair per frame
(largest available baseline at +-6 index distance), single measured depth
per pixel (no cross-pair averaging). Deterministic voxel pick (first point).
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from PIL import Image, ImageDraw

WS = Path("jobs/room1_mvs")
OUT = WS / "stereo_scene"
OUT.mkdir(exist_ok=True)
SCALE = 0.5
STRIDE = 3
NEIGHBOR = 6

rec = pycolmap.Reconstruction(str(WS / "sparse"))
imgs = sorted(rec.images.values(), key=lambda im: im.name)
by_name = {im.name: im for im in imgs}
print(f"registered images: {len(imgs)}", flush=True)

def cam_K(img):
    cam = rec.cameras[img.camera_id]
    fx, fy, cx, cy = cam.params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

def rt(img):
    m = np.eye(4)
    m[:3, :] = img.cam_from_world().matrix()
    return m[:3, :3], m[:3, 3]

sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=160, blockSize=5,
    P1=8 * 3 * 25, P2=32 * 3 * 25, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=200, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

all_pts, all_cols = [], []
t0 = time.time()
done = 0
for i, ref in enumerate(imgs):
    src = None
    for cand in (i + NEIGHBOR, i - NEIGHBOR, i + 3, i - 3):
        if 0 <= cand < len(imgs):
            src = imgs[cand]
            break
    if src is None:
        continue
    im1 = cv2.imread(str(WS / "images" / ref.name), cv2.IMREAD_COLOR)
    im2 = cv2.imread(str(WS / "images" / src.name), cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        continue
    h, w = im1.shape[:2]
    K = np.ascontiguousarray(cam_K(ref).astype(np.float64))
    Rr, tr = rt(ref)
    Rs, ts = rt(src)
    R_rel = np.ascontiguousarray((Rs @ Rr.T).astype(np.float64))
    t_rel = np.ascontiguousarray((ts - R_rel @ tr).astype(np.float64).reshape(3, 1))
    size = (w, h)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, np.zeros(5), K, np.zeros(5), size, R_rel, t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1x, m1y = cv2.initUndistortRectifyMap(K, np.zeros(5), R1, P1, size, cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(K, np.zeros(5), R2, P2, size, cv2.CV_32FC1)
    r1 = cv2.remap(im1, m1x, m1y, cv2.INTER_LINEAR)
    r2 = cv2.remap(im2, m2x, m2y, cv2.INTER_LINEAR)
    small = (int(w * SCALE), int(h * SCALE))
    g1 = cv2.cvtColor(cv2.resize(r1, small), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(r2, small), cv2.COLOR_BGR2GRAY)
    c1 = cv2.resize(r1, small)
    raw = sgbm.compute(g1, g2)
    cv2.filterSpeckles(raw, 0, 200, 24)
    disp = raw.astype(np.float64) / 16.0
    Qs = Q.copy()
    Qs[3, 2] = Q[3, 2] / SCALE  # focal in Q is in full-res px; adjust via f row instead
    # safer: rebuild depth manually: Z = f_rect * B / d, all in small px
    f_rect = P1[0, 0] * SCALE
    B = abs(P2[0, 3] / P1[0, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = f_rect * B / disp
    valid = (disp > 1.0) & np.isfinite(Z) & (Z > 0)
    if valid.sum() < 500:
        continue
    ys, xs = np.mgrid[0:disp.shape[0], 0:disp.shape[1]]
    sel = valid & ((xs % STRIDE == 0) & (ys % STRIDE == 0))
    z = Z[sel]
    # ray in rectified ref frame (small px -> full px by /SCALE)
    fxr, fyr = P1[0, 0], P1[1, 1]
    cxr, cyr = P1[0, 2], P1[1, 2]
    uu = xs[sel] / SCALE
    vv = ys[sel] / SCALE
    xr = (uu - cxr) / fxr * z
    yr = (vv - cyr) / fyr * z
    X_rect = np.stack([xr, yr, z], axis=1)
    X_ref = X_rect @ R1  # R1 rectifies: X_rect = R1 @ X_ref -> X_ref = R1.T @ X_rect => @ R1 works row-wise
    X_w = X_ref @ Rr + (-Rr.T @ tr)
    cols = c1[sel][:, ::-1]  # BGR->RGB
    all_pts.append(X_w.astype(np.float32))
    all_cols.append(cols.astype(np.uint8))
    done += 1
    if done % 20 == 0:
        print(f"{done} pairs, {time.time()-t0:.0f}s", flush=True)

P = np.concatenate(all_pts)
C = np.concatenate(all_cols)
print("raw back-projected points:", len(P), flush=True)
lo = np.percentile(P, 1, axis=0)
hi = np.percentile(P, 99, axis=0)
keep = np.all((P >= lo) & (P <= hi), axis=1)
P, C = P[keep], C[keep]
extent = P.max(axis=0) - P.min(axis=0)
voxel = float(extent.max()) / 240.0
keys = np.floor((P - P.min(axis=0)) / voxel).astype(np.int64)
lin = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
order = np.argsort(lin, kind="stable")
lin_s = lin[order]
first = np.ones(len(order), bool)
first[1:] = lin_s[1:] != lin_s[:-1]
idx = np.sort(order[first])
P, C = P[idx], C[idx]
print("after voxel pick:", len(P), flush=True)
buf = np.empty((len(P), 15), dtype=np.uint8)
buf[:, :12] = P.astype("<f4").view(np.uint8).reshape(len(P), 12)
buf[:, 12:15] = C
buf.tofile(OUT / "points.bin")

center = (P.max(axis=0) + P.min(axis=0)) / 2
radius = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) / 2
def render(az_deg, el_deg, path):
    az, el = np.radians(az_deg), np.radians(el_deg)
    fwd = np.array([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)])
    right = np.array([np.cos(az), 0, -np.sin(az)])
    up = np.cross(right, fwd)
    rel = P - center
    x, y, z = rel @ right, rel @ up, rel @ fwd
    size = 900
    s = size * 0.46 / radius
    px = (size / 2 + x * s).astype(int)
    py = (size / 2 - y * s).astype(int)
    img = Image.new("RGB", (size, size), (10, 10, 14))
    dr = ImageDraw.Draw(img)
    for j in np.argsort(z):
        if 0 <= px[j] < size and 0 <= py[j] < size:
            dr.point((px[j], py[j]), fill=tuple(C[j].tolist()))
    img.save(path)
render(35, 22, OUT / "view.png")
render(0, 89, OUT / "top.png")
print("SGBM_DONE", OUT / "points.bin", len(P), flush=True)
