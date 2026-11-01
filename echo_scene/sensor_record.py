"""ECHO sensor record: the handheld collector's capture-folder format.

One folder per capture. Everything in the folder shares ONE clock:
seconds on the device's monotonic clock since capture start. No stream
carries wall-clock timestamps, so the streams can never drift apart by
timezone/DST/NTP accidents; meta.json maps the monotonic origin to a
UTC instant once, for humans.

Layout
------
    <capture>/
      meta.json     device, firmware, clock, stream inventory, calibration
      video.mp4     the camera recording (absent in simulated captures)
      frames.csv    frame,t_s,file        one row per recorded video frame
      imu.csv       t_s,qw,qx,qy,qz,ax,ay,az,gx,gy,gz
      tof.csv       t_s,z00,z01,...,z63   8x8 zones, millimetres, -1 = no target

Conventions (fixed by this module; the Pi capture program in device/
writes exactly this, and the importer below reads exactly this):

  * Quaternions are scalar-first (qw,qx,qy,qz) and give the rotation
    that maps DEVICE-frame vectors into the IMU's own world frame
    (world-from-device). That world frame is arbitrary -- it is fixed
    by the device at boot. Registering it onto a reconstruction's
    frame is the importer's job (register_imu_frame).
  * Accelerometer: m/s^2 in the device frame, INCLUDING gravity (a
    resting device reports ~+9.81 along its up axis). Gyro: rad/s in
    the device frame.
  * Device frame: x right, y up, z back toward the operator (a board
    held in the hand, screen facing the operator). The camera uses the
    OpenCV convention (x right, y DOWN, z forward). Calibration in
    meta.json gives cam_from_device: the 3x3 rotation R with
    v_cam = R @ v_dev.
  * ToF zones: row-major from the TOP-LEFT of the ranging imager's
    view, zone (r, c) -> column z{r*8+c}. Zone centres tile a square
    field of view (tof_fov_deg in meta.json, full angle, both axes)
    uniformly in tan(angle) space. Distances are the range to the
    nearest surface in the zone, in whole millimetres; -1 means the
    zone saw no target (out of range / no return). This mirrors an
    8x8 multi-zone ToF part (VL53L5CX class).

The payoff derivations at the bottom are why the sensors exist:

  * estimate_scale_from_tof -- TRUE SCALE. The reconstruction is in
    arbitrary units; each ToF zone measures real metres along a ray
    the reconstruction also knows. The metres-per-unit scale is the
    least-squares fit through the origin over every valid
    (scene distance, measured distance) pair -- one explicit global
    objective, no voting, no per-zone decisions.
  * estimate_up_from_imu -- LEVEL. The accelerometer's mean vector,
    carried into the reconstruction frame by the registered IMU
    orientation stream, is the gravity direction; up is its negation.
    This replaces the pipeline's pose-guessing up-alignment with a
    measurement (the two are compared in the sensor payoff report).

Evidence honesty: a capture folder says what the sensors measured.
Simulated captures (meta.json "simulated": true) are fabricated from
an existing reconstruction and prove the data path and the estimator
maths only -- see experiments/sensor_sim/payoff_report.md.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

__all__ = [
    "FORMAT_VERSION",
    "TOF_GRID",
    "TOF_INVALID",
    "write_meta",
    "write_frames",
    "write_imu",
    "write_tof",
    "Capture",
    "read_capture",
    "zone_distances_from_points",
    "register_imu_frame",
    "estimate_up_from_imu",
    "estimate_scale_from_tof",
]

FORMAT_VERSION = 1
TOF_GRID = 8
TOF_INVALID = -1

_FRAMES_HEADER = ["frame", "t_s", "file"]
_IMU_HEADER = ["t_s", "qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz"]
_TOF_HEADER = ["t_s"] + [f"z{r * TOF_GRID + c}" for r in range(TOF_GRID)
                         for c in range(TOF_GRID)]


# ---------------------------------------------------------------------------
# Writers. Each takes plain numpy arrays / dicts and writes one file of
# the format. The Pi program uses its own copies of these (it cannot
# import this package); tests lock the two implementations to the same
# bytes-level contract via the readers here.
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:.6f}"


def write_meta(folder: Path, meta: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("format_version", FORMAT_VERSION)
    path = folder / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return path


def write_frames(folder: Path, frame_ids: np.ndarray, t_s: np.ndarray,
                 files: list) -> Path:
    path = Path(folder) / "frames.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FRAMES_HEADER)
        for i, t, name in zip(frame_ids, t_s, files):
            w.writerow([int(i), _fmt(float(t)), name])
    return path


def write_imu(folder: Path, t_s: np.ndarray, quat_wxyz: np.ndarray,
              accel: np.ndarray, gyro: np.ndarray) -> Path:
    """quat_wxyz: (N,4) scalar-first, world-from-device.
    accel: (N,3) m/s^2 device frame incl. gravity. gyro: (N,3) rad/s."""
    path = Path(folder) / "imu.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_IMU_HEADER)
        for t, q, a, g in zip(t_s, quat_wxyz, accel, gyro):
            w.writerow([_fmt(float(t))] + [_fmt(float(v)) for v in q]
                       + [_fmt(float(v)) for v in a]
                       + [_fmt(float(v)) for v in g])
    return path


def write_tof(folder: Path, t_s: np.ndarray, zones_mm: np.ndarray) -> Path:
    """zones_mm: (N,64) integer millimetres, TOF_INVALID for no target."""
    zones_mm = np.asarray(zones_mm)
    if zones_mm.ndim != 2 or zones_mm.shape[1] != TOF_GRID * TOF_GRID:
        raise ValueError(f"zones_mm must be (N,{TOF_GRID * TOF_GRID})")
    path = Path(folder) / "tof.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_TOF_HEADER)
        for t, row in zip(t_s, zones_mm):
            w.writerow([_fmt(float(t))] + [str(int(v)) for v in row])
    return path


# ---------------------------------------------------------------------------
# Reader.
# ---------------------------------------------------------------------------

class Capture:
    """A read capture folder. Arrays are float64 except tof_mm (int32).

    frames: dict file name -> t_s (plus .frame_t / .frame_files arrays)
    imu_t, imu_quat (N,4 wxyz world-from-device), imu_accel, imu_gyro
    tof_t, tof_mm (M,64)
    cam_from_device: (3,3) from meta calibration, or None.
    """

    def __init__(self, folder: Path):
        folder = Path(folder)
        self.folder = folder
        self.meta = json.loads((folder / "meta.json").read_text())
        if int(self.meta.get("format_version", 0)) != FORMAT_VERSION:
            raise ValueError(
                f"unsupported format_version {self.meta.get('format_version')}")

        with open(folder / "frames.csv", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == _FRAMES_HEADER, "frames.csv header mismatch"
        self.frame_ids = np.array([int(r[0]) for r in rows[1:]], dtype=np.int64)
        self.frame_t = np.array([float(r[1]) for r in rows[1:]])
        self.frame_files = [r[2] for r in rows[1:]]
        self.frame_time = {name: t for name, t in
                           zip(self.frame_files, self.frame_t)}

        with open(folder / "imu.csv", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == _IMU_HEADER, "imu.csv header mismatch"
        imu = np.array([[float(v) for v in r] for r in rows[1:]])
        self.imu_t = imu[:, 0]
        self.imu_quat = imu[:, 1:5]
        self.imu_accel = imu[:, 5:8]
        self.imu_gyro = imu[:, 8:11]

        with open(folder / "tof.csv", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == _TOF_HEADER, "tof.csv header mismatch"
        self.tof_t = np.array([float(r[0]) for r in rows[1:]])
        self.tof_mm = np.array([[int(v) for v in r[1:]] for r in rows[1:]],
                               dtype=np.int32)

        cal = self.meta.get("calibration", {})
        R = cal.get("cam_from_device")
        self.cam_from_device = (np.array(R, dtype=np.float64)
                                if R is not None else None)
        self.tof_fov_deg = float(self.meta.get("tof", {}).get("fov_deg", 45.0))

        for name, arr in (("frames", self.frame_t), ("imu", self.imu_t),
                          ("tof", self.tof_t)):
            if len(arr) > 1 and np.any(np.diff(arr) <= 0):
                raise ValueError(f"{name} timestamps not strictly increasing")

    def imu_rotation(self) -> Rotation:
        """World-from-device rotations at imu_t (scipy, xyzw order)."""
        return Rotation.from_quat(self.imu_quat[:, [1, 2, 3, 0]])


def read_capture(folder: Path) -> Capture:
    return Capture(folder)


# ---------------------------------------------------------------------------
# ToF zone model. Shared by the simulator (device model) and the
# importer: given a point cloud in CAMERA coordinates, the distance an
# ideal 8x8 imager would report per zone -- the low-percentile range of
# the points falling in the zone's angular cell (a point cloud's
# approximation of "nearest surface in the cone").
# ---------------------------------------------------------------------------

def zone_distances_from_points(points_cam: np.ndarray, fov_deg: float = 45.0,
                               percentile: float = 10.0,
                               min_points: int = 3) -> np.ndarray:
    """points_cam: (N,3) OpenCV camera frame (z forward). Returns (64,)
    float distances in the same units as the points, NaN where the zone
    has too few points."""
    out = np.full(TOF_GRID * TOF_GRID, np.nan)
    if len(points_cam) == 0:
        return out
    z = points_cam[:, 2]
    keep = z > 0.05
    if not np.any(keep):
        return out
    p = points_cam[keep]
    half = np.tan(np.radians(fov_deg) / 2.0)
    u = p[:, 0] / p[:, 2] / half  # -1..1 across the FoV
    v = p[:, 1] / p[:, 2] / half  # camera y is DOWN: v>0 is lower half
    inside = (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
    if not np.any(inside):
        return out
    p, u, v = p[inside], u[inside], v[inside]
    col = np.minimum(((u + 1.0) * 0.5 * TOF_GRID).astype(int), TOF_GRID - 1)
    row = np.minimum(((v + 1.0) * 0.5 * TOF_GRID).astype(int), TOF_GRID - 1)
    zone = row * TOF_GRID + col
    dist = np.linalg.norm(p, axis=1)
    order = np.argsort(zone, kind="stable")
    zone_s, dist_s = zone[order], dist[order]
    bounds = np.searchsorted(zone_s, np.arange(TOF_GRID * TOF_GRID + 1))
    for zi in range(TOF_GRID * TOF_GRID):
        seg = dist_s[bounds[zi]:bounds[zi + 1]]
        if len(seg) >= min_points:
            out[zi] = np.percentile(seg, percentile)
    return out


# ---------------------------------------------------------------------------
# Payoff 1: LEVEL from the IMU.
# ---------------------------------------------------------------------------

def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]]).as_matrix()


def register_imu_frame(capture: Capture, poses: dict,
                       cam_from_device: np.ndarray) -> np.ndarray:
    """Rotation R_align mapping the IMU's boot world frame onto the
    reconstruction's world frame (v_recon = R_align @ v_imu_world).

    Pairs each reconstruction pose (cam_from_world, OpenCV convention,
    named frames) with the IMU orientation interpolated at that frame's
    capture time, converts both to world-from-device in their own
    frames (R_wd_recon = R_wc @ cam_from_device), and solves the
    orthogonal Procrustes problem in closed form (Markley's method) --
    one global least-squares objective over all pairs, no voting.
    """
    pairs_recon, pairs_imu = [], []
    slerp = Slerp(capture.imu_t, capture.imu_rotation())
    for name, pose in poses.items():
        t = capture.frame_time.get(name)
        q = pose.get("cam_from_world_quat_xyzw")
        if t is None or q is None:
            continue
        if t < capture.imu_t[0] or t > capture.imu_t[-1]:
            continue
        R_cw = _quat_wxyz_to_matrix(np.array(
            [q[3], q[0], q[1], q[2]]))  # poses store xyzw; to wxyz
        R_wc = R_cw.T
        pairs_recon.append(R_wc @ cam_from_device)
        pairs_imu.append(slerp([t]).as_matrix()[0])
    if len(pairs_recon) < 3:
        raise ValueError("too few pose/IMU pairs to register frames")
    A = np.stack(pairs_recon)   # world-from-device, recon frame
    B = np.stack(pairs_imu)     # world-from-device, IMU boot frame
    M = np.einsum("nij,nkj->ik", A, B)  # sum_i A_i @ B_i^T
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def estimate_up_from_imu(capture: Capture, R_align: np.ndarray) -> np.ndarray:
    """Unit world-up vector in the reconstruction frame.

    Every accelerometer sample is gravity (plus walking motion) written
    in the device frame. Rotating each sample into the reconstruction
    frame with the registered orientation stream and averaging over
    the whole capture cancels the zero-mean walking motion; what
    remains points along gravity. Up is its negation.
    """
    R_wd = capture.imu_rotation().as_matrix()          # boot frame
    R_wd_recon = np.einsum("ij,njk->nik", R_align, R_wd)
    g_recon = np.einsum("nij,nj->ni", R_wd_recon, capture.imu_accel)
    g_mean = g_recon.mean(axis=0)
    return -g_mean / np.linalg.norm(g_mean)


# ---------------------------------------------------------------------------
# Payoff 2: TRUE SCALE from the ToF stream.
# ---------------------------------------------------------------------------

def _camera_centers_and_rotations(poses: dict):
    """name -> (center, cam_from_world matrix) in reconstruction units."""
    out = {}
    for name, pose in poses.items():
        q = pose.get("cam_from_world_quat_xyzw")
        t = pose.get("cam_from_world_t")
        if q is None or t is None:
            continue
        R_cw = _quat_wxyz_to_matrix(np.array([q[3], q[0], q[1], q[2]]))
        center = -R_cw.T @ np.array(t, dtype=np.float64)
        out[name] = (center, R_cw)
    return out


def estimate_scale_from_tof(capture: Capture, poses: dict,
                            points: np.ndarray,
                            max_time_gap_s: float = 0.75,
                            tof_stride: int = 1) -> dict:
    """Metres per reconstruction unit, from ToF vs reconstruction rays.

    Each ToF frame is paired with the reconstruction's camera pose AT
    the ToF timestamp: poses come from running the pipeline on the
    captured video (same clock), and the trajectory between pose knots
    is interpolated (lerp centres, slerp orientations -- the same
    continuous-route construction the replay stage uses). Pairing a
    moving sensor with the NEAREST knot instead measurably biases the
    fit, so interpolation is the honest correspondence. For every zone
    valid on both sides -- the sensor saw a target AND the cloud has
    geometry in that zone -- the pair (scene distance s, measured
    distance m in metres) enters one global least-squares fit through
    the origin: scale = sum(s*m) / sum(s^2). Pairs beyond 3 robust
    sigmas of the fit are dropped and the fit repeated once (the
    pipeline's usual one round of principled trimming, identical in
    spirit to fuse's anchor fit; it removes zones whose surface the
    cloud does not actually contain, e.g. glass).
    """
    cams = _camera_centers_and_rotations(poses)
    if not cams:
        raise ValueError("no usable poses")
    names = sorted((n for n in cams if n in capture.frame_time),
                   key=lambda n: capture.frame_time[n])
    if len(names) < 2:
        raise ValueError("too few poses with capture times")
    pose_t = np.array([capture.frame_time[n] for n in names])
    centers = np.stack([cams[n][0] for n in names])
    slerp_cw = Slerp(pose_t, Rotation.from_matrix(
        np.stack([cams[n][1] for n in names])))

    s_all, m_all = [], []
    used_frames = 0
    for t_tof, zones in list(zip(capture.tof_t, capture.tof_mm))[::tof_stride]:
        if t_tof < pose_t[0] - max_time_gap_s or \
                t_tof > pose_t[-1] + max_time_gap_s:
            continue
        tc = float(np.clip(t_tof, pose_t[0], pose_t[-1]))
        center = np.array([np.interp(tc, pose_t, centers[:, a])
                           for a in range(3)])
        R_cw = slerp_cw([tc]).as_matrix()[0]
        scene = zone_distances_from_points((points - center) @ R_cw.T,
                                           fov_deg=capture.tof_fov_deg)
        meas = zones.astype(np.float64) / 1000.0
        ok = (zones > 0) & np.isfinite(scene)
        if np.any(ok):
            used_frames += 1
            s_all.append(scene[ok])
            m_all.append(meas[ok])
    if not s_all:
        raise ValueError("no overlapping ToF/reconstruction zones")
    s = np.concatenate(s_all)
    m = np.concatenate(m_all)

    def _fit(s, m):
        return float(np.sum(s * m) / np.sum(s * s))

    scale = _fit(s, m)
    resid = m - scale * s
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    keep = np.abs(resid) <= 3.0 * sigma
    dropped = int((~keep).sum())
    if dropped and keep.sum() >= 10:
        s, m = s[keep], m[keep]
        scale = _fit(s, m)
        resid = m - scale * s
    return {
        "scale_m_per_unit": scale,
        "pairs": int(len(s)),
        "pairs_dropped": dropped,
        "tof_frames_used": used_frames,
        "median_ratio": float(np.median(m / s)),
        "residual_rms_m": float(np.sqrt(np.mean(resid ** 2))),
        "residual_median_abs_m": float(np.median(np.abs(resid))),
    }
