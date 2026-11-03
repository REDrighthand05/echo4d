#!/usr/bin/env python3
"""ECHO holder replay v1: the device holder walks the scene again.

Reads a finished job directory (points.bin + poses.json) and renders a
third-person MP4 of a procedural skeleton figure walking the recovered
camera route -- with real limb motion this time:

  * the route is the job's camera path (the registered build_replay_path
    kernel, same as the HTML route replay), laid on the floor;
  * the floor is ONE dominant plane fitted procedurally to the floor
    band of the cloud inside the route corridor (trimmed least squares);
    the figure's feet answer to that plane everywhere;
  * the figure is a segmented point-built humanoid (head / torso /
    upper+fore arms / thighs / shins / feet) driven by an analytic
    gait: hip swing, swing-phase knee flexion, ankle compensation with
    toe-off, opposite arm swing, pelvis bob at twice stride frequency,
    lateral sway, and a ground clamp (a correction curve over the
    route, solved once) that keeps the support foot planted on the
    plane without ever pushing the other foot through it -- no
    gliding, no floating;
  * the camera follows behind and above the figure, smoothed;
  * frames are drawn by the registered MUS rasterizer
    (echo_super.kernels.echo_render) and encoded with the pipeline's
    own ffmpeg pattern.

The figure is generated (a stand-in for the holder, who is never in
their own footage); everything it walks through and on is the job's
real data. Outputs are local files: replay_walk.mp4 + two stills +
report.json in the chosen output directory.

Usage:
  python3 -m echo_scene.replay_walk <job_dir> <out_dir>
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from echo_super.ds_batch1 import build_replay_path
    from echo_super.kernels.echo_render import render_frame
except ImportError:  # running outside the repo root: add it and retry
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from echo_super.ds_batch1 import build_replay_path
    from echo_super.kernels.echo_render import render_frame

from .depth_video import _run_ffmpeg
from .preview import _read_points_bin
from .replay import _camera_route, _route_floor

FIG_SEED = 20261031
FIG_RGB = (255, 166, 66)          # gold accent (the approved sample)
SAMPLES_PER_SEGMENT = 8

# --- figure proportions (fractions of figure height H) -------------------
HIP_Y = 0.530                     # pelvis centre height
HIP_HALF = 0.055                  # hip joint half-width
THIGH = 0.245
SHIN = 0.246
ANKLE_H = 0.039                   # ankle joint above the sole
FOOT_FWD = 0.115                  # ankle axis -> toe contact
FOOT_BACK = 0.045                 # ankle axis -> heel contact
SPINE_TOP = 0.720                 # chest joint height
NECK_Y = 0.835
HEAD_C = 0.920
HEAD_R = 0.075
SHOULDER_Y = 0.800
SHOULDER_HALF = 0.115
UARM = 0.185
FARM = 0.165

# --- gait (analytic, phase = one stride = two steps) ---------------------
STRIDE_LEN = 0.85                 # x H per stride
WALK_SPEED = 0.78                 # x H per second
A_HIP = 0.52                      # rad, thigh swing amplitude
KNEE_SWING = 1.02                 # rad, peak swing-phase knee flexion
KNEE_STANCE = 0.14                # rad, loading-response knee flex
A_ARM = 0.40                      # rad, shoulder swing amplitude
ELBOW_BASE = 0.38
ELBOW_AMP = 0.22
BOB_AMP = 0.020                   # x H, twice per stride
SWAY_AMP = 0.014                  # x H, once per stride
MAX_FLOOR_TILT_DEG = 25.0


# ---------------------------------------------------------------------------
# floor: one dominant plane out of the cloud
# ---------------------------------------------------------------------------

def _fit_plane_trimmed(band: np.ndarray, iters: int = 5):
    """Least-squares plane y = a*x + b*z + c over `band`, MAD-trimmed.

    Returns (a, b, c, inlier_mask_over_band). Deterministic."""
    keep = np.ones(len(band), dtype=bool)
    sol = np.zeros(3)
    for _ in range(iters):
        sel = band[keep]
        if len(sel) < 50:
            break
        A = np.stack([sel[:, 0], sel[:, 2], np.ones(len(sel))], axis=1)
        sol, *_ = np.linalg.lstsq(A, sel[:, 1], rcond=None)
        r = band[:, 1] - (sol[0] * band[:, 0] + sol[1] * band[:, 2]
                          + sol[2])
        mad = float(np.median(np.abs(r[keep] - np.median(r[keep]))))
        thr = max(3.0 * 1.4826 * mad, 1e-4)
        new = np.abs(r) < thr
        if int(new.sum()) < 50 or bool((new == keep).all()):
            keep = new if int(new.sum()) >= 50 else keep
            break
        keep = new
    return float(sol[0]), float(sol[1]), float(sol[2]), keep


def fit_floor_plane(pts: np.ndarray, path: np.ndarray | None = None,
                    fig_h: float | None = None) -> dict:
    """The scene's dominant floor plane.

    Band selection: when a route is given, the floor that matters is
    the one under the walk -- points inside the route corridor whose
    height sits in a window around the route's local floor track.
    Without a route, the global lowest quarter of the cloud is used.
    The fitted slope is clipped to MAX_FLOOR_TILT_DEG so a warped
    cloud cannot tip the world over.
    """
    if path is not None and fig_h:
        track = _route_floor(pts, path)
        chk = np.arange(0, len(path), 10)
        chk_pts, chk_floor = path[chk], track[chk]
        d2 = ((pts[:, None, 0] - chk_pts[None, :, 0]) ** 2
              + (pts[:, None, 2] - chk_pts[None, :, 2]) ** 2)
        idx = np.argmin(d2, axis=1)
        near = np.sqrt(d2[np.arange(len(pts)), idx]) < 0.50 * fig_h
        loc = chk_floor[idx]
        band = pts[near & (pts[:, 1] >= loc - 0.30 * fig_h)
                   & (pts[:, 1] <= loc + 0.20 * fig_h)]
        source = "route corridor"
    else:
        band = pts[pts[:, 1] <= np.percentile(pts[:, 1], 25)]
        source = "global low band"
    if len(band) < 200:                       # degenerate cloud: widen
        band = pts[pts[:, 1] <= np.percentile(pts[:, 1], 40)]
        source += " (widened)"
    a, b, c, keep = _fit_plane_trimmed(band)
    slope = float(np.hypot(a, b))
    max_slope = float(np.tan(np.radians(MAX_FLOOR_TILT_DEG)))
    if slope > max_slope:
        a, b = a / slope * max_slope, b / slope * max_slope
    return {"a": a, "b": b, "c": c,
            "tilt_deg": float(np.degrees(np.arctan(np.hypot(a, b)))),
            "band_points": int(len(band)),
            "inliers": int(keep.sum()),
            "inlier_frac": float(keep.mean()) if len(band) else 0.0,
            "source": source}


def floor_height_at(plane: dict, x, z):
    return plane["a"] * np.asarray(x) + plane["b"] * np.asarray(z) \
        + plane["c"]


# ---------------------------------------------------------------------------
# ground path: the camera route laid on the floor plane
# ---------------------------------------------------------------------------

def ground_path(path: np.ndarray, plane: dict, ds: float):
    """Uniform arc-length resampling of the route on the floor plane.

    Returns (points (M,3) with y on the plane, tangents (M,3) unit,
    cumulative arc (M,)). Tangents are smoothed over a short window;
    near-stationary stretches inherit the last real heading."""
    xz = path[:, [0, 2]]
    seg = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < 1e-6:
        raise ValueError("route has no horizontal extent")
    m = max(int(total / ds), 8)
    s_new = np.linspace(0.0, total, m)
    gx = np.interp(s_new, arc, xz[:, 0])
    gz = np.interp(s_new, arc, xz[:, 1])
    gy = floor_height_at(plane, gx, gz)
    pts = np.stack([gx, gy, gz], axis=1)
    tan = np.gradient(pts, axis=0)
    tan[:, 1] = 0.0
    n = np.linalg.norm(tan, axis=1, keepdims=True)
    ok = n[:, 0] > 1e-9
    tan[ok] = tan[ok] / n[ok]
    k = 9
    padded = np.pad(tan, ((k // 2, k // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, c], np.ones(k) / k,
                                   mode="valid")[:m] for c in range(3)],
                      axis=1)
    sn = np.linalg.norm(smooth, axis=1)
    good = sn > 0.35
    if not good.any():
        raise ValueError("route headings degenerate")
    tan = np.where(good[:, None], smooth / np.maximum(sn, 1e-9)[:, None],
                   tan)
    # fill degenerate headings from the nearest good one (causal pass,
    # then a backward pass for the head)
    last = None
    for i in range(m):
        if good[i]:
            last = tan[i]
        elif last is not None:
            tan[i] = last
    last = None
    for i in range(m - 1, -1, -1):
        if good[i]:
            last = tan[i]
        elif last is not None:
            tan[i] = last
    return pts, tan, s_new


# ---------------------------------------------------------------------------
# gait + skeleton
# ---------------------------------------------------------------------------

def _wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _bump(psi, centre, width):
    """Smooth circular bump (1 at centre, ->0 away)."""
    return np.exp(-(_wrap_pi(psi - centre) / width) ** 2)


def leg_angles(psi):
    """(hip, knee, foot_pitch) for one leg at gait phase psi.

    psi = 0 is heel strike (leg furthest forward). Stance runs to
    ~1.1*pi, swing brings the leg back; the knee bends almost only in
    swing; the foot stays flat in stance, pushes off toe-down at the
    end of stance, and lifts its toes slightly while swinging."""
    hip = A_HIP * np.cos(psi)
    knee = (KNEE_STANCE * _bump(psi, 0.35 * np.pi, 0.45)
            + KNEE_SWING * _bump(psi, 1.52 * np.pi, 0.62))
    foot = (0.55 * _bump(psi, 0.95 * np.pi, 0.30)
            - 0.12 * _bump(psi, 1.45 * np.pi, 0.60))
    return hip, knee, foot


def arm_angles(psi_leg):
    """(shoulder, elbow) for the arm on the same side as a leg: the
    arm swings against its leg, the elbow bends as the arm comes
    forward."""
    shoulder = -A_ARM * np.cos(psi_leg)
    elbow = ELBOW_BASE + ELBOW_AMP * (0.5 - 0.5 * np.cos(psi_leg))
    return shoulder, elbow


class Walker:
    """Articulated point figure walking a ground path.

    Pose is a pure function of travelled distance s. The ground clamp
    is a correction curve over the route arc, solved once at build:
    at each route sample it lowers the figure until the support foot
    meets the plane, but never by more than the lower foot's own
    clearance (so the departing toe at a step transition is not
    pushed through the floor), smoothed over a short window.
    All sizes scale with the figure height H.
    """

    def __init__(self, height: float, plane: dict, gpath: np.ndarray,
                 tangents: np.ndarray, arc: np.ndarray):
        self.H = float(height)
        self.plane = plane
        self.gpath = gpath
        self.tangents = tangents
        self.arc = arc
        self.stride = STRIDE_LEN * self.H
        self.template = _build_template(self.H)
        self._corr = self._build_clamp_table()

    def _clamp_target(self, diag) -> float:
        """Wanted clamp at one sample, from raw (unclamped) diag."""
        H = self.H
        active = [diag[f"clear_{s}"] for s in ("L", "R")
                  if diag[f"stance_{s}"] > 0.2]
        if not active:
            return 0.0
        target = min(active)
        if target > 0.0:
            # lowering budget: the lowest foot (either one) may not
            # be pushed below the plane
            target = min(target, min(diag["clear_L"], diag["clear_R"])
                         + 0.002 * H)
        return float(np.clip(target, -0.030 * H, 0.055 * H))

    def _build_clamp_table(self) -> np.ndarray:
        raw = np.array([self._clamp_target(self._pose_raw(float(s))
                                           ["diag"])
                        for s in self.arc])
        k = 5
        smooth = np.convolve(np.pad(raw, k // 2, mode="edge"),
                             np.ones(k) / k, mode="valid")[:len(raw)]
        return smooth

    # -- route interpolation -------------------------------------------
    def _at(self, s: float):
        s = float(np.clip(s, 0.0, self.arc[-1]))
        pos = np.array([np.interp(s, self.arc, self.gpath[:, k])
                        for k in range(3)])
        tan = np.array([np.interp(s, self.arc, self.tangents[:, k])
                        for k in range(3)])
        tan[1] = 0.0
        n = np.linalg.norm(tan)
        tan = tan / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        return pos, tan

    # -- pose -----------------------------------------------------------
    def _pose_raw(self, s: float) -> dict:
        """Joints + diagnostics at distance s, before the ground
        clamp is applied."""
        H = self.H
        phi = 2.0 * np.pi * s / self.stride
        root_xz, fwd = self._at(s)
        right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
        up = np.array([0.0, 1.0, 0.0])
        floor_y = float(floor_height_at(self.plane, root_xz[0],
                                        root_xz[2]))

        bob = -BOB_AMP * H * (0.5 + 0.5 * np.cos(2.0 * phi))
        sway = SWAY_AMP * H * np.sin(phi)
        pelvis = (root_xz + right * sway
                  + up * (floor_y + HIP_Y * H + bob))
        pelvis_yaw = 0.07 * np.sin(phi)
        chest_yaw = -0.05 * np.sin(phi)
        pelvis_pitch = 0.06 + 0.015 * np.sin(2.0 * phi)
        chest_pitch = 0.10 + 0.02 * np.sin(2.0 * phi)

        def place(origin, side, dy, dz, yaw=0.0, pitch=0.0):
            """Local (side, up, fwd) offset -> world, with the trunk's
            yaw/pitch wiggles applied to the vertical part."""
            r = right * np.cos(yaw) + fwd * np.sin(yaw)
            f = fwd * np.cos(yaw) - right * np.sin(yaw)
            return (origin + r * side
                    + up * (dy * np.cos(pitch)) + f * (dz + dy
                                                       * np.sin(pitch)))

        spine_top = place(pelvis, 0.0, (SPINE_TOP - HIP_Y) * H, 0.0,
                          chest_yaw, chest_pitch)
        neck = place(pelvis, 0.0, (NECK_Y - HIP_Y) * H, 0.0,
                     chest_yaw, chest_pitch)
        head = place(pelvis, 0.0, (HEAD_C - HIP_Y) * H, 0.0,
                     chest_yaw, chest_pitch)

        joints = {"pelvis": pelvis, "spine_top": spine_top,
                  "neck": neck, "head": head}
        diag = {}
        for side_name, sgn, psi in (("L", 1.0, phi),
                                    ("R", -1.0, phi + np.pi)):
            hip_ang, knee_ang, foot_pitch = leg_angles(psi)
            hip_j = place(pelvis, sgn * HIP_HALF * H, 0.0, 0.0,
                          pelvis_yaw, pelvis_pitch)
            d_thigh = fwd * np.sin(hip_ang) + up * (-np.cos(hip_ang))
            knee = hip_j + THIGH * H * d_thigh
            shank = hip_ang - knee_ang
            d_shin = fwd * np.sin(shank) + up * (-np.cos(shank))
            ankle = knee + SHIN * H * d_shin
            # foot: rotate the flat-foot contact offsets by foot_pitch
            # about the lateral axis at the ankle.
            cb, sb = np.cos(foot_pitch), np.sin(foot_pitch)

            def foot_pt(dz_off):
                y = -ANKLE_H * H
                z = dz_off * H
                return ankle + up * (y * cb - z * sb) \
                    + fwd * (y * sb + z * cb)

            toe = foot_pt(FOOT_FWD)
            heel = foot_pt(-FOOT_BACK)
            joints[f"hip_{side_name}"] = hip_j
            joints[f"knee_{side_name}"] = knee
            joints[f"ankle_{side_name}"] = ankle
            joints[f"toe_{side_name}"] = toe
            joints[f"heel_{side_name}"] = heel
            clear = min(
                float(toe[1] - floor_height_at(self.plane, toe[0],
                                               toe[2])),
                float(heel[1] - floor_height_at(self.plane, heel[0],
                                                heel[2])))
            diag[f"clear_{side_name}"] = clear
            diag[f"hip_{side_name}"] = float(hip_ang)
            diag[f"knee_{side_name}"] = float(knee_ang)
            # stance window: wrapped phase in [-0.1pi, 1.1pi],
            # raised sine across it, zero in swing
            w = psi % (2.0 * np.pi)
            if w >= 1.9 * np.pi:
                w -= 2.0 * np.pi
            st = float(np.sin(np.clip((w + 0.1 * np.pi)
                                      / (1.2 * np.pi), 0.0, 1.0)
                              * np.pi)) \
                if -0.1 * np.pi <= w <= 1.1 * np.pi else 0.0
            diag[f"stance_{side_name}"] = st
            # arm on the same side swings against this leg
            sh_ang, el_ang = arm_angles(psi)
            shoulder = place(pelvis, sgn * SHOULDER_HALF * H,
                             (SHOULDER_Y - HIP_Y) * H, 0.0,
                             chest_yaw, chest_pitch)
            d_uarm = fwd * np.sin(sh_ang) + up * (-np.cos(sh_ang))
            elbow = shoulder + UARM * H * d_uarm
            fa = sh_ang + el_ang
            d_farm = fwd * np.sin(fa) + up * (-np.cos(fa))
            wrist = elbow + FARM * H * d_farm
            joints[f"shoulder_{side_name}"] = shoulder
            joints[f"elbow_{side_name}"] = elbow
            joints[f"wrist_{side_name}"] = wrist
            diag[f"shoulder_{side_name}"] = float(sh_ang)

        diag["phase"] = float(phi % (2.0 * np.pi))
        diag["pelvis"] = joints["pelvis"]
        diag["fwd"] = fwd
        return {"joints": joints, "diag": diag}

    def pose(self, s: float) -> dict:
        """Full joint set + diagnostics at travelled distance s,
        with the ground clamp applied: the raw pose shifted down by
        the solved correction (a hard per-pose guard re-checks the
        lowering budget against this pose's own clearances)."""
        out = self._pose_raw(s)
        joints, diag = out["joints"], out["diag"]
        corr = float(np.interp(float(np.clip(s, 0.0, self.arc[-1])),
                               self.arc, self._corr))
        if corr > 0.0:
            corr = min(corr, min(diag["clear_L"], diag["clear_R"])
                       + 0.004 * self.H)
        if abs(corr) > 1e-12:
            up = np.array([0.0, 1.0, 0.0])
            for k in joints:
                joints[k] = joints[k] - up * corr
            for side_name in ("L", "R"):
                diag[f"clear_{side_name}"] -= corr
            diag["pelvis"] = joints["pelvis"]
        diag["clamp"] = corr
        return out

    # -- point figure ----------------------------------------------------
    def points(self, pose: dict) -> np.ndarray:
        """Stamp the canonical surface samples onto the posed bones."""
        j = pose["joints"]
        bones = [
            ("capsule", j["pelvis"] + (j["spine_top"] - j["pelvis"])
             * 0.15, j["neck"], "torso"),
            ("capsule", j["hip_L"], j["hip_R"], "pelvis"),
            ("sphere", j["head"], None, "head"),
            ("capsule", j["spine_top"], j["neck"], "neck"),
        ]
        for s in ("L", "R"):
            bones += [
                ("capsule", j[f"shoulder_{s}"], j[f"elbow_{s}"], "uarm"),
                ("capsule", j[f"elbow_{s}"], j[f"wrist_{s}"], "farm"),
                ("capsule", j[f"hip_{s}"], j[f"knee_{s}"], "thigh"),
                ("capsule", j[f"knee_{s}"], j[f"ankle_{s}"], "shin"),
                ("capsule", j[f"ankle_{s}"], j[f"toe_{s}"], "foot"),
                ("sphere", j[f"knee_{s}"], None, "knee_s"),
                ("sphere", j[f"elbow_{s}"], None, "elbow_s"),
                ("sphere", j[f"wrist_{s}"], None, "wrist_s"),
                ("sphere", j[f"ankle_{s}"], None, "ankle_s"),
            ]
        out = []
        for kind, a, b, key in bones:
            t, cu, su, r = self.template[key]
            if kind == "sphere":
                out.append(a[None, :] + self.template[key + "_dir"] * r)
                continue
            axis = b - a
            length = float(np.linalg.norm(axis))
            if length < 1e-9:
                continue
            d = axis / length
            ref = np.array([0.0, 1.0, 0.0])
            if abs(float(d @ ref)) > 0.92:
                ref = np.array([1.0, 0.0, 0.0])
            e1 = ref - d * float(d @ ref)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(d, e1)
            off = r * (cu[:, None] * e1[None, :] + su[:, None]
                       * e2[None, :])
            out.append(a[None, :] + t[:, None] * axis[None, :] + off)
        return np.vstack(out).astype(np.float64)


def _build_template(H: float) -> dict:
    """Canonical per-bone surface samples (fixed seed -> the same
    figure every frame, only its pose changes)."""
    rng = np.random.default_rng(FIG_SEED)

    def capsule(n, r_frac):
        t = rng.random(n)
        th = rng.uniform(0.0, 2.0 * np.pi, n)
        return t, np.cos(th), np.sin(th), r_frac * H

    def sphere(n, r_frac):
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v, r_frac * H

    tpl = {}
    for key, n, r in (("torso", 650, 0.098), ("pelvis", 240, 0.088),
                      ("neck", 80, 0.045), ("uarm", 200, 0.040),
                      ("farm", 180, 0.033), ("thigh", 360, 0.060),
                      ("shin", 300, 0.046), ("foot", 90, 0.030)):
        tpl[key] = capsule(n, r)
    for key, n, r in (("head", 420, HEAD_R), ("knee_s", 60, 0.047),
                      ("elbow_s", 50, 0.036), ("wrist_s", 40, 0.034),
                      ("ankle_s", 50, 0.038)):
        d, rr = sphere(n, r)
        tpl[key] = (None, None, None, rr)
        tpl[key + "_dir"] = d
    return tpl


# ---------------------------------------------------------------------------
# job loading + follow camera + render
# ---------------------------------------------------------------------------

def load_walk(job_dir: Path):
    """(scene points, scene rgb, walker, info) for a finished job."""
    job_dir = Path(job_dir)
    points_path = job_dir / "points.bin"
    poses_path = job_dir / "poses.json"
    if not points_path.exists():
        raise FileNotFoundError(f"points.bin missing in {job_dir}")
    if not poses_path.exists():
        raise FileNotFoundError(f"poses.json missing in {job_dir}")
    pts, rgb = _read_points_bin(points_path)
    poses = json.loads(poses_path.read_text())
    _stems, centres, targets = _camera_route(job_dir, poses)
    if len(centres) < 2:
        raise ValueError("need at least 2 posed frames for a route")
    path, _look = build_replay_path(centres, targets,
                                     SAMPLES_PER_SEGMENT)
    # figure height from the camera's own height above the local
    # floor track, same calibration as the HTML route replay
    track = _route_floor(pts, path)
    cam_h = float(np.median(path[:, 1] - track))
    fig_h = cam_h / 0.93 if cam_h > 0 else 0.0
    if not np.isfinite(fig_h) or fig_h <= 0:
        fig_h = max(float(np.percentile(pts[:, 1], 95)
                          - np.percentile(pts[:, 1], 5)) * 0.6, 1e-6)
    plane = fit_floor_plane(pts, path, fig_h)
    gpath, tangents, arc = ground_path(path, plane, ds=0.03 * fig_h)
    walker = Walker(fig_h, plane, gpath, tangents, arc)
    info = {"route_length": float(arc[-1]), "fig_height": fig_h,
            "floor": plane, "knots": int(len(centres)),
            "scene_points": int(len(pts))}
    return pts, rgb, walker, info


def _look_camera(eye, target, width, height, fov_y_deg=56.0):
    """Pinhole camera dict for the MUS rasterizer (Y-down rows)."""
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    n = np.linalg.norm(right)
    right = right / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ eye
    fy = (height / 2.0) / np.tan(np.radians(fov_y_deg) / 2.0)
    K = np.array([[fy, 0.0, width / 2.0],
                  [0.0, fy, height / 2.0],
                  [0.0, 0.0, 1.0]])
    return {"K": K, "R": R, "t": t}


def _bone_lines(j: dict) -> np.ndarray:
    """Skeleton segments (M, 2, 3) from a pose's joints -- drawn
    under the points so the figure reads as a figure even where the
    cloud is thin."""
    pairs = [("pelvis", "spine_top"), ("spine_top", "neck"),
             ("neck", "head"), ("hip_L", "hip_R")]
    for s in ("L", "R"):
        pairs += [(f"shoulder_{s}", f"elbow_{s}"),
                  (f"elbow_{s}", f"wrist_{s}"),
                  (f"hip_{s}", f"knee_{s}"), (f"knee_{s}", f"ankle_{s}"),
                  (f"ankle_{s}", f"toe_{s}")]
    pairs += [("pelvis", "hip_L"), ("pelvis", "hip_R"),
              ("spine_top", "shoulder_L"), ("spine_top", "shoulder_R")]
    return np.array([[j[a], j[b]] for a, b in pairs],
                    dtype=np.float64)


class _LosGrid:
    """Coarse occupancy hash of the scene, for camera line-of-sight
    checks: a fused cloud's warped sheets can cross the route, and a
    follow camera that cannot see its subject is useless."""

    def __init__(self, pts: np.ndarray, cell: float):
        self.cell = float(cell)
        keys = np.floor(pts[::3] / self.cell).astype(np.int64)
        self.occ = {tuple(k) for k in keys}

    def blocked(self, eye, target, skip_near_target=0.30) -> int:
        """How many samples along eye->target sit inside occupied
        neighbourhoods (samples stop short of the target so the
        figure's own surroundings do not count)."""
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        span = float(np.linalg.norm(target - eye))
        hits = 0
        for t in np.linspace(0.06, 0.94, 10):
            p = eye + t * (target - eye)
            if float(np.linalg.norm(target - p)) < skip_near_target:
                continue
            base = np.floor(p / self.cell).astype(np.int64)
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if tuple(base + (dx, dy, dz)) in self.occ:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            hits += found
        return hits


def render_walk_replay(job_dir, out_dir, width=1024, height=576,
                       fps=24, max_seconds=40.0) -> dict:
    """Render the walk to out_dir/replay_walk.mp4 (+ stills, report).

    The two stills are the walk's most readable frames (most figure
    pixels on screen among frames with real scene coverage), at
    least 15% of the walk apart. Returns the report dict (also
    written to report.json)."""
    from PIL import Image

    job_dir, out_dir = Path(job_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pts, rgb, walker, info = load_walk(job_dir)
    H = walker.H
    route_len = info["route_length"]
    speed = WALK_SPEED * H
    duration = route_len / speed
    n_frames = int(np.clip(round(duration * fps), 24,
                           int(max_seconds * fps)))
    eff_speed = route_len / (n_frames / fps)
    los = _LosGrid(pts, cell=0.10 * H)
    line_rgb = None

    fig_rgb = None
    cam_eye = None
    cam_tgt = None
    gold_count = np.zeros(n_frames, dtype=np.int64)
    scene_frac = np.zeros(n_frames, dtype=np.float64)
    with tempfile.TemporaryDirectory() as td:
        frames_dir = Path(td) / "frames"
        frames_dir.mkdir()
        for i in range(n_frames):
            s = route_len * i / max(n_frames - 1, 1)
            pose = walker.pose(s)
            j, diag = pose["joints"], pose["diag"]
            fig_pts = walker.points(pose)
            if fig_rgb is None or len(fig_rgb) != len(fig_pts):
                fig_rgb = np.tile(np.array(FIG_RGB, dtype=np.uint8),
                                  (len(fig_pts), 1))
            all_pts = np.vstack([pts, fig_pts])
            all_rgb = np.vstack([rgb, fig_rgb])
            fwd = diag["fwd"]
            root = j["pelvis"] - np.array([0.0, HIP_Y * H, 0.0])
            eye_goal = (root - fwd * (2.0 * H)
                        + np.array([0.0, 1.32 * H, 0.0])
                        + np.cross(np.array([0.0, 1.0, 0.0]), fwd)
                        * (0.55 * H))
            tgt_goal = root + np.array([0.0, 0.60 * H, 0.0]) \
                + fwd * (0.35 * H)
            # lift the camera until it can see the figure (warped
            # sheets crossing the route are this cloud's reality)
            if los.blocked(eye_goal, tgt_goal,
                           skip_near_target=0.30 * H) >= 3:
                for lift in (0.30, 0.60, 0.90, 1.20):
                    cand = eye_goal + np.array([0.0, lift * H, 0.0])
                    if los.blocked(cand, tgt_goal,
                                   skip_near_target=0.30 * H) < 3:
                        eye_goal = cand
                        break
                else:
                    eye_goal = eye_goal + np.array([0.0, 1.20 * H, 0.0])
            if cam_eye is None:
                cam_eye, cam_tgt = eye_goal, tgt_goal
            else:
                a_e = 1.0 - np.exp(-(1.0 / fps) / 0.45)
                a_t = 1.0 - np.exp(-(1.0 / fps) / 0.30)
                cam_eye = cam_eye + a_e * (eye_goal - cam_eye)
                cam_tgt = cam_tgt + a_t * (tgt_goal - cam_tgt)
            cam = _look_camera(cam_eye, cam_tgt, width, height)
            lines = _bone_lines(j)
            if line_rgb is None or len(line_rgb) != len(lines):
                line_rgb = np.tile(
                    np.array([255, 180, 84], dtype=np.uint8),
                    (len(lines), 1))
            frame = render_frame(all_pts, all_rgb,
                                 lines_xyz=lines,
                                 lines_rgb=line_rgb, camera=cam,
                                 width=width, height=height,
                                 point_radius_px=1.6,
                                 glow_strength=0.45, near=0.05 * H)
            fi = frame.astype(np.int32)
            gold_count[i] = int((
                (np.abs(fi[..., 0] - 255) < 60)
                & (np.abs(fi[..., 1] - 166) < 75)
                & (np.abs(fi[..., 2] - 66) < 75)).sum())
            scene_frac[i] = float(
                (np.abs(fi - np.array([5, 7, 14])).sum(axis=2)
                 > 30).mean())
            Image.fromarray(frame).save(frames_dir / f"f_{i + 1:05d}.png")
        # stills: the most readable frames that also show the scene
        # (a fused cloud is a hollow shell -- parts of the route read
        # as a figure in a void, and a preview still should not be
        # one of those). Candidates need real scene coverage and must
        # sit >=15% of the walk apart, away from the unsettled ends.
        margin = max(2, int(0.02 * n_frames))
        cand = [i for i in range(margin, n_frames - margin)
                if scene_frac[i] >= 0.12]
        if len(cand) < 2:
            cand = list(range(margin, n_frames - margin))
        cand.sort(key=lambda i: -gold_count[i])
        pick = [int(cand[0])]
        for c in cand[1:]:
            if abs(int(c) - pick[0]) >= max(2, int(0.15 * n_frames)):
                pick.append(int(c))
                break
        if len(pick) < 2:
            pick.append(int(cand[-1]))
        pick.sort()
        stills = []
        for k, fi_ in enumerate(pick):
            name = f"still_{k + 1:02d}.png"
            Image.open(frames_dir / f"f_{fi_ + 1:05d}.png").save(
                out_dir / name)
            stills.append(name)
        mp4 = out_dir / "replay_walk.mp4"
        _run_ffmpeg(["-y", "-v", "error", "-framerate", str(fps),
                     "-i", str(frames_dir / "f_%05d.png"),
                     "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-crf", "18", str(mp4)])
    report = {"job": str(job_dir), "mp4": "replay_walk.mp4",
              "stills": stills, "still_frames": [int(p) for p in pick],
              "frames": n_frames, "fps": fps,
              "duration_s": round(n_frames / fps, 2),
              "walk_speed_units_s": round(float(eff_speed), 3),
              **info}
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__)
        return 2
    report = render_walk_replay(Path(argv[0]), Path(argv[1]))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
