# DS Flash batch 3 (2026-11-01): sensor-stream utilities for the
# collection-head pre-development; written by DeepSeek Flash from
# echo_super/workorders/WO-*.md specs; to be verified by Muse
# (echo_super/test_ds_batch3.py — 29/29 benches green; registered via
# adapters.register_ds_batch3).


# ---- WO-tof-grid-points ----
import numpy as np

def tof_grid_points(dist_mm, hfov_deg, vfov_deg, max_range_mm):
    dist = np.asarray(dist_mm)
    if dist.shape != (8, 8):
        raise ValueError("dist_mm must have shape (8, 8)")

    hfov = np.deg2rad(float(hfov_deg))
    vfov = np.deg2rad(float(vfov_deg))
    tan_h = np.tan(hfov / 2.0)
    tan_v = np.tan(vfov / 2.0)

    valid = (dist > 0) & (dist <= max_range_mm)
    zones = np.argwhere(valid).astype(np.int64)

    if zones.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)

    r = zones[:, 0].astype(np.float64)
    c = zones[:, 1].astype(np.float64)

    u = (2.0 * (c + 0.5) / 8.0 - 1.0) * tan_h
    v = (1.0 - 2.0 * (r + 0.5) / 8.0) * tan_v

    d = np.stack([u, v, np.ones_like(u)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    dist_valid = dist[zones[:, 0], zones[:, 1]].astype(np.float64)
    points = d * (dist_valid[:, None] / 1000.0)

    return points, zones


# ---- WO-heading ----
import numpy as np

def tilt_compensated_heading(mag, accel):
    # Body frame convention: x forward, y left, z up.
    # Heading is 0 at magnetic north, clockwise positive, in [0, 360) degrees.
    m = np.asarray(mag, dtype=float)
    a = np.asarray(accel, dtype=float)

    if np.linalg.norm(a) == 0.0:
        raise ValueError("accel norm must be non-zero")
    if np.linalg.norm(m) == 0.0:
        raise ValueError("mag norm must be non-zero")

    ax, ay, az = a
    mx, my, mz = m

    phi = np.arctan2(ay, az)
    theta = np.arctan2(-ax, np.sqrt(ay * ay + az * az))

    mx_h = (
        mx * np.cos(theta)
        + my * np.sin(phi) * np.sin(theta)
        + mz * np.cos(phi) * np.sin(theta)
    )
    my_h = my * np.cos(phi) - mz * np.sin(phi)

    heading = (np.degrees(np.arctan2(-my_h, mx_h)) + 360.0) % 360.0
    return float(heading)



# ---- WO-pressure-altitude ----
import math

def pressure_altitude_m(pressure_hpa: float, reference_hpa: float) -> float:
    try:
        if not (pressure_hpa > 0 and reference_hpa > 0):
            raise ValueError("pressure_hpa and reference_hpa must be greater than 0")
    except TypeError:
        raise ValueError("pressure_hpa and reference_hpa must be positive numbers")

    if pressure_hpa == reference_hpa:
        return 0.0

    return 44330.0 * (1.0 - math.pow(pressure_hpa / reference_hpa, 0.1903))


# ---- WO-sound-speed ----
import math

def speed_of_sound_mps(temp_c: float) -> float:
    if not (-60 <= temp_c <= 80):
        raise ValueError("temp_c must be in the range -60..80")
    return 331.3 * math.sqrt(1 + temp_c / 273.15)


# ---- WO-sync-stats ----
import numpy as np

def stream_sync_stats(streams):
    try:
        n_streams = len(streams)
    except TypeError:
        raise ValueError("streams must be a non-empty list of timestamp arrays")
    if n_streams == 0:
        raise ValueError("streams must be a non-empty list of timestamp arrays")

    arrs = []
    for s in streams:
        try:
            a = np.asarray(s, dtype=np.float64)
        except Exception as exc:
            raise ValueError("each stream must be a 1D float64 timestamp array") from exc
        if a.ndim != 1 or a.size < 1:
            raise ValueError("each stream must be a 1D array with at least one element")
        if a.size > 1 and not np.all(np.diff(a) > 0.0):
            raise ValueError("each stream must be strictly increasing")
        arrs.append(a)

    base = arrs[0]

    counts = []
    spans_s = []
    median_interval_s = []
    max_nearest_offset_s = []
    p95_nearest_offset_s = []

    for i, a in enumerate(arrs):
        n = int(a.size)
        counts.append(n)
        spans_s.append(float(a[-1] - a[0]))

        if n == 1:
            median_interval_s.append(0.0)
        else:
            median_interval_s.append(float(np.median(np.diff(a))))

        if i == 0:
            max_nearest_offset_s.append(0.0)
            p95_nearest_offset_s.append(0.0)
        else:
            idx = np.searchsorted(base, a)
            left = np.clip(idx - 1, 0, base.size - 1)
            right = np.clip(idx, 0, base.size - 1)
            offsets = np.minimum(
                np.abs(a - base[left]),
                np.abs(a - base[right])
            )
            max_nearest_offset_s.append(float(np.max(offsets)))
            p95_nearest_offset_s.append(float(np.percentile(offsets, 95.0)))

    return {
        "counts": counts,
        "spans_s": spans_s,
        "median_interval_s": median_interval_s,
        "max_nearest_offset_s": max_nearest_offset_s,
        "p95_nearest_offset_s": p95_nearest_offset_s,
    }


# ---- WO-spectral-ratios ----
import numpy as np

def spectral_ratios(counts):
    arr = np.asarray(counts, dtype=np.float64)
    if arr.shape != (11,):
        raise ValueError("counts must have shape (11,)")
    if np.any(arr < 0):
        raise ValueError("counts must be non-negative")
    clear = arr[8]
    if clear <= 0:
        raise ValueError("Clear channel must be > 0")

    f = arr[:8]
    nir = arr[9]

    return {
        "normalized": np.asarray(f / clear, dtype=np.float64),
        "nir_ratio": float(nir / clear),
        "visible_sum_ratio": float(np.sum(f) / clear),
    }
