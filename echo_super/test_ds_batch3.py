"""Muse's benches for DS batch 3 (specs: workorders/WO-*.md)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from ds_batch3 import (tof_grid_points, tilt_compensated_heading,
                       pressure_altitude_m, speed_of_sound_mps,
                       stream_sync_stats, spectral_ratios)

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

# tof_grid_points: full 1000 mm, 45 deg square FoV
d = np.full((8, 8), 1000, dtype=int)
pts, zones = tof_grid_points(d, 45.0, 45.0, 4000)
check("tof all valid K=64", len(pts) == 64 and zones.shape == (64, 2))
check("tof unit range", np.allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-9))
idx = {tuple(z): p for z, p in zip(zones, pts)}
c = idx[(3, 3)]; corner = idx[(0, 0)]
check("tof centre symmetric", abs(abs(c[0]) - abs(c[1])) < 1e-12 and c[2] > 0)
check("tof corner wider than centre", abs(corner[0]) > abs(c[0]) and abs(corner[1]) > abs(c[1]))
check("tof top row y positive", idx[(0, 4)][1] > 0 and idx[(7, 4)][1] < 0)
d2 = d.copy(); d2[0, 0] = -1; d2[5, 5] = 0; d2[2, 3] = 99999
pts2, zones2 = tof_grid_points(d2, 45.0, 45.0, 4000)
check("tof invalid dropped+aligned", len(pts2) == 61 and (0, 0) not in map(tuple, zones2)
      and (5, 5) not in map(tuple, zones2) and (2, 3) not in map(tuple, zones2))
try:
    tof_grid_points(np.zeros((4, 4), dtype=int), 45.0, 45.0, 4000); check("tof shape raises", False)
except ValueError: check("tof shape raises", True)

# heading: flat benches from the spec
h0 = tilt_compensated_heading(np.array([1.0, 0.0, 0.2]), np.array([0.0, 0.0, 1.0]))
h90 = tilt_compensated_heading(np.array([0.0, -1.0, 0.2]), np.array([0.0, 0.0, 1.0]))
check("heading north = 0", abs(h0) < 1e-6)
check("heading east = 90", abs(h90 - 90.0) < 1e-6)
# yaw the whole rig by +40 deg: mag rotates in body frame, heading must shift +40
th = np.radians(40.0)
Rz = np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
mag_w = np.array([1.0, 0.0, 0.2])
h40 = tilt_compensated_heading(Rz @ mag_w, np.array([0.0, 0.0, 1.0]))
check("heading yaw shift +40", abs(h40 - 40.0) < 1e-4)
# pitch rig 30 deg nose-down about y-axis: build consistent accel+mag in body frame
ph = np.radians(30.0)
Ry = np.array([[np.cos(ph), 0, -np.sin(ph)], [0, 1.0, 0], [np.sin(ph), 0, np.cos(ph)]])
acc_b = Ry @ np.array([0.0, 0.0, 1.0]); mag_b = Ry @ mag_w
hp = tilt_compensated_heading(mag_b, acc_b)
check("heading pitch-compensated = 0", abs(hp) < 1e-4 or abs(hp - 360.0) < 1e-4)
try:
    tilt_compensated_heading(np.zeros(3), np.array([0.0, 0.0, 1.0])); check("heading zero mag raises", False)
except ValueError: check("heading zero mag raises", True)

# pressure altitude
check("pressure p=p0 -> 0", pressure_altitude_m(1013.25, 1013.25) == 0.0)
check("pressure 900hPa ~ 988.7", abs(pressure_altitude_m(900.0, 1013.25) - 988.7) < 1.5)
check("pressure monotonic", pressure_altitude_m(800.0, 1013.25) > pressure_altitude_m(900.0, 1013.25))
try:
    pressure_altitude_m(-1.0, 1013.25); check("pressure raises", False)
except ValueError: check("pressure raises", True)

# speed of sound
check("sound 0C exact", speed_of_sound_mps(0.0) == 331.3)
check("sound 20C ~343.2", abs(speed_of_sound_mps(20.0) - 343.2) < 0.3)
check("sound monotonic", speed_of_sound_mps(30.0) > speed_of_sound_mps(10.0))

# sync stats
base = np.arange(0.0, 1.01, 0.1)
shifted = np.arange(0.03, 1.01, 0.2)
st = stream_sync_stats([base, shifted])
check("sync counts", st["counts"] == [11, 5])
check("sync offset max 0.03", abs(st["max_nearest_offset_s"][1] - 0.03) < 1e-9)
check("sync offset p95 0.03", abs(st["p95_nearest_offset_s"][1] - 0.03) < 1e-9)
check("sync base offset 0", st["max_nearest_offset_s"][0] == 0.0)
check("sync median interval", abs(st["median_interval_s"][0] - 0.1) < 1e-12)
st1 = stream_sync_stats([np.array([0.5]), base])
check("sync single-element ok", st1["counts"][0] == 1 and st1["median_interval_s"][0] == 0.0)
try:
    stream_sync_stats([np.array([0.2, 0.1])]); check("sync non-monotonic raises", False)
except ValueError: check("sync non-monotonic raises", True)

# spectral ratios
counts = np.array([10, 20, 30, 40, 50, 60, 70, 80, 100, 25, 0], dtype=float)
sr = spectral_ratios(counts)
check("spectral normalized", np.allclose(sr["normalized"], np.arange(10, 90, 10) / 100.0, atol=1e-12))
check("spectral nir", abs(sr["nir_ratio"] - 0.25) < 1e-12)
check("spectral visible sum", abs(sr["visible_sum_ratio"] - 3.6) < 1e-12)

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
