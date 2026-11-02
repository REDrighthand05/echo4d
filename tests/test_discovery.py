"""Bench for echo_scene/discovery.py — synthetic cloud, known answers.

Run: .venv/bin/python tests/test_discovery.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_scene.discovery import adjudicate, locate, read_ply_points

FAILS = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        FAILS.append(name)


POSE = {"cam_from_world_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "cam_from_world_t": [0.0, 0.0, 0.0],
        "camera": {"params": [1000.0, 540.0, 960.0, 0.0]}}

rng = np.random.default_rng(7)
# object: 500 pts in a slab centred (0,0,3), size (0.3,0.2,0.1)
obj = np.column_stack([rng.uniform(-0.15, 0.15, 500),
                       rng.uniform(-0.10, 0.10, 500),
                       rng.uniform(2.95, 3.05, 500)])
# background wall at z=5 covering the same view
wall = np.column_stack([rng.uniform(-1.5, 1.5, 800),
                        rng.uniform(-1.5, 1.5, 800),
                        rng.uniform(4.95, 5.05, 800)])
cloud = np.vstack([obj, wall])

BOX = {"xmin": 460, "ymin": 860, "xmax": 620, "ymax": 1060}
m = locate(cloud, POSE, BOX)
check("locate finds the object", m is not None)
check("centre near (0,0,3)",
      m is not None and np.linalg.norm(m["centre"] - [0, 0, 3]) < 0.08)
check("depth layer rejects the wall behind",
      m is not None and m["centre"][2] < 3.5)
check("size roughly (0.3,0.2,0.1)",
      m is not None and abs(m["size"][0] - 0.3) < 0.1
      and abs(m["size"][2] - 0.1) < 0.08)
check("empty box measures nothing",
      locate(cloud, POSE, {"xmin": 0, "ymin": 0, "xmax": 40, "ymax": 40})
      is None)

poses = {"frame_a.jpg": POSE, "frame_b.jpg": POSE}
hit = {"label": "chair", "score": 0.6, "box": BOX}
noms2 = {"frame_a.jpg": {"hits": [hit]}, "frame_b.jpg": {"hits": [hit]}}
res = adjudicate(poses, cloud, noms2, scale_m=0.25)
check("two frames corroborate -> one 已证 entry",
      len(res["objects"]) == 1 and res["objects"][0]["tier"] == "已证")
check("metres conversion present",
      res["objects"] and "size_m" in res["objects"][0])
res1 = adjudicate(poses, cloud, {"frame_a.jpg": {"hits": [hit]}},
                  scale_m=0.25)
check("one frame -> 单见", len(res1["objects"]) == 1
      and res1["objects"][0]["tier"] == "单见")
bad = {"label": "door", "score": 0.7,
       "box": {"xmin": 0, "ymin": 0, "xmax": 30, "ymax": 30}}
res2 = adjudicate(poses, cloud, {"frame_a.jpg": {"hits": [bad]}})
check("unmeasurable nomination is unverified, not listed",
      len(res2["objects"]) == 0 and len(res2["unverified"]) == 1)
low = {"label": "door", "score": 0.05, "box": BOX}
res3 = adjudicate(poses, cloud, {"frame_a.jpg": {"hits": [low]}})
check("below-threshold nomination ignored entirely",
      len(res3["objects"]) == 0 and len(res3["unverified"]) == 0)

# PLY round trip
import struct
tmp = Path(__file__).parent / "_tmp_test.ply"
verts = np.array([[1, 2, 3], [4, 5, 6]], dtype="<f4")
hdr = (b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
       b"property float x\nproperty float y\nproperty float z\n"
       b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
       b"end_header\n")
rows = b"".join(verts[i].tobytes() + bytes([9, 9, 9]) for i in range(2))
tmp.write_bytes(hdr + rows)
got = read_ply_points(tmp)
check("ply reader round trip",
      got.shape == (2, 3) and np.allclose(got[1], [4, 5, 6]))
tmp.unlink()

print("FAILURES:", FAILS if FAILS else "none")
sys.exit(1 if FAILS else 0)
