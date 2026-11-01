"""Studio benches: project model math, renderer on the real room2
job, and a tiny end-to-end export. Run: .venv/bin/python
tests/test_studio.py — plain asserts, prints ALL PASS at the end."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo_app.studio import StudioProject

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


# ---- project model ---------------------------------------------------
p = StudioProject(duration=20.0)
check("default attach is orbit", p.attach_at(5.0) == "orbit")
p.set_attach(4.0, 12.0, "route")
check("attach set", p.attach_at(6.0) == "route" and p.attach_at(2.0) == "orbit")
p.set_attach(8.0, 16.0, "photo")
check("attach cut/override",
      p.attach_at(6.0) == "route" and p.attach_at(10.0) == "photo"
      and p.attach_at(14.0) == "photo" and p.attach_at(18.0) == "orbit")

check("no-slow identity mapping",
      abs(p.output_duration() - 20.0) < 1e-9
      and abs(p.to_source(7.5) - 7.5) < 1e-9)
p.set_slow(5.0, 10.0, 2.0)
check("slow stretches output", abs(p.output_duration() - 25.0) < 1e-9)
check("slow mapping inside", abs(p.to_source(9.0) - 7.0) < 1e-9)
check("slow mapping after", abs(p.to_source(20.0) - 15.0) < 1e-9)
check("roundtrip source<->output",
      all(abs(p.to_source(p.to_output(s)) - s) < 1e-6
          for s in (0.0, 3.0, 7.5, 12.0, 19.9)))
p.set_slow(5.0, 10.0, None)
check("slow cleared", abs(p.output_duration() - 20.0) < 1e-9)

p.set_crush(10.0, 0.2)
check("crush key holds", abs(p.crush_at(10.0) - 0.2) < 1e-6)
check("crush monotone travel",
      p.crush_at(2.0) > p.crush_at(6.0) > p.crush_at(10.0))

with tempfile.TemporaryDirectory() as td:
    p.save(Path(td))
    q = StudioProject.load(Path(td))
    check("save/load roundtrip",
          q.to_dict() == p.to_dict())

# ---- renderer on the real job -----------------------------------------
JOB = Path(__file__).resolve().parent.parent / "jobs" / "room2"
if JOB.exists():
    from echo_app.studio_render import StudioRenderer, export_mp4

    proj = StudioProject(duration=24.0)
    r = StudioRenderer(JOB, proj)
    check("events cover the build", len(r.events) > 50
          and r.events[0].kind == "selected"
          and r.events[-1].kind == "fused")

    img_early = r.render(1.0, 320, 180)      # select/poses phase
    img_mid = r.render(12.0, 320, 180)       # depth phase
    img_late = r.render(23.9, 320, 180)      # fused
    check("render shapes", img_early.shape == (180, 320, 3)
          and img_mid.shape == (180, 320, 3)
          and img_late.shape == (180, 320, 3))
    check("mid render has content", img_mid.std() > 1.0)
    check("late render has content", img_late.std() > 1.0)
    check("phases differ", not np.array_equal(img_mid, img_late))

    proj2 = StudioProject(duration=24.0)
    proj2.elements["grid"] = False
    img_nogrid = StudioRenderer(JOB, proj2).render(23.9, 320, 180)
    check("grid toggle changes image",
          not np.array_equal(img_late, img_nogrid))

    proj3 = StudioProject(duration=24.0)
    proj3.set_attach(0.0, 24.0, "route")
    img_route = StudioRenderer(JOB, proj3).render(18.0, 320, 180)
    img_orbit = StudioRenderer(JOB, StudioProject(duration=24.0)).render(
        18.0, 320, 180)
    check("route camera differs from orbit",
          not np.array_equal(img_route, img_orbit))

    proj4 = StudioProject(duration=24.0)
    proj4.elements["photos"] = False
    img_nophoto = StudioRenderer(JOB, proj4).render(12.0, 320, 180)
    check("photos off empties depth phase",
          img_nophoto.std() < img_mid.std())

    with tempfile.TemporaryDirectory() as td:
        out = export_mp4(JOB, StudioProject(duration=2.0), Path(td) / "s.mp4",
                         fps=4, width=320, height=180)
        check("export writes mp4", out.exists() and out.stat().st_size > 1000)
else:
    print("SKIP renderer benches (jobs/room2 missing)")

print("FAILURES:", FAILS if FAILS else "none")
print("ALL PASS" if not FAILS else "SOME FAILED")
sys.exit(1 if FAILS else 0)
