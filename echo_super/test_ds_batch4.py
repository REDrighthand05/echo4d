"""Muse's benches for DS batch 4 (spec: workorders/WO-sample-track.md)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ds_batch4 import sample_track

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

lin = [(0.0, 0.0, 'linear'), (2.0, 10.0, 'linear')]
check("clamp before", sample_track(lin, -1.0) == 0.0)
check("clamp after", sample_track(lin, 5.0) == 10.0)
check("linear midpoint", sample_track(lin, 1.0) == 5.0)
check("linear quarter", abs(sample_track(lin, 0.5) - 2.5) < 1e-12)
check("single key", sample_track([(3.0, 7.0, 'linear')], 0.0) == 7.0
      and sample_track([(3.0, 7.0, 'linear')], 99.0) == 7.0)

hold = [(0.0, 4.0, 'hold'), (1.0, 9.0, 'linear')]
check("hold keeps until next key", sample_track(hold, 0.999) == 4.0)
check("hold jumps at next key", sample_track(hold, 1.0) == 9.0)

ei = [(0.0, 0.0, 'ease_in'), (1.0, 1.0, 'linear')]
eo = [(0.0, 0.0, 'ease_out'), (1.0, 1.0, 'linear')]
eio = [(0.0, 0.0, 'ease_in_out'), (1.0, 1.0, 'linear')]
check("ease_in(0.5) ~ 0.316", 0.30 < sample_track(ei, 0.5) < 0.33)
check("ease_out(0.5) ~ 0.684", 0.67 < sample_track(eo, 0.5) < 0.70)
check("ease_in_out(0.5) = 0.5", abs(sample_track(eio, 0.5) - 0.5) < 1e-6)
check("ease mirror symmetry",
      abs(sample_track(eo, 0.3) - (1.0 - sample_track(ei, 0.7))) < 1e-4)
mono = all(sample_track(eio, u / 20) <= sample_track(eio, (u + 1) / 20) + 1e-12
           for u in range(20))
check("ease_in_out monotonic", mono)
check("easing endpoints exact",
      sample_track(ei, 0.0) == 0.0 and abs(sample_track(ei, 0.999999) - 1.0) < 1e-3)

multi = [(0.0, 0.0, 'linear'), (1.0, 10.0, 'hold'), (3.0, 30.0, 'linear')]
check("segment 1 linear", sample_track(multi, 0.5) == 5.0)
check("segment 2 holds own easing", sample_track(multi, 2.5) == 10.0)
check("segment 2 end value", sample_track(multi, 3.0) == 30.0)
check("non-uniform times",
      abs(sample_track([(0.0, 0.0, 'linear'), (0.25, 1.0, 'linear'),
                        (2.25, 3.0, 'linear')], 1.25) - 2.0) < 1e-12)
check("deterministic", sample_track(eio, 0.37) == sample_track(eio, 0.37))
try:
    sample_track([], 0.0)
    check("empty keys raises", False)
except ValueError:
    check("empty keys raises", True)

print()
print(f"{len(fails)} failures" if fails else "ALL PASS")
sys.exit(1 if fails else 0)
