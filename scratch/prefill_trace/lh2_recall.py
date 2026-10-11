#!/usr/bin/env python3
"""Live look-ahead recall from hub_lh2 (decode) records: per N in 1..6, hits_paged_any_rN / paged_mirror_nonres (the
hintable box-2 misses the predictor's top-N caught) and hits_late_any_rN / paged_late_total (of the reads a lane
stalled on); plus per-step volumes.
  lh2_recall.py TXT"""
import sys
F, R = None, []
for line in open(sys.argv[1]):
    if line.startswith("#F hub_lh2 "):
        F = line.split()[2:]
    elif line.startswith("hub_lh2 ") and F:
        R.append(dict(zip(F, (float(x) for x in line.split()[1:]))))
R = [r for r in R if all(v == v for v in r.values())]
s = lambda k: sum(r.get(k, 0.0) for r in R)
print(f"steps {len(R)}; per step: paged {s('lh2_paged_total') / max(len(R), 1):.2f}, hintable (mirror non-resident) "
      f"{s('lh2_paged_mirror_nonres') / max(len(R), 1):.2f}, late {s('lh2_paged_late_total') / max(len(R), 1):.2f}")
for n in range(1, 7):
    a = s(f"lh2_hits_paged_any_r{n}") / max(s("lh2_paged_mirror_nonres"), 1)
    b = s(f"lh2_hits_late_any_r{n}") / max(s("lh2_paged_late_total"), 1)
    print(f"  top-{n}: recall of hintable misses {a:.2f}; of stalled reads {b:.2f}")
