#!/usr/bin/env python3
"""Series of one 'ms.stage' counter (e.g. sub.picks_swapped) from hub logs, per 10-minute bucket: mean ms_per_step.
  stage_series.py STAGE LOG [LOG...]"""
import collections, re, sys
A = re.compile(r"\x1b\[[0-9;]*m")
stage = sys.argv[1]
b = collections.defaultdict(list)
for p in sys.argv[2:]:
    for l in open(p, errors="replace"):
        l = A.sub("", l)
        if f'stage="{stage}"' not in l:
            continue
        m = re.search(r'ms_per_step="([0-9.]+)"', l)
        if m:
            b[l[:15] + "0"].append(float(m.group(1)))
for k in sorted(b):
    v = b[k]
    print(f"{k}  n {len(v):4d}  {stage} per step {sum(v) / len(v):8.1f}")
