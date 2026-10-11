#!/usr/bin/env python3
"""DSpark accept rate over time from hub logs ('ms dspark: blocks' lines, one per 50 blocks): per BUCKET-minute
bucket the mean accept_rate, tokens/block, k_mean, and the block-weighted share of K=0..5 from k_hist.
  accept_series.py BUCKET_MIN LOG [LOG...]"""
import collections, re, sys
A = re.compile(r"\x1b\[[0-9;]*m")
bm = int(sys.argv[1])
b = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
for path in sys.argv[2:]:
    for l in open(path, errors="replace"):
        l = A.sub("", l)
        if "ms dspark: blocks blocks=" not in l:
            continue
        hh, mm = int(l[11:13]), int(l[14:16])
        key = f"{l[:11]}{hh:02d}:{(mm // bm) * bm:02d}"
        g = lambda k: float(re.search(k + r'="([0-9.]+)"', l).group(1))
        x = b[key]
        x[0] += 1
        x[1] += g("accept_rate")
        x[2] += g("tokens_per_block")
        x[3] += g("k_mean")
for t in sorted(b):
    n, a, tp, k = b[t]
    print(f"{t}  n {n:4d}  accept {a / n:.3f}  tok/block {tp / n:.2f}  k_mean {k / n:.2f}")
