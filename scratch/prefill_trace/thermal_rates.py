#!/usr/bin/env python3
"""Throughput vs temperature per NVMe drive from nvme_thermal.sh logs: per hottest-sensor temperature band, the
active seconds, GB/s p50/p90/max and queue depth p50; plus the hottest temperature reached and how many seconds at or
above THROTTLE_C. Throttling shows as a falling GB/s ceiling at high temperature with deep queues.
  thermal_rates.py LOG [THROTTLE_C=80] [NAME=dev ...]"""
import collections, sys
path = sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].replace(".", "").isdigit() else 80.0
names = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
S = collections.defaultdict(list)
for l in open(path):
    p = l.split()
    if len(p) < 6:
        continue
    temps = [int(x) for x in p[5:8] if x.lstrip("-").isdigit()]
    S[p[1]].append((float(p[0]), int(p[2]), int(p[3]), max(temps) if temps else 0))
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
for dev, v in sorted(S.items()):
    bands = collections.defaultdict(list)
    hot = 0
    tmax = 0
    for a, b in zip(v, v[1:]):
        dt = b[0] - a[0]
        if dt <= 0 or dt > 3:
            continue
        gb = (b[1] - a[1]) * 512 / 1e9 / dt
        qd = (b[2] - a[2]) / 1e3 / dt
        t = b[3]
        tmax = max(tmax, t)
        hot += t >= thr
        if gb > 0.05:
            bands[(t // 5) * 5].append((gb, qd))
    print(f"{names.get(dev, dev)}: hottest {tmax} C; seconds >= {thr:.0f} C: {hot}")
    for k in sorted(bands):
        g = [x for x, _ in bands[k]]
        d = [y for _, y in bands[k]]
        print(f"   {k:3d}-{k + 4:3d} C: active s {len(g):5d}  GB/s p50 {q(g, .5):.2f} p90 {q(g, .9):.2f} max {max(g):.2f}  QD p50 {q(d, .5):.0f} p90 {q(d, .9):.0f}")
