#!/usr/bin/env python3
"""Per-second read-rate distribution per NVMe device from disksample.sh logs: active seconds (> 0.05 GB/s), GB/s
p50/p90/p99/max, average queue depth (ms_reading delta / 1000 per second), average IO size; the fastest seconds.
Does the drive ever deliver its rated speed, or is it capped (thermal throttle, IO size, ...)?
  disk_rates.py DISK_LOG [NAME=dev ...]"""
import collections, sys
S = collections.defaultdict(list)
for l in open(sys.argv[1]):
    p = l.split()
    if len(p) == 5:
        S[p[1]].append((float(p[0]), int(p[2]), int(p[3]), int(p[4])))
names = dict(a.split("=") for a in sys.argv[2:])
q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
for dev, v in sorted(S.items()):
    rates, qd, sz, top = [], [], [], []
    for a, b in zip(v, v[1:]):
        dt = b[0] - a[0]
        if dt <= 0 or dt > 3:
            continue
        gb = (b[1] - a[1]) * 512 / 1e9 / dt
        if gb > 0.05:
            d = (b[2] - a[2]) / 1e3 / dt
            rates.append(gb)
            qd.append(d)
            top.append((gb, d))
            if b[3] > a[3]:
                sz.append((b[1] - a[1]) * 512 / (b[3] - a[3]) / 1024)
    rates.sort(); qd.sort(); sz.sort(); top.sort()
    print(f"{names.get(dev, dev)}: active seconds {len(rates)}; GB/s p50 {q(rates, .5):.2f} p90 {q(rates, .9):.2f} "
          f"p99 {q(rates, .99):.2f} max {rates[-1] if rates else 0:.2f}; queue depth p50 {q(qd, .5):.0f} p90 {q(qd, .9):.0f}; avg IO {q(sz, .5):.0f} KB")
    print("   fastest seconds (GB/s, QD):", [(round(r, 2), round(d)) for r, d in top[-5:]])
