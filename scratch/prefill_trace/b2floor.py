#!/usr/bin/env python3
"""Box-2 prefill request FLOOR cost (p10 per bin) vs distinct experts and vs rows, plus merged-vs-single requests
(box 2's own same-layer merge = a natural experiment for cross-unit batching).  b2floor.py B2WIN_TXT"""
import sys

fields, rows = None, []
for line in open(sys.argv[1]):
    if line.startswith("#F b2_req "):
        fields = line.split()[2:]
    elif line.startswith("b2_req ") and fields:
        rows.append(dict(zip(fields, (float(x) for x in line.split()[1:]))))
pre = [r for r in rows if r["b"] >= 64 and r["n_miss"] == 0 and r["exec_gpu_us"] > 0]


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


print(f"hit-only prefill requests: {len(pre)}")
print("\nby distinct experts (b == 512, not merged):")
print(f"{'distinct':>10} {'n':>6} {'gpu p10':>8} {'gpu p50':>8} {'p10 us/expert':>14}")
single = [r for r in pre if r["b"] == 512 and r["merged"] == 0]
for lo in range(80, 200, 20):
    v = [r["exec_gpu_us"] for r in single if lo <= r["n_distinct"] < lo + 20]
    if len(v) >= 20:
        print(f"{lo:4d}-{lo + 19:<4d} {len(v):6d} {q(v, .1):8.0f} {q(v, .5):8.0f} {q(v, .1) / (lo + 10):14.1f}")
print("\nby rows b (not merged):")
print(f"{'b':>10} {'n':>6} {'distinct p50':>12} {'gpu p10':>8} {'gpu p50':>8}")
for lo, hi in [(64, 256), (256, 511), (512, 512), (513, 1024), (1025, 4096)]:
    v = [r for r in pre if lo <= r["b"] <= hi and r["merged"] == 0]
    if len(v) >= 10:
        print(f"{lo:4d}-{hi:<5d} {len(v):6d} {q([r['n_distinct'] for r in v], .5):12.0f} "
              f"{q([r['exec_gpu_us'] for r in v], .1):8.0f} {q([r['exec_gpu_us'] for r in v], .5):8.0f}")
print("\nmerged (box 2 ran two same-layer requests as one pass) vs single, b == 512 leaders:")
for m in (0, 1):
    v = [r for r in pre if r["merged"] == m and r["b"] >= 256]
    if v:
        pb = [r["partner_b"] for r in v if r["partner_b"] == r["partner_b"]]
        print(f"  merged={m}: n {len(v)}, b p50 {q([r['b'] for r in v], .5):.0f}, partner_b p50 {q(pb, .5) if pb else float('nan'):.0f}, "
              f"distinct p50 {q([r['n_distinct'] for r in v], .5):.0f}, gpu p10 {q([r['exec_gpu_us'] for r in v], .1):.0f} "
              f"p50 {q([r['exec_gpu_us'] for r in v], .5):.0f}")
