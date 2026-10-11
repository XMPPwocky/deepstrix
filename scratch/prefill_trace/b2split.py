#!/usr/bin/env python3
"""Box-2 prefill requests (b >= 64) split by time segment (cuts = box-2 t_hdr instants where >= GAP_S passes with no
prefill request, e.g. a decode burst) and by hit/miss: count, exec_gpu / page / server p50/p90 (ms), picks, decode
requests interleaved, and box-2 busy share (union of [t_run_start, t_run_end] over the segment's wall).
  b2split.py B2REQ_TXT [GAP_S=5]"""
import sys
fields, rows = None, []
for line in open(sys.argv[1]):
    if line.startswith("#F b2_req "):
        fields = line.split()[2:]
    elif line.startswith("b2_req ") and fields:
        rows.append(dict(zip(fields, (float(x) for x in line.split()[1:]))))
gap = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
rows.sort(key=lambda r: r["t_hdr"])
pre = [r for r in rows if r["b"] >= 64]
dec = [r for r in rows if r["b"] < 64]


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


segs, cur = [], [pre[0]]
for a, b in zip(pre, pre[1:]):
    if (b["t_hdr"] - a["t_hdr"]) / 1e9 >= gap:
        segs.append(cur)
        cur = []
    cur.append(b)
segs.append(cur)
for k, s in enumerate(segs):
    t0, t1 = s[0]["t_hdr"], s[-1]["t_ready"]
    iv = sorted((r["t_run_start"], r["t_run_end"]) for r in s)
    busy, ce = 0.0, None
    for a, b in iv:
        if ce is None or a > ce:
            busy += b - a
            ce = b
        elif b > ce:
            busy += b - ce
            ce = b
    nd = sum(1 for r in dec if t0 <= r["t_hdr"] <= t1)
    print(f"segment {k}: {len(s)} prefill requests over {(t1 - t0) / 1e9:.1f} s, box-2 GPU busy {busy / (t1 - t0):.0%}, decode requests inside {nd}")
    for name, sub in (("hit", [r for r in s if r["n_miss"] == 0]), ("miss", [r for r in s if r["n_miss"] > 0])):
        if not sub:
            continue
        f = lambda key: (q([r[key] / 1e3 for r in sub], .5), q([r[key] / 1e3 for r in sub], .9))
        print(f"  {name:4} n {len(sub):4d}  gpu {f('exec_gpu_us')[0]:6.1f}/{f('exec_gpu_us')[1]:6.1f}  page {f('page_us')[0]:6.1f}/{f('page_us')[1]:6.1f}"
              f"  server {f('server_us')[0]:6.1f}/{f('server_us')[1]:6.1f}  picks p50 {q([r['n_sel'] for r in sub], .5):.0f}"
              f"  distinct {q([r['n_distinct'] for r in sub], .5):.0f}  misses p50 {q([r['n_miss'] for r in sub], .5):.0f}"
              f"  idle_before p50 {q([r['idle_before_us'] / 1e3 for r in sub], .5):.1f} ms")
