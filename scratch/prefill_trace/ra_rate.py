#!/usr/bin/env python3
"""Run ON box 2 over b2win text (b2_req,b2_read) from stdin: per prefill burst (b2_req b >= 64, split at gaps > 20 s):
span, prefill requests, read-ahead words carried, readahead (src 5) reads + their GB/s, hold (hint -> land), read time,
wait-for-land time (read end -> land start), and the readers' busy share (sum of read walls / (span * 2 readers)).
  b2win.py ... b2_req,b2_read | ra_rate.py"""
import statistics, sys
F = {}
reqs, reads = [], []
for line in sys.stdin:
    if line.startswith("#F "):
        p = line.split()
        F[p[1]] = p[2:]
        continue
    k = line.split(" ", 1)[0]
    if k not in F:
        continue
    r = dict(zip(F[k], (float(x) for x in line.split()[1:])))
    if k == "b2_req" and r["b"] >= 64:
        reqs.append(r)
    elif k == "b2_read" and int(r["src"]) == 5:
        reads.append(r)
reqs.sort(key=lambda r: r["t_dequeue"])
bursts, cur = [], []
for r in reqs:
    if cur and r["t_dequeue"] - cur[-1]["t_dequeue"] > 20e9:
        bursts.append(cur)
        cur = []
    cur.append(r)
if cur:
    bursts.append(cur)
p50 = lambda v: statistics.median(v) if v else float("nan")
for b in bursts:
    t0, t1 = b[0]["t_dequeue"], b[-1]["t_ready"]
    rr = [x for x in reads if t0 <= x["t_hint"] <= t1]
    span = (t1 - t0) / 1e9
    words = sum(x["n_prefetch_words"] for x in b)
    busy = sum(x["t_read_end"] - x["t_read_start"] for x in rr) / 1e9
    print(f"burst {span:6.1f} s: {len(b)} prefill reqs, {words:.0f} read-ahead words carried, {len(rr)} readahead reads "
          f"({len(rr) * 18.8e-3 / max(span, 1e-9):.2f} GB/s); p50 ms: hint->land {p50([(x['t_land_end'] - x['t_hint']) / 1e6 for x in rr]):.1f}, "
          f"queue {p50([(x['t_read_start'] - x['t_hint']) / 1e6 for x in rr]):.1f}, read {p50([(x['t_read_end'] - x['t_read_start']) / 1e6 for x in rr]):.1f}, "
          f"read end->land {p50([(x['t_land_start'] - x['t_read_end']) / 1e6 for x in rr]):.1f}, pause {p50([x['pause_ns'] / 1e6 for x in rr]):.1f}; "
          f"readers busy {busy / max(span * 2, 1e-9):.0%}")
