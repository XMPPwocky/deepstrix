#!/usr/bin/env python3
"""Run ON box 2 over b2win text (b2_req) from stdin: for prefill-shaped requests (b >= 64), and separately b == 1024:
n, background words dropped (sum, per request), demand misses per request (p50 / mean), page_us p50 / mean,
prefetch wait (d_prefetch_wait_ns) mean ms, server_us p50.
  b2win.py ... b2_req | b2_drops.py"""
import statistics, sys
F = None
rows = []
for line in sys.stdin:
    if line.startswith("#F b2_req"):
        F = line.split()[2:]
        continue
    if not line.startswith("b2_req ") or F is None:
        continue
    rows.append(dict(zip(F, map(float, line.split()[1:]))))
ok = lambda v: v == v
for name, sel in (("b>=64", lambda r: r["b"] >= 64), ("b=1024", lambda r: r["b"] == 1024)):
    rr = [r for r in rows if sel(r)]
    if not rr:
        print(name, "n 0")
        continue
    g = lambda k: [r[k] for r in rr if ok(r[k])]
    dr, miss, page, pw, srv = g("pf_d_dropped"), g("n_miss"), g("page_us"), g("d_prefetch_wait_ns"), g("server_us")
    print(f"{name}: n {len(rr)}; dropped {sum(dr):.0f} ({sum(dr) / len(rr):.1f}/req); n_miss p50 {statistics.median(miss):.0f} "
          f"mean {statistics.mean(miss):.1f}; page_us p50 {statistics.median(page):.0f} mean {statistics.mean(page):.0f}; "
          f"prefetch wait mean {statistics.mean(pw) / 1e6 if pw else float('nan'):.1f} ms; server_us p50 {statistics.median(srv):.0f}")
