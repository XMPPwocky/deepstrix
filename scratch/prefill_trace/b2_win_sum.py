#!/usr/bin/env python3
"""Run ON box 2 over b2win text (b2_req,b2_read) from stdin for ONE window span: one line -- requests, demand misses,
summed page / prefetch-wait / service ms over requests (box-2 serial work), background reads by class (src).
  b2win.py ... SPAN b2_req,b2_read | b2_win_sum.py LABEL"""
import collections, sys
F = {}
n = miss = 0
dist = 0.0
page = wait = srv = 0.0
reads = collections.Counter()
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
        n += 1
        miss += r["n_miss"]
        dist += r["n_distinct"]
        page += r["page_us"] / 1e3
        srv += r["server_us"] / 1e3
        if r["d_prefetch_wait_ns"] == r["d_prefetch_wait_ns"]:
            wait += r["d_prefetch_wait_ns"] / 1e6
    elif k == "b2_read":
        reads[int(r["src"])] += 1
print(f"{sys.argv[1]}: reqs {n} distinct/req {dist / max(n, 1):.0f} miss {miss:.0f} page {page:.0f} ms pf_wait {wait:.0f} ms service {srv:.0f} ms | bg reads "
      + " ".join(f"{c}:{reads[c]}" for c in sorted(reads)))
