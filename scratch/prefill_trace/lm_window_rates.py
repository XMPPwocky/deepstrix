#!/usr/bin/env python3
"""Layer-major window throughput by window size, from the hub log (prefill_lm_window open/done pairs per job).
Also: how many prefill jobs never opened a window (chunked only), by prompt size.
  lm_window_rates.py SERVER_LOG [SINCE_ISO]"""
import collections, datetime, re, sys
A = re.compile(r"\x1b\[[0-9;]*m")
since = sys.argv[2] if len(sys.argv) > 2 else ""
iso = lambda s: datetime.datetime.fromisoformat(s.rstrip("Z")).timestamp()
opens, wins, jobs_lm, prog = {}, [], set(), {}
for l in open(sys.argv[1], errors="replace"):
    l = A.sub("", l)
    m = re.match(r"^(\S+)Z ", l)
    if not m or m.group(1) < since:
        continue
    t = m.group(1)
    if "prefill_lm_window open" in l:
        g = dict(re.findall(r"(\w+)=([0-9]+)", l))
        opens[g["tokens_total"]] = (t, int(g["rows"]), int(g["tokens_done"]), g.get("layer_outer"))
        jobs_lm.add(g["tokens_total"])
    elif "prefill_lm_window done" in l:
        g = dict(re.findall(r"(\w+)=([0-9]+)", l))
        o = opens.pop(g["tokens_total"], None)
        if o:
            wins.append((int(g["rows"]), iso(t) - iso(o[0]), int(g["tokens_total"]), o[2]))
    elif "prefill_job_progress" in l:
        g = dict(re.findall(r"(\w+)=([0-9]+)", l))
        if "tokens_total" in g:
            prog.setdefault(g["tokens_total"], t)
buckets = [(1, 2048), (2049, 2560), (2561, 3072), (3073, 3584), (3585, 4095), (4096, 4096)]
print("window rows        n   s p50   rows/s p50   (first windows of a job excluded: cold box-2 / setup)")
for lo, hi in buckets:
    v = sorted((d, r) for r, d, tt, done in wins if lo <= r <= hi and done > 0 and d < 30)
    if not v:
        continue
    d50 = v[len(v) // 2][0]
    rate = sorted(r / d for d, r in v)[len(v) // 2]
    print(f"  {lo:5d}-{hi:5d}  {len(v):5d}  {d50:6.2f}   {rate:8.0f}")
v = sorted((d, r) for r, d, tt, done in wins if done == 0 and d < 60)
if v:
    print(f"  first windows: n {len(v)}, s p50 {v[len(v) // 2][0]:.2f}, rows p50 {sorted(r for _, r in v)[len(v) // 2]}")
sz = collections.Counter()
for tt in prog:
    n = int(tt)
    b = "<=2048" if n <= 2048 else "2049-4096" if n <= 4096 else "4097-16K" if n <= 16384 else ">16K"
    sz[(b, tt in jobs_lm)] += 1
print("\nprefill jobs (prompt size, used layer-major windows?):")
for b in ("<=2048", "2049-4096", "4097-16K", ">16K"):
    print(f"  {b:10} LM {sz[(b, True)]:5d}   chunked-only {sz[(b, False)]:5d}")
