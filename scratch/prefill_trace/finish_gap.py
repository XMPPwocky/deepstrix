#!/usr/bin/env python3
"""Prefill FINISH time per job: last layer-major window done -> 'stream admitted' (multistream), matched by prompt
size (admitted prompt - restored == tokens_total). Also every log line in between for the first N jobs (what runs).
  finish_gap.py N_SHOW LOG [LOG...]"""
import datetime, re, sys
A = re.compile(r"\x1b\[[0-9;]*m")
iso = lambda s: datetime.datetime.fromisoformat(s.rstrip("Z")).timestamp()
show = int(sys.argv[1])
lines = []
for p in sys.argv[2:]:
    lines += [A.sub("", l).rstrip("\n") for l in open(p, errors="replace")]
last_done = {}
out = []
for i, l in enumerate(lines):
    if "prefill_lm_window done" in l:
        g = dict(re.findall(r"(\w+)=([0-9]+)", l))
        if g.get("tokens_done") == g.get("tokens_total"):
            last_done[int(g["tokens_total"])] = (iso(l.split()[0]), i)
    elif "stream admitted" in l:
        g = dict(re.findall(r"(\w+)=([0-9]+)", l))
        n = int(g["prompt"]) - int(g["restored"])
        if n in last_done:
            t0, i0 = last_done.pop(n)
            out.append((l[11:19], n, iso(l.split()[0]) - t0, int(g.get("prefill_ms", 0)), i0, i))
v = sorted(x[2] for x in out)
if v:
    print(f"jobs {len(v)}: finish s p10 {v[len(v) // 10]:.2f} p50 {v[len(v) // 2]:.2f} p90 {v[9 * len(v) // 10]:.2f}")
for ts, n, d, pms, i0, i1 in out[-12:]:
    print(f"  {ts} job {n:6d} finish {d:5.2f} s (prefill_ms {pms})")
for ts, n, d, pms, i0, i1 in out[-show:] if show else []:
    print(f"\n--- {ts} job {n} ({d:.2f} s) ---")
    for l in lines[i0:i1 + 1]:
        if re.search(r"ms\.step|ms dspark|ms\.stage|ms lanes", l):
            continue
        print("   ", l[11:23], l[l.find(":", 30) + 1:][:200] if "INFO" in l or "WARN" in l else l[:200])
