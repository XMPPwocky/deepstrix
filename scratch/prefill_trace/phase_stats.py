#!/usr/bin/env python3
"""Prefill/decode phase behaviour of the multistream scheduler from hub logs ('ms.phase' switch lines: from, to, live,
prefills, queued, burst_ms = how long the phase that ENDS lasted, next_budget_ms, starved):
  * PREFILL phases that froze live streams (live > 0 at the switch back to decode): durations = decode stalls;
  * DECODE phases while prefills/queue waited: durations = added prefill wait (TTFT);
  * how phases end: budget elapsed vs ran out of work (burst_ms < budget);
  * share of wall time in each phase.
  phase_stats.py LOG [LOG...]"""
import re, sys
A = re.compile(r"\x1b\[[0-9;]*m")
rows = []
for p in sys.argv[1:]:
    for l in open(p, errors="replace"):
        l = A.sub("", l)
        if "ms.phase" not in l:
            continue
        g = dict(re.findall(r"(\w+)=([A-Za-z0-9.]+)", l))
        rows.append((l[:23], g.get("from"), g.get("to"), int(g.get("live", 0)), int(g.get("prefills", 0)),
                     int(g.get("queued", 0)), int(g.get("burst_ms", 0)), g.get("starved")))
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
pf_with_live = [r[6] / 1e3 for r in rows if r[1] == "Prefill" and r[3] > 0]
pf_all = [r[6] / 1e3 for r in rows if r[1] == "Prefill"]
dec_with_wait = [r[6] / 1e3 for r in rows if r[1] == "Decode" and (r[4] > 0 or r[5] > 0)]
dec_all = [r[6] / 1e3 for r in rows if r[1] == "Decode"]
print(f"phase switches: {len(rows)}")
for name, v in (("prefill phases (all)", pf_all), ("prefill phases that froze live streams", pf_with_live),
                ("decode phases (all)", dec_all), ("decode phases while prefills/queue waited", dec_with_wait)):
    if v:
        print(f"  {name:44} n {len(v):4d}  s p50 {q(v, .5):6.1f} p90 {q(v, .9):6.1f} max {max(v):6.1f}  total {sum(v) / 60:6.1f} min")
n_pf_budget = sum(1 for x in pf_all if x >= 119.5)
n_dec_budget = sum(1 for x in dec_all if x >= 29.5)
print(f"  prefill phases ending at the 120 s budget: {n_pf_budget} of {len(pf_all)}; decode phases ending at the 30 s budget: {n_dec_budget} of {len(dec_all)}")
print(f"  starved switches: {sum(1 for r in rows if r[7] == 'true')}")
fz = sorted(pf_with_live)[-5:]
print("  longest stream freezes (s):", [round(x, 1) for x in fz])
