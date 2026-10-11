#!/usr/bin/env python3
"""Every layer-major window from hub logs with its job size: (time, prompt tokens, window index, rows, seconds),
then medians for NON-first windows split by job size < / >= SPLIT_MIN tokens (the long-job prefill split threshold).
  win_by_job.py SPLIT_MIN LOG [LOG...]"""
import datetime, re, sys
A = re.compile(r"\x1b\[[0-9;]*m")
lim = int(sys.argv[1])
iso = lambda s: datetime.datetime.fromisoformat(s.rstrip("Z")).timestamp()
op, rows = {}, []
for p in sys.argv[2:]:
    for l in open(p, errors="replace"):
        l = A.sub("", l)
        if "prefill_lm_window" not in l:
            continue
        g = dict(re.findall(r"(\w+)=([0-9]+)", l))
        t = l.split()[0]
        if " open " in l:
            op[g["tokens_total"]] = (t, g)
        elif " done " in l and g.get("tokens_total") in op:
            t0, go = op.pop(g["tokens_total"])
            rows.append((t0[11:19], int(g["tokens_total"]), int(go["tokens_done"]), int(go["rows"]), iso(t) - iso(t0)))
for small in (True, False):
    v = sorted(d for _, tt, done, r, d in rows if done > 0 and r >= 3584 and (tt < lim) == small)
    if v:
        print(f"{'<' if small else '>='} {lim} tokens: non-first windows >= 3584 rows: n {len(v)}, s p10 {v[len(v) // 10]:.2f} p50 {v[len(v) // 2]:.2f} p90 {v[9 * len(v) // 10]:.2f}")
print("\nlast 25 windows (time, job tokens, tokens_done, rows, s):")
for r in rows[-25:]:
    print(f"  {r[0]}  job {r[1]:6d}  done {r[2]:6d}  rows {r[3]:5d}  {r[4]:6.2f}")
