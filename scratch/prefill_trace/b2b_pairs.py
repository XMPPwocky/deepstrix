#!/usr/bin/env python3
"""Back-to-back prefills: a job's FIRST window opening within MAX_GAP_S of another job's last window closing. Prints
each pair: gap, previous job (tokens, its last window s), the new job's first window (rows, s) and its second window
if any -- is the new job's first window slow even with box 2 warm from the previous job?
  b2b_pairs.py MAX_GAP_S LOG [LOG...]"""
import datetime, re, sys
A = re.compile(r"\x1b\[[0-9;]*m")
iso = lambda s: datetime.datetime.fromisoformat(s.rstrip("Z")).timestamp()
mx = float(sys.argv[1])
op, W = {}, []
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
            W.append(dict(t0=iso(t0), t1=iso(t), ts=t0[11:19], job=int(g["tokens_total"]), done=int(go["tokens_done"]), rows=int(go["rows"]),
                          last=int(g["tokens_done"]) == int(g["tokens_total"])))
W.sort(key=lambda w: w["t0"])
for i, w in enumerate(W):
    if w["done"] != 0:
        continue
    prev = [x for x in W[:i] if x["last"] and 0 <= w["t0"] - x["t1"] <= mx]
    if not prev:
        continue
    pv = prev[-1]
    nxt = [x for x in W[i + 1:] if x["job"] == w["job"] and x["done"] > 0][:1]
    s2 = f"{nxt[0]['t1'] - nxt[0]['t0']:5.2f} s ({nxt[0]['rows']} rows)" if nxt else "-"
    print(f"{w['ts']}  gap {w['t0'] - pv['t1']:5.2f} s  prev job {pv['job']:6d} last window {pv['t1'] - pv['t0']:5.2f} s"
          f"  ->  new job {w['job']:6d} first window {w['rows']:5d} rows {w['t1'] - w['t0']:5.2f} s, second {s2}")
