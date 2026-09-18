#!/usr/bin/env python3
"""Parse het.token.summary / pager lines out of a server log into a decode report."""
import re, sys, statistics as st

path = sys.argv[1]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # skip first N summaries (warmup)
txt = open(path, errors="replace").read()
# strip ANSI
txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)

def kv(line):
    return dict(re.findall(r"(\w+)=([-\w./]+)", line))

toks = [kv(l) for l in txt.splitlines() if "het.token.summary" in l]
stages = {}
for l in txt.splitlines():
    if "het.stage" in l:
        d = kv(l)
        key = (d.get("device"), d.get("stage"))
        stages.setdefault(key, []).append((int(d["total_us"]), int(d["calls"]), int(d.get("token_pos", 0))))

sel = toks[lo:]
if not sel:
    print("no token summaries"); sys.exit(0)
def col(k, cast=int):
    return [cast(t[k]) for t in sel if k in t]
n = len(sel)
print(f"tokens: {n}  pos {sel[0].get('token_pos')}..{sel[-1].get('token_pos')}")
for k in ("total_us","dgpu_busy_us","igpu_busy_us","sel_sync_us","pager_ensure_us",
          "pager_read_us","pager_h2d_us","pager_misses","host_us","sync_us"):
    v = col(k)
    if not v: continue
    print(f"  {k:18s} mean {st.mean(v):10.1f}  median {st.median(v):10.1f}  min {min(v):9d}  max {max(v):9d}  sum {sum(v)}")
tot = sum(col("total_us"))
print(f"  ms/token {tot/n/1000:.1f}   tok/s {1e6*n/tot:.2f}")
if stages:
    print("\nper-stage (mean us/token over profiled tokens):")
    rows = []
    for (dev, name), v in stages.items():
        us = [x[0] for x in v]; calls = [x[1] for x in v]
        rows.append((st.mean(us), dev, name, st.mean(calls), len(v)))
    for m, dev, name, c, ntok in sorted(rows, reverse=True):
        print(f"  {dev:5s} {name:34s} {m:9.1f} us  calls {c:6.1f}  ntok {ntok}")
    print(f"  TOTAL dgpu {sum(m for m,d,_,_,_ in rows if d=='dgpu'):9.1f} us   igpu {sum(m for m,d,_,_,_ in rows if d=='igpu'):9.1f} us")
print()
for pat in ("expert pager (request)", "expert pager (cumulative)", "decode miss phases",
            "decode heartbeat", "expert read phases", "ced_replay", "expert pager:"):
    for l in txt.splitlines():
        if pat in l:
            print(re.sub(r"^.*?(deepstrix_server|v4flash_kernels)\S*:?\s*", "", l).strip()[:400])
