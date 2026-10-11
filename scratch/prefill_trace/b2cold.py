#!/usr/bin/env python3
"""Why is box 2 cold at a job's first window? From b2win text (b2_read + b2_req, box-2 clock) over a span that ends
with the window W = [W0, W1] (box-2 wall ISO): every (layer, expert) box 2 READ inside W, classified by its history in
the span before W:
  never_seen   -- not read in the span before W (and not evicted in it): the earlier turns did not use it (or it was
                  resident the whole time -- then it would not be read in W)
  evicted      -- read earlier in the span, then EVICTED before W: by whom (the evicting read's request size: prefill
                  b >= 64 vs decode b < 64, via the evicting read's seq -> b2_req.b), and how long before W
  For the evicted ones, the evictor's layer distribution and time histogram.
  b2cold.py TXT W0_ISO W1_ISO   (box-2 wall clock; b2win prints its raw ranges, this maps wall -> raw the same way)
  --fields: print one b2_read record's fields and exit"""
import collections, datetime, sys, time

path = sys.argv[1]
F, reads, reqs = {}, [], {}
for line in open(path):
    if line.startswith("#F "):
        p = line.split()
        F[p[1]] = p[2:]
        continue
    k = line.split(" ", 1)[0]
    if k == "b2_read" and k in F:
        reads.append(dict(zip(F[k], (float(x) for x in line.split()[1:]))))
    elif k == "b2_req" and k in F:
        r = dict(zip(F[k], (float(x) for x in line.split()[1:])))
        reqs[int(r["seq"])] = r
if "--fields" in sys.argv:
    print({k: v for k, v in reads[0].items()})
    sys.exit(0)
# wall -> raw (this script runs on box 1 against box-2 data: use the raw ranges b2win printed instead)
rng = [l for l in open(path) if l.startswith("# ranges raw ")][0].split()[-1]
lo, hi = (int(x) for x in rng.split(":"))
w0w, w1w = sys.argv[2], sys.argv[3]
# the span's wall bounds are the b2win args; W is given in wall time inside it: interpolate linearly (raw is 1:1 with wall)
span0 = None
for a in sys.argv[4:5]:
    span0 = a
iso = lambda s: datetime.datetime.fromisoformat(s).timestamp()
S0 = iso(span0)
W0 = lo + (iso(w0w) - S0) * 1e9
W1 = lo + (iso(w1w) - S0) * 1e9
t = lambda r: r["t_read_start"] if r["t_read_start"] > 0 else r["t_pop"]
inW = [r for r in reads if W0 <= t(r) <= W1]
before = sorted((r for r in reads if t(r) < W0), key=t)
last_read, last_evict = {}, {}
for r in before:
    last_read[(int(r["layer"]), int(r["expert"]))] = r
    if r["victim_layer"] >= 0:
        last_evict[(int(r["victim_layer"]), int(r["victim_expert"]))] = r
cls = collections.Counter()
ev_by, ev_age, ev_layer = collections.Counter(), [], collections.Counter()
seen = set()
for r in inW:
    key = (int(r["layer"]), int(r["expert"]))
    if key in seen:
        continue
    seen.add(key)
    lr, le = last_read.get(key), last_evict.get(key)
    if le is not None and (lr is None or t(le) >= t(lr)):
        cls["evicted"] += 1
        rq = reqs.get(int(le["seq"]))
        kind = "unknown-seq" if rq is None else ("prefill" if rq["b"] >= 64 else "decode")
        if le["src"] != 0 and rq is None:
            kind = f"src{int(le['src'])}"
        ev_by[kind] += 1
        ev_age.append((W0 - t(le)) / 1e9)
        ev_layer[int(le["layer"])] += 1
    elif lr is not None:
        cls["read_before_not_evicted?"] += 1
    else:
        cls["never_seen"] += 1
print(f"W: {len(inW)} reads, {len(seen)} distinct (layer, expert)")
print("classes:", dict(cls))
print("evicted by (the evicting read's request):", dict(ev_by))
if ev_age:
    ev_age.sort()
    print(f"evicted how long before W: p10 {ev_age[len(ev_age) // 10]:.0f} s, p50 {ev_age[len(ev_age) // 2]:.0f} s, p90 {ev_age[9 * len(ev_age) // 10]:.0f} s")
    print("evictor's layer (top):", ev_layer.most_common(10))
print("src values in W:", collections.Counter(int(r["src"]) for r in inW), "| span reads", len(reads), "reqs", len(reqs))
