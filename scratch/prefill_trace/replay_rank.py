#!/usr/bin/env python3
"""Rank-based restore sizing for option 2: with the persisted DECODE pick counts (expert_stats.json, 40 x 384), how
many of the replay's misses would a top-K-per-layer restore cover? Two candidate pools per layer >= 20:
  displaced -- experts evicted during the phase (resident at its start), ranked by decode count;
  any       -- every expert of the layer not displaced-and-missed-proof... i.e. all 384 ranked by decode count, minus
               those box 2 never reads (approximated: the experts box 2 read at least once in the span = box-2-owned).
Prints coverage of replay misses and GB to read per K.
  replay_rank.py TXT SPAN0_ISO R0_ISO R1_ISO STATS_JSON"""
import datetime, json, sys
path, s0, r0, r1, stats = sys.argv[1:6]
F, reads = {}, []
for line in open(path):
    if line.startswith("#F "):
        p = line.split()
        F[p[1]] = p[2:]
        continue
    if line.startswith("b2_read ") and "b2_read" in F:
        reads.append(dict(zip(F["b2_read"], (float(x) for x in line.split()[1:]))))
lo = int([l for l in open(path) if l.startswith("# ranges raw ")][0].split()[-1].split(":")[0])
iso = lambda s: datetime.datetime.fromisoformat(s).timestamp()
R0 = lo + (iso(r0) - iso(s0)) * 1e9
R1 = lo + (iso(r1) - iso(s0)) * 1e9
t = lambda r: r["t_read_start"] if r["t_read_start"] > 0 else r["t_pop"]
reads.sort(key=t)
dec = json.load(open(stats))["decode"]["counts"]
cnt = lambda l, e: dec[l * 384 + e]
disp = {}
for r in reads:
    if t(r) >= R0:
        break
    disp.pop((int(r["layer"]), int(r["expert"])), None)
    if r["victim_layer"] == r["victim_layer"] and r["victim_layer"] >= 20:
        disp[(int(r["victim_layer"]), int(r["victim_expert"]))] = 1
miss = {(int(r["layer"]), int(r["expert"])) for r in reads if R0 <= t(r) <= R1 and int(r["src"]) == 0 and r["layer"] >= 20}
owned = {(int(r["layer"]), int(r["expert"])) for r in reads} | set(disp)
layers = sorted({l for l, _ in miss})
print(f"replay misses {len(miss)} over layers {layers[0]}-{layers[-1]}; box-2 candidate experts seen {sum(1 for k in owned if k[0] >= 20)}")
for pool in ("displaced", "owned"):
    for K in (10, 20, 30, 40, 60, 80):
        sel = set()
        for l in range(20, 40):
            cand = [e for (ll, e) in (disp if pool == "displaced" else owned) if ll == l]
            cand.sort(key=lambda e: -cnt(l, e))
            sel |= {(l, e) for e in cand[:K]}
        cov = len(miss & sel)
        print(f"  {pool:9} top-{K:2d}/layer: restore {len(sel):5d} ({len(sel) * 18.8 / 1000:5.1f} GB) -> covers {cov:4d} of {len(miss)} = {cov / max(len(miss), 1):4.0%}")
