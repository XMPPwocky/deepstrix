#!/usr/bin/env python3
"""agg.py results/final_run*.txt  -> per (tag, variant): medians of med_us across runs, p10s, ratio vs base
Reads the KBJSON lines the harness prints (kb::ab)."""
import json, sys, collections, statistics
runs = sys.argv[1:]
data = collections.defaultdict(lambda: collections.defaultdict(list))   # tag -> variant -> [(med, p10)]
for path in runs:
    for line in open(path):
        if not line.startswith("KBJSON {\"tag\""):
            continue
        j = json.loads(line[7:])
        data[j["tag"]][j["variant"]].append((j["med_us"], j["p10_us"]))
for tag, vs in data.items():
    print(f"\n== {tag}   ({len(runs)} runs)")
    base = None
    for v, xs in vs.items():
        if base is None:
            base = xs
        meds = [x[0] for x in xs]; p10s = [x[1] for x in xs]
        bmeds = [x[0] for x in base]
        ratio = statistics.median(meds) / statistics.median(bmeds)
        print(f"  {v:52s} n={len(xs)} med {statistics.median(meds):7.2f} [{min(meds):6.2f}..{max(meds):6.2f}]  p10 {statistics.median(p10s):7.2f}  x{ratio:.3f}")
