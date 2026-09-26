#!/usr/bin/env python3
"""Summarise rocprofv3 PMC csv(s): per kernel-name, median counter value over dispatches
(dispatches 0-1 discarded as warm-up). Usage: pmc_summary.py dir [dir ...]"""
import csv, sys, os, statistics
from collections import defaultdict

for d in sys.argv[1:]:
    p = os.path.join(d, "run_counter_collection.csv")
    if not os.path.exists(p):
        print(f"{d}: no csv"); continue
    rows = list(csv.DictReader(open(p)))
    # columns: Dispatch_Id, Kernel_Name, Counter_Name, Counter_Value, ...
    per = defaultdict(lambda: defaultdict(list))
    seen = defaultdict(set)
    for r in rows:
        k = r["Kernel_Name"]; c = r["Counter_Name"]; v = float(r["Counter_Value"]); did = int(r["Dispatch_Id"])
        seen[k].add(did)
        per[(k, did)][c].append(v)
    for k in sorted(seen):
        dids = sorted(seen[k])
        if len(dids) > 2: dids = dids[2:]
        agg = defaultdict(list)
        for did in dids:
            for c, vs in per[(k, did)].items():
                agg[c].append(sum(vs))
        print(f"{d}: {k[:60]} dispatches={len(dids)}")
        for c in sorted(agg):
            print(f"    {c:28s} med={statistics.median(agg[c]):.6g}  min={min(agg[c]):.6g} max={max(agg[c]):.6g}")
