#!/usr/bin/env python3
"""summ.py FILE... : compact table of kb::ab KBJSON lines, one row per tag, variants side by side
(med/p10 in us and ratio vs variant 0). With several files (separate runs), prints per-run rows and
a cross-run summary: median of per-run median ratios and median of per-run p10 ratios."""
import json, sys, collections, statistics

runs = []
for fn in sys.argv[1:]:
    d = collections.OrderedDict()
    for line in open(fn, errors="replace"):
        if not line.startswith("KBJSON {\"tag\""):
            continue
        j = json.loads(line[7:])
        d.setdefault(j["tag"], []).append(j)
    runs.append((fn, d))

tags = []
for _, d in runs:
    for t in d:
        if t not in tags:
            tags.append(t)

for t in tags:
    print(f"== {t}")
    per_var = collections.OrderedDict()
    for fn, d in runs:
        if t not in d:
            continue
        rows = d[t]
        base = rows[0]
        cells = []
        for r in rows:
            rm = r["med_us"] / base["med_us"]
            rp = r["p10_us"] / base["p10_us"]
            cells.append(f"{r['variant'][:34]:34s} {r['med_us']:8.2f} {r['p10_us']:8.2f} x{rm:5.3f}/{rp:5.3f}")
            per_var.setdefault(r["variant"], []).append((rm, rp, r["med_us"], r["p10_us"]))
        if len(runs) == 1:
            for c in cells:
                print("   " + c)
    if len(runs) > 1:
        for v, lst in per_var.items():
            rms = [x[0] for x in lst]
            rps = [x[1] for x in lst]
            meds = [x[2] for x in lst]
            print(f"   {v[:40]:40s} n={len(lst)} med_us={statistics.median(meds):8.2f} "
                  f"ratio_med={statistics.median(rms):6.3f} [{min(rms):.3f}..{max(rms):.3f}] "
                  f"ratio_p10={statistics.median(rps):6.3f}")
