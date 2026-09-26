#!/usr/bin/env python3
"""summarize.py <files...> — per (tag, variant): median-of-run-medians, min/max over runs, ratio vs the
tag's first variant (the baseline). Reads the KBJSON lines the harness prints."""
import json, sys, statistics as st
from collections import OrderedDict
rows = OrderedDict()
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith("KBJSON "): continue
        d = json.loads(line[7:])
        if "med_us" not in d: continue
        rows.setdefault(d["tag"], OrderedDict()).setdefault(d["variant"], []).append(d["med_us"])
for tag, vs in rows.items():
    print(f"== {tag}")
    base = None
    for v, meds in vs.items():
        m = st.median(meds)
        if base is None: base = m
        print(f"  {v:52s} runs={len(meds)} med={m:8.2f}  min={min(meds):8.2f} max={max(meds):8.2f}  ratio={m/base:.3f}")
