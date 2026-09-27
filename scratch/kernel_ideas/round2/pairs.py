#!/usr/bin/env python3
"""pairs.py [--pairs "OLDRE=>NEWRE" ...] FILE... : cross-run crossover table from kb::ab KBJSON lines.
For every tag, each (old, new) variant pair given by regexes (default: variant 0 vs every other one)
gets: runs, old med us, new med us, median over runs of new/old med ratio, [min..max], median of the
per-run p10 ratio. Ratios are computed per run (interleaved A/B), then aggregated."""
import json, re, statistics, sys, collections

args = sys.argv[1:]
pairs = []
tagf = None
while args and args[0].startswith("--"):
    if args[0] == "--pairs":
        for p in args[1].split(";;"):
            o, n = p.split("=>")
            pairs.append((re.compile(o), re.compile(n)))
        args = args[2:]
    elif args[0] == "--tag":
        tagf = re.compile(args[1]); args = args[2:]
    else:
        break

runs = []
for fn in args:
    d = collections.OrderedDict()
    for line in open(fn, errors="replace"):
        i = line.find("KBJSON {\"tag\"")
        if i < 0:
            continue
        try:
            j = json.loads(line[i + 7:])
        except json.JSONDecodeError:
            continue  # a line interleaved with stderr
        d.setdefault(j["tag"], collections.OrderedDict())[j["variant"]] = j
    runs.append(d)

tags = []
for d in runs:
    for t in d:
        if t not in tags and (tagf is None or tagf.search(t)):
            tags.append(t)

for t in tags:
    rows = collections.OrderedDict()
    for d in runs:
        if t not in d:
            continue
        vs = d[t]
        names = list(vs)
        prs = []
        if pairs:
            for o, n in pairs:
                on = [x for x in names if o.search(x)]
                nn = [x for x in names if n.search(x)]
                if on and nn:
                    prs.append((on[0], nn[0]))
        else:
            prs = [(names[0], x) for x in names[1:]]
        for o, n in prs:
            a, b = vs[o], vs[n]
            rows.setdefault((o, n), []).append((b["med_us"] / a["med_us"], b["p10_us"] / a["p10_us"], a["med_us"], b["med_us"]))
    for (o, n), lst in rows.items():
        rm = [x[0] for x in lst]; rp = [x[1] for x in lst]
        print(f"{t[:60]:60s} | {n[:34]:34s} vs {o[:28]:28s} n={len(lst)} old={statistics.median([x[2] for x in lst]):8.2f} "
              f"new={statistics.median([x[3] for x in lst]):8.2f} r={statistics.median(rm):.3f} [{min(rm):.3f}..{max(rm):.3f}] p10r={statistics.median(rp):.3f}")
