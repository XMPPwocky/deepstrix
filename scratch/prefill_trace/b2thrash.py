#!/usr/bin/env python3
"""Thrash check over a b2win span (b2_read + b2_req): for every DEMAND read (src 0) after the first SKIP_S seconds,
was that (layer, expert) evicted earlier in the span -- and by a read of which class (src 0 demand, 1/2 certain
background, 3 speculative = layer-ahead prefetch words, 4 LIKELY)? Also per class: how many landed reads were
'wanted' and how many landed pages were evicted again before any demand use.
  b2thrash.py TXT [SKIP_S=0]"""
import collections, sys
F, reads = {}, []
for line in open(sys.argv[1]):
    if line.startswith("#F "):
        p = line.split()
        F[p[1]] = p[2:]
        continue
    if line.startswith("b2_read ") and "b2_read" in F:
        reads.append(dict(zip(F["b2_read"], (float(x) for x in line.split()[1:]))))
skip = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
t = lambda r: r["t_read_start"] if r["t_read_start"] > 0 else r["t_pop"]
reads.sort(key=t)
T0 = t(reads[0])
evicted_by = {}   # key -> src of the read whose landing evicted it
landed_src = {}   # key -> src of the read that last landed it
used = set()
cls = collections.Counter()
by = collections.Counter()
wasted = collections.Counter()
for r in reads:
    key = (int(r["layer"]), int(r["expert"]))
    s = int(r["src"])
    if s == 0 and (t(r) - T0) / 1e9 >= skip:
        if key in evicted_by:
            cls["demand miss: evicted earlier in span"] += 1
            by[evicted_by[key]] += 1
        else:
            cls["demand miss: first touch in span"] += 1
    if r["victim_layer"] == r["victim_layer"] and r["victim_layer"] >= 0:
        vk = (int(r["victim_layer"]), int(r["victim_expert"]))
        evicted_by[vk] = s
        if vk in landed_src and vk not in used:
            wasted[landed_src[vk]] += 1
        landed_src.pop(vk, None)
        used.discard(vk)
    landed_src[key] = s
    if s == 0:
        used.add(key)
    evicted_by.pop(key, None)
print(dict(cls))
print("evicted by read class (0 demand, 1/2 certain bg, 3 speculative, 4 LIKELY):", dict(by))
print("reads per class:", dict(collections.Counter(int(r["src"]) for r in reads)))
print("wanted share per class:", {s: f"{sum(1 for r in reads if int(r['src']) == s and r['wanted'] > 0) / max(1, sum(1 for r in reads if int(r['src']) == s)):.0%}"
                                  for s in sorted({int(r['src']) for r in reads})})
print("landed then evicted with no demand read in between, per landing class:", dict(wasted))
