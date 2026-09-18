#!/usr/bin/env python3
"""Per-layer decode cache behaviour, simulated on the real pick trace.

Decode gives every layer exactly 6 picks per token, so per-layer miss RATES are
directly comparable -- unlike the server's live `by_layer` counts, which mix
prefill (where CED gives layers 0-19 ~80x the rows of 20-39) into the same
histogram.
"""
import sys, collections, json

N_LAYER, N_EXPERT = 40, 384
BOX1_MILLI, SLOTS = 397, 4070

def box2(l, e):
    return ((l * 1000003 + e * 7919) % 1000) >= BOX1_MILLI

toks, cur = [], []
for line in open(sys.argv[1]):
    p = line.split()
    if not p or p[0] != 'D':
        continue
    l = int(p[1]); ids = tuple(int(x) for x in p[2:] if int(x) >= 0)
    if l == 0 and cur:
        if len(cur) == N_LAYER: toks.append(cur)
        cur = []
    cur.append(ids)
if len(cur) == N_LAYER: toks.append(cur)

cache = collections.OrderedDict()
acc = [0]*N_LAYER; miss = [0]*N_LAYER; cold = [0]*N_LAYER
seen = set()
distinct = [set() for _ in range(N_LAYER)]
WARM = 2000
for t, tk in enumerate(toks):
    for L, ids in enumerate(tk):
        for e in ids:
            distinct[L].add(e)
            if box2(L, e):
                continue
            k = (L, e)
            counted = t >= WARM
            if counted: acc[L] += 1
            if k in cache:
                cache.move_to_end(k)
            else:
                if counted:
                    miss[L] += 1
                    if k not in seen: cold[L] += 1
                if len(cache) >= SLOTS: cache.popitem(last=False)
                cache[k] = 1
            seen.add(k)

rows = []
for L in range(N_LAYER):
    hit = 1 - miss[L]/max(1, acc[L])
    rows.append(dict(layer=L, hit=hit, miss_rate=miss[L]/max(1,acc[L]),
                     misses=miss[L], acc=acc[L],
                     cold_share=cold[L]/max(1,miss[L]),
                     distinct=len(distinct[L])))
print(f"tokens {len(toks):,} (scored after {WARM} warm-up)")
print(f"{'L':>3} {'hit':>8} {'miss/tok':>9} {'cold%':>7} {'distinct':>9}")
for r in rows:
    print(f"{r['layer']:>3} {r['hit']:>8.4f} {r['misses']/(len(toks)-WARM):>9.3f} "
          f"{100*r['cold_share']:>6.1f}% {r['distinct']:>9}")
tot_m = sum(miss); tot_a = sum(acc)
print(f"\nALL {1-tot_m/tot_a:>8.4f} {tot_m/(len(toks)-WARM):>9.3f}")
enc = sum(miss[:20])/sum(acc[:20]); dec = sum(miss[20:])/sum(acc[20:])
print(f"encoder L0-19 miss {100*enc:.2f}%   decoder L20-39 miss {100*dec:.2f}%   "
      f"ratio {dec/enc:.2f}x")
json.dump(rows, open(sys.argv[2], "w"))
