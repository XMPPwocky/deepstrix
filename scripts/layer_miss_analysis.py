#!/usr/bin/env python3
"""Per-layer MISS rate (cost), with the two nulls that could fake a trend.

  null 1: our own hash partition gives each layer a different number of
          box-1-owned experts, so a layer with more owned experts has a bigger
          working set and should miss more -- nothing to do with the model.
  null 2: Poisson noise. ~500 misses per layer means +-4.5% counting error.
"""
import sys, json, math, collections
N_LAYER, N_EXPERT, BOX1_MILLI, SLOTS = 40, 384, 397, 4070
def box2(l, e): return ((l*1000003 + e*7919) % 1000) >= BOX1_MILLI

rows = json.load(open(sys.argv[1]))
owned = [sum(1 for e in range(N_EXPERT) if not box2(L, e)) for L in range(N_LAYER)]
mr = [r["miss_rate"] for r in rows]
mis = [r["misses"] for r in rows]
acc = [r["acc"] for r in rows]

def pearson(a, b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((y-mb)**2 for y in b))
    return num/(da*db) if da and db else 0.0

lo, hi = min(mr), max(mr)
print(f"miss rate  min {100*lo:.2f}% (L{mr.index(lo)})   max {100*hi:.2f}% (L{mr.index(hi)})"
      f"   spread {hi/lo:.2f}x")
print(f"misses per layer: min {min(mis)}  max {max(mis)}  -> Poisson sd ~{math.sqrt(sum(mis)/N_LAYER):.0f}"
      f" ({100/math.sqrt(sum(mis)/N_LAYER):.1f}% relative)")
print(f"\nowned experts/layer (hash partition): min {min(owned)} max {max(owned)} "
      f"mean {sum(owned)/N_LAYER:.1f}")
print(f"  corr(miss_rate, owned)      {pearson(mr, owned):+.3f}   <- null 1")
print(f"  corr(miss_rate, layer idx)  {pearson(mr, list(range(N_LAYER))):+.3f}   <- depth trend")
print(f"  corr(miss_rate, distinct)   {pearson(mr, [r['distinct'] for r in rows]):+.3f}")

# Residual after removing the partition effect: miss rate per OWNED expert.
adj = [m/o for m, o in zip(mr, owned)]
alo, ahi = min(adj), max(adj)
print(f"\nafter dividing out owned-count: spread {ahi/alo:.2f}x "
      f"(was {hi/lo:.2f}x)")
print(f"  corr(adjusted, layer idx)   {pearson(adj, list(range(N_LAYER))):+.3f}")

# Is the spread beyond Poisson? Chi-square against a constant rate.
p = sum(mis)/sum(acc)
chi = sum((m - a*p)**2/(a*p) for m, a in zip(mis, acc))
print(f"\nchi-square vs a CONSTANT miss rate: {chi:.1f} on {N_LAYER-1} dof")
print(f"  (expected ~{N_LAYER-1} if the spread were pure counting noise)")
for r, o in zip(rows, owned):
    r["owned"] = o
    r["sd"] = math.sqrt(max(1, r["misses"]))/max(1, r["acc"])
json.dump(rows, open(sys.argv[1], "w"))
