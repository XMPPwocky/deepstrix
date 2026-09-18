#!/usr/bin/env python3
"""How often does ONE (layer, token) take multiple misses?

Two boxes = two SSDs, so misses within a layer-token are only serialised if one
box owns them all. Cost is max(box1, box2) per leg, not the sum -- so what
matters is the DISTRIBUTION of misses per layer-token, not the total.
A layer-token with m misses costs m*c if one box serves them all, and
ceil(m/2)*c if they are split evenly. Gain exists only where m >= 2.
"""
import sys, collections
N_LAYER, N_EXPERT, BOX1_MILLI = 40, 384, 397
B1_SLOTS, B2_SLOTS = 4070, 6160
def box2(l, e): return ((l*1000003 + e*7919) % 1000) >= BOX1_MILLI

toks, cur = [], []
for line in open(sys.argv[1]):
    p = line.split()
    if not p or p[0] != 'D': continue
    l = int(p[1]); ids = tuple(int(x) for x in p[2:] if int(x) >= 0)
    if l == 0 and cur:
        if len(cur) == N_LAYER: toks.append(cur)
        cur = []
    cur.append(ids)
if len(cur) == N_LAYER: toks.append(cur)

c1 = collections.OrderedDict(); c2 = collections.OrderedDict()
h1 = collections.Counter()   # box-1 misses per layer-token
h2 = collections.Counter()
hb = collections.Counter()   # combined misses per layer-token
WARM = 1000
for t, tk in enumerate(toks):
    counted = t >= WARM
    for L, ids in enumerate(tk):
        m1 = m2 = 0
        for e in ids:
            k = (L, e)
            if box2(L, e):
                if k in c2: c2.move_to_end(k)
                else:
                    m2 += 1
                    if len(c2) >= B2_SLOTS: c2.popitem(last=False)
                    c2[k] = 1
            else:
                if k in c1: c1.move_to_end(k)
                else:
                    m1 += 1
                    if len(c1) >= B1_SLOTS: c1.popitem(last=False)
                    c1[k] = 1
        if counted:
            h1[m1] += 1; h2[m2] += 1; hb[m1+m2] += 1

n = sum(hb.values())
print(f"layer-tokens scored {n:,}  ({n/N_LAYER:,.0f} tokens x {N_LAYER} layers)\n")
print(f"{'misses in one layer-token':>26} {'box 1':>12} {'box 2':>12} {'combined':>12}")
for m in range(0, 7):
    if hb[m] == 0 and m > 4: break
    print(f"{m:>26} {100*h1[m]/n:>11.3f}% {100*h2[m]/n:>11.3f}% {100*hb[m]/n:>11.3f}%")

tot_b1 = sum(m*c for m, c in h1.items())
tot_all = sum(m*c for m, c in hb.items())
# Serial on whichever box owns them vs split evenly across the two SSDs.
serial = sum(max(m1, 0)*c for m1, c in h1.items())          # box 1's own, today
import math
ideal = sum(math.ceil(m/2)*c for m, c in hb.items())
print(f"\ncombined misses total        {tot_all:,}")
print(f"  served serially on one box  {tot_all:,} reads on the critical path")
print(f"  split evenly across 2 SSDs  {ideal:,}  -> {100*(1-ideal/tot_all):.1f}% shorter")
multi = sum(c for m, c in hb.items() if m >= 2)
print(f"\nlayer-tokens with >=2 combined misses: {multi:,} ({100*multi/n:.3f}% of all)")
print(f"  share of ALL misses living in them : "
      f"{100*sum(m*c for m,c in hb.items() if m>=2)/tot_all:.1f}%")
