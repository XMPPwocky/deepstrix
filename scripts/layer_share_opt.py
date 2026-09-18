#!/usr/bin/env python3
"""Per-LAYER partition shares: does rebalancing who-owns-what by layer pay?

Both boxes are simulated, each with its own LRU over its own partition, because
moving experts off box 1 is not free -- box 2 then holds more and misses more.
Shares are fitted on the FIRST half and scored on the SECOND, since a static
placement that is only scored in-sample has burned us before (placements
retained ~30% of in-sample gain).

Kept deliberately low-dimensional: 40 per-layer thresholds, not a 15,360-entry
per-expert LUT. The per-layer miss pattern is what replicates (r=0.80); the
per-expert one is where overfitting lives.
"""
import sys, collections
N_LAYER, N_EXPERT = 40, 384
B1_SLOTS, B2_SLOTS = 4070, 6160
BASE_M = 397

def h(l, e): return (l * 1000003 + e * 7919) % 1000

def load(path):
    toks, cur = [], []
    for line in open(path):
        p = line.split()
        if not p or p[0] != 'D': continue
        l = int(p[1]); ids = tuple(int(x) for x in p[2:] if int(x) >= 0)
        if l == 0 and cur:
            if len(cur) == N_LAYER: toks.append(cur)
            cur = []
        cur.append(ids)
    if len(cur) == N_LAYER: toks.append(cur)
    return toks

def sim(toks, m, warm=1000):
    """-> (box1 misses, box2 misses, per-layer box1 miss rate)"""
    c1 = collections.OrderedDict(); c2 = collections.OrderedDict()
    m1 = [0]*N_LAYER; a1 = [0]*N_LAYER; miss2 = 0
    for t, tk in enumerate(toks):
        counted = t >= warm
        for L, ids in enumerate(tk):
            for e in ids:
                if h(L, e) >= m[L]:                      # box 2's
                    k = (L, e)
                    if k in c2: c2.move_to_end(k)
                    else:
                        if counted: miss2 += 1
                        if len(c2) >= B2_SLOTS: c2.popitem(last=False)
                        c2[k] = 1
                else:                                     # box 1's
                    k = (L, e)
                    if counted: a1[L] += 1
                    if k in c1: c1.move_to_end(k)
                    else:
                        if counted: m1[L] += 1
                        if len(c1) >= B1_SLOTS: c1.popitem(last=False)
                        c1[k] = 1
    return sum(m1), miss2, [x/max(1, y) for x, y in zip(m1, a1)]

toks = load(sys.argv[1]); half = len(toks)//2
tr, te = toks[:half], toks[half:]

base = [BASE_M]*N_LAYER
b1_tr, b2_tr, rate = sim(tr, base)
b1_te, b2_te, _ = sim(te, base)
print(f"tokens {len(toks):,}  fit on {len(tr):,}  score on {len(te):,}")
print(f"{'':<22} {'box1 miss':>10} {'box2 miss':>10} {'total':>10}")
print(f"{'BASELINE (uniform)':<22} {b1_te:>10,} {b2_te:>10,} {b1_te+b2_te:>10,}")

# Fit: shift share AWAY from layers whose box-1 miss rate is high, keeping the
# mean share fixed so overall box1/box2 load is unchanged.
m = list(base)
mean_rate = sum(rate)/N_LAYER
for it in range(4):
    for L in range(N_LAYER):
        # high miss rate -> lower threshold -> box 1 owns fewer of this layer
        adj = -220.0 * (rate[L] - mean_rate) / max(1e-9, mean_rate)
        m[L] = int(max(150, min(700, m[L] + adj)))
    off = BASE_M - sum(m)/N_LAYER            # hold the mean at the baseline
    m = [int(max(150, min(700, x + off))) for x in m]
    _, _, rate = sim(tr, m)
    mean_rate = sum(rate)/N_LAYER

o1_tr, o2_tr, _ = sim(tr, m)
o1_te, o2_te, _ = sim(te, m)
print(f"{'per-layer, IN-sample':<22} {o1_tr:>10,} {o2_tr:>10,} {o1_tr+o2_tr:>10,}"
      f"   box1 {100*(b1_tr-o1_tr)/b1_tr:+.1f}%")
print(f"{'per-layer, HELD OUT':<22} {o1_te:>10,} {o2_te:>10,} {o1_te+o2_te:>10,}"
      f"   box1 {100*(b1_te-o1_te)/b1_te:+.1f}%  total {100*((b1_te+b2_te)-(o1_te+o2_te))/(b1_te+b2_te):+.1f}%")
keep = (b1_tr-o1_tr)/max(1,b1_tr)
print(f"\nretained out-of-sample: {100*((b1_te-o1_te)/max(1,b1_te))/max(1e-9,keep):.0f}% of the in-sample box-1 gain")
print(f"share range fitted: {min(m)}..{max(m)} (baseline {BASE_M}), mean {sum(m)/N_LAYER:.0f}")
