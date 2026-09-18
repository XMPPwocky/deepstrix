#!/usr/bin/env python3
"""Two questions the first economics pass left open.

1. BREAK-EVEN PRECISION. Prefetching to hide fraction h of demand misses at
   precision p issues h*m/p extra reads, raising drive utilisation and
   inflating every REMAINING demand miss by 1/(1-rho). Tabulate net gain over
   (h, p) to find how wrong a predictor may be before it stops paying.

2. THE CED PREDICTOR. Layer 20 is the encoder/decoder seam, and 19->20 was the
   strongest cross-layer pair in the trace study (7x over marginal on the MISS
   subset). Predicting ALL of layers 20-39 from layer 19's picks gives half the
   network as lead time. Does it fire on the right population?
"""
import sys, collections
from array import array

N_LAYER, N_EXPERT = 40, 384
BOX1_MILLI, SLOTS = 397, 4070
T_TOKEN_MS, C_MISS_MS = 97.0, 8.85
M_BASE = 1.17
C_S = C_MISS_MS * (1 - (1.67 * C_MISS_MS / T_TOKEN_MS))

def box2(l, e):
    return ((l * 1000003 + e * 7919) % 1000) >= BOX1_MILLI

# ---------------- 1. break-even surface ----------------------------------
print("1. NET GAIN vs BASELINE, over recall h and precision p")
print(f"   m={M_BASE} demand misses/token, c_s={C_S:.2f} ms, T={T_TOKEN_MS} ms")
rho0 = M_BASE * C_S / T_TOKEN_MS
base = M_BASE * C_S / (1 - rho0)
print(f"   baseline exposed {base:.2f} ms/token (rho {rho0:.3f})\n")
ps = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
print("      p:  " + "".join(f"{p:>8.0%}" for p in ps))
for h in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
    row = f"  h={h:>4.0%}  "
    for p in ps:
        rho = ((1 - h) * M_BASE + h * M_BASE / p) * C_S / T_TOKEN_MS
        if rho >= 1.0:
            row += f"{'SAT':>8}"
            continue
        cost = (1 - h) * M_BASE * C_S / (1 - rho)
        row += f"{100*(base-cost)/base:>7.0f}%"
    print(row)
print("\n  SAT = prefetch stream saturates the drive.")

# ---------------- 2. the CED predictor -----------------------------------
def tokens(path):
    toks, cur = [], []
    for line in open(path):
        p = line.split()
        if not p or p[0] != 'D':
            continue
        l = int(p[1]); ids = tuple(int(x) for x in p[2:] if int(x) >= 0)
        if l == 0 and cur:
            if len(cur) == N_LAYER: toks.append(cur)
            cur = []
        cur.append(ids)
    if len(cur) == N_LAYER: toks.append(cur)
    return toks

toks = tokens(sys.argv[1]); T = len(toks); half = T // 2
SRC = 19
DST = list(range(20, N_LAYER))
NT = len(DST) * N_EXPERT

# Lead time from a layer-19 prediction: layer L is reached at L/40*T, so
# hideable iff (L-19)/40*T >= C_MISS. Everything below is counted only there.
HIDE = [L for L in DST if (L - SRC) / N_LAYER * T_TOKEN_MS >= C_MISS_MS]
print(f"\n2. CED PREDICTOR  L{SRC} -> L20..39")
print(f"   hideable from L{SRC}: L{HIDE[0]}..{HIDE[-1]} ({len(HIDE)}/{len(DST)} decoder layers)")

cond = [array('i', bytes(4 * NT)) for _ in range(N_EXPERT)]
for tk in toks[:half]:
    tgt = [(L - 20) * N_EXPERT + e for L in DST for e in tk[L]]
    for a in tk[SRC]:
        row = cond[a]
        for j in tgt:
            row[j] += 1
# Ship-able table: top-M targets per source expert.
TOPM = 128
top = []
for a in range(N_EXPERT):
    row = cond[a]
    idx = sorted(range(NT), key=lambda j: row[j], reverse=True)[:TOPM]
    top.append([(j, row[j]) for j in idx if row[j] > 0])
del cond

def run(n_pred):
    cache = collections.OrderedDict()
    demand = iss = used = 0
    counted = 0
    for t, tk in enumerate(toks):
        for L in range(N_LAYER):
            if L == 20 and n_pred:                     # fire at the seam
                sc = collections.Counter()
                for a in tk[SRC]:
                    for j, c in top[a]:
                        sc[j] += c
                cand = []
                for j, _ in sc.most_common():
                    Lj, e = 20 + j // N_EXPERT, j % N_EXPERT
                    if box2(Lj, e) or (Lj, e) in cache:
                        continue
                    cand.append((Lj, e))
                    if len(cand) >= n_pred:
                        break
                for (Lj, e) in cand:
                    iss += 1
                    if len(cache) >= SLOTS: cache.popitem(last=False)
                    cache[(Lj, e)] = 1
                    if t >= 2000 and Lj in HIDE and e in tk[Lj]:
                        used += 1
            for e in tk[L]:
                if box2(L, e): continue
                k = (L, e)
                if k in cache: cache.move_to_end(k)
                else:
                    if t >= 2000: demand += 1
                    if len(cache) >= SLOTS: cache.popitem(last=False)
                    cache[k] = 1
        if t >= 2000: counted += 1
    return demand / counted, iss / counted, used / max(1, iss)

d0, _, _ = run(0)
print(f"\n   {'n_pred':>7} {'demand/tok':>11} {'prefetch/tok':>13} "
      f"{'precision':>10} {'rho':>6} {'exposed':>9} {'vs base':>9}")
b_rho = d0 * C_S / T_TOKEN_MS
b_cost = d0 * C_S / (1 - b_rho)
print(f"   {'0':>7} {d0:>11.2f} {0:>13.2f} {'-':>10} {b_rho:>6.3f} "
      f"{b_cost:>8.2f}ms {'baseline':>9}")
for n in (8, 32, 128):
    d, i, pr = run(n)
    rho = (d + i) * C_S / T_TOKEN_MS
    if rho >= 1.0:
        print(f"   {n:>7} {d:>11.2f} {i:>13.2f} {pr:>9.1%} {rho:>6.3f}   SATURATES")
        continue
    cost = d * C_S / (1 - rho)
    print(f"   {n:>7} {d:>11.2f} {i:>13.2f} {pr:>9.1%} {rho:>6.3f} "
          f"{cost:>8.2f}ms {100*(b_cost-cost)/b_cost:>8.1f}%")
