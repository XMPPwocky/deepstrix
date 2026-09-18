#!/usr/bin/env python3
"""What explains the 2.2x per-layer miss-rate spread, if not depth or partition?

Candidate: temporal locality. A layer whose routing CONCENTRATES (low entropy,
high top-k share) re-picks the same experts sooner, so its reuse distances are
short and the shared LRU holds them. Measured per layer and correlated.
"""
import sys, json, math, collections
N_LAYER, N_EXPERT, BOX1_MILLI = 40, 384, 397
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

freq = [collections.Counter() for _ in range(N_LAYER)]
last = [dict() for _ in range(N_LAYER)]
gaps = [[] for _ in range(N_LAYER)]
for t, tk in enumerate(toks):
    for L, ids in enumerate(tk):
        for e in ids:
            if box2(L, e): continue
            freq[L][e] += 1
            if e in last[L]: gaps[L].append(t - last[L][e])
            last[L][e] = t

rows = json.load(open(sys.argv[2]))
ent, top16, med_gap, p90_gap = [], [], [], []
for L in range(N_LAYER):
    c = freq[L]; tot = sum(c.values())
    ps = [v/tot for v in c.values()]
    ent.append(-sum(p*math.log2(p) for p in ps))
    top16.append(sum(sorted(c.values(), reverse=True)[:16])/tot)
    g = sorted(gaps[L])
    med_gap.append(g[len(g)//2]); p90_gap.append(g[int(len(g)*.9)])

def pearson(a, b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((y-mb)**2 for y in b))
    return num/(da*db) if da and db else 0.0

mr = [r["miss_rate"] for r in rows]
print(f"{'variable':<28} {'corr with miss rate':>20}")
for name, v in (("routing entropy (bits)", ent), ("top-16 expert share", top16),
                ("median reuse gap (tokens)", med_gap), ("p90 reuse gap (tokens)", p90_gap)):
    print(f"  {name:<26} {pearson(mr, v):>+20.3f}")
print(f"\n{'L':>3} {'miss%':>7} {'entropy':>8} {'top16':>7} {'gap p50':>8} {'gap p90':>8}")
for L in (1, 29, 39, 0, 20, 27, 28, 31):
    print(f"{L:>3} {100*mr[L]:>7.2f} {ent[L]:>8.3f} {top16[L]:>7.3f} "
          f"{med_gap[L]:>8} {p90_gap[L]:>8}")
for L, r in enumerate(rows):
    r.update(entropy=ent[L], top16=top16[L], gap_p50=med_gap[L], gap_p90=p90_gap[L])
json.dump(rows, open(sys.argv[2], "w"))
