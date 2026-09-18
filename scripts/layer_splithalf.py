#!/usr/bin/env python3
"""Is the per-layer miss pattern a property of the MODEL, or of this session?

Split-half reliability: simulate the cache independently on the first and second
half of the trace and correlate the 40 per-layer miss rates. A stable model
property replicates (r high). Session-specific content does not.
"""
import sys, json, math, collections
N_LAYER, N_EXPERT, BOX1_MILLI, SLOTS = 40, 384, 397, 4070
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

def sim(ts, warm=1000):
    cache = collections.OrderedDict(); acc=[0]*N_LAYER; miss=[0]*N_LAYER
    for t, tk in enumerate(ts):
        for L, ids in enumerate(tk):
            for e in ids:
                if box2(L, e): continue
                k=(L,e); c = t>=warm
                if c: acc[L]+=1
                if k in cache: cache.move_to_end(k)
                else:
                    if c: miss[L]+=1
                    if len(cache)>=SLOTS: cache.popitem(last=False)
                    cache[k]=1
    return [m/max(1,a) for m,a in zip(miss,acc)]

def pearson(a,b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((y-mb)**2 for y in b))
    return num/(da*db) if da and db else 0.0

h = len(toks)//2
a, b = sim(toks[:h]), sim(toks[h:])
r = pearson(a, b)
print(f"tokens {len(toks):,}  split at {h:,}")
print(f"first half  miss {100*min(a):.2f}%..{100*max(a):.2f}%  spread {max(a)/min(a):.2f}x")
print(f"second half miss {100*min(b):.2f}%..{100*max(b):.2f}%  spread {max(b)/min(b):.2f}x")
print(f"\nsplit-half correlation of the 40 per-layer miss rates: r = {r:+.3f}")
print(f"  Spearman-Brown full-length reliability: {2*r/(1+r):+.3f}" if r>0 else "")
print("\n  r near 1.0 -> a stable model property, worth acting on per layer")
print("  r near 0.0 -> this session's content; the per-layer shape will not replicate")
top_a = sorted(range(N_LAYER), key=lambda L: -a[L])[:5]
top_b = sorted(range(N_LAYER), key=lambda L: -b[L])[:5]
print(f"\nworst 5 layers, first half : {top_a}")
print(f"worst 5 layers, second half: {top_b}")
print(f"overlap: {len(set(top_a)&set(top_b))}/5")
rows=json.load(open(sys.argv[2]))
for L,rw in enumerate(rows): rw.update(miss_h1=a[L], miss_h2=b[L])
json.dump(rows, open(sys.argv[2],"w"))
