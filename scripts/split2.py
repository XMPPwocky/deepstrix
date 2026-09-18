#!/usr/bin/env python3
"""Box 1's decode LRU as an INCLUSIVE front cache over box 2's per-layer regions.

A pick hits box 1 (free, local RAM) -> done. Otherwise it goes to box 2, which
hits from its own region or MISSES to its disk. Both caches fill independently,
so sizing box 1 can never make box 2 worse -- which the previous model got
wrong by letting box 1 steal box 2's fills.

Validation: at box1=25 slots this must reproduce the shipped 20.4 / measured 20.2.
"""
import sys, array, collections
N_LAYER, TOPK, N_EXPERT = 40, 6, 384
buf = array.array('H'); buf.frombytes(open(sys.argv[1],'rb').read())
per_tok = N_LAYER*TOPK; ntok = len(buf)//per_tok; t0 = ntok//2
picks=[]
for t in range(ntok):
    rows=[]
    for l in range(N_LAYER):
        seen=[]
        for k in range(TOPK):
            e=buf[t*per_tok+l*TOPK+k]
            if e<N_EXPERT and e not in seen: seen.append(e)
        rows.append(seen)
    picks.append(rows)
B2 = [260]*20 + [68]*20

def sim(b1_total, cover):
    lru2=[collections.OrderedDict() for _ in range(N_LAYER)]
    g1=collections.OrderedDict()
    miss=0; b1hit=0; served=0
    for ti,tk in enumerate(picks):
        for l,row in enumerate(tk):
            for e in row:
                if ti>=t0: served+=1
                k=(l,e)
                if l in cover and k in g1:
                    g1.move_to_end(k)
                    if ti>=t0: b1hit+=1
                    continue
                a=lru2[l]
                if e in a: a.move_to_end(e)
                else:
                    if ti>=t0: miss+=1
                    a[e]=1
                    if len(a)>B2[l]: a.popitem(last=False)
                if l in cover and b1_total>0:      # inclusive fill, independent of box 2
                    g1[k]=1
                    if len(g1)>b1_total: g1.popitem(last=False)
    w=ntok-t0
    return miss/w, b1hit/w, served/w

dec=set(range(20,40)); allL=set(range(40))
for name, cover in [("box 1 covers DECODER layers 20-39", dec), ("box 1 covers ALL 40 layers", allL)]:
    print(f"\n{name}")
    print(f"{'box1 slots':>11} {'GB':>6} {'prefill win':>12} {'miss/tok':>9} {'b1 hits/tok':>12} {'ms @6.6':>8} {'saved':>8}")
    base=None
    for n in [25,128,256,512,768,1024,1536,2048,2688]:
        win=max(0,(2766-n)//128)
        m,h,s = sim(n, cover)
        if base is None: base=m
        print(f"{n:11d} {n*18.8/1000:6.1f} {win:6d} of 21 {m:9.1f} {h:12.1f} {m*6.6:8.0f} {(base-m)*6.6:7.0f}ms")
