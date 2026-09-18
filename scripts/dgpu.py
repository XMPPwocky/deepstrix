#!/usr/bin/env python3
"""dGPU hot tier as an EXCLUSIVE top-rank band, on the real routing trace.

17.1 GB of dGPU VRAM is idle during decode. At 18.8 MB/expert that is ~900
experts = 22 per layer. The dGPU statically holds the top-k by frequency; box 2
then LRUs only what the dGPU did NOT serve, so its 68 decoder slots stop being
spent on the hot set and start covering the tail. Exclusive by construction.

Free in wall-clock too: the decode structure is max(box1 iGPU MoE, box2), box 2
is the long pole at 35 ms vs 18.2, so dGPU MoE work up to ~35 ms hides entirely
-- and every pick it takes SHORTENS that long pole.
"""
import sys, array, collections
N_LAYER, TOPK, N_EXPERT = 40, 6, 384
buf=array.array('H'); buf.frombytes(open(sys.argv[1],'rb').read())
per_tok=N_LAYER*TOPK; ntok=len(buf)//per_tok; t0=ntok//2
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
freq=[collections.Counter() for _ in range(N_LAYER)]
for tk in picks[t0:]:
    for l,row in enumerate(tk):
        for e in row: freq[l][e]+=1
hot=[[e for e,_ in freq[l].most_common()] for l in range(N_LAYER)]
B2=[260]*20+[68]*20

def sim(k):
    dg=[set(hot[l][:k]) for l in range(N_LAYER)]
    lru2=[collections.OrderedDict() for _ in range(N_LAYER)]
    miss=0; served=0; dghit=0
    for ti,tk in enumerate(picks):
        for l,row in enumerate(tk):
            for e in row:
                if ti>=t0: served+=1
                if e in dg[l]:
                    if ti>=t0: dghit+=1
                    continue
                a=lru2[l]
                if e in a: a.move_to_end(e)
                else:
                    if ti>=t0: miss+=1
                    a[e]=1
                    if len(a)>B2[l]: a.popitem(last=False)
    w=ntok-t0
    return miss/w, dghit/w, served/w

print(f"{'k/layer':>8} {'experts':>8} {'VRAM GB':>8} {'dGPU serves':>12} {'% of picks':>11} {'box2 miss/tok':>14} {'ms @6.6':>8} {'saved':>8}")
base=None
for k in [0,4,6,8,12,16,22,32]:
    n=k*N_LAYER
    m,h,s=sim(k)
    if base is None: base=m
    print(f"{k:8d} {n:8d} {n*18.8/1000:8.1f} {h:12.1f} {100*h/s:10.1f}% {m:14.1f} {m*6.6:8.0f} {(base-m)*6.6:7.0f}ms")
