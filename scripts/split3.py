#!/usr/bin/env python3
"""EXCLUSIVE partition: box 1 pins the experts box 2 does NOT hold.

The inclusive front cache failed because it duplicated box 2's hot set -- 96.8
hits/token bought only 1.6 fewer misses. The fix is disjointness: box 2 keeps
the top-68 per decoder layer, box 1 STATICALLY pins the next K by frequency
(the same freq-ranked placement already shipped for encoder layers). A pick in
box 1's band never reaches box 2 at all, and box 1 never pages -- it holds a
fixed set, so there is no LRU churn and no disk read on either box for it.
"""
import sys, array, collections
N_LAYER, TOPK, N_EXPERT = 40, 6, 384
buf = array.array('H'); buf.frombytes(open(sys.argv[1],'rb').read())
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

# frequency rank per layer, over the WARM half (what a placement file would be built from)
freq=[collections.Counter() for _ in range(N_LAYER)]
for tk in picks[t0:]:
    for l,row in enumerate(tk):
        for e in row: freq[l][e]+=1
rank=[{e:i for i,(e,_) in enumerate(freq[l].most_common())} for l in range(N_LAYER)]

B2=[260]*20+[68]*20
def sim(k_per_decoder):
    """box 1 pins ranks [68, 68+k) on each DECODER layer; box 2 LRUs the rest."""
    lru2=[collections.OrderedDict() for _ in range(N_LAYER)]
    b1={l:set(e for e,r in rank[l].items() if B2[l] <= r < B2[l]+k_per_decoder)
        for l in range(20,40)}
    miss=0; b1hit=0
    for ti,tk in enumerate(picks):
        for l,row in enumerate(tk):
            for e in row:
                if l>=20 and e in b1[l]:
                    if ti>=t0: b1hit+=1
                    continue
                a=lru2[l]
                if e in a: a.move_to_end(e)
                else:
                    if ti>=t0: miss+=1
                    a[e]=1
                    if len(a)>B2[l]: a.popitem(last=False)
    w=ntok-t0
    return miss/w, b1hit/w

print(f"{'k/decoder layer':>16} {'box1 slots':>11} {'GB':>6} {'prefill win':>12} {'miss/tok':>9} {'b1 serves':>10} {'ms @6.6':>8} {'saved':>8}")
base=None
for k in [0, 16, 32, 48, 64, 96, 128, 160, 192]:
    n=k*20
    win=max(0,(2766-n)//128)
    m,h=sim(k)
    if base is None: base=m
    print(f"{k:16d} {n:11d} {n*18.8/1000:6.1f} {win:6d} of 21 {m:9.1f} {h:10.1f} {m*6.6:8.0f} {(base-m)*6.6:7.0f}ms")
