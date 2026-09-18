#!/usr/bin/env python3
"""How much is box 1's decode LRU worth, as a function of how many slots it gets?

Box 1's 52 GB pool = 2766 slots. Today 2688 are PINNED as 21 packed prefill
windows (stride 128, covering all 20 CED encoder layers) and decode's LRU gets
the ~25-78 remainder. Prefill and decode never run at once, so the pin is a
policy choice, not a constraint -- but every slot decode takes is a window slot
prefill must re-page next request.
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

B2 = [260]*20 + [68]*20            # box 2, fixed by prefill's stride-128 rule

def sim(b1_total, b1_layers):
    """box 1 = one GLOBAL LRU over b1_layers (it has one pool, not per-layer regions)."""
    lru2=[collections.OrderedDict() for _ in range(N_LAYER)]
    g1=collections.OrderedDict()
    cover=set(b1_layers); miss=0
    for ti,tk in enumerate(picks):
        for l,row in enumerate(tk):
            for e in row:
                if l in cover:
                    k=(l,e)
                    if k in g1: g1.move_to_end(k); continue
                a=lru2[l]
                if e in a: a.move_to_end(e); continue
                if ti>=t0: miss+=1
                if l in cover and b1_total>0:
                    g1[(l,e)]=1
                    if len(g1)>b1_total: g1.popitem(last=False)
                else:
                    a[e]=1
                    if len(a)>B2[l]: a.popitem(last=False)
    return miss/(ntok-t0)

dec=list(range(20,40))
print(f"{'box1 decode slots':>18} {'GB':>6} {'prefill windows left':>21} {'miss/tok':>9} {'ms @6.6':>8} {'vs today':>9}")
base=None
for n in [25, 128, 256, 512, 768, 1024, 1536, 2048, 2688]:
    win = max(0, (2766-n)//128)
    m = sim(n, dec)
    if base is None: base=m
    print(f"{n:18d} {n*18.8/1000:6.1f} {win:11d} of 21{'':7} {m:9.1f} {m*6.6:8.0f} {(m-base)*6.6:+8.0f}ms")
