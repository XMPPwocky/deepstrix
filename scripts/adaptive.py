#!/usr/bin/env python3
"""Static band vs adaptive VICTIM cache, both scored HELD-OUT.

Static placement retains only ~30% of its in-sample coverage, so the box-1
"exclusive frequency band" needs re-pricing on held-out ranks. The adaptive
alternative is a victim cache: box 1 catches what box 2 EVICTS. Exclusive by
construction, and it tracks drift instead of assuming stationarity.
"""
import sys, array, collections
N_LAYER, TOPK, N_EXPERT = 40, 6, 384
buf=array.array('H'); buf.frombytes(open(sys.argv[1],'rb').read())
per_tok=N_LAYER*TOPK; ntok=len(buf)//per_tok; half=ntok//2
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
# ranks fit on the FIRST half only
fit=[collections.Counter() for _ in range(N_LAYER)]
for tk in picks[:half]:
    for l,row in enumerate(tk):
        for e in row: fit[l][e]+=1
rankfit=[[e for e,_ in fit[l].most_common()] for l in range(N_LAYER)]
B2=[260]*20+[68]*20

def run(mode, cap):
    """cap = total box-1 slots. mode: 'static' band after box2's depth, or 'victim'."""
    lru2=[collections.OrderedDict() for _ in range(N_LAYER)]
    b1=collections.OrderedDict()
    band={l:set(rankfit[l][B2[l]:B2[l]+cap//20]) for l in range(20,40)} if mode=="static" else None
    miss=0; b1hit=0
    for ti,tk in enumerate(picks):
        warm = ti>=half
        for l,row in enumerate(tk):
            for e in row:
                k=(l,e)
                if mode=="static":
                    if l>=20 and e in band[l]:
                        if warm: b1hit+=1
                        continue
                else:
                    if k in b1:
                        b1.move_to_end(k)
                        if warm: b1hit+=1
                        continue
                a=lru2[l]
                if e in a:
                    a.move_to_end(e); continue
                if warm: miss+=1
                a[e]=1
                if len(a)>B2[l]:
                    vic,_=a.popitem(last=False)          # box 2 evicts
                    if mode=="victim" and cap>0:          # box 1 catches it
                        b1[(l,vic)]=1
                        if len(b1)>cap: b1.popitem(last=False)
    w=ntok-half
    return miss/w, b1hit/w

print(f"{'box1 slots':>11} {'GB':>6} | {'STATIC band':>12} {'ms':>6} | {'VICTIM cache':>13} {'ms':>6} | {'b1 serves':>10}")
base=None
for cap in [0,320,640,960,1280,1920,2688]:
    ms,_hs = run("static", cap)
    mv,hv  = run("victim", cap)
    if base is None: base=ms
    print(f"{cap:11d} {cap*18.8/1000:6.1f} | {ms:9.1f} {ms*6.6:8.0f} | {mv:10.1f} {mv*6.6:8.0f} | {hv:10.1f}")
print(f"\nbaseline (box 2 alone, held-out) = {base:.1f} miss/tok = {base*6.6:.0f} ms")
