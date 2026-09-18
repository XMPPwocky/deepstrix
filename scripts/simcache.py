#!/usr/bin/env python3
"""Replay a V41_PICK_TRACE through cache policies. Reports DECODE misses/token
after a warm-up of the first W requests (prefill rows are replayed as accesses
too, since box 2 sees them, but only decode misses are scored)."""
import sys, json, collections
from collections import OrderedDict
TR=sys.argv[1] if len(sys.argv)>1 else 'picks.trace'
WARM=int(sys.argv[2]) if len(sys.argv)>2 else 2
NL,NE=40,384
# ---- parse ----
reqs=[]; cur=[]
for line in open(TR):
    p=line.split()
    if not p: continue
    if p[0]=='REQ':
        if cur: reqs.append(cur)
        cur=[]; continue
    if p[0]=='D':
        l=int(p[1]); ids={int(x) for x in p[2:] if 0<=int(x)<NE}
        cur.append(('D',l,ids))
    elif p[0]=='P':
        l=int(p[1]); ids={int(x) for x in p[3:] if 0<=int(x)<NE}
        cur.append(('P',l,ids))
if cur: reqs.append(cur)
def tokens(req): return sum(1 for ev in req if ev[0]=='D' and ev[1]==0)
print(f"trace: {len(reqs)} requests, decode tokens per request: {[tokens(r) for r in reqs]}")
# global frequency from the cumulative stats file
try:
    st=json.load(open('/home/claude-code/.cache/deepstrix/expert_stats.json'))['decode']['counts']
    freq={(l,e):st[l*NE+e] for l in range(NL) for e in range(NE)}
except Exception: freq=None

class LRU:
    def __init__(s,n): s.n=n; s.d=OrderedDict()
    def access(s,k):
        if k in s.d: s.d.move_to_end(k); return True
        if len(s.d)>=s.n: s.d.popitem(last=False)
        s.d[k]=1; return False
class LRUPinned(LRU):
    """top-K by global frequency are pinned (never evicted); LRU over the rest."""
    def __init__(s,n,pinned): s.n=n-len(pinned); s.d=OrderedDict(); s.pin=set(pinned)
    def access(s,k):
        if k in s.pin: return True
        return LRU.access(s,k)
class LFU:
    def __init__(s,n): s.n=n; s.c=collections.Counter(); s.res=set()
    def access(s,k):
        s.c[k]+=1
        if k in s.res: return True
        if len(s.res)>=s.n:
            v=min(s.res,key=lambda x:s.c[x]); s.res.discard(v)
        s.res.add(k); return False
class TwoTier:
    """L1 LRU demand-fill (n1), L2 LRU victim-fill (n2), exclusive. Miss = in neither."""
    def __init__(s,n1,n2): s.l1=OrderedDict(); s.l2=OrderedDict(); s.n1=n1; s.n2=n2
    def access(s,k):
        if k in s.l1: s.l1.move_to_end(k); return True
        hit = k in s.l2
        if hit: del s.l2[k]
        if len(s.l1)>=s.n1:
            v,_=s.l1.popitem(last=False)
            if len(s.l2)>=s.n2: s.l2.popitem(last=False)
            s.l2[v]=1
        s.l1[k]=1
        return hit
def run(mk):
    c=mk(); dmiss=0; dtok=0
    for i,req in enumerate(reqs):
        score = i>=WARM
        for kind,l,ids in req:
            for e in ids:
                h=c.access((l,e))
                if score and kind=='D' and not h: dmiss+=1
            if score and kind=='D' and l==0: dtok+=1
    return dmiss, dtok
pinned_top = None
if freq:
    def top_per_layer(k):
        out=[]
        for l in range(NL):
            row=sorted(range(NE),key=lambda e:-freq[(l,e)])[:k]
            out+= [(l,e) for e in row]
        return out
cfgs=[]
for n in (6160, 7670, 10600, 12000, 13000):
    cfgs.append((f"LRU global      {n:6d}", lambda n=n: LRU(n)))
for n in (6160, 10600):
    cfgs.append((f"LFU running     {n:6d}", lambda n=n: LFU(n)))
if freq:
    for n,k in ((6160,100),(6160,130),(10600,200),(10600,230),(10600,247)):
        cfgs.append((f"pin top-{k}/layer+LRU {n:6d}", lambda n=n,k=k: LRUPinned(n, top_per_layer(k))))
    for k in (154,200,247,265):
        cfgs.append((f"pin top-{k}/layer ONLY (static)", lambda k=k: LRUPinned(k*NL+1, top_per_layer(k))))
cfgs.append(("2-tier excl 1510+6160", lambda: TwoTier(1510,6160)))
cfgs.append(("2-tier excl 4454+6160", lambda: TwoTier(4454,6160)))
print(f"scored requests: {len(reqs)-WARM}")
for name,mk in cfgs:
    m,t=run(mk)
    print(f"{name:34s} decode misses/token = {m/max(t,1):6.2f}   (misses {m} over {t} tokens)")

# ---- hash partition: each (layer,e) has one home box; per-box LRU over its own stream ----
class Partition:
    def __init__(s,n1,n2):
        s.c=[LRU(n1),LRU(n2)]; s.n=(n1,n2); s.miss=[0,0]
    def home(s,k):
        h=(k[0]*1000003+k[1]*7919)%1000
        return 0 if h < 1000*s.n[0]//(s.n[0]+s.n[1]) else 1
    def access(s,k):
        b=s.home(k); h=s.c[b].access(k)
        if not h: s.miss[b]+=1
        return h
print("--- hash partition (box1 share proportional to capacity) ---")
for n1 in (1510, 3942, 4454):
    c=Partition(n1,6160); dtok=0; m0=m1=0
    for i,req in enumerate(reqs):
        if i==WARM: c.miss=[0,0]
        for kind,l,ids in req:
            for e in ids:
                h=c.access((l,e))
            if i>=WARM and kind=='D' and l==0: dtok+=1
    # misses counted on all accesses incl P rows after warm; recount decode-only:
    c=Partition(n1,6160); dm=[0,0]; dtok=0
    for i,req in enumerate(reqs):
        for kind,l,ids in req:
            for e in ids:
                h=c.access((l,e))
                if i>=WARM and kind=='D' and not h: dm[c.home((l,e))]+=1
            if i>=WARM and kind=='D' and l==0: dtok+=1
    print(f"partition box1={n1:5d} box2=6160: decode misses/token box1={dm[0]/dtok:5.2f} box2={dm[1]/dtok:5.2f} total={(dm[0]+dm[1])/dtok:5.2f}")
