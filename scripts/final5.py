import pickle, bisect, collections, re
tracks, lv = pickle.load(open('leaves.pkl','rb'))
calls=pickle.load(open('calls.pkl','rb'))
for u in lv: lv[u].sort()
starts={u:[x[0] for x in lv[u]] for u in lv}
DG=0x4450475500000001; IG=0x4950475500000001; RE=0x52454d5400000001
DX=0x4450475500000002; IX=0x4950475500000002
def win(u,b,e):
    i=bisect.bisect_left(starts[u],b); j=bisect.bisect_left(starts[u],e)
    return [(max(x,b),min(y,e),n) for (x,y,n) in lv[u][max(0,i-1):j] if min(y,e)>max(x,b)]
def merge(iv):
    iv=sorted(iv); out=[]
    for b,e in iv:
        if out and b<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([b,e])
    return out
def sub(a,b):  # a minus b, both merged lists
    out=[];  j=0
    for s,e in a:
        cur=s
        for bs,be in b:
            if be<=cur: continue
            if bs>=e: break
            if bs>cur: out.append((cur,min(bs,e)))
            cur=max(cur,be)
            if cur>=e: break
        if cur<e: out.append((cur,e))
    return out
def tot(iv): return sum(e-b for b,e in iv)
V=[c for c in calls[27:] if c['B']==6]; N=len(V)
A=B=C=W=0
for c in V:
    ws,we=c['wstart'],c['wend']; W+=we-ws
    gpu=merge([(x,y) for u in (DG,IG,DX,IX) for (x,y,n) in win(u,ws,we) if '.wait' not in n])
    idle=sub([[ws,we]],gpu)
    rem=merge([(x,y) for (x,y,n) in win(RE,ws,we) if 'wait L' in n])
    blocked=[ (max(a,b),min(c2,d)) for a,c2 in idle for b,d in rem if min(c2,d)>max(a,b)]
    blocked=merge(blocked)
    A+=tot(gpu); B+=tot(blocked); C+=tot(idle)-tot(blocked)
print(f"VERIFY B=6 steady state, n={N}, mean wall {W/N/1e6:.1f} ms")
print(f"  (a) any-device BUSY                      {A/N/1e6:7.2f} ms  {100*A/W:5.1f}%")
print(f"  (b) device idle, host BLOCKED on box 2   {B/N/1e6:7.2f} ms  {100*B/W:5.1f}%")
print(f"  (c) device idle, host CPU (not blocked)  {C/N/1e6:7.2f} ms  {100*C/W:5.1f}%")
print()
# Where does (c) sit? attribute host-CPU idle to the dgpu gap it falls in
agg=collections.Counter(); cnt=collections.Counter()
for c in V:
    ws,we=c['wstart'],c['wend']
    gpu=merge([(x,y) for u in (DG,IG,DX,IX) for (x,y,n) in win(u,ws,we) if '.wait' not in n])
    idle=sub([[ws,we]],gpu)
    rem=merge([(x,y) for (x,y,n) in win(RE,ws,we) if 'wait L' in n])
    host=sub(idle,rem)
    seg=win(DG,ws,we); pe=None; pn='<start>'; spans=[]
    for (x,y,n) in seg:
        if pe is not None and x>pe: spans.append((pe,x,pn,n))
        pe=y if pe is None else max(pe,y); pn=n
    for (hb,he) in host:
        for (sb,se,p,q) in spans:
            o=min(he,se)-max(hb,sb)
            if o>0: agg[(p,q)]+=o; cnt[(p,q)]+=1
print("host-CPU idle (c) attributed to the dgpu.compute gap it falls in:")
for (p,q),t in agg.most_common(12):
    print(f"   {t/N/1e6:7.2f} ms/call  n={cnt[(p,q)]/N:5.1f}  {p} -> {q}")
