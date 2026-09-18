import pickle, bisect, sys
tracks, lv = pickle.load(open('leaves.pkl','rb'))
steps = pickle.load(open('steps.pkl','rb'))
layers,cls = pickle.load(open('layers.pkl','rb'))
for u in lv: lv[u].sort()
short={0x4450475500000001:'DGC',0x4450475500000002:'DGX',0x4950475500000001:'IGC',0x4950475500000002:'IGX',0x52454d5400000001:'REM',0x5041474500000001:'PGR'}
# step 3 window
V=[(i,b,e) for i,(c,b,e,n) in enumerate(steps) if c and n==40]
si=int(sys.argv[1]); nl=int(sys.argv[2]) if len(sys.argv)>2 else 3
idx,b,e=[v for v in V if v[0]==si][0]
# find layer boundaries inside
lb=[(x,y) for (x,y) in layers if x>=b and x<e]
print(f"step {si} wall {(e-b)/1e6:.2f} ms, {len(lb)} layers")
t0=lb[0][0]
for li in range(nl):
    x,y=lb[li]
    print(f"--- layer {li}  [{(x-t0)/1e6:8.3f} .. {(y-t0)/1e6:8.3f}] dur {(y-x)/1e3:8.1f} us")
    ev=[]
    for u in lv:
        st=[s[0] for s in lv[u]]
        i=bisect.bisect_left(st,x); j=bisect.bisect_left(st,y)
        for (sb,se,n) in lv[u][i:j]: ev.append((sb,se,short[u],n))
    ev.sort()
    prev=x
    for (sb,se,tr,n) in ev:
        gap=(sb-prev)/1e3
        gs=f"  <gap {gap:8.1f} us>" if gap>50 else ""
        print(f"   {(sb-t0)/1e6:9.3f} +{(se-sb)/1e3:8.1f}us {tr:4} {n}{gs}")
        prev=max(prev,se)
