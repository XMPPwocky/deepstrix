import pickle, bisect, collections
tracks, lv = pickle.load(open('leaves.pkl','rb'))
calls=pickle.load(open('calls.pkl','rb'))
for u in lv: lv[u].sort()
starts={u:[x[0] for x in lv[u]] for u in lv}
DG=0x4450475500000001; IG=0x4950475500000001
def win(u,b,e):
    i=bisect.bisect_left(starts[u],b); j=bisect.bisect_left(starts[u],e)
    return [(max(x,b),min(y,e),n) for (x,y,n) in lv[u][max(0,i-1):j] if min(y,e)>max(x,b)]
V=[c for c in calls[27:] if c['B']==6]
P=[c for c in calls if c['B']==33]
def means(u,cs):
    tot=collections.Counter(); cnt=collections.Counter()
    for c in cs:
        for (x,y,n) in win(u,c['wstart'],c['wend']):
            if '.wait' in n: continue
            tot[n]+=y-x; cnt[n]+=1
    return {s: (tot[s]/cnt[s]/1000.0, cnt[s]) for s in tot}
print("KERNEL COST vs BATCH (mean us per launch).  lane B=3 (verify) vs lane B=16.5 (prefill chunk)")
print(f"{'stage':<38} {'B=3 us':>9} {'B~16.5 us':>10} {'ratio':>7} {'ideal 5.5x?':>12}")
for u in (DG,IG):
    mv=means(u,V); mp=means(u,P)
    rows=[]
    for s,(a,ca) in mv.items():
        if s in mp and ca>=30:
            b,cb=mp[s]
            rows.append((a*ca/32, s,a,b,b/a))
    rows.sort(reverse=True)
    print(f"  [{tracks[u]}]")
    for _,s,a,b,r in rows[:20]:
        verdict = "BATCH-1 SHAPED" if r<1.6 else ("sublinear" if r<4 else "scales")
        print(f"   {s:<38} {a:>9.1f} {b:>10.1f} {r:>7.2f}x  {verdict}")
