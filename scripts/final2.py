import pickle, bisect, collections
tracks, lv = pickle.load(open('leaves.pkl','rb'))
calls=pickle.load(open('calls.pkl','rb'))
for u in lv: lv[u].sort()
starts={u:[x[0] for x in lv[u]] for u in lv}
DG=0x4450475500000001; IG=0x4950475500000001; DX=0x4450475500000002; IX=0x4950475500000002
def win(u,b,e):
    i=bisect.bisect_left(starts[u],b); j=bisect.bisect_left(starts[u],e)
    return [(max(x,b),min(y,e),n) for (x,y,n) in lv[u][max(0,i-1):j] if min(y,e)>max(x,b)]
V=[c for c in calls[27:] if c['B']==6]; D=[c for c in calls if c['kind']=='decode']
def table(u,cs,label,top=25):
    tot=collections.Counter(); cnt=collections.Counter()
    for c in cs:
        for (x,y,n) in win(u,c['wstart'],c['wend']):
            if '.wait' in n: continue
            tot[n]+=y-x; cnt[n]+=1
    n=len(cs); allt=sum(tot.values())
    print(f"--- {tracks[u]} | {label} | n={n} calls | total device busy {allt/n/1e6:.2f} ms/call")
    print(f"    {'ms/call':>9} {'calls/call':>10} {'mean us':>9} {'%dev':>6}  stage")
    for s,t in tot.most_common(top):
        print(f"    {t/n/1e6:>9.3f} {cnt[s]/n:>10.1f} {t/cnt[s]/1000:>9.1f} {100*t/allt:>6.1f}  {s}")
    print()
table(DG,V,"VERIFY B=6 (req2)")
table(IG,V,"VERIFY B=6 (req2)",12)
table(DX,V,"VERIFY B=6 (req2)",8)
table(IX,V,"VERIFY B=6 (req2)",8)
table(DG,D,"DECODE B=1 (4 calls, graph-captured)",25)
table(IG,D,"DECODE B=1",12)
