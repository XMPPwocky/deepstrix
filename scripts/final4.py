import pickle, bisect, collections, re
tracks, lv = pickle.load(open('leaves.pkl','rb'))
calls=pickle.load(open('calls.pkl','rb'))
for u in lv: lv[u].sort()
starts={u:[x[0] for x in lv[u]] for u in lv}
DG=0x4450475500000001; IG=0x4950475500000001; RE=0x52454d5400000001
def win(u,b,e):
    i=bisect.bisect_left(starts[u],b); j=bisect.bisect_left(starts[u],e)
    return [(max(x,b),min(y,e),n) for (x,y,n) in lv[u][max(0,i-1):j] if min(y,e)>max(x,b)]
V=[c for c in calls[27:] if c['B']==6]
N=len(V)
def gaps(u,label,top=18):
    agg=collections.Counter(); cnt=collections.Counter(); big=[]
    tot=0
    for c in V:
        seg=win(u,c['wstart'],c['wend'])
        pe=None; pn='<call start>'
        for (x,y,n) in seg:
            if pe is not None and x>pe:
                g=x-pe; agg[(pn,n)]+=g; cnt[(pn,n)]+=1; tot+=g; big.append((g,pn,n))
            pe=y if pe is None else max(pe,y); pn=n
    print(f"--- GAPS on {tracks[u]} across {N} verify calls: total idle {tot/N/1e6:.2f} ms/call")
    print(f"    {'ms/call':>9} {'n/call':>7} {'mean us':>9}  prev -> next")
    for (p,n),t in agg.most_common(top):
        print(f"    {t/N/1e6:>9.2f} {cnt[(p,n)]/N:>7.1f} {t/cnt[(p,n)]/1000:>9.1f}  {p}  ->  {n}")
    big.sort(reverse=True)
    print("    single largest gaps:", ", ".join(f"{g/1e6:.1f}ms({p}->{n})" for g,p,n in big[:5]))
    print()
gaps(DG,'dgpu')
gaps(IG,'igpu')
# remote breakdown
sub=0; wat=0; rtt=0; rem=0; link=0; nwait=0
per=[]
for c in V:
    s=w=0; R=L=M=0; k=0
    for (x,y,n) in win(RE,c['wstart'],c['wend']):
        if n.startswith('submit') or n.startswith('decode submit'): s+=y-x
        else:
            w+=y-x
            m=re.search(r'rtt=(\d+)us link=(\d+)us remote=(\d+)us',n)
            if m: R+=int(m.group(1)); L+=int(m.group(2)); M+=int(m.group(3)); k+=1
    sub+=s; wat+=w; rtt+=R; link+=L; rem+=M; nwait+=k
    per.append((w/1e6,R/1e3,M/1e3,k))
print(f"remote.expert per verify call: submit-block {sub/N/1e6:.2f} ms, wait-block {wat/N/1e6:.2f} ms over {nwait/N:.0f} waits")
print(f"  reported rtt sum {rtt/N/1e3:.1f} ms/call ({rtt/max(nwait,1)/1e3:.2f} ms avg/layer-lane), link {link/N/1e3:.1f} ms, remote(box2) {rem/N/1e3:.1f} ms ({rem/max(nwait,1)/1e3:.2f} ms avg)")
