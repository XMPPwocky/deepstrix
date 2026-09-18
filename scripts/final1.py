import pickle, bisect, collections, statistics, re
tracks, lv = pickle.load(open('leaves.pkl','rb'))
calls=pickle.load(open('calls.pkl','rb'))
for u in lv: lv[u].sort()
starts={u:[x[0] for x in lv[u]] for u in lv}
short={0x4450475500000001:'dgpu.compute',0x4450475500000002:'dgpu.xfer',0x4950475500000001:'igpu.compute',0x4950475500000002:'igpu.xfer',0x52454d5400000001:'remote.expert',0x5041474500000001:'expert pager'}
def win(u,b,e):
    i=bisect.bisect_left(starts[u],b); j=bisect.bisect_left(starts[u],e)
    r=[]
    for (sb,se,n) in lv[u][max(0,i-1):j]:
        ob,oe=max(sb,b),min(se,e)
        if oe>ob: r.append((ob,oe,n))
    return r
def unions(iv):
    iv=sorted(iv); tot=0; cb=ce=-1
    for b,e in iv:
        if b>ce:
            if ce>cb: tot+=ce-cb
            cb,ce=b,e
        else: ce=max(ce,e)
    if ce>cb: tot+=ce-cb
    return tot
def isw(n): return '.wait' in (n or '')
groups={
 'VERIFY B=6 (all 53)':[c for c in calls if c['B']==6],
 'VERIFY B=6 (req2, calls 27-58)':[c for c in calls[27:] if c['B']==6],
 'DECODE B=1 (4)':[c for c in calls if c['kind']=='decode'],
}
for gname,cs in groups.items():
    n=len(cs); wall=sum(c['wend']-c['wstart'] for c in cs)
    print(f"##### {gname}: n={n}, mean wall {wall/n/1e6:.1f} ms")
    devint=[]
    for u in short:
        tb=tw=0; dev=[]
        for c in cs:
            for (x,y,nm) in win(u,c['wstart'],c['wend']):
                if isw(nm): tw+=y-x
                else:
                    tb+=y-x
                    if u in (0x4450475500000001,0x4950475500000001): dev.append((x,y))
        devint+=dev
        print(f"   {short[u]:<14} busy {tb/n/1e6:8.2f} ms/call ({100*tb/wall:5.1f}%)  wait_slices {tw/n/1e6:7.2f} ms")
    print(f"   ANY-GPU busy (union dgpu+igpu compute): {unions(devint)/n/1e6:.2f} ms/call ({100*unions(devint)/wall:.1f}%)")
    print(f"   => GPU IDLE {100-100*unions(devint)/wall:.1f}% of the call")
    print()
