import pickle, re, bisect, collections
tracks, lv = pickle.load(open('leaves.pkl','rb'))
RE=0x52454d5400000001; DG=0x4450475500000001
s=sorted(lv[RE])
marks=[]
for (b,e,n) in s:
    m=re.match(r'(decode )?submit L(\d+)(?: b=(\d+))?$', n)
    if m: marks.append((b,e,'decode' if m.group(1) else 'batch', int(m.group(2)), int(m.group(3)) if m.group(3) else 1))
calls=[]; cur=None; prevL=None
for (b,e,k,L,bb) in marks:
    if cur is None or (L==0 and prevL==39):
        if cur: calls.append(cur)
        cur={'start':b,'kind':k,'bs':collections.Counter(),'n':0,'end':e}
    cur['bs'][bb]+=1; cur['n']+=1; cur['end']=e; prevL=L
calls.append(cur)
print("calls:", len(calls))
# window: from previous call end to this call's last event; use start-of-next as end
dgs=sorted(lv[DG]); dst=[x[0] for x in dgs]
for i,c in enumerate(calls):
    j=bisect.bisect_left(dst,c['start']); k=j
    while k>0 and dgs[k-1][2]!='dgpu.mhc_pre_attn': k-=1
    c['wstart']=dgs[k-1][0] if k>0 else c['start']
for i,c in enumerate(calls):
    c['wend']=calls[i+1]['wstart'] if i+1<len(calls) else c['end']
agg=collections.Counter()
for i,c in enumerate(calls):
    key=(c['kind'], tuple(sorted(c['bs'].items())), c['n'])
    agg[key]+=1
for k,v in agg.most_common(): print(" ",v,"x",k)
pickle.dump(calls, open('calls.pkl','wb'))
print()
for i,c in enumerate(calls):
    B=sum(b*(n//40) for b,n in c['bs'].items())
    print(f"{i:>3} {c['kind']:>6} B={B:>3} submits={c['n']:>3} wall={(c['wend']-c['wstart'])/1e6:9.2f} ms")
