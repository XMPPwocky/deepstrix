import pickle, collections, bisect
tracks, lv = pickle.load(open('leaves.pkl','rb'))
steps = pickle.load(open('steps.pkl','rb'))
DG=0x4450475500000001; DX=0x4450475500000002
IG=0x4950475500000001; IX=0x4950475500000002
RE=0x52454d5400000001; PG=0x5041474500000001
order=[(DG,'dgpu.compute'),(IG,'igpu.compute'),(DX,'dgpu.xfer'),(IX,'igpu.xfer'),(RE,'remote.expert'),(PG,'expert pager')]
for u,_ in order: lv[u].sort()
starts={u:[x[0] for x in lv[u]] for u,_ in order}
def iswait(n): return n is not None and '.wait' in n
def window(u,b,e):
    i=bisect.bisect_left(starts[u],b)
    j=bisect.bisect_left(starts[u],e)
    # include one before in case it straddles
    k=max(0,i-1)
    res=[]
    for (sb,se,n) in lv[u][k:j]:
        ob=max(sb,b); oe=min(se,e)
        if oe>ob: res.append((ob,oe,n))
    return res
def union(iv):
    iv=sorted(iv); tot=0; ce=-1; cb=-1
    for b,e in iv:
        if b>ce:
            if ce>cb: tot+=ce-cb
            cb,ce=b,e
        else: ce=max(ce,e)
    if ce>cb: tot+=ce-cb
    return tot

agg=collections.defaultdict(lambda: collections.defaultdict(float))
cnt=collections.Counter()
rows=[]
for idx,(c,b,e,n) in enumerate(steps):
    if n!=40: continue
    key='VERIFY' if c else 'decode'
    cnt[key]+=1
    W=e-b
    r={'wall':W}
    devb=[]
    for u,nm in order:
        sls=window(u,b,e)
        busy=sum(y-x for (x,y,s) in sls if not iswait(s))
        wait=sum(y-x for (x,y,s) in sls if iswait(s))
        r[nm+'_busy']=busy; r[nm+'_wait']=wait
        if nm in ('dgpu.compute','igpu.compute'):
            devb += [(x,y) for (x,y,s) in sls if not iswait(s)]
    r['anyGPU']=union(devb)
    rows.append((idx,key,r))
    for k,v in r.items(): agg[key][k]+=v

print(f"{'':6} {'n':>4} " + " ".join(f"{nm:>16}" for nm in ['wall','anyGPUbusy','dgpu.compute','igpu.compute','dgpu.xfer','igpu.xfer','remote.expert']))
for key in ('VERIFY','decode'):
    a=agg[key]; n=cnt[key]
    f=lambda k: a[k]/n/1e6
    print(f"{key:6} {n:>4} " + " ".join(f"{v:>16.2f}" for v in [f('wall'),f('anyGPU'),f('dgpu.compute_busy'),f('igpu.compute_busy'),f('dgpu.xfer_busy'),f('igpu.xfer_busy'),f('remote.expert_busy')]))
print()
print("per-VERIFY-step detail (ms):")
for idx,key,r in rows:
    if key!='VERIFY': continue
    print(f" step {idx}: wall {r['wall']/1e6:.2f} anyGPU {r['anyGPU']/1e6:.2f} ({100*r['anyGPU']/r['wall']:.1f}%) dgpu {r['dgpu.compute_busy']/1e6:.2f} igpu {r['igpu.compute_busy']/1e6:.2f} dgpu.wait {r['dgpu.compute_wait']/1e6:.2f} igpu.wait {r['igpu.compute_wait']/1e6:.2f} remote_busy {r['remote.expert_busy']/1e6:.2f} pager {r['expert pager_busy']/1e6:.3f}")
print()
print("decode steps: median wall & anyGPU")
import statistics
dw=sorted(r['wall']/1e6 for i,k,r in rows if k=='decode')
dg=sorted(r['anyGPU']/1e6 for i,k,r in rows if k=='decode')
print(" wall p10/p50/p90:", f"{dw[len(dw)//10]:.1f} {dw[len(dw)//2]:.1f} {dw[9*len(dw)//10]:.1f}")
print(" anyGPU p10/p50/p90:", f"{dg[len(dg)//10]:.1f} {dg[len(dg)//2]:.1f} {dg[9*len(dg)//10]:.1f}")
pickle.dump(rows, open('rows.pkl','wb'))
