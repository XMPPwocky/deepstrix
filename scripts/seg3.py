import pickle, collections, bisect
tracks, sl = pickle.load(open('slices.pkl','rb'))
DG=0x4450475500000001
s=sl[DG]
pre=[ (b,e) for (b,e,n) in s if n=='dgpu.mhc_pre_attn' ]
pre.sort()
xq=sorted(b for (b,e,n) in s if n=='dgpu.moe_xq_pre')
print("n mhc_pre_attn", len(pre), "n moe_xq_pre", len(xq))
# layer spans
layers=[]
for i,(b,e) in enumerate(pre):
    end = pre[i+1][0] if i+1<len(pre) else e
    layers.append((b,end))
# classify
def isverify(b,en):
    i=bisect.bisect_left(xq,b)
    return i<len(xq) and xq[i]<en
cls=[isverify(b,e) for (b,e) in layers]
print("verify layers:", sum(cls), "decode layers:", len(cls)-sum(cls))
# group into runs of same class
runs=[]
cur=cls[0]; st=0
for i in range(1,len(cls)):
    if cls[i]!=cur:
        runs.append((cur,st,i-1)); cur=cls[i]; st=i
runs.append((cur,st,len(cls)-1))
print("runs (class, nlayers, wall_ms):")
for c,a,b in runs:
    n=b-a+1
    w=(layers[b][1]-layers[a][0])/1e6
    print(f"  {'VERIFY' if c else 'decode'}  layers={n:>5}  wall={w:9.2f} ms  ({w/max(n/40,1e-9):7.2f} ms/step)  idx {a}..{b}")
pickle.dump((layers,cls), open('layers.pkl','wb'))
