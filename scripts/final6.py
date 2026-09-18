import pickle, bisect, collections
tracks, lv = pickle.load(open('leaves.pkl','rb'))
calls=pickle.load(open('calls.pkl','rb'))
for u in lv: lv[u].sort()
starts={u:[x[0] for x in lv[u]] for u in lv}
DG=0x4450475500000001; IG=0x4950475500000001; RE=0x52454d5400000001
def win(u,b,e):
    i=bisect.bisect_left(starts[u],b); j=bisect.bisect_left(starts[u],e)
    return [(max(x,b),min(y,e),n) for (x,y,n) in lv[u][max(0,i-1):j] if min(y,e)>max(x,b)]
V=[c for c in calls[27:] if c['B']==6]; N=len(V)
head=tail=body=0
rw_tail=0
for c in V:
    ws,we=c['wstart'],c['wend']
    seg=win(DG,ws,we)
    f,l=seg[0][0],max(y for _,y,_ in seg)
    head+= f-ws; tail+= we-l; body += l-f
print(f"verify call wall {sum(c['wend']-c['wstart'] for c in V)/N/1e6:.1f} ms =")
print(f"   head (before first dgpu kernel) {head/N/1e6:6.2f} ms")
print(f"   body (first..last dgpu kernel)  {body/N/1e6:6.2f} ms")
print(f"   TAIL (after last dgpu kernel)   {tail/N/1e6:6.2f} ms   <- head/sampler/accept/draft host work")
print()
# last stages before the tail
c2=collections.Counter()
for c in V:
    seg=win(DG,c['wstart'],c['wend'])
    c2[seg[-1][2]]+=1
print("last dgpu stage of each verify call:", c2.most_common(5))
# per lane-layer structure: use remote submit times to slice
print()
print("per-layer-lane timing inside the body (mean of 80 lane-layers/call):")
print(f"   dgpu busy {sum(y-x for c in V for (x,y,n) in win(DG,c['wstart'],c['wend']) if '.wait' not in n)/N/80/1000:.1f} us")
print(f"   igpu busy {sum(y-x for c in V for (x,y,n) in win(IG,c['wstart'],c['wend']) if '.wait' not in n)/N/80/1000:.1f} us")
print(f"   wall/lane-layer {body/N/80/1000:.1f} us")
