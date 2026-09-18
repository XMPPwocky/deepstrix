import pickle, re, bisect, collections
tracks, lv = pickle.load(open('leaves.pkl','rb'))
_t, sl = pickle.load(open('slices.pkl','rb'))
DG=0x4450475500000001
calls=pickle.load(open('calls.pkl','rb'))
pre=sorted(x[0] for x in sl[DG] if x[2]=='dgpu.mhc_pre_attn')
for c in calls:
    i=bisect.bisect_left(pre,c['start'])
    c['wstart']=pre[i-1] if i>0 else c['start']
for i,c in enumerate(calls):
    c['wend']=calls[i+1]['wstart'] if i+1<len(calls) else c['end']
    c['B']=sum(b*(n//40) for b,n in c['bs'].items())
pickle.dump(calls,open('calls.pkl','wb'))
w=[( (c['wend']-c['wstart'])/1e6, c['kind'], c['B']) for c in calls]
for i,(x,k,B) in enumerate(w): print(f"{i:>3} {k:>6} B={B:>3} wall={x:9.2f} ms")
import statistics
v=sorted(x for x,k,B in w if B==6)
print("\nB=6 verify calls: n=%d  mean %.1f  p10 %.1f p50 %.1f p90 %.1f  min %.1f max %.1f"%(len(v),statistics.mean(v),v[len(v)//10],v[len(v)//2],v[9*len(v)//10],v[0],v[-1]))
