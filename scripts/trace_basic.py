import array, collections, statistics
N_LAYER, TOPK, N_EXPERT = 40, 6, 384
buf = array.array('H'); buf.frombytes(open('/home/claude-code/.cache/deepstrix/v41/expert_trace_v41_decode313.bin','rb').read())
per = N_LAYER*TOPK; ntok = len(buf)//per
print("tokens", ntok)
# per-layer stats
dist_per_layer=[]; pick_per_layer=[]
freq=[collections.Counter() for _ in range(N_LAYER)]
dup=0; tot=0; invalid=0
for t in range(ntok):
    for l in range(N_LAYER):
        picks=[buf[t*per+l*TOPK+k] for k in range(TOPK)]
        tot+=TOPK
        for e in picks:
            if e>=N_EXPERT: invalid+=1
        s=set(e for e in picks if e<N_EXPERT)
        dup+=TOPK-len(s)
        for e in s: freq[l][e]+=1
print("invalid picks",invalid,"dup picks (same expert twice in a layer)",dup,"of",tot)
print("\nlayer distinct  top6%  top8%  top16%  top40%  top116% top154% top268%  (in-sample static coverage, whole trace)")
for l in range(N_LAYER):
    c=freq[l]; n=sum(c.values()); mc=c.most_common()
    cov=lambda N: sum(v for _,v in mc[:N])/n
    print(f"L{l:2d} {len(c):4d}  {cov(6)*100:5.1f} {cov(8)*100:5.1f} {cov(16)*100:6.1f} {cov(40)*100:6.1f} {cov(116)*100:7.1f} {cov(154)*100:7.1f} {cov(268)*100:7.1f}")
alld=sum(len(c) for c in freq)
print("distinct pairs total",alld,"of",N_LAYER*N_EXPERT, " mean/layer", alld/N_LAYER)
# per-layer, 384-slot LRU, distinct experts within sliding windows
