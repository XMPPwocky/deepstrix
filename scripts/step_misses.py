import struct
from collections import OrderedDict
P="/home/claude-code/.cache/deepstrix/v41/expert_trace_v4flash.bin"
raw=open(P,'rb').read(); n=len(raw)//2; L=43;K=6;NE=256; T=n//(L*K)
v=struct.unpack('<%dH'%n, raw)
tok=[[v[(t*L+l)*K:(t*L+l)*K+K] for l in range(L)] for t in range(T)]
def run(res, S, seg, warm=1000):
    cap=int(res*L*NE); lru=OrderedDict(); start=seg[0]
    steps=0; dm=0; layer_events=0; per_layer_hist={}
    t=max(0,start-warm)
    while t+S<=seg[-1]+1:
        for l in range(L):
            new=set()
            for i in range(S):
                for e in tok[t+i][l]:
                    key=l*NE+e
                    if key in lru: lru.move_to_end(key)
                    else:
                        new.add(key); lru[key]=1
                        if len(lru)>cap: lru.popitem(last=False)
            if t>=start:
                dm+=len(new); layer_events+= len(new)>0
                per_layer_hist[len(new)]=per_layer_hist.get(len(new),0)+1
        if t>=start: steps+=1
        t+=S
    return dm/steps, layer_events/steps, per_layer_hist
for res in (0.66,0.75,0.85):
    for S in (1,4,6):
        a=run(res,S,list(range(1000,2560))); b=run(res,S,list(range(2560,T)))
        print(f"res={res:.2f} step={S} pos: distinct misses/step EN={a[0]:.2f} ZH={b[0]:.2f}; layers-with-miss/step EN={a[1]:.1f} ZH={b[1]:.1f}; miss-count-per-layer hist(ZH)={dict(sorted(b[2].items()))}")
