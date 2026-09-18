import struct
from collections import OrderedDict
P="/home/claude-code/.cache/deepstrix/v41/expert_trace_v41_decode313.bin"
raw=open(P,'rb').read(); n=len(raw)//2; L=40; K=6; NE=384
vals=struct.unpack('<%dH'%n, raw); T=n//(L*K)
tok=[[vals[(t*L+l)*K:(t*L+l)*K+K] for l in range(L)] for t in range(T)]
WARM=range(0,128); EVAL=range(128,T)
def run(cap, local):
    lru=OrderedDict(); miss=0; req=0; rtt=0
    for t in range(T):
        for l in range(L):
            layer_remote=False
            for e in tok[t][l]:
                if not local(l,e): layer_remote=True; continue
                k=(l,e)
                if t in EVAL: req+=1
                if k in lru: lru.move_to_end(k)
                else:
                    if t in EVAL: miss+=1
                    lru[k]=1
                    if len(lru)>cap: lru.popitem(last=False)
            if t in EVAL and layer_remote: rtt+=1
    n=len(EVAL); return req/n, miss/n, rtt/n
assign={
 "no split (box1 all)":        lambda l,e: True,
 "uniform all:250-383":        lambda l,e: e<250,
 "skew L0-19:116-383 only":    lambda l,e: (e<116) if l<20 else True,
 "skew L0-19:200-383 + L20-39:300-383": lambda l,e: (e<200) if l<20 else (e<300),
}
for cap in (2186,3274,4096):
    print(f"--- box1 slots={cap}")
    for name,f in assign.items():
        req,miss,rtt=run(cap,f)
        print(f"  {name:38s} box1 req/tok {req:6.1f}  box1 miss/tok {miss:5.1f}  remote rtts/tok {rtt:4.1f}")
