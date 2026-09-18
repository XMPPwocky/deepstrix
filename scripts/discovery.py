#!/usr/bin/env python3
"""Is 12.4 misses/token a real floor, or just the key-discovery rate at t=128-256?"""
import sys, array, collections, math
N_LAYER, TOPK = 40, 6
buf = array.array('H'); buf.frombytes(open(sys.argv[1],'rb').read())
per_tok = N_LAYER*TOPK; ntok = len(buf)//per_tok
toks=[]
for t in range(ntok):
    s=set()
    for l in range(N_LAYER):
        seen=[]
        for k in range(TOPK):
            e=buf[t*per_tok+l*TOPK+k]
            if e<384 and e not in seen: seen.append(e); s.add((l,e))
    toks.append(s)

seen=set(); dist=[]; new=[]
for s in toks:
    n=len(s-seen); new.append(n); seen|=s; dist.append(len(seen))
print(f"{ntok} tokens, {len(seen)} distinct (layer,expert) of {N_LAYER*384} = {100*len(seen)/(N_LAYER*384):.1f}%")
print(f"\n{'token window':>14} {'new keys/token':>15} {'cum distinct':>13}")
W=32
for i in range(0,ntok,W):
    print(f"{i:6d}-{min(i+W,ntok):<7d} {sum(new[i:i+W])/len(new[i:i+W]):15.1f} {dist[min(i+W,ntok)-1]:13d}")

# Heaps' law fit  D(T) = k*T^beta  on the second half (past the cold burst)
import statistics
xs=[math.log(t+1) for t in range(ntok//4,ntok)]; ys=[math.log(dist[t]) for t in range(ntok//4,ntok)]
mx,my=statistics.mean(xs),statistics.mean(ys)
beta=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
k=math.exp(my-beta*mx)
print(f"\nHeaps fit: distinct(T) = {k:.1f} * T^{beta:.3f}")
print(f"{'T tokens':>10} {'distinct pred':>14} {'new keys/token = dD/dT':>24}")
for T in [256,512,1024,4096,16384,65536,262144]:
    D=k*T**beta; d=k*beta*T**(beta-1)
    cap=N_LAYER*384
    print(f"{T:10d} {min(D,cap):14.0f} {d:24.2f}{'  (saturated)' if D>cap else ''}")
