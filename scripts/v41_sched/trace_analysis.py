#!/usr/bin/env python3
# Routing predictability in expert_trace_v4flash.bin: u16 ids, 6/layer, 43 layers/token.
import struct, sys, math, random
from collections import defaultdict, OrderedDict
P="/home/claude-code/.cache/deepstrix/v41/expert_trace_v4flash.bin"
raw=open(P,'rb').read(); n=len(raw)//2
L=43; K=6; NE=256
T=n//(L*K)
vals=struct.unpack('<%dH'%n, raw)
# tok[t][l] = tuple of 6 ids
tok=[[vals[(t*L+l)*K:(t*L+l)*K+K] for l in range(L)] for t in range(T)]
print(f"tokens={T} layers={L} k={K} maxid={max(vals)}")
A=range(0,2560); B=range(2560,T)

# ---------- 1. temporal locality (same layer): fraction of picks at t seen in union of last W tokens
for W in (1,2,4,16,64):
    hit=tot=0
    for t in range(64,T):
        for l in range(L):
            recent=set()
            for u in range(max(0,t-W),t): recent.update(tok[u][l])
            for e in tok[t][l]:
                tot+=1; hit+= e in recent
    print(f"temporal W={W:3d}: P(pick at t in picks of last W tokens, same layer) = {hit/tot:.3f}")

# ---------- 2. LRU miss rate vs residency (global LRU over (layer,expert) slots), warmup 1000 tok
def lru_misses(resident_frac, seg, warm=1000):
    cap=int(resident_frac*L*NE)
    lru=OrderedDict()
    # warm from tokens before seg
    misses=0; cnt=0
    start=seg[0]
    for t in range(max(0,start-warm), seg[-1]+1):
        for l in range(L):
            for e in tok[t][l]:
                key=l*NE+e
                if key in lru: lru.move_to_end(key)
                else:
                    if t>=start: misses+=1
                    lru[key]=1
                    if len(lru)>cap: lru.popitem(last=False)
        if t>=start: cnt+=1
    return misses/cnt
for r in (0.50,0.60,0.66,0.75,0.85):
    print(f"LRU resident={r:.2f}: misses/token  A(EN)={lru_misses(r,list(A)[1000:]):.2f}  B(ZH)={lru_misses(r,list(B)):.2f}   (V4-Flash slots; x240/258 for V4.1 picks)")

# ---------- 3. cross-layer predictor: P(e' at l+k | e at l), trained on A, tested on B (and swap)
def train(train_rng,k):
    C=[defaultdict(lambda: defaultdict(int)) for _ in range(L)]
    F=[defaultdict(int) for _ in range(L)]
    for t in train_rng:
        for l in range(L-k):
            for e2 in tok[t][l+k]: F[l+k][e2]+=1
            for e in tok[t][l]:
                row=C[l][e]
                for e2 in tok[t][l+k]: row[e2]+=1
    return C,F
def predict(C,F,l,picks,k,Kp,evidence_layers=1):
    # score e' = sum over evidence picks of P(e'|e); evidence = picks at layers l-evidence_layers+1..l
    score=defaultdict(float)
    for j,pl in picks:  # (layer, picks)
        for e in pl:
            row=C[j][e]; s=sum(row.values()) or 1
            # weight nearer layers more? keep uniform
            for e2,c in row.items(): score[e2]+=c/s
    return [e for e,_ in sorted(score.items(), key=lambda x:-x[1])[:Kp]]
def eval_pred(train_rng,test_rng,k,Kp,ev=1):
    C,F=train(train_rng,k)
    # but for evidence layers j<l we need tables at horizon (l+k-j); build tables for horizons k..k+ev-1
    tabs={h:train(train_rng,h)[0] for h in range(k,k+ev)}
    hit=tot=0; fhit=0
    for t in test_rng:
        for l in range(ev-1,L-k):
            score=defaultdict(float)
            for d in range(ev):
                j=l-d; h=k+d
                for e in tok[t][j]:
                    row=tabs[h][j][e]; s=sum(row.values()) or 1
                    for e2,c in row.items(): score[e2]+=c/s
            top=[e for e,_ in sorted(score.items(), key=lambda x:-x[1])[:Kp]]
            ftop=[e for e,_ in sorted(F[l+k].items(), key=lambda x:-x[1])[:Kp]]
            for e in tok[t][l+k]:
                tot+=1; hit+= e in top; fhit+= e in ftop
    return hit/tot, fhit/tot
print("\ncross-layer predictor (train EN->test ZH): recall of the 6 true picks at layer l+k within top-K' predicted")
print(" k  K'  ev | recall(pred)  recall(freq baseline)   random=K'/256")
for k in (1,2,5,10):
    for Kp in (6,12,24,48):
        r,f=eval_pred(A,B,k,Kp,ev=1)
        print(f"{k:2d} {Kp:3d}  1  |   {r:.3f}        {f:.3f}                 {Kp/256:.3f}")
for k in (1,5):
    for Kp in (12,24):
        r,f=eval_pred(A,B,k,Kp,ev=3)
        print(f"{k:2d} {Kp:3d}  3  |   {r:.3f}        {f:.3f}     (evidence = picks of 3 layers)")
print("swap (train ZH -> test EN), k=5:")
for Kp in (12,24,48):
    r,f=eval_pred(B,A,5,Kp,ev=1); print(f" 5 {Kp:3d}  1  |   {r:.3f}        {f:.3f}")

# ---------- 4. can the k-ahead predictor catch LRU MISSES specifically? (resident 0.66, test on B, trained on A)
def miss_pred(resident_frac,k,Kp,train_rng,test_rng,warm=1000):
    tabs=train(train_rng,k)[0]
    cap=int(resident_frac*L*NE); lru=OrderedDict()
    start=test_rng[0]; nm=0; caught=0; prefetch_issued=0; ntok=0
    for t in range(max(0,start-warm), test_rng[-1]+1):
        # predictions made at layer l for layer l+k, before the token's LRU updates at l+k
        preds={}
        for l in range(L-k):
            score=defaultdict(float)
            for e in tok[t][l]:
                row=tabs[l][e]; s=sum(row.values()) or 1
                for e2,c in row.items(): score[e2]+=c/s
            preds[l+k]=[e for e,_ in sorted(score.items(), key=lambda x:-x[1])[:Kp]]
        for l in range(L):
            if t>=start and l in preds:
                # prefetch candidates not resident at prediction time (approx: state now)
                prefetch_issued+=sum(1 for e in preds[l] if (l*NE+e) not in lru)
            for e in tok[t][l]:
                key=l*NE+e
                if key in lru: lru.move_to_end(key)
                else:
                    if t>=start:
                        nm+=1
                        if l in preds and e in preds[l]: caught+=1
                    lru[key]=1
                    if len(lru)>cap: lru.popitem(last=False)
        if t>=start: ntok+=1
    return nm/ntok, caught/max(nm,1), prefetch_issued/ntok
print("\nLRU misses caught by the k-ahead id predictor (resident 0.66, warm LRU, train EN test ZH):")
print(" k  K' | misses/tok  frac caught  prefetches issued/tok (non-resident predicted)")
for k in (1,3,5,8):
    for Kp in (6,12,24):
        m,c,pi=miss_pred(0.66,k,Kp,A,B)
        print(f"{k:2d} {Kp:3d} |   {m:.2f}       {c:.3f}         {pi:.1f}")
