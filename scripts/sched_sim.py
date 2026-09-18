#!/usr/bin/env python3
# List-scheduling sim of one decode step on: dgpu, igpu1, igpu2, nvme1, nvme2 (+ link latency).
# Sub-batches trail each other by one layer (KV dependency). Anchors from PLAN 7a/7a.1.
import random, math, sys
def poisson(lam):
    L=math.exp(-lam); k=0; p=1.0
    while True:
        p*=random.random()
        if p<L: return k
        k+=1
ATT=lambda b: 0.55+0.03*(b-1)+0.03      # dGPU attention chain (launch-bound) + router
PICK=0.088                              # 18.8 MB @ 214 GB/s
SH1,SH2,SHD=0.38,0.47,0.05              # pick shares: box1 iGPU, box2 iGPU, dGPU hot (rest = SSD tier, misses)
MISS=4.82                               # ms per cold expert, NVMe single-server (bandwidth-bound)
LINK_BW=1.5e6                           # bytes/ms
def simulate(subs, rtt, p, miss_per_pos, steps=12, drafter=5.2, seed=1, nvme_split=(0.4,0.6)):
    random.seed(seed)
    K=sum(subs)-1; toks=(1-p**(K+1))/(1-p) if K>0 else 1.0
    free={'dgpu':0,'igpu1':0,'igpu2':0,'nvme1':0,'nvme2':0}
    busy={k:0.0 for k in free}; link_busy=0.0; inflight=[]  # (start,end) of remote requests
    t0=0.0; T=0.0
    def run(res, ready, dur):
        nonlocal busy
        s=max(ready, free[res]); free[res]=s+dur; busy[res]+=dur; return s+dur
    for step in range(steps):
        post_prev=[t0]*len(subs); attn_prev_sub=[None]*40
        # simple layer-major issue order: for l, for s  (list scheduling by readiness emerges from max())
        for l in range(40):
            for si,b in enumerate(subs):
                dep=post_prev[si]
                if si>0: dep=max(dep, attn_prev_sub[l])  # KV of previous sub-batch at this layer
                a_end=run('dgpu', dep, ATT(b)); attn_prev_sub[l]=a_end
                picks=6*b
                # box1 local
                lam1=miss_per_pos*b/40*nvme_split[0]; n1=poisson(lam1) if miss_per_pos>0 else 0
                e1=run('igpu1', a_end, picks*SH1*PICK)
                m1=a_end
                for _ in range(n1): m1=max(m1, run('nvme1', a_end, MISS))
                if n1: e1=run('igpu1', max(e1,m1), n1*PICK)
                # box2 remote
                out=a_end+rtt/2+ (b*6000)/LINK_BW; link_busy+=(b*6000)/LINK_BW
                lam2=miss_per_pos*b/40*nvme_split[1]; n2=poisson(lam2) if miss_per_pos>0 else 0
                e2=run('igpu2', out, picks*SH2*PICK)
                m2=out
                for _ in range(n2): m2=max(m2, run('nvme2', out, MISS))
                if n2: e2=run('igpu2', max(e2,m2), n2*PICK)
                back=e2+rtt/2+(b*20000)/LINK_BW; link_busy+=(b*20000)/LINK_BW
                inflight.append((a_end,back))
                # dgpu shared + hot
                e3=run('dgpu', a_end, 0.06+picks*SHD*PICK*214/640)
                post_prev[si]=run('dgpu', max(e1,back,e3), 0.02)
        head=run('dgpu', max(post_prev), 1.4 if K>0 else 1.4)
        if K>0:
            # drafter: dGPU attn 1.0 -> igpu1 experts 3.2 -> dGPU head+markov 1.0 (serial)
            d1=run('dgpu', head+0.1, 1.0); d2=run('igpu1', d1, 3.2); t0=run('dgpu', d2, 1.0)
        else: t0=head+0.05
        T=t0
    # link in-flight fraction
    ev=sorted(inflight); cov=0; cur=None
    for s,e in ev:
        if cur is None or s>cur[1]:
            if cur: cov+=cur[1]-cur[0]
            cur=[s,e]
        else: cur[1]=max(cur[1],e)
    if cur: cov+=cur[1]-cur[0]
    ms=T/steps
    return ms, toks, toks/ms*1000, {k:busy[k]/T for k in busy}, cov/T
def row(name, subs, rtt, p, mp):
    ms,toks,tps,bz,lk=simulate(subs,rtt,p,mp)
    return f"{name:14s} rtt={rtt:.2f} p={p:.2f} miss/pos={mp:.1f} | {ms:6.1f} ms/step {toks:4.2f} tok/step -> {tps:5.1f} tok/s | busy dGPU {bz['dgpu']:.2f} iGPU1 {bz['igpu1']:.2f} iGPU2 {bz['igpu2']:.2f} nvme {bz['nvme1']:.2f}/{bz['nvme2']:.2f} link-inflight {lk:.2f}"
cfgs=[("single",[1]),("batched K=2",[3]),("batched K=5",[6]),("wave 1+1",[1,1]),("wave 2+2",[2,2]),("wave 3+3",[3,3]),("wave 2+2+2",[2,2,2]),("wave 3+2+1",[3,2,1])]
print("=== no misses (residency solved), p=0.75 ===")
for rtt in (0.10,0.30):
    for n,s in cfgs: print(row(n,s,rtt,0.75,0.0))
print("=== misses 3.5/position (66% resident, trace-measured), p=0.75 ===")
for rtt in (0.10,0.30):
    for n,s in cfgs: print(row(n,s,rtt,0.75,3.5))
print("=== misses 1.5/position (75% resident), p=0.75, rtt 0.1 ===")
for n,s in cfgs: print(row(n,s,0.10,0.75,1.5))
print("=== acceptance sensitivity, rtt 0.1, no misses ===")
for p in (0.6,0.75,0.85,0.95):
    for n,s in [("batched K=2",[3]),("wave 2+2",[2,2]),("wave 3+3",[3,3]),("wave 3+2+1",[3,2,1])]: print(row(n,s,0.10,p,0.0))
print("=== acceptance sensitivity, rtt 0.1, misses 3.5 ===")
for p in (0.6,0.75,0.85):
    for n,s in [("batched K=2",[3]),("wave 2+2",[2,2]),("wave 3+3",[3,3])]: print(row(n,s,0.10,p,3.5))
