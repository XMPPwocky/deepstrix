#!/usr/bin/env python3
# DAG + event-driven list scheduler (tasks start in readiness order per resource; res=None = pure latency).
import random, math, heapq
def poisson(lam):
    L=math.exp(-lam); k=0; p=1.0
    while True:
        p*=random.random()
        if p<L: return k
        k+=1
ATT=lambda b: 0.55+0.03*(b-1)+0.03
PICK=0.088; SH1,SH2,SHD=0.38,0.47,0.05; MISS=4.82; LINK_BW=1.5e6
class Task:
    __slots__=('res','dur','preds','succs','npend','end','name','st')
    def __init__(s,res,dur,name=''): s.res=res; s.dur=dur; s.preds=[]; s.succs=[]; s.npend=0; s.end=None; s.name=name
def dep(a,b):  # a before b
    a.succs.append(b); b.preds.append(a); b.npend+=1
def simulate(subs, rtt, p, miss_per_pos, steps=10, seed=1, nvme_split=(0.4,0.6), drafter=(1.0,3.2,1.0), remote_share=SH2, local_share=SH1):
    random.seed(seed)
    K=sum(subs)-1; toks=(1-p**(K+1))/(1-p) if K>0 else 1.0
    tasks=[]; 
    def T(res,dur,name=''): t=Task(res,dur,name); tasks.append(t); return t
    root=T(None,0,'root'); prev_step_end=root
    remote_reqs=[]
    for step in range(steps):
        post_prev=[prev_step_end]*len(subs); attn_at=[None]*40
        for l in range(40):
            for si,b in enumerate(subs):
                a=T('dgpu',ATT(b),f'attn s{si} l{l}'); dep(post_prev[si],a)
                if si>0: dep(attn_at[l],a)
                attn_at[l]=a
                picks=6*b
                # local
                e1=T('igpu1',picks*local_share*PICK); dep(a,e1)
                n1=poisson(miss_per_pos*b/40*nvme_split[0]) if miss_per_pos>0 else 0
                j1=T(None,0); dep(e1,j1)
                for _ in range(n1):
                    m=T('nvme1',MISS); dep(a,m); dep(m,j1)
                e1b=T('igpu1',n1*PICK); dep(j1,e1b)
                # remote
                out=T(None,rtt/2+b*6000/LINK_BW,'out'); dep(a,out)
                e2=T('igpu2',picks*remote_share*PICK); dep(out,e2)
                n2=poisson(miss_per_pos*b/40*nvme_split[1]) if miss_per_pos>0 else 0
                j2=T(None,0); dep(e2,j2)
                for _ in range(n2):
                    m=T('nvme2',MISS); dep(out,m); dep(m,j2)
                e2b=T('igpu2',n2*PICK); dep(j2,e2b)
                back=T(None,rtt/2+b*20000/LINK_BW,'back'); dep(e2b,back)
                remote_reqs.append((a,back))
                e3=T('dgpu',0.06+picks*SHD*PICK*214/640); dep(a,e3)
                post=T('dgpu',0.02); dep(e1b,post); dep(back,post); dep(e3,post)
                post_prev[si]=post
        head=T('dgpu',1.4,'head')
        for pp in post_prev: dep(pp,head)
        if K>0:
            acc=T(None,0.1); dep(head,acc)
            d1=T('dgpu',drafter[0]); dep(acc,d1); d2=T('igpu1',drafter[1]); dep(d1,d2); d3=T('dgpu',drafter[2]); dep(d2,d3)
            prev_step_end=d3
        else:
            s=T(None,0.05); dep(head,s); prev_step_end=s
    # event loop
    free={'dgpu':0,'igpu1':0,'igpu2':0,'nvme1':0,'nvme2':0}; busy={k:0.0 for k in free}
    pq=[]; cnt=0
    heapq.heappush(pq,(0.0,cnt,root)); cnt+=1
    T_end=0
    while pq:
        r,_,t=heapq.heappop(pq)
        if t.res is None or t.dur==0: st=r
        else: st=max(r,free[t.res]); free[t.res]=st+t.dur; busy[t.res]+=t.dur
        t.end=st+t.dur; T_end=max(T_end,t.end)
        for s in t.succs:
            s.npend-=1
            if s.npend==0:
                heapq.heappush(pq,(max(x.end for x in s.preds),cnt,s)); cnt+=1
    total=prev_step_end.end
    # link in-flight coverage
    iv=sorted((a.end,b.end) for a,b in remote_reqs); cov=0; cur=None
    for s,e in iv:
        if cur is None or s>cur[1]:
            if cur: cov+=cur[1]-cur[0]
            cur=[s,e]
        else: cur[1]=max(cur[1],e)
    if cur: cov+=cur[1]-cur[0]
    ms=total/steps
    return ms,toks,toks/ms*1000,{k:busy[k]/total for k in busy},cov/total
def row(name,subs,rtt,p,mp,**kw):
    ms,toks,tps,bz,lk=simulate(subs,rtt,p,mp,**kw)
    return f"{name:13s} rtt={rtt:.2f} p={p:.2f} miss/pos={mp:.1f} | {ms:6.1f} ms/step {toks:4.2f} tok/step -> {tps:5.1f} tok/s | busy dGPU {bz['dgpu']:.2f} iGPU1 {bz['igpu1']:.2f} iGPU2 {bz['igpu2']:.2f} nvme {bz['nvme1']:.2f}/{bz['nvme2']:.2f} link-inflight {lk:.2f}"
cfgs=[("single",[1]),("batched K=2",[3]),("batched K=5",[6]),("wave 1+1",[1,1]),("wave 2+1",[2,1]),("wave 2+2",[2,2]),("wave 3+3",[3,3]),("wave 2+2+2",[2,2,2]),("wave 3+2+1",[3,2,1])]
if __name__=='__main__':
    import sys
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    print("=== no misses, p=0.75 ===")
    for rtt in (0.10,0.30):
        for n,s in cfgs: print(row(n,s,rtt,0.75,0.0))
    print("=== misses 3.5/position (66% resident), p=0.75 ===")
    for rtt in (0.10,0.30):
        for n,s in cfgs: print(row(n,s,rtt,0.75,3.5))
    print("=== misses 1.5/position (75% resident), p=0.75, rtt 0.1 ===")
    for n,s in cfgs: print(row(n,s,0.10,0.75,1.5))
    print("=== acceptance sensitivity, rtt 0.1, no misses / 1.5 / 3.5 ===")
    for mp in (0.0,1.5,3.5):
        for p in (0.6,0.75,0.85,0.95):
            for n,s in [("batched K=2",[3]),("wave 2+1",[2,1]),("wave 2+2",[2,2]),("wave 3+3",[3,3])]: print(row(n,s,0.10,p,mp))
    print("=== rebalance: wave 3+3 with remote share raised (box2 takes more picks, box1 fewer), rtt 0.1/0.3, no misses ===")
    for rtt in (0.1,0.3):
        for rs,ls in ((0.47,0.38),(0.55,0.30),(0.60,0.25)):
            print(row(f"3+3 r={rs:.2f}",[3,3],rtt,0.75,0.0,remote_share=rs,local_share=ls))
    print("=== drafter cost sensitivity, wave 3+3, rtt 0.1, no misses, p=0.75 ===")
    for d in ((0.5,1.0,0.5),(1.0,3.2,1.0),(1.5,5.0,1.5)):
        print(row(f"3+3 draft={sum(d):.1f}",[3,3],0.1,0.75,0.0,drafter=d))
