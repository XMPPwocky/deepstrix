import sched_sim2 as S, heapq
# monkeypatch: rerun one step of 3+3 and dump attn/post timings of the first 6 layers
def trace(subs, rtt=0.1):
    S.random.seed(1)
    tasks=[]
    def T(res,dur,name=''): t=S.Task(res,dur,name); tasks.append(t); return t
    root=T(None,0,'root'); post_prev=[root]*len(subs); attn_at=[None]*40; log=[]
    for l in range(40):
        for si,b in enumerate(subs):
            a=T('dgpu',S.ATT(b),f'attn{si}L{l}'); S.dep(post_prev[si],a)
            if si>0: S.dep(attn_at[l],a)
            attn_at[l]=a; picks=6*b
            e1=T('igpu1',picks*S.SH1*S.PICK,f'loc{si}L{l}'); S.dep(a,e1)
            out=T(None,rtt/2+b*6000/S.LINK_BW,'out'); S.dep(a,out)
            e2=T('igpu2',picks*S.SH2*S.PICK,f'rem{si}L{l}'); S.dep(out,e2)
            back=T(None,rtt/2+b*20000/S.LINK_BW,'back'); S.dep(e2,back)
            e3=T('dgpu',0.06+picks*S.SHD*S.PICK*214/640,f'sh{si}L{l}'); S.dep(a,e3)
            post=T('dgpu',0.02,f'post{si}L{l}'); S.dep(e1,post); S.dep(back,post); S.dep(e3,post)
            post_prev[si]=post; log+= [a,e2,post]
    free={'dgpu':0,'igpu1':0,'igpu2':0}; pq=[]; cnt=0; heapq.heappush(pq,(0.0,cnt,root)); cnt+=1
    while pq:
        r,_,t=heapq.heappop(pq)
        if t.res is None: st=r
        else: st=max(r,free[t.res]); free[t.res]=st+t.dur
        t.end=st+t.dur; t.st=st
        for s in t.succs:
            s.npend-=1
            if s.npend==0: heapq.heappush(pq,(max(x.end for x in s.preds),cnt,s)); cnt+=1
    for t in log[:6*len(subs)*3]: print(f"{t.name:10s} start {t.st:6.2f} end {t.end:6.2f}")
    print("layer 39 post ends:", [round(p.end,2) for p in post_prev])
print("--- 3+3"); trace([3,3])
print("--- 2+2"); trace([2,2])
