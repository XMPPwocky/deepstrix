import sys; sys.argv=[sys.argv[0]]
from sim import *
import random

# ---- claim-capped cost: given nh_res resident-on-host and d_res resident-on-dGPU picks (disjoint), choose how many to claim ----
class ClaimModel:
    def __init__(self, base, cap=True):
        self.base=base; self.name=base.name+('+claim' if cap else ''); self.cap=cap
        self.tab={}
        for nh in range(7):
            for d in range(7-nh):
                if cap:
                    self.tab[(nh,d)] = min(base.tab[(a,b)] for a in range(nh+1) for b in range(d+1))
                else:
                    self.tab[(nh,d)] = base.tab[(nh,d)]
    def f(self, nh, d=0): return self.tab[(nh,d)]
CM = {k: ClaimModel(v) for k,v in MODELS.items()}
for k,M in CM.items(): print(f"{k:8s} claim-capped f(n) = {' '.join(f'n{n}:{M.f(n):.0f}' for n in range(7))}  | with d=2 dGPU: {' '.join(f'n{n}:{M.f(n,2):.0f}' for n in range(5))}")
M = CM['lin73']

def evaluate2(Hs, Ds, M, toks=EVAL):
    tot=0.0; nbar=[0.0]*N_LAYER; nd=[0.0]*N_LAYER; nt=len(toks)
    for l in range(N_LAYER):
        s=0.0; sn=0.0; sd=0.0
        for t in toks:
            p=picks[t][l]; d=sum(1 for e in p if e in Ds[l]); nh=sum(1 for e in p if e in Hs[l] and e not in Ds[l])
            s+=M.f(nh,d); sn+=nh; sd+=d
        tot+=s/nt; nbar[l]=sn/nt; nd[l]=sd/nt
    return tot, nbar, nd
def rep(name, Hs, Ds, M=M, toks=EVAL):
    tot, nb, nd = evaluate2(Hs, Ds, M, toks)
    print(f"{name:60s} host={sum(len(h) for h in Hs):5d} dgpu={sum(len(d) for d in Ds):4d} | phase {tot/1000:6.2f} ms/tok  n_h {sum(nb)/40:.2f} (enc {sum(nb[:20])/20:.2f} dec {sum(nb[20:])/20:.2f}) n_d {sum(nd)/40:.2f} | tok/s {1000/(61.0-25.4+tot/1000):5.2f}")
    return tot

def greedy2(M, S_host, S_dgpu_per_layer, toks, allow_dgpu=True, layers=range(N_LAYER), min_gain=0.05):
    toks=list(toks); nt=len(toks)
    where=collections.defaultdict(list)
    for i,t in enumerate(toks):
        for l in layers:
            for e in picks[t][l]: where[(l,e)].append(i)
    nh=[[0]*nt for _ in range(N_LAYER)]; nd=[[0]*nt for _ in range(N_LAYER)]
    Hs=empty(); Ds=empty()
    def gain(l,e,tier):
        g=0.0
        for i in where[(l,e)]:
            a,b=nh[l][i],nd[l][i]
            g += M.f(a,b) - (M.f(a+1,b) if tier=='h' else M.f(a,b+1))
        return g/nt
    heap=[]
    for (l,e) in where:
        heapq.heappush(heap,(-gain(l,e,'h'),l,e,'h'))
        if allow_dgpu: heapq.heappush(heap,(-gain(l,e,'d'),l,e,'d'))
    used_h=0; used_d=[0]*N_LAYER; total=sum(M.f(0,0) for _ in range(N_LAYER)); traj=[(0,0,total)]; placed=set()
    while heap:
        ng,l,e,tier=heapq.heappop(heap)
        if (l,e) in placed: continue
        if tier=='h' and used_h>=S_host: continue
        if tier=='d' and used_d[l]>=S_dgpu_per_layer: continue
        g=gain(l,e,tier)
        if heap and g < -heap[0][0]-1e-9: heapq.heappush(heap,(-g,l,e,tier)); continue
        if g <= min_gain: break   # us/token per slot: stop at negligible gain
        placed.add((l,e))
        if tier=='h':
            Hs[l].add(e); used_h+=1
            for i in where[(l,e)]: nh[l][i]+=1
        else:
            Ds[l].add(e); used_d[l]+=1
            for i in where[(l,e)]: nd[l][i]+=1
        total-=g; traj.append((used_h,sum(used_d),total))
    return traj,Hs,Ds

print("\n=== CLAIM-CAPPED (n* = 4): baselines and oracle ===")
rep("today (ids<116 touched on enc, 25-slot LRU dec)", today(), empty())
rep("nothing local (mode 2)", empty(), empty())
rep("everything resident, claim 4", [set(range(N_EXP)) for _ in range(N_LAYER)], empty())
traj,Hs,Ds = greedy2(M, 10**9, 0, EVAL, allow_dgpu=False)
byslot={s:tot for s,d,tot in traj}
print("  greedy in-sample (host only) by budget:")
prev=None
for s in (0,128,256,384,512,768,1024,1280,1536,2048,2560,3072,4096):
    if s>traj[-1][0]: break
    v=min(byslot[k] for k in byslot if k<=s)
    extra = f"   marginal {((prev[1]-v)/1000)/((s-prev[0])/128):5.2f} ms/tok per 128-slot window" if prev else ""
    print(f"   host {s:5d} slots ({s*18.8/1000:5.1f} GB) -> {v/1000:6.2f} ms/tok  tok/s {1000/(61.0-25.4+v/1000):5.2f}{extra}")
    prev=(s,v)
print(f"   greedy stops at {traj[-1][0]} slots (gain < 0.05 us/tok/slot): {traj[-1][2]/1000:.2f} ms/tok")
rep("greedy in-sample unconstrained", Hs, Ds)
print("   |H| per layer:", ' '.join(str(len(h)) for h in Hs))
for S in (512, 1024):
    traj,Hs2,Ds2 = greedy2(M, S, 0, EVAL, allow_dgpu=False)
    tot,nb,nd = evaluate2(Hs2,Ds2,M)
    print(f"   S={S}: n_bar per layer:", ' '.join(f"{x:.1f}" for x in nb))
print("\n  greedy in-sample WITH dGPU 6/layer:")
for S in (25, 409, 1024, 10**9):
    traj,Hs2,Ds2 = greedy2(M, S, 6, EVAL)
    rep(f"greedy host<={S if S<10**9 else 'inf'} + dgpu 6/layer", Hs2, Ds2)
print("\n  split-sample (fit 0..127, eval 128..255):")
for S in (512, 1024, 10**9):
    traj,Hs2,Ds2 = greedy2(M, S, 0, range(0,H), allow_dgpu=False)
    rep(f"greedy fit-first-half host<={S if S<10**9 else 'inf'}", Hs2, Ds2)
traj,Hs2,Ds2 = greedy2(M, 1024, 6, range(0,H))
rep("greedy fit-first-half host<=1024 + dgpu 6/layer", Hs2, Ds2)
fr_fit=freq_tables(range(0,H)); fr_in=freq_tables(EVAL)
for n in (24, 40, 64):
    rep(f"static top-{n}/layer fit-first-half", static_topn(fr_fit,[n]*40), empty())

print("\n=== ZERO-WINDOW-COST variant: encoder windows hold box 1's 116-share ranked by frequency; decoder = 409 host + dGPU 12/dec-layer ===")
for label, fit in (("in-sample", EVAL), ("fit-first-half", range(0,H))):
    fr = freq_tables(fit)
    Hs = empty()
    for l in range(20): Hs[l] = set(e for e,_ in fr[l].most_common(116))
    rep(f"[{label}] enc top-116 in windows only", Hs, empty())
    traj,Hd,Dd = greedy2(M, 409, 12, fit, layers=range(20,40))
    for l in range(20): Hd[l]=Hs[l]
    rep(f"[{label}] enc top-116 + dec 409 host + dgpu 12/dec-layer", Hd, Dd)
    traj,Hd,Dd = greedy2(M, 409, 0, fit, allow_dgpu=False, layers=range(20,40))
    for l in range(20): Hd[l]=Hs[l]
    rep(f"[{label}] enc top-116 + dec 409 host, no dgpu", Hd, Dd)
    traj,Hd,Dd = greedy2(M, 409+530, 12, fit, layers=range(20,40))
    for l in range(20): Hd[l]=Hs[l]
    rep(f"[{label}] enc top-116 + dec 939 host (+10 GB pool) + dgpu 12/dec", Hd, Dd)
Hs = today_uniform116()
traj,Hd,Dd = greedy2(M, 409, 12, EVAL, layers=range(20,40))
for l in range(20): Hd[l]=Hs[l]
rep("[in-sample] enc CONTIGUOUS ids<116 (today's split) + dec 409 host + dgpu 12/dec", Hd, Dd)

print("\n=== model sensitivity, claim-capped, greedy in-sample unconstrained vs today ===")
for k,Mx in CM.items():
    traj,Hs2,Ds2 = greedy2(Mx, 10**9, 0, EVAL, allow_dgpu=False)
    t0,_,_ = evaluate2(today(), empty(), Mx); t1,nb,_ = evaluate2(Hs2,Ds2,Mx); t2,_,_ = evaluate2(empty(),empty(),Mx)
    print(f"  {k:8s} mode2 {t2/1000:6.2f}  today {t0/1000:6.2f} -> oracle {t1/1000:6.2f} ms at {traj[-1][0]} slots (n_h avg {sum(nb)/40:.2f}); saving vs today {(t0-t1)/1000:5.2f} ms/tok")

# ---- box 2 side: LRU misses on the complement stream ----
print("\n=== box 2 side: per-layer LRU over the picks box 1 does NOT claim (warm half), misses/token ===")
def box2_misses(Hs, R_enc, R_dec, claim_cap=4):
    miss=0
    for l in range(N_LAYER):
        R = R_enc if l<20 else R_dec
        lru=collections.OrderedDict()
        for t in range(ntok):
            p=picks[t][l]
            res=[e for e in p if e in Hs[l]]; claimed=set(res[:claim_cap])
            for e in p:
                if e in claimed: continue
                if e in lru: lru.move_to_end(e)
                else:
                    if t>=H: miss+=1
                    lru[e]=1
                    if len(lru)>R: lru.popitem(last=False)
    return miss/(ntok-H)
for R_dec in (40, 80, 154):
    print(f"  box2 dec slots/layer={R_dec:3d} enc=268: box1 empty -> {box2_misses(empty(),268,R_dec):5.1f} miss/tok ; box1 today -> {box2_misses(today(),268,R_dec):5.1f} ; box1 oracle -> {box2_misses(Hs,268,R_dec):5.1f}")
print("  (each box-2 miss adds ~2.5-6.3 ms to that layer's box-2 leg, on the critical path)")
