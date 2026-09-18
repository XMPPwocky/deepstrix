import sys; sys.argv=[sys.argv[0]]
from sim import *
from sim2 import CM, evaluate2
import random
M = CM['lin73']

def online2(S_host, S_dgpu_per_layer=0, tau=128.0, theta=2.0, hyst_tokens=16, margin=0.25, a_max=6, lag=1,
            dgpu_layers=range(N_LAYER), M=M, seq=None, bins=None):
    """Windowed-LFU residency + claim cap (in M). Admission: score>=theta; victim = lowest score, hysteresis, margin."""
    seq = seq if seq is not None else picks
    decay=0.5**(1.0/tau); score={}
    def s_get(k,t):
        v,lt=score.get(k,(0.0,t)); return v*decay**(t-lt)
    def s_bump(k,t): score[k]=(s_get(k,t)+1.0,t)
    Hs=empty(); Ds=empty(); admit_t={}; pending=[]; free=S_host; stats=collections.Counter()
    cost_tok=[]; nh_tok=[]; nd_tok=[]
    for t in range(len(seq)):
        still=[]
        for (rt,l,e) in pending:
            if rt<=t: Hs[l].add(e)
            else: still.append((rt,l,e))
        pending=still
        tc=0.0; tn=0; td=0
        for l in range(N_LAYER):
            p=seq[t][l]; d=sum(1 for e in p if e in Ds[l]); nh=sum(1 for e in p if e in Hs[l] and e not in Ds[l])
            tc+=M.f(nh,d); tn+=nh; td+=d
            for e in p: s_bump((l,e),t)
        cost_tok.append(tc); nh_tok.append(tn/40); nd_tok.append(td/40)
        cands=[]
        for l in range(N_LAYER):
            for e in seq[t][l]:
                if e in Hs[l] or ((l,e) in admit_t and admit_t[(l,e)]+lag>t): continue
                sc=s_get((l,e),t)
                if sc>=theta: cands.append((sc,l,e))
        cands.sort(reverse=True); admitted=0
        for sc,l,e in cands:
            if admitted>=a_max: stats['throttled']+=1; break
            if free>0: free-=1
            else:
                best=None
                for l2 in range(N_LAYER):
                    for e2 in Hs[l2]:
                        if admit_t.get((l2,e2),-10**9)+hyst_tokens>t: continue
                        v=s_get((l2,e2),t)
                        if best is None or v<best[0]: best=(v,l2,e2)
                if best is None: break
                v,l2,e2=best
                if sc<=v*(1+margin): stats['rejected']+=1; continue
                Hs[l2].discard(e2); Ds[l2].discard(e2); stats['evict']+=1
            pending.append((t+lag,l,e)); admit_t[(l,e)]=t; admitted+=1; stats['admit']+=1
        if S_dgpu_per_layer>0 and t%8==0:
            for l in dgpu_layers:
                ranked=sorted(Hs[l],key=lambda e:-s_get((l,e),t))[:S_dgpu_per_layer]; cur=Ds[l]
                for e in ranked:
                    if e in cur: continue
                    if len(cur)<S_dgpu_per_layer: cur.add(e); stats['dgpu_promote']+=1; continue
                    weakest=min(cur,key=lambda x:s_get((l,x),t))
                    if s_get((l,e),t)>1.5*s_get((l,weakest),t): cur.discard(weakest); cur.add(e); stats['dgpu_promote']+=1
    return dict(cost_tok=cost_tok, nh_tok=nh_tok, nd_tok=nd_tok, stats=stats, Hs=Hs, Ds=Ds, used=sum(len(h) for h in Hs))

def show(name,r):
    ct=r['cost_tok']; warm=sum(ct[H:])/len(ct[H:]); alls=sum(ct)/len(ct)
    print(f"{name:48s} used={r['used']:5d} | warm {warm/1000:6.2f} ms/tok  all {alls/1000:6.2f} | n_h {sum(r['nh_tok'][H:])/len(ct[H:]):.2f} n_d {sum(r['nd_tok'][H:])/len(ct[H:]):.2f} | admits/tok {r['stats']['admit']/len(ct):.1f} evict {r['stats']['evict']} rej {r['stats']['rejected']} thr {r['stats']['throttled']} dgpu_prom {r['stats']['dgpu_promote']}")

print("=== ONLINE windowed-LFU + claim cap 4 (async admission, lag 1, <=6 admits/token) ===")
for S in (256, 409, 512, 1024, 1536, 2969):
    show(f"WLFU host={S}", online2(S))
print("--- + dGPU 6/layer all layers ---")
for S in (409, 1024, 2969):
    show(f"WLFU host={S} + dgpu 6/layer", online2(S, 6))
print("--- zero-window-cost: host 409 for decoder layers only (enc via windows: assume enc top-116 static) + dGPU 12/dec-layer ---")
# emulate: encoder layers pre-resident (static top-116 fit on first half), host budget for decoder only
def online_dec_only(S_host, dg, enc_fit):
    fr=freq_tables(enc_fit)
    r=online2(S_host, dg, dgpu_layers=range(20,40))
    # recompute costs with encoder layers replaced by static top-116 (claim-capped)
    Hs=r['Hs']; Ds=r['Ds']
    for l in range(20): Hs[l]=set(e for e,_ in fr[l].most_common(116)); Ds[l]=set()
    return Hs,Ds
# NOTE: online2 spent host budget on all 40 layers; restrict by giving it decoder-only sequences
seq_dec=[[ (picks[t][l] if l>=20 else []) for l in range(N_LAYER)] for t in range(ntok)]
for S,dg in ((409,0),(409,12),(939,12)):
    r=online2(S, dg, dgpu_layers=range(20,40), seq=seq_dec)
    Hs=r['Hs']; Ds=r['Ds']; fr=freq_tables(range(0,H))
    for l in range(20): Hs[l]=set(e for e,_ in fr[l].most_common(116)); Ds[l]=set()
    tot,nb,nd=evaluate2(Hs,Ds,M)
    print(f"enc static top-116(fit first half) + WLFU dec host={S} dgpu={dg}/dec-layer{'':8s} used={r['used']:5d} | warm {tot/1000:6.2f} ms/tok  n_h {sum(nb)/40:.2f} (enc {sum(nb[:20])/20:.2f} dec {sum(nb[20:])/20:.2f}) n_d {sum(nd)/40:.2f} | tok/s {1000/(61.0-25.4+tot/1000):5.2f}  admits/tok {r['stats']['admit']/ntok:.1f}")

print("\n--- parameter sensitivity at host=1024 ---")
for tau in (32, 64, 256, 1e9): show(f"  tau={tau}", online2(1024, tau=tau))
for theta in (1.0, 1.5, 3.0): show(f"  theta={theta}", online2(1024, theta=theta))
for amax in (2, 12, 100): show(f"  a_max={amax}", online2(1024, a_max=amax))
show("  lag=4", online2(1024, lag=4)); show("  margin=0", online2(1024, margin=0.0)); show("  hyst=0", online2(1024, hyst_tokens=0))

# ---- domain shift: second half with a per-layer random permutation of expert ids (total shift, worst case) ----
print("\n=== DOMAIN SHIFT test: tokens 0..127 original, 128..255 with expert ids permuted per layer (total shift) ===")
random.seed(1)
perm=[list(range(N_EXP)) for _ in range(N_LAYER)]
for l in range(N_LAYER): random.shuffle(perm[l])
shifted=[[ (picks[t][l] if t<H else [perm[l][e] for e in picks[t][l]]) for l in range(N_LAYER)] for t in range(ntok)]
def bins_of(ct, w=32): return [sum(ct[i:i+w])/w/1000 for i in range(0,len(ct),w)]
# static table fit on first half, evaluated on shifted
fr=freq_tables(range(0,H)); Hst=static_topn(fr,[40]*40)
ct_static=[]
for t in range(ntok):
    tc=0.0
    for l in range(N_LAYER):
        p=shifted[t][l]; nh=sum(1 for e in p if e in Hst[l]); tc+=M.f(nh,0)
    ct_static.append(tc)
print("  bin(32 tok):        ", ' '.join(f"{i*32:4d}" for i in range(8)))
print("  static top-40 (fit A):", ' '.join(f"{x:5.2f}" for x in bins_of(ct_static)))
for tau in (32, 128, 512):
    r=online2(1600, tau=tau, seq=shifted); print(f"  WLFU 1600 tau={tau:4d}:   ", ' '.join(f"{x:5.2f}" for x in bins_of(r['cost_tok'])), f" admits/tok {r['stats']['admit']/ntok:.1f}")
r=online2(1600, tau=128, a_max=12, seq=shifted); print(f"  WLFU 1600 tau=128 a12:", ' '.join(f"{x:5.2f}" for x in bins_of(r['cost_tok'])), f" admits/tok {r['stats']['admit']/ntok:.1f}")
r=online2(1600, tau=128, theta=1.0, seq=shifted); print(f"  WLFU 1600 tau=128 th1:", ' '.join(f"{x:5.2f}" for x in bins_of(r['cost_tok'])), f" admits/tok {r['stats']['admit']/ntok:.1f}")
r=online2(1600, seq=picks); print(f"  WLFU 1600 NO shift:   ", ' '.join(f"{x:5.2f}" for x in bins_of(r['cost_tok'])))
