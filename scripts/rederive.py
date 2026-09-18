import sys; sys.argv=[sys.argv[0]]
from sim import *            # picks, EVAL, H, today, empty, freq_tables, static_topn, Model
from sim2 import ClaimModel, evaluate2, greedy2
import online2 as O          # online2() and the module-level runs print; suppress by importing quietly? (it prints) -> acceptable
def box2_lin_old(m): return 0.0 if m==0 else 202.0+72.0*m
def box2_corr(K):   return lambda m: 0.0 if m==0 else K+95.0*m
PTS = {1:177.0, 2:279.0, 3:370.0, 6:650.0}; PTS[4]=370+(650-370)/3; PTS[5]=370+2*(650-370)/3
def box2_meas(K_extra): return lambda m: 0.0 if m==0 else PTS[m]+K_extra      # measured GPU points + link/clock
grid = {}
for c1 in (73.0, 98.0):
    grid[f"c1={c1:.0f} old 202+72m"]        = Model('', c1=c1, box2=box2_lin_old)
    grid[f"c1={c1:.0f} corr 80+95m (GPU)"]  = Model('', c1=c1, box2=box2_corr(80.0))
    grid[f"c1={c1:.0f} corr 175+95m (prod)"]= Model('', c1=c1, box2=box2_corr(175.0))
    grid[f"c1={c1:.0f} meas pts +95 (prod)"]= Model('', c1=c1, box2=box2_meas(95.0))
def fcont(M, x):
    a=int(math.floor(x)); b=min(a+1,6); w=x-a; return M.f(a)*(1-w)+M.f(b)*w
fr_fit = freq_tables(range(0,H))
print(f"{'model':32s} | f(n) n=0..6                              | n* | mode2  today | oracle(claim) slots n_h  Jensen | split-greedy | top40-fit | prize(today-oracle) prize(mode2-oracle)")
for name, Mb in grid.items():
    M = ClaimModel(Mb)                       # claim-capped at the model's own argmin
    nstar = min(range(7), key=Mb.f)
    t_m2,_,_ = evaluate2(empty(), empty(), M)
    t_td,_,_ = evaluate2(today(), empty(), M)
    traj,Hs,Ds = greedy2(M, 10**9, 0, EVAL, allow_dgpu=False)
    t_or, nb, _ = evaluate2(Hs, Ds, M)
    jens = (t_or - sum(fcont(M, n) for n in nb))/1000
    traj2,Hs2,Ds2 = greedy2(M, 10**9, 0, range(0,H), allow_dgpu=False)
    t_sp,_,_ = evaluate2(Hs2, Ds2, M)
    t_40,_,_ = evaluate2(static_topn(fr_fit,[40]*40), empty(), M)
    fs = ' '.join(f"{Mb.f(n):4.0f}" for n in range(7))
    print(f"{name:32s} | {fs} | {nstar}  | {t_m2/1000:5.2f} {t_td/1000:6.2f} | {t_or/1000:6.2f} {traj[-1][0]:5d} {sum(nb)/40:4.2f} {jens:+5.2f} | {t_sp/1000:6.2f} | {t_40/1000:6.2f} | {(t_td-t_or)/1000:5.2f} {(t_m2-t_or)/1000:5.2f}")
print("\nJensen for the STATIC top-40 table (E[f] - f(E[n])), same grid:")
for name, Mb in grid.items():
    M = ClaimModel(Mb); Hs = static_topn(fr_fit,[40]*40)
    t,nb,_ = evaluate2(Hs, empty(), M); print(f"  {name:32s} {(t - sum(fcont(M,n) for n in nb))/1000:+5.2f} ms/tok")
print("\nUncapped (no claim rule) oracle vs capped, corrected prod constants, to show the cap's value:")
for name in ("c1=73 corr 175+95m (prod)", "c1=98 corr 175+95m (prod)"):
    Mb = grid[name]; Mu = ClaimModel(Mb, cap=False); Mc = ClaimModel(Mb)
    traj,Hs,Ds = greedy2(Mu, 10**9, 0, EVAL, allow_dgpu=False); tu,nbu,_ = evaluate2(Hs,Ds,Mu)
    traj,Hs,Ds = greedy2(Mc, 10**9, 0, EVAL, allow_dgpu=False); tc,nbc,_ = evaluate2(Hs,Ds,Mc)
    print(f"  {name}: uncapped {tu/1000:.2f} (n_h {sum(nbu)/40:.2f}) vs capped {tc/1000:.2f} (n_h {sum(nbc)/40:.2f})")
print("\nRecommended zero-window config + online WLFU at 1024, corrected prod constants (fit first half / online):")
for name in ("c1=73 corr 175+95m (prod)", "c1=98 corr 175+95m (prod)"):
    Mb = grid[name]; M = ClaimModel(Mb)
    r = O.online2(1024, M=M); ct=r['cost_tok']; print(f"  {name}: WLFU 1024 warm {sum(ct[H:])/len(ct[H:])/1000:.2f} ms/tok")
    fr=freq_tables(range(0,H)); seq_dec=[[ (picks[t][l] if l>=20 else []) for l in range(N_LAYER)] for t in range(ntok)]
    r=O.online2(409, 12, dgpu_layers=range(20,40), seq=seq_dec, M=M); Hs=r['Hs']; Ds=r['Ds']
    for l in range(20): Hs[l]=set(e for e,_ in fr[l].most_common(116)); Ds[l]=set()
    tot,nb,nd=evaluate2(Hs,Ds,M); print(f"  {name}: enc top-116(fit) + WLFU dec 409 + dGPU 12/dec: {tot/1000:.2f} ms/tok (n_h {sum(nb)/40:.2f}, n_d {sum(nd)/40:.2f})")
