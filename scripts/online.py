import collections, math, sys
sys.argv=[sys.argv[0]]
from sim import *

M = MODELS['lin73']

def online(S_host, S_dgpu_per_layer=0, tau=128.0, theta=2.0, n_cap=3.8, hyst_tokens=16, margin=0.25,
           a_max=6, lag=1, evict_over=0.4, ncap_tau=32.0, M=M, toks_eval=EVAL, dgpu_layers=range(N_LAYER), verbose=False):
    """Kink-capped windowed-LFU with async admission. Returns dict of results."""
    decay = 0.5 ** (1.0/tau)
    score = {}      # (l,e) -> (value, last_t)
    def s_get(k, t):
        v, lt = score.get(k, (0.0, t))
        return v * decay ** (t - lt)
    def s_bump(k, t):
        score[k] = (s_get(k, t) + 1.0, t)
    Hs = empty(); Ds = empty()
    admit_t = {}    # (l,e) -> token admitted (resident from admit_t+lag)
    pending = []    # (ready_t, l, e)
    free = S_host
    nbar = [0.0]*N_LAYER; a_n = 0.5 ** (1.0/ncap_tau)
    stats = collections.Counter()
    cost_eval = 0.0; cost_all = 0.0; nh_eval = [0.0]*N_LAYER; nd_eval=[0.0]*N_LAYER
    n_eval = 0
    for t in range(ntok):
        # activate pending admissions
        still = []
        for (rt, l, e) in pending:
            if rt <= t: Hs[l].add(e)
            else: still.append((rt,l,e))
        pending = still
        tok_cost = 0.0
        for l in range(N_LAYER):
            p = picks[t][l]
            nh = sum(1 for e in p if e in Hs[l]); d = sum(1 for e in p if e in Ds[l])
            c = M.f(nh, d); tok_cost += c
            if t in toks_eval: nh_eval[l] += nh; nd_eval[l] += d
            for e in p: s_bump((l,e), t)
            nbar[l] = a_n*nbar[l] + (1-a_n)*(nh+d)
        cost_all += tok_cost
        if t in toks_eval: cost_eval += tok_cost; n_eval += 1
        # ---- admission pass ----
        admitted = 0
        # over-kink demotion: layers whose nbar exceeds cap -> evict their coldest resident
        for l in range(N_LAYER):
            if nbar[l] > n_cap + evict_over and Hs[l]:
                victim = min(Hs[l], key=lambda e: s_get((l,e), t))
                Hs[l].discard(victim); free += 1; stats['demote_overkink'] += 1
        cands = []
        for l in range(N_LAYER):
            if nbar[l] >= n_cap: continue
            for e in picks[t][l]:
                if e in Hs[l] or e in Ds[l] or (l,e) in admit_t and admit_t[(l,e)] + lag > t: continue
                sc = s_get((l,e), t)
                if sc >= theta: cands.append((sc, l, e))
        cands.sort(reverse=True)
        for sc, l, e in cands:
            if admitted >= a_max: stats['admit_throttled'] += 1; break
            if free > 0:
                free -= 1
            else:
                # victim: lowest score resident, not recently admitted; prefer over-cap layers
                best = None
                for l2 in range(N_LAYER):
                    for e2 in Hs[l2]:
                        if admit_t.get((l2,e2), -10**9) + hyst_tokens > t: continue
                        v = s_get((l2,e2), t)
                        key = (0 if nbar[l2] > n_cap else 1, v)
                        if best is None or key < best[0]: best = (key, l2, e2)
                if best is None: stats['no_victim'] += 1; break
                (_, v), l2, e2 = best
                if sc <= v * (1+margin): stats['admit_rejected_margin'] += 1; continue
                Hs[l2].discard(e2); stats['evict'] += 1
                if (l2,e2) in Ds[l2]: pass
            pending.append((t+lag, l, e)); admit_t[(l,e)] = t; admitted += 1; stats['admit'] += 1
        # ---- dGPU tier: top-k per layer by score among host-resident, with hysteresis ----
        if S_dgpu_per_layer > 0 and t % 8 == 0:
            for l in dgpu_layers:
                ranked = sorted(Hs[l], key=lambda e: -s_get((l,e), t))[:S_dgpu_per_layer]
                cur = Ds[l]
                # swap only if new candidate beats the weakest current by 1.5x
                want = set(ranked)
                for e in list(cur):
                    if e not in Hs[l]: cur.discard(e); stats['dgpu_drop']+=1
                for e in ranked:
                    if e in cur: continue
                    if len(cur) < S_dgpu_per_layer: cur.add(e); stats['dgpu_promote'] += 1; continue
                    weakest = min(cur, key=lambda x: s_get((l,x), t))
                    if s_get((l,e), t) > 1.5 * s_get((l,weakest), t):
                        cur.discard(weakest); cur.add(e); stats['dgpu_promote'] += 1
    res = dict(cost_eval=cost_eval/max(n_eval,1), cost_all=cost_all/ntok, nh=[x/max(n_eval,1) for x in nh_eval],
               nd=[x/max(n_eval,1) for x in nd_eval], used=sum(len(h) for h in Hs), stats=stats, Hs=Hs, Ds=Ds)
    return res

def show(name, r):
    nh = r['nh']; nd = r['nd']
    print(f"{name:52s} used={r['used']:5d} | warm {r['cost_eval']/1000:6.2f} ms/tok  all-256 {r['cost_all']/1000:6.2f} | n_h {sum(nh)/40:.2f} (enc {sum(nh[:20])/20:.2f} dec {sum(nh[20:])/20:.2f}) n_d {sum(nd)/40:.2f} | admits/tok {r['stats']['admit']/ntok:.1f} evict {r['stats']['evict']} demote {r['stats']['demote_overkink']} throttled {r['stats']['admit_throttled']}")

print("=== online kink-capped windowed-LFU (async admission, lag 1 token, <=6 admits/token) ===")
for S in (256, 512, 1024, 1536, 2048, 2969):
    show(f"KLFU host={S}", online(S))
print("\n--- with dGPU tier 6/layer (all layers) ---")
for S in (256, 512, 1024, 2969):
    show(f"KLFU host={S} + dgpu 6/layer", online(S, 6))
print("\n--- parameter sensitivity at host=1024 ---")
for tau in (32, 64, 256, 1e9):
    show(f"  tau={tau}", online(1024, tau=tau))
for theta in (1.0, 1.5, 3.0):
    show(f"  theta={theta}", online(1024, theta=theta))
for ncap in (3.0, 3.4, 4.2, 6.0):
    show(f"  n_cap={ncap}", online(1024, n_cap=ncap))
for amax in (2, 12, 100):
    show(f"  a_max={amax}", online(1024, a_max=amax))
show(f"  lag=4 tokens", online(1024, lag=4))
print("\n--- no kink cap, plain LFU with same budget (what 'maximise hit rate' does) ---")
for S in (1024, 2969):
    show(f"LFU no cap host={S}", online(S, n_cap=99.0))
print("\n--- first-touch LRU (today's pager semantics: admit on first touch, evict LRU) for reference ---")
def first_touch_lru(S):
    lru = collections.OrderedDict(); Hs = empty()
    cost_eval=0.0; n_eval=0; nh_eval=[0.0]*N_LAYER
    for t in range(ntok):
        tc=0.0
        for l in range(N_LAYER):
            p=picks[t][l]; nh=sum(1 for e in p if (l,e) in lru); tc+=M.f(nh,0)
            if t in EVAL: nh_eval[l]+=nh
            for e in p:
                k=(l,e)
                if k in lru: lru.move_to_end(k)
                else:
                    lru[k]=1
                    if len(lru)>S: lru.popitem(last=False)
        if t in EVAL: cost_eval+=tc; n_eval+=1
    return cost_eval/n_eval, [x/n_eval for x in nh_eval]
for S in (25, 256, 1024, 2969):
    c, nh = first_touch_lru(S)
    print(f"first-touch LRU host={S:5d} | warm {c/1000:6.2f} ms/tok  n_h {sum(nh)/40:.2f} (enc {sum(nh[:20])/20:.2f} dec {sum(nh[20:])/20:.2f})")
