import sys; sys.argv=[sys.argv[0]]
from sim import *
M = MODELS['lin73']
print("=== greedy oracle allocation shape under tight budgets (in-sample warm half) ===")
for S in (512, 1024, 2506):
    traj, Hs, Ds = greedy(M, S, 0, EVAL, allow_dgpu=False)
    tot, nbar, ph, nd = evaluate(Hs, Ds, M)
    print(f"\nS={S}: phase {tot/1000:.2f} ms/tok")
    print("  layer :", ' '.join(f"{l:3d}" for l in range(N_LAYER)))
    print("  |H|   :", ' '.join(f"{len(Hs[l]):3d}" for l in range(N_LAYER)))
    print("  n_bar :", ' '.join(f"{nbar[l]:3.1f}" for l in range(N_LAYER)))
    # marginal pick-probability of the last-admitted expert per layer (frequency rank)
    fr = freq_tables(EVAL)
    pmin = []
    for l in range(N_LAYER):
        if Hs[l]: pmin.append(min(fr[l][e] for e in Hs[l])/len(EVAL))
        else: pmin.append(0)
    print("  p_min :", ' '.join(f"{x:3.2f}" for x in pmin), f"  (mean {sum(pmin)/40:.3f}, spread {min(pmin):.3f}-{max(pmin):.3f})")
    # distribution of n_t across tokens (all layers)
    dist = collections.Counter()
    for l in range(N_LAYER):
        for t in EVAL: dist[sum(1 for e in picks[t][l] if e in Hs[l])] += 1
    totn = sum(dist.values())
    print("  P(n_t):", ' '.join(f"n{k}={dist[k]/totn:.2f}" for k in range(7)))

print("\n=== marginal value curve of host slots (in-sample greedy, ms/token saved per 128-slot window) ===")
traj, Hs, Ds = greedy(M, 10**9, 0, EVAL, allow_dgpu=False)
byslot = {s:tot for s,d,tot in traj}
prev = None
for s in (0, 128, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2506):
    v = min(byslot[k] for k in byslot if k <= s)
    if prev is not None:
        print(f"  {prev[0]:4d}->{s:4d} slots: {(prev[1]-v)/1000:5.2f} ms/tok total, {(prev[1]-v)/1000/((s-prev[0])/128):5.2f} ms/tok per window")
    prev = (s, v)

print("\n=== zero-window-cost placement: encoder hot set INSIDE the prefill windows, decoder on dGPU+rotating ===")
fr_fit = freq_tables(range(0,H)); fr_in = freq_tables(EVAL)
def enc_variant(fr, k=116):
    Hs = empty()
    for l in range(20): Hs[l] = set(e for e,_ in fr[l].most_common(k))
    return Hs
for label, fr in (("in-sample", fr_in), ("fit 0..127", fr_fit)):
    Hs = enc_variant(fr)
    report(f"enc top-116 [{label}] windows only, dec nothing", Hs, empty(), M)
    # decoder: greedy under 409 host slots (25 LRU + 384 rotating) + dGPU 12/layer on layers 20-39 only
    traj, Hd, Dd = greedy(M, 409, 12, EVAL if label=="in-sample" else range(0,H))
    # keep only decoder layers from the greedy, drop encoder entries
    for l in range(20): Hd[l] = Hs[l]; Dd[l] = set()
    report(f"enc top-116 [{label}] + dec greedy 409 host + dgpu 12/dec-layer", Hd, Dd, M)
    traj, Hd, Dd = greedy(M, 409, 0, EVAL if label=="in-sample" else range(0,H), allow_dgpu=False)
    for l in range(20): Hd[l] = Hs[l]
    report(f"enc top-116 [{label}] + dec greedy 409 host only", Hd, Dd, M)
# and today's contiguous share for comparison on encoder layers
Hs = today_uniform116()
traj, Hd, Dd = greedy(M, 409, 12, EVAL)
for l in range(20): Hd[l] = Hs[l]; Dd[l] = set()
report("enc contiguous ids<116 (today) + dec greedy 409 host + dgpu 12/dec-layer", Hd, Dd, M)

print("\n=== Jensen: E[f(n_t)] vs f(E[n]) — how much the per-token randomness of n costs ===")
for name, (Hs, Ds) in (("oracle unconstrained", greedy(M, 10**9, 0, EVAL, allow_dgpu=False)[1:]), ("static top-40", (static_topn(fr_in,[40]*40), empty()))):
    tot, nbar, ph, nd = evaluate(Hs, Ds, M)
    fbar = sum(M.f(int(round(n))) for n in nbar)
    # continuous interpolation of f at E[n]
    def fcont(x):
        a=int(math.floor(x)); b=min(a+1,6); w=x-a
        return M.f(a)*(1-w)+M.f(b)*w
    fc = sum(fcont(n) for n in nbar)
    print(f"  {name:22s}: E[f(n_t)] = {tot/1000:.2f} ms, f(E[n]) interp = {fc/1000:.2f} ms  -> Jensen gap {(tot-fc)/1000:.2f} ms/tok")
