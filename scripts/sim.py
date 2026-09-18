import array, collections, heapq, math, sys
N_LAYER, TOPK, N_EXP = 40, 6, 384
buf = array.array('H'); buf.frombytes(open('/home/claude-code/.cache/deepstrix/v41/expert_trace_v41_decode313.bin','rb').read())
per_tok = N_LAYER*TOPK; ntok = len(buf)//per_tok
picks = [[list(buf[t*per_tok + l*TOPK : t*per_tok + l*TOPK + TOPK]) for l in range(N_LAYER)] for t in range(ntok)]
H = ntok//2   # eval = warm half
EVAL = range(H, ntok)

# ---------------- cost models ----------------
def box2_lin(m, b=202.0, c=72.0): return 0.0 if m == 0 else b + c*m
B2M = {0:0.0, 1:274.0, 2:315.5, 3:357.0, 4:449.7, 5:542.3, 6:635.0}   # loopback measured k=1,3,6; linear between
def box2_meas(m): return B2M[m]
class Model:
    def __init__(self, name, c1=73.0, box2=box2_lin, dg=29.0, shared=96.0):
        self.name=name; self.c1=c1; self.box2=box2; self.dg=dg; self.shared=shared
        self.tab = {}
        for nh in range(7):
            for d in range(7-nh):
                m = 6-nh-d
                self.tab[(nh,d)] = max(c1*nh, shared + dg*d, box2(m))
    def f(self, nh, d=0): return self.tab[(nh,d)]
MODELS = {
  'lin73':   Model('lin73'),
  'meas73':  Model('meas73', box2=box2_meas),
  'lin98':   Model('lin98', c1=98.0),
  'meas98':  Model('meas98', c1=98.0, box2=box2_meas),
  'lin73_x': Model('lin73_x', box2=lambda m: box2_lin(m, 250.0, 72.0)),  # +45us cross-box link on the intercept
}
def show_f(M):
    return ' '.join(f"n{n}:{M.f(n):.0f}" for n in range(7))
for k,M in MODELS.items(): print(f"{k:8s} f(n) = {show_f(M)}   argmin n={min(range(7), key=M.f)}")

# ---------------- evaluation ----------------
def evaluate(Hset, Dset, M, toks=EVAL):
    """Hset/Dset: list per layer of sets. Returns (total_us_per_token, per-layer mean n_h, per-layer mean phase, E[f] - f(E[n]) jensen gap)"""
    tot = 0.0; nbar = [0.0]*N_LAYER; ph = [0.0]*N_LAYER; nd = [0.0]*N_LAYER
    nt = len(toks)
    for l in range(N_LAYER):
        hs, ds = Hset[l], Dset[l]
        s = 0.0; sn = 0.0; sd = 0.0
        for t in toks:
            p = picks[t][l]
            nh = sum(1 for e in p if e in hs); d = sum(1 for e in p if e in ds)
            s += M.f(nh, d); sn += nh; sd += d
        ph[l] = s/nt; nbar[l] = sn/nt; nd[l] = sd/nt; tot += s/nt
    return tot, nbar, ph, nd

def report(name, Hset, Dset, M, toks=EVAL, detail=False):
    tot, nbar, ph, nd = evaluate(Hset, Dset, M, toks)
    slots = sum(len(h) for h in Hset); dsl = sum(len(d) for d in Dset)
    jens = sum(ph[l] - M.f(round(nbar[l])) for l in range(N_LAYER))/N_LAYER
    print(f"{name:44s} host={slots:5d} dgpu={dsl:4d} | phase {tot/1000:6.2f} ms/tok  n_h avg {sum(nbar)/40:.2f} (enc {sum(nbar[:20])/20:.2f} dec {sum(nbar[20:])/20:.2f})  n_d avg {sum(nd)/40:.2f} | tok/s {1000/(61.0-25.4+tot/1000):5.2f}")
    if detail:
        print("   per-layer n_h:", ' '.join(f"{x:.1f}" for x in nbar))
        print("   per-layer |H|:", ' '.join(f"{len(h)}" for h in Hset))
    return tot

# ---------------- baselines ----------------
def empty(): return [set() for _ in range(N_LAYER)]
# "today": encoder layers hold ids<116 touched in the trace (stand-in for prefill-touched window contents);
# decoder layers: a 25-slot global first-touch LRU (the rotating window ignored)
def today():
    Hs = empty()
    for l in range(20):
        touched = set(e for t in range(ntok) for e in picks[t][l] if e < 116)
        Hs[l] = touched
    lru = collections.OrderedDict()
    # first-touch LRU over decoder layers (ensure semantics: only free slots are filled under catch-all -> after fill, frozen)
    for t in range(ntok):
        for l in range(20, 40):
            for e in picks[t][l]:
                k=(l,e)
                if k in lru: continue
                if len(lru) < 25: lru[k]=1
    for (l,e) in lru: Hs[l].add(e)
    return Hs
def today_uniform116():
    Hs = empty()
    for l in range(20): Hs[l] = set(range(116))
    return Hs

# static top-N by frequency, fit on tokens in `fit`
def freq_tables(fit):
    fr = [collections.Counter() for _ in range(N_LAYER)]
    for t in fit:
        for l in range(N_LAYER): fr[l].update(picks[t][l])
    return fr
def static_topn(fr, n_per_layer):
    return [set(e for e,_ in fr[l].most_common(n_per_layer[l])) for l in range(N_LAYER)]

# ---------------- greedy oracle under the max() objective ----------------
def greedy(M, S_host, S_dgpu_per_layer, toks, fr_hint=None, allow_dgpu=True):
    """lazy greedy over (layer, expert, tier). Returns trajectory [(slots_host, slots_dgpu, total_us)] and final sets."""
    toks = list(toks); nt=len(toks)
    # per (l,e): token indices where picked
    where = collections.defaultdict(list)
    for i,t in enumerate(toks):
        for l in range(N_LAYER):
            for e in picks[t][l]: where[(l,e)].append(i)
    nh = [[0]*nt for _ in range(N_LAYER)]; nd = [[0]*nt for _ in range(N_LAYER)]
    Hs = empty(); Ds = empty()
    def gain(l,e,tier):
        g = 0.0
        for i in where[(l,e)]:
            a,b = nh[l][i], nd[l][i]
            g += M.f(a,b) - (M.f(a+1,b) if tier=='h' else M.f(a,b+1))
        return g/nt
    heap = []
    for (l,e) in where:
        heapq.heappush(heap, (-gain(l,e,'h'), l, e, 'h'))
        if allow_dgpu: heapq.heappush(heap, (-gain(l,e,'d'), l, e, 'd'))
    used_h = 0; used_d = [0]*N_LAYER
    base = sum(M.f(0,0) for _ in range(N_LAYER))
    total = base
    traj = [(0,0,total)]
    placed = set()
    while heap:
        ng, l, e, tier = heapq.heappop(heap)
        if (l,e) in placed: continue
        if tier=='h' and used_h >= S_host: continue
        if tier=='d' and used_d[l] >= S_dgpu_per_layer: continue
        g = gain(l,e,tier)
        if heap and g < -heap[0][0] - 1e-9:
            heapq.heappush(heap, (-g, l, e, tier)); continue
        if g <= 0: break
        placed.add((l,e))
        if tier=='h':
            Hs[l].add(e); used_h += 1
            for i in where[(l,e)]: nh[l][i] += 1
        else:
            Ds[l].add(e); used_d[l] += 1
            for i in where[(l,e)]: nd[l][i] += 1
        total -= g
        traj.append((used_h, sum(used_d), total))
    return traj, Hs, Ds

if __name__ == '__main__':
    M = MODELS['lin73']
    print("\n=== baselines (eval = warm half, tokens 128..255; tok/s assumes 61 ms/token today with a 25.4 ms expert phase) ===")
    report("today (ids<116 touched on enc, 25-slot LRU dec)", today(), empty(), M)
    report("today, enc windows = all ids<116", today_uniform116(), empty(), M)
    report("nothing local (mode 2)", empty(), empty(), M)
    report("everything local (n=6)", [set(range(N_EXP)) for _ in range(N_LAYER)], empty(), M)
    fr_in = freq_tables(EVAL); fr_fit = freq_tables(range(0,H))
    print("\n=== static top-N per layer (uniform N), in-sample vs split-sample ===")
    for n in (6, 12, 24, 32, 40, 48, 64, 96):
        report(f"static top-{n}/layer IN-SAMPLE", static_topn(fr_in, [n]*N_LAYER), empty(), M)
        report(f"static top-{n}/layer fit on 0..127", static_topn(fr_fit, [n]*N_LAYER), empty(), M)
    print("\n=== greedy oracle (in-sample on warm half) under global host budget, no dGPU ===")
    traj, Hs, Ds = greedy(M, 10**9, 0, EVAL, allow_dgpu=False)
    marks = [25, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 2969]
    for s,d,tot in traj:
        if s in marks: print(f"  host slots {s:5d} ({s*18.8/1000:5.1f} GB, {s/128:4.1f} windows) -> phase {tot/1000:6.2f} ms/tok  tok/s {1000/(61.0-25.4+tot/1000):5.2f}")
    print(f"  greedy stops at {traj[-1][0]} slots (marginal gain <= 0): phase {traj[-1][2]/1000:.2f} ms/tok")
    report("greedy oracle, unconstrained", Hs, Ds, M, detail=True)
    print("\n=== greedy oracle in-sample WITH dGPU tier (6/layer) ===")
    for S in (25, 256, 512, 1024, 10**9):
        traj, Hs, Ds = greedy(M, S, 6, EVAL)
        report(f"greedy host<={S if S<10**9 else 'inf'} + dgpu 6/layer", Hs, Ds, M)
    print("\n=== split-sample greedy: fit on 0..127, score on 128..255 ===")
    for S in (256, 512, 1024, 10**9):
        traj, Hs, Ds = greedy(M, S, 0, range(0,H), allow_dgpu=False)
        report(f"greedy fit 0..127 host<={S if S<10**9 else 'inf'}", Hs, Ds, M)
    traj, Hs, Ds = greedy(M, 1024, 6, range(0,H))
    report("greedy fit 0..127 host<=1024 + dgpu 6/layer", Hs, Ds, M)
    print("\n=== model sensitivity (greedy in-sample, unconstrained host, no dGPU) ===")
    for k,Mx in MODELS.items():
        traj, Hs, Ds = greedy(Mx, 10**9, 0, EVAL, allow_dgpu=False)
        t0,_,_,_ = evaluate(today(), empty(), Mx)
        t1,nb,_,_ = evaluate(Hs, Ds, Mx)
        print(f"  {k:8s} today {t0/1000:6.2f} ms -> oracle {t1/1000:6.2f} ms at {traj[-1][0]} slots (n_h avg {sum(nb)/40:.2f}); saving {(t0-t1)/1000:5.2f} ms/tok")
