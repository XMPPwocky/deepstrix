#!/usr/bin/env python3
"""What does prediction actually BUY, once a wrong guess is charged for?

A prefetch does not remove a fetch (see belady_bound.py -- at fixed capacity
OPT bounds the fetch COUNT). It moves one off the critical path. So the payoff
is exposed latency, and a wrong guess is charged twice:
  1. its bytes raise drive utilisation, inflating EVERY remaining demand miss
     through the queue term 1/(1-rho);
  2. it takes a slot and evicts something, which can CREATE demand misses.

Both are simulated here rather than assumed: prefetched experts are admitted to
the same LRU and evict exactly like demand fills, and the resulting demand-miss
count is measured against the no-prefetch baseline on the same trace.
"""
import sys, collections

N_LAYER, N_EXPERT = 40, 384
BOX1_MILLI, SLOTS = 397, 4070
T_TOKEN_MS, C_MISS_MS = 97.0, 8.85          # live heartbeat p50
C_SERVICE_MS = C_MISS_MS * (1 - (1.67 * C_MISS_MS / T_TOKEN_MS))

def box2(layer, e):
    return ((layer * 1000003 + e * 7919) % 1000) >= BOX1_MILLI

def tokens(path):
    toks, cur = [], []
    for line in open(path):
        p = line.split()
        if not p or p[0] != 'D':
            continue
        layer = int(p[1])
        ids = tuple(int(x) for x in p[2:] if int(x) >= 0)
        if layer == 0 and cur:
            if len(cur) == N_LAYER:
                toks.append(cur)
            cur = []
        cur.append(ids)
    if len(cur) == N_LAYER:
        toks.append(cur)
    return toks

# Layers reached before a prefetch issued at token start can complete are NOT
# hideable: layer L is reached at (L/40)*T, and a read takes C_MISS_MS.
HIDEABLE = [L for L in range(N_LAYER) if (L / N_LAYER) * T_TOKEN_MS >= C_MISS_MS]

def run(toks, predict, warm=2000):
    """predict(t, tok_prev, cache, marg) -> {(layer, expert)} to prefetch at the
    START of token t. Prefetched entries are admitted (and evict) like any fill."""
    cache = collections.OrderedDict()
    marg = [collections.Counter() for _ in range(N_LAYER)]
    demand = pref_issued = pref_used = 0
    toks_counted = 0
    for t, tk in enumerate(toks):
        if predict is not None and t > 0:
            for (L, e) in predict(t, toks[t - 1], cache, marg):
                if (L, e) in cache:
                    continue
                pref_issued += 1
                if len(cache) >= SLOTS:
                    cache.popitem(last=False)
                cache[(L, e)] = 1
                if t >= warm and L in HIDEABLE and e in tk[L] and not box2(L, e):
                    pref_used += 1
        counted = t >= warm
        if counted:
            toks_counted += 1
        for L, ids in enumerate(tk):
            for e in ids:
                if box2(L, e):
                    continue
                k = (L, e)
                if k in cache:
                    cache.move_to_end(k)
                else:
                    if counted:
                        demand += 1
                    if len(cache) >= SLOTS:
                        cache.popitem(last=False)
                    cache[k] = 1
            marg[L].update(ids)
    return demand, pref_issued, pref_used, toks_counted

def report(name, base_m, d, iss, used, n):
    m = d / n                      # demand misses per token, still EXPOSED
    pf = iss / n                   # extra reads per token
    rho = (m + pf) * C_SERVICE_MS / T_TOKEN_MS
    if rho >= 1.0:
        print(f"{name:<34} SATURATES the drive (rho={rho:.2f})")
        return
    cost = m * C_SERVICE_MS / (1 - rho)
    print(f"{name:<34} demand/tok {m:>5.2f}  prefetch/tok {pf:>7.2f}  "
          f"prec {used/max(1,iss):>6.1%}  rho {rho:>5.2f}  "
          f"exposed {cost:>6.2f} ms  vs base {base_m:>6.2f}  "
          f"{'+' if cost<base_m else ''}{100*(base_m-cost)/base_m:>5.1f}%")

toks = tokens(sys.argv[1])
print(f"tokens {len(toks):,}   hideable layers {HIDEABLE[0]}..39 "
      f"({len(HIDEABLE)}/40 have >= {C_MISS_MS} ms of lead from token start)")
print(f"service time c_s = {C_SERVICE_MS:.2f} ms (from c={C_MISS_MS} at rho_d=0.15)\n")

d, i, u, n = run(toks, None)
m0 = d / n
rho0 = m0 * C_SERVICE_MS / T_TOKEN_MS
base = m0 * C_SERVICE_MS / (1 - rho0)
print(f"{'BASELINE (no prefetch)':<34} demand/tok {m0:>5.2f}  "
      f"rho {rho0:>5.2f}  exposed {base:>6.2f} ms\n")

# --- oracle: prefetch EXACTLY what this token will miss -------------------
def oracle(t, prev, cache, marg):
    return [(L, e) for L, ids in enumerate(toks[t]) for e in ids
            if not box2(L, e) and (L, e) not in cache]
report("ORACLE (perfect, p=1)", base, *run(toks, oracle))

# --- realistic: last token's picks (recency) -----------------------------
def recency(t, prev, cache, marg):
    return [(L, e) for L, ids in enumerate(prev) for e in ids if not box2(L, e)]
report("recency (last token's picks)", base, *run(toks, recency))

# --- realistic: top-k by frequency at each layer -------------------------
for k in (2, 8, 32):
    def freq(t, prev, cache, marg, k=k):
        out = []
        for L in range(N_LAYER):
            for e, _ in marg[L].most_common(k):
                if not box2(L, e):
                    out.append((L, e))
        return out
    report(f"frequency top-{k}/layer", base, *run(toks, freq))
