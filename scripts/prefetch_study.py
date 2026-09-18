#!/usr/bin/env python3
"""Is expert prefetch plausible? Per-token analysis of a V41_PICK_TRACE.

analyze_picks.py answers cache-policy questions (Zipf, stack distance) but
explicitly skips cross-layer correlation because it never grouped rows into
tokens. Decode rows DO group cleanly (strict layer 0..39 per token), so this
script answers the question that matters for prefetch:

    a miss is prefetchable only if it is KNOWN before the router that
    selects it runs. That needs either (a) the same (layer,expert) to have
    been used recently enough to predict by recency, or (b) an EARLIER
    layer's picks in the SAME token to predict a later layer's picks.

Everything here is measured against the production partition (box 1 owns
39.7% of ids by hash) and box 1's real decode LRU capacity (4070 slots).
"""
import sys, collections

N_LAYER, N_EXPERT = 40, 384
BOX1_MILLI = 397          # from the server's own startup line
BOX1_SLOTS = 4070

def partition_box2(layer, e):
    return ((layer * 1000003 + e * 7919) % 1000) >= BOX1_MILLI

def load_tokens(path):
    """-> list of tokens; each token is a list of 40 tuples of expert ids."""
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

def sec(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)

def main():
    path = sys.argv[1]
    toks = load_tokens(path)
    T = len(toks)
    print(f"decode tokens {T:,}   layers {N_LAYER}   picks/layer {len(toks[0][0])}")

    # ---- 1. token-to-token overlap at the SAME layer ----------------------
    # The naive prefetch idea: "fetch what the last token used". Its ceiling
    # is this overlap. Baseline = expected overlap if consecutive tokens drew
    # independently from each layer's OWN empirical marginal, so a high number
    # means real temporal structure, not just a skewed distribution.
    sec("1. SAME-LAYER TOKEN-TO-TOKEN OVERLAP  (of 6 picks)")
    appear = [collections.Counter() for _ in range(N_LAYER)]
    for tk in toks:
        for L, ids in enumerate(tk):
            for e in ids:
                appear[L][e] += 1
    ov = [0] * N_LAYER
    for t in range(1, T):
        for L in range(N_LAYER):
            ov[L] += len(set(toks[t][L]) & set(toks[t - 1][L]))
    print(f"{'layer':>6} {'overlap':>8} {'chance':>8} {'lift':>7}")
    tot_ov = tot_ch = 0.0
    for L in range(N_LAYER):
        m = ov[L] / (T - 1)
        ch = sum((c / T) ** 2 for c in appear[L].values())
        tot_ov += m; tot_ch += ch
        if L < 4 or L in (19, 20, 21) or L >= 37:
            print(f"{L:>6} {m:>8.2f} {ch:>8.2f} {m/ch:>6.1f}x")
    print(f"{'ALL':>6} {tot_ov/N_LAYER:>8.2f} {tot_ch/N_LAYER:>8.2f} "
          f"{tot_ov/tot_ch:>6.1f}x   <- mean over all 40 layers")

    # ---- 2. does LRU already own that overlap? ---------------------------
    # Box 1 sees only its partition's share of the 240 accesses/token. Its
    # 4070 slots therefore retain a fixed number of TOKENS of history; any
    # reuse inside that horizon is already a hit, and prefetching it is a
    # no-op. This is the number that decides whether recency prefetch can
    # possibly do anything.
    sec("2. HOW MANY TOKENS OF HISTORY DOES BOX 1's LRU HOLD?")
    b1 = sum(1 for tk in toks[0] for e in tk if not partition_box2(0, e))
    acc_tok = sum(1 for L, ids in enumerate(toks[0]) for e in ids
                  if not partition_box2(L, e))
    print(f"box-1 accesses per token        {acc_tok}  (of 240)")
    print(f"box-1 decode slots             {BOX1_SLOTS}")
    print(f"=> retention horizon           {BOX1_SLOTS/max(1,acc_tok):.0f} tokens "
          f"of distinct-access history")

    # ---- 3. simulate box 1's LRU; how OLD is each miss? ------------------
    # For every real miss, how many tokens ago was that (layer,expert) last
    # used? If the answer is far beyond the horizon above, no recency-based
    # prefetcher can see it coming.
    sec("3. AGE OF EACH MISS  (tokens since that expert was last picked)")
    cache = collections.OrderedDict()
    last_tok = {}
    ages, misses, acc = [], 0, 0
    cold = 0
    miss_per_tok = [0] * T
    for t, tk in enumerate(toks):
        for L, ids in enumerate(tk):
            for e in ids:
                if partition_box2(L, e):
                    continue
                k = (L, e)
                acc += 1
                if k in cache:
                    cache.move_to_end(k)
                else:
                    misses += 1
                    miss_per_tok[t] += 1
                    if k in last_tok:
                        ages.append(t - last_tok[k])
                    else:
                        cold += 1
                    if len(cache) >= BOX1_SLOTS:
                        cache.popitem(last=False)
                    cache[k] = 1
                last_tok[k] = t
    print(f"box-1 accesses {acc:,}   misses {misses:,}  "
          f"({100*misses/acc:.2f}%)   = {misses/T:.2f} misses/token")
    print(f"  of those, COLD (never seen before): {cold:,} ({100*cold/misses:.1f}%)")
    print(f"  CAPACITY (seen before, evicted):    {len(ages):,} ({100*len(ages)/misses:.1f}%)")
    if ages:
        ages.sort()
        print("  age of a capacity miss, in tokens:")
        for p in (1, 5, 10, 25, 50, 75, 90, 99):
            print(f"    p{p:<2}: {ages[len(ages)*p//100]:>8,}")
        for w in (1, 2, 4, 8, 16, 32, 64):
            n = sum(1 for a in ages if a <= w)
            print(f"    within last {w:>3} tokens: {n:>7,}  "
                  f"({100*n/misses:>5.2f}% of ALL misses)")

    # ---- 4. cross-layer: does an EARLY layer predict a LATE one? ---------
    # This is the only source of WITHIN-token lead time. Learn, from the first
    # half of the trace, P(e' picked at L+d | e picked at L); score on the
    # second half by pooling the 6 observed picks at L and taking the top-6
    # predicted experts at L+d. Chance recall for top-6 of 384 is 6*6/384.
    sec("4. CROSS-LAYER PREDICTION WITHIN ONE TOKEN  (recall@6 of 6 true picks)")
    half = T // 2
    print(f"{'src':>4} {'dst':>4} {'recall':>8} {'chance':>8} {'lift':>7}")
    for src, dst in ((0, 1), (0, 5), (0, 10), (0, 20), (0, 39),
                     (10, 11), (10, 20), (19, 20), (20, 21), (20, 30), (30, 39)):
        cond = [collections.Counter() for _ in range(N_EXPERT)]
        for tk in toks[:half]:
            for a in tk[src]:
                cond[a].update(tk[dst])
        hit = tot = 0
        for tk in toks[half:]:
            score = collections.Counter()
            for a in tk[src]:
                score.update(cond[a])
            pred = {e for e, _ in score.most_common(6)}
            hit += len(pred & set(tk[dst])); tot += len(tk[dst])
        ch = 6 / N_EXPERT
        print(f"{src:>4} {dst:>4} {hit/tot:>8.3f} {ch:>8.3f} {(hit/tot)/ch:>6.1f}x")

    # ---- 5. what a PERFECT one-token-lookahead oracle would buy ----------
    # Upper bound: at the start of token t, an oracle names every (layer,expert)
    # token t will miss and issues all reads at once. It removes no bytes --
    # the same experts are read -- it only moves them off the critical path.
    sec("5. CEILING: a PERFECT one-token-ahead prefetcher")
    nz = sum(1 for m in miss_per_tok if m)
    print(f"tokens with >=1 box-1 miss   {nz:,} / {T:,}  ({100*nz/T:.1f}%)")
    print(f"mean misses on such a token  {misses/max(1,nz):.2f}")
    print(f"max misses in one token      {max(miss_per_tok)}")
    print("bytes moved: UNCHANGED (prefetch reorders reads, it does not remove them)")

main()
