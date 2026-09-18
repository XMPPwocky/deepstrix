#!/usr/bin/env python3
"""Controls for prefetch_study.py sections 3-4, plus the prefill case.

Two nulls that the first pass got wrong:
  * cross-layer "lift over 6/384" assumes a UNIFORM marginal. Routing is
    Zipfian, so a predictor that ignores the source layer entirely and names
    the target layer's 6 most frequent experts already scores far above that.
    That marginal-only predictor is the real null.
  * a predictor scored on ALL accesses is scored almost entirely on HITS.
    Prefetch only pays on MISSES, so the same predictor is re-scored on the
    miss subset alone.
"""
import sys, collections

N_LAYER, N_EXPERT = 40, 384
BOX1_MILLI, BOX1_SLOTS = 397, 4070

def partition_box2(layer, e):
    return ((layer * 1000003 + e * 7919) % 1000) >= BOX1_MILLI

def load(path, phase):
    rows = []
    for line in open(path):
        p = line.split()
        if not p or p[0] != phase:
            continue
        off = 2 if phase == 'D' else 3
        rows.append((int(p[1]), tuple(int(x) for x in p[off:] if int(x) >= 0)))
    return rows

def tokens(rows):
    toks, cur = [], []
    for layer, ids in rows:
        if layer == 0 and cur:
            if len(cur) == N_LAYER: toks.append(cur)
            cur = []
        cur.append(ids)
    if len(cur) == N_LAYER: toks.append(cur)
    return toks

def sec(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)

path = sys.argv[1]
toks = tokens(load(path, 'D'))
T = len(toks); half = T // 2
print(f"decode tokens {T:,}")

# ---- A. cross-layer prediction against the RIGHT null -------------------
sec("A. CROSS-LAYER, vs a marginal-only (frequency) predictor")
marg = [collections.Counter() for _ in range(N_LAYER)]
for tk in toks[:half]:
    for L, ids in enumerate(tk):
        marg[L].update(ids)
print(f"{'src':>4} {'dst':>4} {'cond':>7} {'margonly':>9} {'gain':>7}")
for src, dst in ((0,1),(0,20),(0,39),(10,20),(19,20),(20,21),(30,39)):
    cond = [collections.Counter() for _ in range(N_EXPERT)]
    for tk in toks[:half]:
        for a in tk[src]:
            cond[a].update(tk[dst])
    base = {e for e,_ in marg[dst].most_common(6)}
    hc = hb = tot = 0
    for tk in toks[half:]:
        sc = collections.Counter()
        for a in tk[src]: sc.update(cond[a])
        truth = set(tk[dst])
        hc += len({e for e,_ in sc.most_common(6)} & truth)
        hb += len(base & truth); tot += len(truth)
    print(f"{src:>4} {dst:>4} {hc/tot:>7.3f} {hb/tot:>9.3f} "
          f"{(hc/tot)/(hb/tot):>6.2f}x")

# ---- B. the same predictors, scored ONLY on accesses that MISS ----------
sec("B. SAME PREDICTORS, SCORED ONLY ON BOX-1 MISSES")
cache, miss_at = collections.OrderedDict(), []
for t, tk in enumerate(toks):
    mset = set()
    for L, ids in enumerate(tk):
        for e in ids:
            if partition_box2(L, e): continue
            k = (L, e)
            if k in cache: cache.move_to_end(k)
            else:
                mset.add(k)
                if len(cache) >= BOX1_SLOTS: cache.popitem(last=False)
                cache[k] = 1
    miss_at.append(mset)
tot_m = sum(len(m) for m in miss_at)
print(f"box-1 misses {tot_m:,} over {T:,} tokens")
for src, dst in ((0,20),(19,20),(0,39)):
    cond = [collections.Counter() for _ in range(N_EXPERT)]
    for tk in toks[:half]:
        for a in tk[src]:
            cond[a].update(tk[dst])
    base = {e for e,_ in marg[dst].most_common(32)}
    hc = hb = tot = 0
    for t in range(half, T):
        want = {e for (L, e) in miss_at[t] if L == dst}
        if not want: continue
        sc = collections.Counter()
        for a in toks[t][src]: sc.update(cond[a])
        pred = {e for e,_ in sc.most_common(32)}
        hc += len(pred & want); hb += len(base & want); tot += len(want)
    if tot:
        print(f"  L{src}->L{dst}: misses at dst {tot:>5}  "
              f"recall@32 cond {hc/tot:>.3f}   margonly {hb/tot:>.3f}")

# ---- C. prefill: how much of a layer does ONE chunk touch? --------------
# Prefill has a structural advantage decode does not: a 512-row chunk selects
# top-6 for EVERY row, so its union at one layer may be most of the 384. If it
# is, "page the whole layer ahead of the layer that needs it" is a correct and
# nearly-precise prefetch -- no prediction required.
sec("C. PREFILL: distinct experts touched by ONE chunk at ONE layer")
pre = load(path, 'P')
per = collections.defaultdict(set)
order = []
seen = set()
for layer, ids in pre:
    if layer not in seen:
        seen.add(layer); order.append(layer)
    per[layer].update(ids)
    if len(seen) == N_LAYER and layer == order[-1]:
        pass
# chunk boundaries: layer resets to 0
chunks, cur, curlayer = [], collections.defaultdict(set), None
prev = None
for layer, ids in pre:
    if prev is not None and layer < prev:
        chunks.append(cur); cur = collections.defaultdict(set)
    cur[layer].update(ids); prev = layer
if cur: chunks.append(cur)
print(f"prefill chunks {len(chunks)}")
import statistics
for L in (0, 10, 19, 20, 30, 39):
    v = [len(c[L]) for c in chunks if L in c and c[L]]
    if v:
        v.sort()
        print(f"  L{L:<2} distinct experts per chunk: "
              f"min {v[0]:>3}  p50 {v[len(v)//2]:>3}  max {v[-1]:>3}  "
              f"({100*v[len(v)//2]/N_EXPERT:.0f}% of the layer)")
