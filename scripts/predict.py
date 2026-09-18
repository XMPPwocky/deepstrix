#!/usr/bin/env python3
"""How predictable are a layer's picks from information available EARLIER?

Prefetch needs lead time. Three candidate predictors, in increasing lead:
  prev-token same-layer : token t's layer-L picks predict t+1's layer-L picks.
                          Lead = a whole token (~90 ms). Strongest if routing is
                          temporally stable.
  union-K prev tokens   : union of the last K tokens' layer-L picks.
  global top-N          : the static hot set (what placement already does).
"""
import sys, collections
path = sys.argv[1]
# group decode rows into tokens: layers arrive 0..39 in order, so a layer <= prev
# layer starts a new token.
tokens = []; cur = {}; prev_layer = -1
for line in open(path):
    p = line.split()
    if len(p) < 3 or p[0] != 'D': continue
    layer = int(p[1]); ids = set(int(x) for x in p[2:] if int(x) >= 0)
    if layer <= prev_layer and cur:
        tokens.append(cur); cur = {}
    cur[layer] = ids; prev_layer = layer
if cur: tokens.append(cur)
print(f"tokens {len(tokens):,}")

freq = collections.Counter()
for t in tokens:
    for l, s in t.items():
        for e in s: freq[(l, e)] += 1

def hit_rate(pred_fn, warm=8):
    hit = tot = 0
    for i in range(warm, len(tokens)):
        for l, actual in tokens[i].items():
            pred = pred_fn(i, l)
            if pred is None: continue
            tot += len(actual); hit += len(actual & pred)
    return 100.0 * hit / max(1, tot), tot

def prev_tok(i, l):
    return tokens[i-1].get(l)
def union_k(k):
    def f(i, l):
        u = set()
        for j in range(i-k, i):
            u |= tokens[j].get(l, set())
        return u
    return f
TOPN = {}
def global_top(n):
    def f(i, l):
        if (l, n) not in TOPN:
            cand = [(c, e) for (ll, e), c in freq.items() if ll == l]
            cand.sort(reverse=True)
            TOPN[(l, n)] = set(e for _, e in cand[:n])
        return TOPN[(l, n)]
    return f

print("\npredictor                       recall   (of picks covered)")
r, n = hit_rate(prev_tok);        print(f"  prev token, same layer   (6)   {r:>6.1f}%   n={n:,}")
for k in (2, 4, 8, 16):
    r, _ = hit_rate(union_k(k));  print(f"  union of last {k:<2} tokens  ({6*k:>3})  {r:>6.1f}%")
for n_ in (16, 32, 64, 128):
    r, _ = hit_rate(global_top(n_)); print(f"  global top-{n_:<3} per layer       {r:>6.1f}%")
