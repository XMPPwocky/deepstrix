#!/usr/bin/env python3
"""What is multi-stream decode worth, and does it wreck the expert cache?

Costs split into what does and does not scale with S:
  * link (40 remote calls/step, ONE per layer) is B-INVARIANT -> divides by S
  * DENSE weights (~5.8 GB of attention / shared-expert / router, dGPU-resident)
    are also B-INVARIANT: a GEMM reads its weight matrix once whether B is 1 or
    32, and every stream in the batch uses it. This is the term the first
    version of this model wrongly held CONSTANT per token, which understated
    multi-stream by ~10 tok/s.
  * expert weight streaming divides by S only as far as streams SHARE experts,
    AND the two boxes stream their shares on INDEPENDENT memory systems, so the
    cost is max(box1, box2) = 0.6x the total, not the sum
  * KV / attention is genuinely per-token (each stream has its own cache) and is
    the one term batching can never amortise -- it sets the asymptote
  * misses depend on whether S live working sets fit the pool -- simulated, not
    assumed, because the production data point everyone quotes (50 agents ->
    3.37 misses/token) was SERIAL serving with a context switch per request,
    which is the opposite of multi-stream.

S=1 must reproduce the measured ~82 ms token anatomy or the model is wrong.
"""
import sys, random, collections
N_LAYER, BOX1_MILLI, SLOTS = 40, 397, 4070
EXP_MB, BW_GBS, C_MISS = 18.8, 256.0, 8.85
LINK_MS, OTHER_MS = 40.0, 10.0
def box2(l, e): return ((l*1000003 + e*7919) % 1000) >= BOX1_MILLI

def load(path):
    toks, cur = [], []
    for line in open(path):
        p = line.split()
        if not p or p[0] != 'D': continue
        l = int(p[1]); ids = tuple(int(x) for x in p[2:] if int(x) >= 0)
        if l == 0 and cur:
            if len(cur) == N_LAYER: toks.append(cur)
            cur = []
        cur.append(ids)
    if len(cur) == N_LAYER: toks.append(cur)
    return toks

def run(toks, S, steps=3000, seed=3, warm=400):
    rng = random.Random(seed); T = len(toks)
    starts = [rng.randrange(0, T-steps-1) for _ in range(S)]
    cache = collections.OrderedDict(); miss = 0; ut = 0
    for i in range(steps):
        for L in range(N_LAYER):
            u = set()
            for s in starts: u.update(toks[s+i][L])
            if i >= warm: ut += len(u)
            for e in u:
                if box2(L, e): continue
                k = (L, e)
                if k in cache: cache.move_to_end(k)
                else:
                    if i >= warm: miss += 1
                    if len(cache) >= SLOTS: cache.popitem(last=False)
                    cache[k] = 1
    return miss/((steps-warm)*S), ut/((steps-warm)*N_LAYER)

if __name__ == "__main__":
    toks = load(sys.argv[1])
    print(f"{'S':>4} {'union/L':>8} {'miss/tok':>9} {'link':>6} {'bw':>6} {'miss':>6} "
          f"{'other':>6} {'ms/tok':>7} {'tok/s':>7}")
    for S in (1, 2, 4, 8, 16, 32):
        mpt, U = run(toks, S)
        link = LINK_MS/S
        bw = (N_LAYER*U*EXP_MB/1024)/S/BW_GBS*1000
        tot = link + bw + mpt*C_MISS + OTHER_MS
        print(f"{S:>4} {U:>8.1f} {mpt:>9.2f} {link:>6.1f} {bw:>6.1f} {mpt*C_MISS:>6.1f} "
              f"{OTHER_MS:>6.1f} {tot:>7.1f} {1000/tot:>7.1f}")
