#!/usr/bin/env python3
"""Replay the decode pick trace as S CONCURRENT streams against the pager pools.

The real batched engine is LAYER-MAJOR: one step touches layer L for all S rows,
so the access stream is (step, layer, union-of-S-rows) and duplicate picks inside
one (step,layer) cost ONE lookup, not S. That dedup is exactly the quantity the
multi-stream throughput model needs, and it is not derivable from the B=1 trace.

Streams are disjoint contiguous segments of real agent traffic, i.e. as
decorrelated as anything we have.
"""
import sys, collections

N_LAYER = 40
B1_CAP, B2_CAP = 4454, 6160

def box2(layer, e):          # het/remote_experts.rs partition_box2
    return ((layer * 1000003 + e * 7919) % 1000) >= 397

def load_tokens(path):
    toks, cur, last = [], [None]*N_LAYER, -1
    for line in open(path):
        if not line.startswith('D '): continue
        p = line.split()
        L = int(p[1])
        if L <= last:                       # wrapped -> token boundary
            toks.append(cur); cur = [None]*N_LAYER
        cur[L] = [int(x) for x in p[2:] if int(x) >= 0]
        last = L
    if any(c is not None for c in cur): toks.append(cur)
    return [t for t in toks if all(x is not None for x in t)]

def sim(tokens, S):
    seglen = len(tokens) // S
    segs = [tokens[i*seglen:(i+1)*seglen] for i in range(S)]
    pools = [collections.OrderedDict(), collections.OrderedDict()]
    caps  = [B1_CAP, B2_CAP]
    miss  = [0, 0]; look = [0, 0]
    union_sum = [0, 0]; nlayer = 0
    for t in range(seglen):
        for L in range(N_LAYER):
            u = [[], []]
            seen = set()
            for s in range(S):
                for e in segs[s][t][L]:
                    if e in seen: continue
                    seen.add(e)
                    u[1 if box2(L, e) else 0].append(e)
            nlayer += 1
            for b in (0, 1):
                union_sum[b] += len(u[b])
                pool, cap = pools[b], caps[b]
                for e in u[b]:
                    k = (L, e); look[b] += 1
                    if k in pool: pool.move_to_end(k)
                    else:
                        miss[b] += 1
                        if len(pool) >= cap: pool.popitem(last=False)
                        pool[k] = 1
    toks_done = seglen * S
    return dict(S=S, steps=seglen, tokens=toks_done,
                b1_miss_pct=100*miss[0]/look[0], b2_miss_pct=100*miss[1]/look[1],
                b1_miss_per_tok=miss[0]/toks_done, b2_miss_per_tok=miss[1]/toks_done,
                b1_u=union_sum[0]/nlayer, b2_u=union_sum[1]/nlayer,
                b1_miss_step=miss[0]/seglen, b2_miss_step=miss[1]/seglen)

if __name__ == "__main__":
    toks = load_tokens(sys.argv[1])
    print(f"decode tokens {len(toks):,}  (layers {N_LAYER}, top-{len(toks[0][0])})\n")
    hdr = f"{'S':>2} {'steps':>6} | {'b1 U/lyr':>8} {'b1 miss%':>8} {'b1 m/tok':>8} | {'b2 U/lyr':>8} {'b2 miss%':>8} {'b2 m/tok':>8}"
    print(hdr); print('-'*len(hdr))
    for S in (1, 2, 3, 4, 6, 8):
        r = sim(toks, S)
        print(f"{r['S']:>2} {r['steps']:>6} | {r['b1_u']:>8.2f} {r['b1_miss_pct']:>8.3f} {r['b1_miss_per_tok']:>8.3f} |"
              f" {r['b2_u']:>8.2f} {r['b2_miss_pct']:>8.3f} {r['b2_miss_per_tok']:>8.3f}")
