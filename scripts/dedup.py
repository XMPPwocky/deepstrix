#!/usr/bin/env python3
"""Distinct-expert scaling for a DSpark verify batch.

A verify step processes B CONSECUTIVE tokens in one forward. The MoE legs are
weight-bandwidth bound, so their cost tracks DISTINCT experts touched per layer,
not tokens. m_moe(B) = distinct(B) / (B * distinct(1)) * B = distinct(B)/distinct(1).
"""
import sys, array

N_LAYER, TOPK, N_EXPERT = 40, 6, 384
path = sys.argv[1]
buf = array.array('H'); buf.frombytes(open(path, 'rb').read())
per_tok = N_LAYER * TOPK
ntok = len(buf) // per_tok
print(f"trace {path.split('/')[-1]}: {ntok} tokens x {N_LAYER} layers x top-{TOPK}\n")

# picks[t][l] = set of valid expert ids
picks = [[set() for _ in range(N_LAYER)] for _ in range(ntok)]
for t in range(ntok):
    base = t * per_tok
    for l in range(N_LAYER):
        s = picks[t][l]
        off = base + l * TOPK
        for k in range(TOPK):
            e = buf[off + k]
            if e < N_EXPERT:
                s.add(e)

Bs = [1, 2, 3, 4, 5, 6, 8, 12, 16]
print(f"{'B':>3} {'distinct/layer':>15} {'vs B=1':>8} {'ideal(=B)':>10} {'dedup save':>11} {'m_moe':>7}")
base1 = None
for B in Bs:
    tot = 0.0; nwin = 0
    for t0 in range(0, ntok - B + 1, B):          # disjoint windows, as verify steps are
        for l in range(N_LAYER):
            u = set()
            for t in range(t0, t0 + B):
                u |= picks[t][l]
            tot += len(u)
        nwin += 1
    avg = tot / (nwin * N_LAYER)
    if base1 is None: base1 = avg
    ratio = avg / base1
    print(f"{B:3d} {avg:15.2f} {ratio:8.2f}x {float(B):10.2f} "
          f"{100*(1 - ratio/B):10.1f}% {ratio:7.2f}")

print("\nm_moe = MoE-leg cost multiplier for a B-token verify step (weight-BW model)")
print("per-token MoE cost vs B=1 = m_moe / B  (lower is better)")
print(f"\n{'B':>3} {'MoE cost/token':>15}")
base1 = None
for B in Bs:
    tot = 0.0; nwin = 0
    for t0 in range(0, ntok - B + 1, B):
        for l in range(N_LAYER):
            u = set()
            for t in range(t0, t0 + B):
                u |= picks[t][l]
            tot += len(u)
        nwin += 1
    avg = tot / (nwin * N_LAYER)
    if base1 is None: base1 = avg
    print(f"{B:3d} {avg/base1/B:15.3f}")
