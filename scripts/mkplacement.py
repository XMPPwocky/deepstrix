#!/usr/bin/env python3
"""Expert-selection trace(s) -> frequency-ranked placement file.

Format matches `weights::parse_hot_expert_file` (and so `Assignment::from_placement_file`):
one line per layer, comma-separated `id:count`, descending. The consumer applies a
GLOBAL greedy budget of k_avg * N_LAYER over all (layer, expert) pairs by count, so
skewed layers get more slots than flat ones. Dedup per (token, layer) exactly as the
pager does, or ids picked twice in one step get double weight.

  usage: mkplacement.py OUT.txt TRACE.bin [TRACE.bin ...]
"""
import sys, array, collections
N_LAYER, TOPK = 40, 6
out_path, traces = sys.argv[1], sys.argv[2:]
freq = [collections.Counter() for _ in range(N_LAYER)]
tot_tok = 0
for path in traces:
    buf = array.array('H'); buf.frombytes(open(path, 'rb').read())
    per = N_LAYER * TOPK; ntok = len(buf) // per; tot_tok += ntok
    for t in range(ntok):
        for l in range(N_LAYER):
            seen = []
            for k in range(TOPK):
                e = buf[t*per + l*TOPK + k]
                if e < 384 and e not in seen: seen.append(e)
            for e in seen: freq[l][e] += 1
with open(out_path, 'w') as f:
    for l in range(N_LAYER):
        f.write(','.join(f"{e}:{c}" for e, c in freq[l].most_common()) + '\n')
allp = sorted(((c, l, e) for l in range(N_LAYER) for e, c in freq[l].items()), reverse=True)
tot = sum(c for c, _, _ in allp)
print(f"{out_path}: {tot_tok} tokens from {len(traces)} trace(s); "
      f"{len(allp)} distinct pairs of {N_LAYER*384}")
for k in (77, 154, 200):
    b = k * N_LAYER
    print(f"  k_avg={k:3d} -> {b:5d} experts, {b*18.8/1000:6.1f} GB, "
          f"in-sample coverage {100*sum(c for c,_,_ in allp[:b])/tot:.1f}%")
