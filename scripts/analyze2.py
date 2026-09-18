"""Cold-expert prefetch / caching feasibility, v2.

Fixes vs v1: (a) rank slots by PRODUCTION stats (1.8M decode tokens of real
traffic), not by the trace's own frequencies -- v1 was circular; (b) warm the
LRU before measuring, so cold-start first-touches are not counted as misses.
"""
import json, sys, numpy as np
from collections import OrderedDict

TRACE = "expert_trace.bin"
STATS = "/home/claude-code/.cache/deepstrix/models/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS-00001-of-00004/expert_stats.json"
N_LAYER, N_USED, N_EXPERT = 43, 6, 256
WARM = 1000                                  # tokens used only to warm caches

raw = np.fromfile(TRACE, dtype="<u2")
ntok = raw.size // (N_LAYER * N_USED)
sel = raw[: ntok * N_LAYER * N_USED].reshape(ntok, N_LAYER, N_USED).astype(np.int32)

d = json.load(open(STATS))
prod = np.array(d["decode"]["counts"], dtype=np.int64).reshape(N_LAYER, N_EXPERT)
print(f"trace {ntok} tokens (measuring on the last {ntok-WARM}); "
      f"production ranking from {d['decode']['tokens']:,} decode tokens\n")

order = np.argsort(prod.ravel())             # ascending: coldest first
def mask_coldest(frac):
    k = int(frac * order.size)
    m = np.zeros(order.size, dtype=bool); m[order[:k]] = True
    return m.reshape(N_LAYER, N_EXPERT)

meas = slice(WARM, ntok); nmeas = ntok - WARM

print("=== 1. cold-tail load, cold defined by PRODUCTION frequency ===")
for f in (0.05, 0.10, 0.25):
    m = mask_coldest(f)
    c = sum(m[l][sel[meas, l, :]].sum() for l in range(N_LAYER))
    print(f"  coldest {int(f*100):2d}% of slots: {c/nmeas:5.2f} picks/token in this trace")

print("\n=== 2. does the trace's working set match production's hot set? ===")
for f in (0.75, 0.85, 0.90):
    M = int(f * order.size)
    keep = np.zeros(order.size, dtype=bool); keep[order[-M:]] = True
    keep = keep.reshape(N_LAYER, N_EXPERT)
    miss = sum((~keep[l][sel[meas, l, :]]).sum() for l in range(N_LAYER))
    print(f"  static top-{int(f*100)}% by production stats: {miss/nmeas:5.2f} misses/token")

print("\n=== 3. static (production-ranked) vs LRU, equal budget, warmed ===")
print(f"  {'resident':>8} {'M':>6} {'static miss/tok':>16} {'LRU miss/tok':>13}")
for f in (0.75, 0.85, 0.90):
    M = int(f * order.size)
    keep = np.zeros(order.size, dtype=bool); keep[order[-M:]] = True
    keep = keep.reshape(N_LAYER, N_EXPERT)
    static_miss = sum((~keep[l][sel[meas, l, :]]).sum() for l in range(N_LAYER)) / nmeas
    lru = OrderedDict(); miss = 0
    for t in range(ntok):
        counting = t >= WARM
        for l in range(N_LAYER):
            for e in sel[t, l]:
                k = (l, int(e))
                if k in lru: lru.move_to_end(k)
                else:
                    if counting: miss += 1
                    lru[k] = 1
                    if len(lru) > M: lru.popitem(last=False)
    print(f"  {100*f:7.0f}% {M:6d} {static_miss:16.2f} {miss/nmeas:13.2f}")

print("\n=== 4. are the misses predictable from recent history? ===")
for f in (0.75, 0.85):
    M = int(f * order.size)
    keep = np.zeros(order.size, dtype=bool); keep[order[-M:]] = True
    keep = keep.reshape(N_LAYER, N_EXPERT)
    for W in (1, 8, 32):
        pred = tot = 0
        for l in range(N_LAYER):
            s = sel[:, l, :]
            for t in range(WARM, ntok):
                prev = set(s[max(0,t-W):t].ravel().tolist())
                for e in s[t]:
                    if not keep[l, e]:
                        tot += 1; pred += (e in prev)
        if tot:
            print(f"  resident {100*f:.0f}%  W={W:2d}: {tot/nmeas:5.2f} misses/tok, "
                  f"{100*pred/tot:4.0f}% seen in last {W} tok "
                  f"-> {(1-pred/tot)*tot/nmeas:5.2f} unpredicted reads/tok")

print("\n=== 5. hybrid: static hot set + small LRU victim cache for the rest ===")
for f in (0.75, 0.85):
    M = int(f * order.size)
    keep = np.zeros(order.size, dtype=bool); keep[order[-M:]] = True
    keep = keep.reshape(N_LAYER, N_EXPERT)
    base = sum((~keep[l][sel[meas, l, :]]).sum() for l in range(N_LAYER)) / nmeas
    for C in (128, 512, 2048):
        vic = OrderedDict(); miss = 0
        for t in range(ntok):
            counting = t >= WARM
            for l in range(N_LAYER):
                for e in sel[t, l]:
                    if keep[l, e]: continue
                    k = (l, int(e))
                    if k in vic: vic.move_to_end(k)
                    else:
                        if counting: miss += 1
                        vic[k] = 1
                        if len(vic) > C: vic.popitem(last=False)
        print(f"  resident {100*f:.0f}% + victim C={C:4d} ({C*8.69/1024:.2f} GiB): "
              f"{base:5.2f} -> {miss/nmeas:5.2f} SSD reads/token "
              f"({100*(1-miss/max(base,1e-9)):.0f}% of misses absorbed)")
