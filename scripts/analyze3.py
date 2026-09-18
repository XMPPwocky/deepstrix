"""v3: two conversations (EN technical ~2560 tok, then ZH history/poetry ~2100).
Tests whether dynamic caching survives a hard topic+language switch."""
import json, numpy as np
from collections import OrderedDict
N_LAYER, N_USED, N_EXPERT = 43, 6, 256
STATS = "/home/claude-code/.cache/deepstrix/models/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS-00001-of-00004/expert_stats.json"
raw = np.fromfile("expert_trace.bin", dtype="<u2")
ntok = raw.size // (N_LAYER * N_USED)
sel = raw[: ntok*N_LAYER*N_USED].reshape(ntok, N_LAYER, N_USED).astype(np.int32)
SWITCH = 2560                                   # token index where the topic changes
prod = np.array(json.load(open(STATS))["decode"]["counts"], dtype=np.int64).reshape(N_LAYER, N_EXPERT)
order = np.argsort(prod.ravel())
print(f"{ntok} tokens; segment A = EN technical [0,{SWITCH}), B = ZH history/poetry [{SWITCH},{ntok})\n")

def run(M, policy, seg, warm):
    keep = np.zeros(order.size, dtype=bool); keep[order[-M:]] = True
    keep = keep.reshape(N_LAYER, N_EXPERT)
    lo, hi = seg
    if policy == "static":
        return sum((~keep[l][sel[lo:hi, l, :]]).sum() for l in range(N_LAYER)) / (hi-lo)
    lru = OrderedDict(); miss = 0
    for t in range(warm, hi):
        counting = t >= lo
        for l in range(N_LAYER):
            for e in sel[t, l]:
                k = (l, int(e))
                if k in lru: lru.move_to_end(k)
                else:
                    if counting: miss += 1
                    lru[k] = 1
                    if len(lru) > M: lru.popitem(last=False)
    return miss / (hi-lo)

print(f"{'resident':>8} {'segment':>28} {'static':>9} {'LRU':>8}  {'gain':>6}")
for f in (0.75, 0.85):
    M = int(f*order.size)
    for name, seg, warm in [
        ("A steady state",        (1000, SWITCH),    0),
        ("B: first 300 after switch", (SWITCH, SWITCH+300), 0),
        ("B steady state",        (SWITCH+300, ntok), 0),
    ]:
        st = run(M, "static", seg, warm); lr = run(M, "lru", seg, warm)
        print(f"{100*f:7.0f}% {name:>28} {st:9.2f} {lr:8.2f}  {st/max(lr,1e-9):5.1f}x")

print("\n=== working-set overlap between the two conversations ===")
A = set(); B = set()
for l in range(N_LAYER):
    for e in np.unique(sel[:SWITCH, l, :]): A.add((l, int(e)))
    for e in np.unique(sel[SWITCH:, l, :]): B.add((l, int(e)))
print(f"  A touches {len(A)} slots, B touches {len(B)}, shared {len(A&B)} "
      f"({100*len(A&B)/len(A|B):.0f}% of the union)")
print(f"  slots unique to B: {len(B-A)}  (these are what a static hot set from A would miss)")

print("\n=== hybrid: static top-85% + LRU victim cache, across the switch ===")
M = int(0.85*order.size)
keep = np.zeros(order.size, dtype=bool); keep[order[-M:]] = True
keep = keep.reshape(N_LAYER, N_EXPERT)
for C in (128, 256, 512, 1024):
    for name, lo, hi in [("A", 1000, SWITCH), ("B-switch", SWITCH, SWITCH+300), ("B", SWITCH+300, ntok)]:
        vic = OrderedDict(); miss = 0
        for t in range(0, hi):
            counting = t >= lo
            for l in range(N_LAYER):
                for e in sel[t, l]:
                    if keep[l, e]: continue
                    k = (l, int(e))
                    if k in vic: vic.move_to_end(k)
                    else:
                        if counting: miss += 1
                        vic[k] = 1
                        if len(vic) > C: vic.popitem(last=False)
        print(f"  C={C:4d} ({C*8.69/1024:4.2f} GiB) {name:>9}: {miss/(hi-lo):5.2f} SSD reads/token")
