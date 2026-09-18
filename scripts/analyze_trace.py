"""Cold-expert prefetch feasibility from a decode routing trace.

Trace layout: u16 LE, N_USED per layer, N_LAYER layers per token, decode order.
"""
import sys, numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "expert_trace.bin"
N_LAYER, N_USED, N_EXPERT = 43, 6, 256

raw = np.fromfile(PATH, dtype="<u2")
ntok = raw.size // (N_LAYER * N_USED)
sel = raw[: ntok * N_LAYER * N_USED].reshape(ntok, N_LAYER, N_USED).astype(np.int32)
print(f"trace: {ntok} tokens x {N_LAYER} layers x {N_USED} picks\n")

# global frequency -> cold definition
flat = sel.reshape(-1, N_USED)
counts = np.zeros((N_LAYER, N_EXPERT), dtype=np.int64)
for l in range(N_LAYER):
    counts[l] = np.bincount(sel[:, l, :].ravel(), minlength=N_EXPERT)
glob = counts.ravel()
order = np.argsort(glob)                       # ascending = coldest first
rank = np.empty_like(order); rank[order] = np.arange(order.size)
picks_total = glob.sum()

def coldmask(frac):
    """bool [N_LAYER, N_EXPERT]: the coldest `frac` of (layer,expert) slots globally."""
    k = int(frac * order.size)
    m = np.zeros(order.size, dtype=bool); m[order[:k]] = True
    return m.reshape(N_LAYER, N_EXPERT)

print("=== 1. cold-tail share of decode picks (this trace) ===")
for f in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
    m = coldmask(f)
    share = glob.reshape(N_LAYER, N_EXPERT)[m].sum() / picks_total
    print(f"  coldest {int(f*100):2d}% of slots -> {100*share:6.3f}% of picks "
          f"({share*N_LAYER*N_USED:5.2f} cold picks/token)")

print("\n=== 2. consecutive-token overlap at the same layer ===")
inter = np.zeros(ntok - 1)
for t in range(1, ntok):
    for l in range(N_LAYER):
        inter[t-1] += len(set(sel[t, l]) & set(sel[t-1, l]))
print(f"  mean |picks(t,l) & picks(t-1,l)| = {inter.mean()/N_LAYER:.2f} of {N_USED}"
      f"  ({100*inter.mean()/(N_LAYER*N_USED):.1f}%)")

print("\n=== 3. prefetch: union of last W tokens' picks at layer l ===")
print("   (hit = token t's pick was in that union; cost = experts in the union)")
for W in (1, 2, 4, 8, 16, 32):
    hits = tot = 0; union_sz = 0.0; n = 0
    for l in range(N_LAYER):
        s = sel[:, l, :]
        for t in range(W, ntok):
            prev = set(s[t-W:t].ravel().tolist())
            cur = s[t]
            hits += sum(1 for e in cur if e in prev); tot += N_USED
            union_sz += len(prev); n += 1
    print(f"  W={W:3d}: hit {100*hits/tot:5.1f}%   mean union {union_sz/n:5.1f} experts/layer")

print("\n=== 4. the number that matters: prefetch hit rate ON COLD PICKS ===")
print("   a cold pick that was predicted = prefetched during the previous token = free")
for f in (0.05, 0.10, 0.25):
    m = coldmask(f)
    for W in (1, 4, 16):
        hits = tot = 0
        for l in range(N_LAYER):
            s = sel[:, l, :]
            for t in range(W, ntok):
                prev = set(s[t-W:t].ravel().tolist())
                for e in s[t]:
                    if m[l, e]:
                        tot += 1; hits += (e in prev)
        if tot:
            print(f"  cold={int(f*100):2d}%  W={W:2d}:  {100*hits/tot:5.1f}% of cold picks "
                  f"already seen  ({tot/(ntok-W):.2f} cold picks/token, "
                  f"{(1-hits/tot)*tot/(ntok-W):.2f} still stall)")

print("\n=== 5. THE question: static hot-set vs LRU at the same memory budget ===")
print("   M resident slots out of", N_LAYER*N_EXPERT, "; miss = an SSD read")
from collections import OrderedDict
print(f"   {'resident':>9} {'M':>6} {'static miss/tok':>16} {'LRU miss/tok':>13} {'LRU hit on':>11}")
for frac in (0.95, 0.90, 0.85, 0.75, 0.70):
    M = int(frac * N_LAYER * N_EXPERT)
    # static: keep the M globally hottest slots
    keep = np.zeros(order.size, dtype=bool); keep[order[-M:]] = True
    keep = keep.reshape(N_LAYER, N_EXPERT)
    static_miss = sum(
        (~keep[l][sel[:, l, :]]).sum() for l in range(N_LAYER)
    ) / ntok
    # LRU over the same M slots, keyed (layer, expert)
    lru = OrderedDict(); miss = 0
    for t in range(ntok):
        for l in range(N_LAYER):
            for e in sel[t, l]:
                key = (l, int(e))
                if key in lru: lru.move_to_end(key)
                else:
                    miss += 1
                    lru[key] = 1
                    if len(lru) > M: lru.popitem(last=False)
    lru_miss = miss / ntok
    print(f"   {100*frac:8.0f}% {M:6d} {static_miss:16.2f} {lru_miss:13.2f} "
          f"{100*(1-lru_miss/max(static_miss,1e-9)):10.0f}%")
print("   (LRU is warmed in-place; first-touch misses are counted, so this is pessimistic)")

print("\n=== 6. of the remaining misses, how many are predictable 1 token ahead? ===")
for frac in (0.90, 0.75):
    M = int(frac * N_LAYER * N_EXPERT)
    keep = np.zeros(order.size, dtype=bool); keep[order[-M:]] = True
    keep = keep.reshape(N_LAYER, N_EXPERT)
    pred = tot = 0
    for l in range(N_LAYER):
        s = sel[:, l, :]
        for t in range(1, ntok):
            prev = set(s[t-1].tolist())
            for e in s[t]:
                if not keep[l, e]:
                    tot += 1; pred += (e in prev)
    if tot:
        print(f"   resident {100*frac:.0f}%: {tot/(ntok-1):.2f} misses/token, "
              f"{100*pred/tot:.0f}% were also picked at t-1 "
              f"-> {(1-pred/tot)*tot/(ntok-1):.2f} unpredicted SSD reads/token")
