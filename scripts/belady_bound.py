#!/usr/bin/env python3
"""How much could ANY prediction reduce our expert miss rate?

For a FIXED capacity C the minimum number of disk fetches over a trace is
exactly Belady's OPT (evict the resident item whose next use is furthest in the
future). Prefetching cannot beat it: a prefetched expert occupies a slot, and
OPT already assumes every slot is used optimally, so prefetch moves a fetch
EARLIER but never removes it. Therefore

    miss_LRU - miss_OPT = the entire headroom available to ANY predictor,
                          whether it predicts for replacement or for prefetch

and the compulsory (first-touch) misses are a floor no policy of any kind can
go below. This prints that decomposition at several capacities.
"""
import sys, collections, heapq

N_LAYER, N_EXPERT = 40, 384
BOX1_MILLI = 397

def partition_box2(layer, e):
    return ((layer * 1000003 + e * 7919) % 1000) >= BOX1_MILLI

def stream(path, box1_only=True):
    out = []
    for line in open(path):
        p = line.split()
        if not p or p[0] != 'D':
            continue
        layer = int(p[1])
        for x in p[2:]:
            e = int(x)
            if e < 0:
                continue
            if box1_only and partition_box2(layer, e):
                continue
            out.append((layer, e))
    return out

def sim_lru(s, cap):
    c = collections.OrderedDict(); m = 0
    for k in s:
        if k in c:
            c.move_to_end(k)
        else:
            m += 1
            if len(c) >= cap:
                c.popitem(last=False)
            c[k] = 1
    return m

def sim_opt(s, cap):
    """Exact Belady. next_use[i] = index of the next access to the same key
    (len(s) = never again). Evict the resident key with the largest next_use,
    via a max-heap with lazy invalidation."""
    n = len(s)
    nxt = [n] * n
    last = {}
    for i in range(n - 1, -1, -1):
        k = s[i]
        nxt[i] = last.get(k, n)
        last[k] = i
    res = {}          # key -> its current next_use
    heap = []         # (-next_use, key)
    m = 0
    for i, k in enumerate(s):
        if k in res:
            res[k] = nxt[i]
            heapq.heappush(heap, (-nxt[i], k))
            continue
        m += 1
        if len(res) >= cap:
            while True:
                nu, vk = heapq.heappop(heap)
                if vk in res and res[vk] == -nu:
                    del res[vk]
                    break
        res[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return m

path = sys.argv[1]
s = stream(path)
n = len(s)
distinct = len(set(s))
# A first touch must come from disk at ANY capacity under ANY policy.
compulsory = distinct
print(f"box-1 decode accesses {n:,}   distinct (layer,expert) {distinct:,}")
print(f"compulsory floor (first touch of each pair) {compulsory:,} "
      f"= {100*compulsory/n:.3f}% of accesses\n")
print(f"{'cap':>7} {'LRU':>9} {'OPT':>9} {'LRU%':>7} {'OPT%':>7} "
      f"{'headroom':>9} {'of LRU':>8}")
for cap in (2000, 4070, 6000, 10230, 15360):
    ml = sim_lru(s, cap)
    mo = sim_opt(s, cap)
    print(f"{cap:>7} {ml:>9,} {mo:>9,} {100*ml/n:>6.3f}% {100*mo/n:>6.3f}% "
          f"{ml-mo:>9,} {100*(ml-mo)/ml:>7.1f}%")
