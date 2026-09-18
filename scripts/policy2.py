#!/usr/bin/env python3
"""Pager replacement-policy study. All policies share one resident set of S slots
and count a miss per (layer,expert) request the pager would have to service.

  belady : optimal offline (evict the entry used furthest in the future) -- the CEILING
  lru    : today's pager
  lru2q  : protect entries seen at least twice (2Q / LRU-K approximation)
  flfu   : frequency-pinned core + LRU tail, learned online with decay
  static : top-N by frequency from a TRAIN split, pinned; LRU for the tail
"""
import sys, array, collections, heapq

L, K, N_EXPERT = 40, 6, 384

def load(path):
    buf = array.array('H'); buf.frombytes(open(path,'rb').read())
    per = L*K; nt = len(buf)//per
    seq = []          # flat request stream, with a token index per request
    for t in range(nt):
        for l in range(L):
            seen = []
            for k in range(K):
                e = buf[t*per + l*K + k]
                if e < N_EXPERT and e not in seen:
                    seen.append(e); seq.append(((l, e), t))
    return seq, nt

def lru(seq, S, t0):
    res = collections.OrderedDict(); m = w = 0
    for key, t in seq:
        if key in res: res.move_to_end(key)
        else:
            m += 1; w += (t >= t0)
            res[key] = 1
            if len(res) > S: res.popitem(last=False)
    return m, w

def lru2q(seq, S, t0, hot_frac=0.75):
    """A1 (probationary, 1-hit) + Am (protected, >=2 hits). Evict A1 first."""
    a1 = collections.OrderedDict(); am = collections.OrderedDict()
    cap_m = int(S*hot_frac); m = w = 0
    for key, t in seq:
        if key in am: am.move_to_end(key)
        elif key in a1:
            del a1[key]; am[key] = 1
            if len(am) > cap_m:
                v, _ = am.popitem(last=False); a1[v] = 1
        else:
            m += 1; w += (t >= t0)
            a1[key] = 1
        while len(a1) + len(am) > S:
            if a1: a1.popitem(last=False)
            else: am.popitem(last=False)
    return m, w

def pinned_lru(seq, S, t0, is_pinned, repin=None, pin_n=0, decay=0.5):
    """One LRU order over all residents; a victim search skips pinned keys.
    `is_pinned` is a callable(key)->bool, refreshed every `repin` tokens when given."""
    res = collections.OrderedDict(); m = w = 0
    cnt = collections.Counter(); pin = set(); last_t = -1
    for key, t in seq:
        if repin is not None:
            cnt[key] += 1
            if t != last_t and t and t % repin == 0 and t != last_t:
                pin = {k for k, _ in cnt.most_common(pin_n)}
                for k in list(cnt):
                    c = cnt[k]*decay
                    if c < 0.25: del cnt[k]
                    else: cnt[k] = c
            last_t = t
            pinned = pin
        else:
            pinned = is_pinned
        if key in res: res.move_to_end(key)
        else:
            m += 1; w += (t >= t0)
            if len(res) >= S:
                victim = None
                for cand in res:                       # LRU order
                    if cand not in pinned: victim = cand; break
                if victim is None: victim = next(iter(res))
                del res[victim]
            res[key] = 1
    return m, w

def belady(seq, S, t0):
    """Optimal: evict the resident whose next use is furthest away."""
    nxt = [0]*len(seq)
    last = {}
    for i in range(len(seq)-1, -1, -1):
        k = seq[i][0]
        nxt[i] = last.get(k, len(seq))
        last[k] = i
    res = set(); heap = []          # max-heap on next-use, lazily validated
    pos = {}                        # key -> its current next-use
    m = w = 0
    for i, (key, t) in enumerate(seq):
        if key in res:
            pos[key] = nxt[i]; heapq.heappush(heap, (-nxt[i], key))
        else:
            m += 1; w += (t >= t0)
            if len(res) >= S:
                while heap:
                    negn, cand = heapq.heappop(heap)
                    if cand in res and pos.get(cand) == -negn:
                        res.discard(cand); pos.pop(cand, None); break
            res.add(key); pos[key] = nxt[i]; heapq.heappush(heap, (-nxt[i], key))
    return m, w

if __name__ == "__main__":
    seq, nt = load(sys.argv[1])
    sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1418,2186,3072,4096").split(",")]
    t0 = nt//2; n = nt - t0
    train = collections.Counter()
    for key, t in seq:
        if t < t0: train[key] += 1
    order = [k for k, _ in train.most_common()]
    print(f"{sys.argv[1]}: {nt} tokens, {len(seq)} requests ({len(seq)/nt:.0f}/token); "
          f"train [0,{t0}) test [{t0},{nt})\n")
    print(f"{'slots':>6} {'belady':>8} {'LRU':>8} {'2Q':>8} {'flfu':>8} {'static(holdout)':>16}   misses/token on TEST")
    for S in sizes:
        b = belady(seq, S, t0)[1]/n
        l = lru(seq, S, t0)[1]/n
        q = lru2q(seq, S, t0)[1]/n
        f = pinned_lru(seq, S, t0, None, repin=16, pin_n=int(S*0.75))[1]/n
        s = pinned_lru(seq, S, t0, set(order[:int(S*0.75)]))[1]/n
        print(f"{S:6d} {b:8.1f} {l:8.1f} {q:8.1f} {f:8.1f} {s:16.1f}   "
              f"| LRU is {100*(l-b)/max(b,1e-9):+.0f}% over optimal; best realisable gain {100*(1-b/l):.0f}%")
