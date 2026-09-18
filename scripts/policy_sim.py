#!/usr/bin/env python3
"""Replay DEEPSTRIX_EXPERT_TRACE under candidate pager policies.

Policies
  lru    : today's pager (pure LRU over (layer, expert))
  oracle : top-N by frequency measured on the TRAIN split, pinned; LRU for the tail
  flfu   : the shippable online policy -- the pager counts its own picks, and every
           `repin` tokens promotes the top `pin_frac * slots` pairs by DECAYED count
           into an eviction-immune core; everything else is LRU.  No offline file,
           so it adapts to whatever the server is actually being asked.
"""
import sys, array, collections

L, K, N_EXPERT = 40, 6, 384


def load(path):
    buf = array.array('H'); buf.frombytes(open(path, 'rb').read())
    per = L * K; nt = len(buf) // per
    toks = []
    for t in range(nt):
        rows = []
        for l in range(L):
            seen = []
            for k in range(K):
                e = buf[t * per + l * K + k]
                if e < N_EXPERT and e not in seen:
                    seen.append(e)
            rows.append([(l, e) for e in seen])
        toks.append(rows)
    return toks


def run_lru(toks, S, t0):
    lru = collections.OrderedDict(); miss = wmiss = 0
    for ti, tk in enumerate(toks):
        for row in tk:
            for key in row:
                if key in lru:
                    lru.move_to_end(key)
                else:
                    miss += 1
                    if ti >= t0: wmiss += 1
                    lru[key] = 1
                    if len(lru) > S: lru.popitem(last=False)
    return miss, wmiss


def run_pinned(toks, S, t0, pinned):
    """LRU, but a slot holding a pinned key is never evicted."""
    lru = collections.OrderedDict(); res = set(); miss = wmiss = 0
    for ti, tk in enumerate(toks):
        for row in tk:
            for key in row:
                if key in res:
                    if key in lru: lru.move_to_end(key)
                else:
                    miss += 1
                    if ti >= t0: wmiss += 1
                    if len(res) >= S:
                        if not lru:      # every slot pinned: fall back
                            break
                        v, _ = lru.popitem(last=False); res.discard(v)
                    res.add(key)
                    if key not in pinned: lru[key] = 1
    return miss, wmiss


def run_flfu(toks, S, t0, pin_frac=0.75, repin=64, decay=0.5):
    n_pin = int(S * pin_frac)
    cnt = collections.Counter(); pinned = set()
    lru = collections.OrderedDict(); res = set(); miss = wmiss = 0
    for ti, tk in enumerate(toks):
        for row in tk:
            for key in row:
                cnt[key] += 1
                if key in res:
                    if key in lru: lru.move_to_end(key)
                else:
                    miss += 1
                    if ti >= t0: wmiss += 1
                    if len(res) >= S:
                        v = None
                        while lru:
                            c, _ = lru.popitem(last=False)
                            if c in res: v = c; break
                        if v is None:
                            for c in list(res):
                                if c not in pinned: v = c; break
                        if v is None: v = next(iter(res))
                        res.discard(v)
                    res.add(key)
                    if key not in pinned: lru[key] = 1
        if (ti + 1) % repin == 0:
            new = {k for k, _ in cnt.most_common(n_pin)}
            # keys that just lost their pin re-enter the LRU at the back
            for k in pinned - new:
                if k in res and k not in lru: lru[k] = 1
            for k in new:
                lru.pop(k, None)
            pinned = new
            for k in list(cnt):
                c = cnt[k] * decay
                if c < 0.25: del cnt[k]
                else: cnt[k] = c
    return miss, wmiss


if __name__ == "__main__":
    toks = load(sys.argv[1])
    nt = len(toks)
    sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1418,2186,3072,4096").split(",")]
    t0 = nt // 2
    print(f"trace {sys.argv[1]}: {nt} tokens, train = [0,{t0}), test = [{t0},{nt})")

    # TRUE HOLDOUT: rank on the train split only, score on the test split.
    train = collections.Counter()
    for tk in toks[:t0]:
        for row in tk:
            for key in row: train[key] += 1
    order = [k for k, _ in train.most_common()]

    print(f"\n{'slots':>6} | {'LRU':>8} {'oracle-holdout':>15} {'flfu(online)':>13}   (misses per generated token, TEST half)")
    for S in sizes:
        _, l = run_lru(toks, S, t0)
        _, o = run_pinned(toks, S, t0, set(order[:int(S * 0.75)]))
        _, f = run_flfu(toks, S, t0)
        n = nt - t0
        print(f"{S:6d} | {l/n:8.1f} {o/n:15.1f} {f/n:13.1f}   "
              f"(flfu {100*(1-f/l):+.0f}% vs LRU, oracle {100*(1-o/l):+.0f}%)")
