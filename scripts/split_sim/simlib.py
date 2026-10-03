#!/usr/bin/env python3
"""Shared pieces of the hot-split simulator: cache loading, clocks, today's
ownership (hot_set) replica, LRU pools.

Everything here mirrors crates/v4flash-kernels/src/het/expert_pager.rs
(`partition_box2`, `hot_set::refresh`) as of worktree lm-prefill-prod a7799a7.
"""
import calendar
import pickle
import time
from collections import OrderedDict

N_LAYER = 40
N_EXPERT = 384
N_USED = 6


def load_cache(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def lane_rows(b, n):
    return [b // n + (1 if i < b % n else 0) for i in range(n)]


# ---------------------------------------------------------------- clocks

class Clock:
    """CLOCK_MONOTONIC_RAW ns of an .evt file <-> unix seconds, from its header."""

    def __init__(self, mono_at_open, real_at_open_ns):
        self.m0 = mono_at_open
        self.r0 = real_at_open_ns

    def to_unix(self, mono_ns):
        return (self.r0 + (mono_ns - self.m0)) / 1e9


def parse_iso(ts):
    """'2026-10-03T19:02:18.326283Z' -> unix seconds."""
    ts = ts.rstrip('Z')
    main, frac = (ts.split('.') + ['0'])[:2]
    t = calendar.timegm(time.strptime(main, '%Y-%m-%dT%H:%M:%S'))
    return t + float('0.' + frac)


def load_refresh_times(path):
    out = []
    for line in open(path):
        p = line.split()
        if p:
            out.append(parse_iso(p[0]))
    return out


# ---------------------------------------------------------------- today's ownership

def hash_box2(layer, e, milli=420):
    """`partition_box2` before the hot set is warm (the 420 default: `set_partition_share` is not called on this path, MEASURED 99.7% match)."""
    h = (layer * 1000003 + e * 7919) % 1000
    return h >= milli


class HotSet:
    """Replica of `expert_pager::hot_set` (counts of router decode picks,
    refresh = top-K with hysteresis + change cap, counts halved)."""

    def __init__(self, per_layer=103, hyst=100, max_change=3, min_picks=20000, hash_milli=420):
        self.k = per_layer
        self.hyst = hyst
        self.cap = max_change
        self.min_picks = min_picks
        self.hash_milli = hash_milli
        self.counts = [[0] * N_EXPERT for _ in range(N_LAYER)]
        self.own = [[False] * N_EXPERT for _ in range(N_LAYER)]
        self.warm = False
        self.total = 0

    def note(self, layer, e):
        self.counts[layer][e] += 1
        self.total += 1

    def box2(self, layer, e):
        if not self.warm:
            return hash_box2(layer, e, self.hash_milli)
        return not self.own[layer][e]

    def refresh(self):
        if self.total < self.min_picks:
            return None
        k, hyst, warm = self.k, self.hyst, self.warm
        changed = 0
        for l in range(N_LAYER):
            cnt = self.counts[l]
            order = sorted(range(N_EXPERT), key=lambda e: (-cnt[e], e))
            rank = [0] * N_EXPERT
            for i, e in enumerate(order):
                rank[e] = i
            own = [False] * N_EXPERT
            n_own = 0
            prev = self.own[l]
            if warm:
                for e in range(N_EXPERT):
                    if prev[e] and rank[e] < k + hyst and cnt[e] > 0 and n_own < k:
                        own[e] = True
                        n_own += 1
            for i, e in enumerate(order):
                c = cnt[e]
                if c == 0 or i >= k + hyst:
                    break
                if not own[e] and n_own < k and i < k:
                    own[e] = True
                    n_own += 1
            if warm and self.cap > 0:
                newc = [(cnt[e], e) for e in range(N_EXPERT) if own[e] and not prev[e]]
                if len(newc) > self.cap:
                    newc.sort(key=lambda x: (-x[0], x[1]))
                    dep = [(cnt[e], e) for e in range(N_EXPERT) if not own[e] and prev[e]]
                    dep.sort(key=lambda x: (-x[0], x[1]))
                    excess = len(newc) - self.cap
                    for _, e in newc[self.cap:]:
                        own[e] = False
                    for _, e in dep[:excess]:
                        own[e] = True
            for e in range(N_EXPERT):
                if own[e] != prev[e]:
                    changed += 1
                cnt[e] //= 2
            self.own[l] = own
        self.total //= 2
        self.warm = True
        return changed


# ---------------------------------------------------------------- pools

class LruPool:
    """Per-layer LRU of `cap` slots over the ids not in `pinned`."""

    def __init__(self, caps, pinned=None):
        self.caps = list(caps)
        self.lru = [OrderedDict() for _ in range(N_LAYER)]
        self.pinned = pinned if pinned is not None else [set() for _ in range(N_LAYER)]

    def resident(self, layer, e):
        return e in self.pinned[layer] or e in self.lru[layer]

    def touch(self, layer, e):
        """Use e. Returns True on a miss (and admits it, evicting the LRU)."""
        if e in self.pinned[layer]:
            return False
        d = self.lru[layer]
        if e in d:
            d.move_to_end(e)
            return False
        cap = self.caps[layer]
        if cap <= 0:
            return True
        d[e] = True
        if len(d) > cap:
            d.popitem(last=False)
        return True
