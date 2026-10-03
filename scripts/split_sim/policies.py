#!/usr/bin/env python3
"""Split policies: which box computes each pick, what each box holds.

Every policy exposes
  begin_step(step, t_unix)            (refresh hooks)
  split(layer, picks_by_row, b, lanes_ctx) -> (box1 ids list, box2 ids list, replicated ids list)
  observe(layer, router_rows)         (learning from router picks, decode only)
  box2_pinned(layer) -> set           (box-2 slots that never miss)
  box2_lru_cap(layer) -> int
  box1_miss(layer, e) -> bool         (box-1 owned but not resident: a sync read)

Replicated picks (policy d) are routed per lane-layer by the replay engine
(`route_replicated`), which knows the lane's rows and the other legs.
"""
import math

import simlib

NL, NE = simlib.N_LAYER, simlib.N_EXPERT


def rank_by_counts(counts):
    """counts: [layer][e] -> per layer list of ids, hottest first (ties by id)."""
    return [sorted(range(NE), key=lambda e: (-counts[l][e], e)) for l in range(NL)]


# ---------------------------------------------------------------- (a) today

class Today:
    """Live hot set (simlib.HotSet, refreshed at the logged times or every
    `refresh_every` steps), box 2 = everything else: pinned top `pin2` of box-2's
    own ids by the same decayed counts (stand-in for the pin ledger) + LRU."""

    name = 'a_today'

    def __init__(self, refresh_times=None, refresh_every=None, per_layer=103, pin2=85, b2_slots=106,
                 b1_slots=135):
        self.hs = simlib.HotSet(per_layer=per_layer)
        self.refresh_times = refresh_times or []
        self.ri = 0
        self.refresh_every = refresh_every
        self.n = 0
        self.pin2 = pin2
        self.b2_slots = b2_slots
        self.pinned = [set() for _ in range(NL)]
        self.b1_extra = b1_slots - per_layer  # box-1 slots beyond the owned set (recently de-owned)
        self.b1_resident = [set() for _ in range(NL)]
        self.b1_lru = [[] for _ in range(NL)]
        self.refreshes = 0

    def _refresh(self):
        before = [list(o) for o in self.hs.own] if self.hs.warm else None
        if self.hs.refresh() is None:
            return
        self.refreshes += 1
        # box-1 residency: newly owned are NOT resident until first use (sync read);
        # de-owned stay resident in the spare slots (LRU of b1_extra)
        for l in range(NL):
            own = self.hs.own[l]
            if before is None:
                self.b1_resident[l] = {e for e in range(NE) if own[e]}  # warm start: assume filled
                self.b1_lru[l] = []
            else:
                for e in range(NE):
                    if before[l][e] and not own[e]:
                        # de-owned: keep as spare
                        if e in self.b1_lru[l]:
                            self.b1_lru[l].remove(e)
                        self.b1_lru[l].append(e)
                        while len(self.b1_lru[l]) > self.b1_extra:
                            ev = self.b1_lru[l].pop(0)
                            if not own[ev]:
                                self.b1_resident[l].discard(ev)
            # box-2 pins: hottest box-2 ids by the (pre-halving) counts ~ halved counts order
            cnt = self.hs.counts[l]
            cand = sorted((e for e in range(NE) if not own[e]), key=lambda e: (-cnt[e], e))
            self.pinned[l] = set(cand[:self.pin2])

    def begin_step(self, step_idx, t_unix):
        self.n += 1
        if self.refresh_every:
            if self.n % self.refresh_every == 0:
                self._refresh()
        else:
            while self.ri < len(self.refresh_times) and self.refresh_times[self.ri] <= t_unix:
                self._refresh()
                self.ri += 1

    def owner1(self, l, e):
        return not self.hs.box2(l, e)

    def observe(self, l, e):
        self.hs.note(l, e)

    def box2_pinned(self, l):
        return self.pinned[l] if self.hs.warm else set()

    def box2_lru_cap(self, l):
        return self.b2_slots - (len(self.pinned[l]) if self.hs.warm else 0)

    def box1_miss(self, l, e):
        """Owned e picked: True if it must be read now (then it is resident)."""
        if not self.hs.warm:
            return False
        if e in self.b1_resident[l]:
            return False
        self.b1_resident[l].add(e)
        return True

    def replicated(self, l, e):
        return False


# ---------------------------------------------------------------- static placements

class Static:
    """A fixed placement learned on the first half:
      own1[l]: ids box 1 owns (resident, never miss);
      pin2[l]: ids box 2 pins (never miss);
      rep[l]:  ids resident on BOTH (counted in both own1 and pin2);
      everything else goes to box 2's LRU of (b2_slots - |pin2|) slots.
    """

    def __init__(self, name, own1, pin2, rep=None, b2_slots=106, note=''):
        self.name = name
        self.own1 = own1
        self.pin2 = pin2
        self.rep = rep or [set() for _ in range(NL)]
        self.b2_slots = b2_slots
        self.note = note

    def begin_step(self, step_idx, t_unix):
        pass

    def owner1(self, l, e):
        return e in self.own1[l] and e not in self.rep[l]

    def replicated(self, l, e):
        return e in self.rep[l]

    def observe(self, l, e):
        pass

    def box2_pinned(self, l):
        return self.pin2[l]

    def box2_lru_cap(self, l):
        return self.b2_slots - len(self.pin2[l])

    def box1_miss(self, l, e):
        return False


def mass_of(counts_l, ids):
    tot = sum(counts_l) or 1
    return sum(counts_l[e] for e in ids) / tot


def policy_static_today(counts, k1=103, pin2=85, b2_slots=106):
    """(a') today's split as a STATIC placement learned on the first half."""
    rk = rank_by_counts(counts)
    own1 = [set(rk[l][:k1]) for l in range(NL)]
    pin = [set(rk[l][k1:k1 + pin2]) for l in range(NL)]
    return Static(f'a_static_top{k1}', own1, pin, b2_slots=b2_slots)


def policy_swap(counts, pin2=90, b1_slots=133, b2_slots=106):
    """(b) box 2 pins the head (top pin2), box 1 owns the next b1_slots ranks;
    the rest is box 2's LRU (b2_slots - pin2)."""
    rk = rank_by_counts(counts)
    pin = [set(rk[l][:pin2]) for l in range(NL)]
    own1 = [set(rk[l][pin2:pin2 + b1_slots]) for l in range(NL)]
    return Static(f'b_swap_pin{pin2}', own1, pin, b2_slots=b2_slots)


def split_head(head, q):
    """Deal head ranks (hottest first) to box 1 at rate q (box 1 takes rank 0;
    q = 1/2 alternates 1,2,1,2..., q = 1/3 deals 1,2,2,1,2,2...)."""
    h1, h2 = [], []
    acc = 1.0 - q + 1e-9
    for e in head:
        acc += q
        if acc >= 1.0:
            h1.append(e)
            acc -= 1.0
        else:
            h2.append(e)
    return h1, h2


def interleave_layer(order, cnt, target, b1_slots, b2_pin_max, rep_k=0):
    """Mass-target interleave for one layer.
    order: ids hottest first. Ranks [0, rep_k) are replicated (resident on both).
    The next H ranks (the head) are dealt between the boxes -- alternating
    (q = 1/2), or 2:1 / 3:1 in box 2's favour when the target needs it; box 1
    then fills its remaining slots with the ranks right after the head (mid
    tail); box 2 pins its head share (at most b2_pin_max) and keeps its other
    slots as an LRU for everything else. (H, q) is chosen so that box 1's owned
    share of the NON-replicated mass (first-half counts) is closest to
    `target`, preferring the plain alternation."""
    rep = order[:rep_k]
    rest = order[rep_k:]
    tot = sum(cnt[e] for e in rest) or 1
    best = None
    cap1 = b1_slots - rep_k
    for qi, q in enumerate((0.5, 0.4, 1.0 / 3.0, 0.25)):
        for H in range(0, len(rest) + 1, 2):
            h1, h2 = split_head(rest[:H], q)
            if len(h2) > b2_pin_max or len(h1) > cap1:
                break
            fill = rest[H:H + (cap1 - len(h1))]
            own1 = h1 + fill
            m1 = sum(cnt[e] for e in own1) / tot
            err = abs(m1 - target) + 0.002 * qi  # prefer alternation unless clearly better
            if best is None or err < best[0] - 1e-12:
                best = (err, set(own1), set(h2), m1, H, q)
    _, own1, pin2, m1, H, q = best
    return own1, pin2, set(rep), m1


def policy_interleave(counts, target, b1_slots=133, b2_slots=106, b2_lru_min=16, rep_k=0, name=None):
    """(c) / (d): box 1 takes alternating head ranks, then mid-tail fill, to hit
    `target` of the layer's (non-replicated) mass; box 2 pins its head share
    (at most b2_slots - b2_lru_min - rep_k) plus the replicated head; the rest of
    box 2's slots are an LRU."""
    rk = rank_by_counts(counts)
    own1, pin2, rep = [], [], []
    for l in range(NL):
        o1, p2, r, _ = interleave_layer(rk[l], counts[l], target, b1_slots, b2_slots - b2_lru_min - rep_k, rep_k)
        own1.append(o1 | r)
        pin2.append(p2 | r)
        rep.append(r)
    nm = name or (f'c_interleave_{target:.2f}' if rep_k == 0 else f'd_rep{rep_k}_{target:.2f}')
    return Static(nm, own1, pin2, rep=rep, b2_slots=b2_slots)
