#!/usr/bin/env python3
"""Exact Python port of crates/v4flash-kernels/src/het/hot_split.rs (the pure
placement `interleave_layer` and the replicated-pick leg choice `choose_legs`),
as of a44685b. Keep it line-for-line comparable with the Rust; `selftest()`
re-runs the Rust host tests' assertions.
"""

B2, B1, B2H, REP = 0, 1, 2, 3
SIDE_NAME = {B2: 'B2', B1: 'B1', B2H: 'B2H', REP: 'REP'}


def box2_home(s):
    return s == B2 or s == B2H


def box1_resident(s):
    return s == B1 or s == REP


def keep(s):
    return s == B2H or s == REP


class IlParams:
    def __init__(self, n1=103, k=0, p2=60, target=0.60, tol=0.02, moves=3, hyst=40, rep_hyst=5):
        self.n1, self.k, self.p2, self.target, self.tol = n1, k, p2, target, tol
        self.moves, self.hyst, self.rep_hyst = moves, hyst, rep_hyst


class LayerPlan:
    __slots__ = ('sides', 'share1', 'swaps', 'b1_new', 'b2_new')

    def __init__(self, sides, share1, swaps, b1_new, b2_new):
        self.sides, self.share1, self.swaps, self.b1_new, self.b2_new = sides, share1, swaps, b1_new, b2_new


def order(counts):
    """Hottest first, ties by id."""
    return sorted(range(len(counts)), key=lambda e: (-counts[e], e))


def interleave_layer(counts, prev, p):
    ne = len(counts)
    assert len(prev) == ne
    n1 = min(p.n1, ne)
    k = min(p.k, n1)
    ord_ = order(counts)
    rank = [ne] * ne
    for i, e in enumerate(ord_):
        rank[e] = i
    side = [B2] * ne
    moves_left = p.moves

    # 1. Replicated set.
    rep = []
    rep_set = set()
    for e in ord_:
        if len(rep) == k:
            break
        if prev[e] == REP and counts[e] > 0 and rank[e] < k + p.rep_hyst:
            rep.append(e)
            rep_set.add(e)
    rep_left = p.moves
    for e in ord_[:k]:
        if len(rep) == k or counts[e] == 0:
            break
        if e not in rep_set and rep_left > 0 and (box1_resident(prev[e]) or moves_left > 0):
            rep.append(e)
            rep_set.add(e)
            rep_left -= 1
            if not box1_resident(prev[e]):
                moves_left -= 1
    if len(rep) < k:
        for e in ord_:
            if len(rep) == k:
                break
            if prev[e] == REP and e not in rep_set:
                rep.append(e)
                rep_set.add(e)
    for e in rep:
        side[e] = REP

    # 2./3. The region and its sides. Non-replicated ranks only.
    cap1 = n1 - len(rep)
    cap2 = p.p2
    region = cap1 + cap2
    nonrep = [e for e in ord_ if side[e] != REP]
    nr_rank = [ne] * ne
    for i, e in enumerate(nonrep):
        nr_rank[e] = i
    mass = max(1.0, float(sum(counts[e] for e in nonrep)))
    n_b1 = n_b2 = 0
    m1 = 0.0
    for e in nonrep:
        if counts[e] == 0 or nr_rank[e] >= region + p.hyst:
            continue
        pe = prev[e]
        if (pe == B1 or pe == REP) and n_b1 < cap1:
            side[e] = B1
            n_b1 += 1
            m1 += counts[e]
        elif (pe == B2H or pe == REP) and n_b2 < cap2:
            side[e] = B2H
            n_b2 += 1
    for e in nonrep:
        if n_b1 == cap1 and n_b2 == cap2:
            break
        if counts[e] == 0 or nr_rank[e] >= region:
            break
        if side[e] != B2:
            continue
        want1 = m1 / mass < p.target
        b1_ok = n_b1 < cap1 and (box1_resident(prev[e]) or moves_left > 0)
        if (want1 or n_b2 == cap2) and b1_ok:
            if not box1_resident(prev[e]):
                moves_left -= 1
            side[e] = B1
            n_b1 += 1
            m1 += counts[e]
        elif n_b2 < cap2:
            side[e] = B2H
            n_b2 += 1

    # 4. Balance.
    swaps = 0
    while moves_left > 0:
        dev = m1 / mass - p.target
        if abs(dev) <= p.tol:
            break
        ones = [e for e in range(ne) if side[e] == B1]
        twos = [e for e in range(ne) if side[e] == B2H]
        best_in = None
        best_any = None
        for a in ones:
            ca = counts[a]
            for b in twos:
                after = (m1 - ca + counts[b]) / mass - p.target
                if abs(after) + 1e-12 >= abs(dev):
                    continue
                if abs(after) <= p.tol:
                    moved = float(ca + counts[b])
                    if best_in is None or moved < best_in[0] - 1e-12 or (abs(moved - best_in[0]) <= 1e-12 and (a, b) < (best_in[1], best_in[2])):
                        best_in = (moved, a, b)
                else:
                    key = abs(after)
                    if best_any is None or key < best_any[0] - 1e-12 or (abs(key - best_any[0]) <= 1e-12 and (a, b) < (best_any[1], best_any[2])):
                        best_any = (key, a, b)
        best = best_in if best_in is not None else best_any
        if best is None:
            break
        _, a, b = best
        side[a] = B2H
        side[b] = B1
        m1 += counts[b] - counts[a]
        swaps += 1
        moves_left -= 1

    b1_new = [e for e in range(ne) if box1_resident(side[e]) and not box1_resident(prev[e])]
    b2_new = [e for e in range(ne) if keep(side[e]) and not keep(prev[e])]
    return LayerPlan(side, m1 / mass, swaps, b1_new, b2_new)


class LegCosts:
    def __init__(self, ig0=0.100, ig_d=0.0774, ig_n=0.0046, ig_b=0.0397, s0_us=116.0, s_d_us=99.9, s_b_us=15.6,
                 link_us=(58.0, 87.0, 118.0, 276.0, 295.0, 318.0), open_ms=0.2):
        self.ig0, self.ig_d, self.ig_n, self.ig_b = ig0, ig_d, ig_n, ig_b
        self.s0_us, self.s_d_us, self.s_b_us = s0_us, s_d_us, s_b_us
        self.link_us = tuple(link_us)
        self.open_ms = open_ms

    def ig(self, b, n, d):
        return self.ig0 + ((self.ig_d * d + self.ig_n * n) if d > 0 else 0.0) + self.ig_b * b

    def b2(self, b, d):
        if d == 0:
            return 0.0
        link = self.link_us[min(max(b, 1), 6) - 1]
        return (link + self.s0_us + self.s_d_us * d + self.s_b_us * b) / 1000.0

    def b2_leg(self, b, d, d0):
        if d == 0:
            return 0.0
        return self.b2(b, d) + (self.open_ms if d0 == 0 else 0.0)


def choose_legs(c, b, rep, n1, d1, n2, d2):
    """rep: [(expert, picks)]; returns per entry True = box 2."""
    idx = sorted(range(len(rep)), key=lambda i: (-rep[i][1], rep[i][0]))
    out = [False] * len(rep)
    d2_0 = d2
    for i in idx:
        kp = rep[i][1]
        t1 = max(c.ig(b, n1 + kp, d1 + 1), c.b2_leg(b, d2, d2_0))
        t2 = max(c.ig(b, n1, d1), c.b2_leg(b, d2 + 1, d2_0))
        if t1 <= t2:
            n1 += kp
            d1 += 1
        else:
            n2 += kp
            d2 += 1
            out[i] = True
    return out


# ---------------------------------------------------------------- the Rust host tests

def selftest():
    def zipf(ne):
        return [100000 // (i + 1) for i in range(ne)]

    def params():
        return IlParams(n1=103, k=10, p2=60, target=0.60, tol=0.02, moves=3, hyst=100, rep_hyst=5)

    def top_k(c, k):
        s = [B2] * len(c)
        for e in order(c)[:k]:
            s[e] = B1
        return s

    def count(s, x):
        return sum(1 for v in s if v == x)

    c = zipf(384)
    prev = top_k(c, 103)
    for _ in range(30):
        p = interleave_layer(c, prev, params())
        r = count(p.sides, REP)
        assert r <= 10 and count(p.sides, B1) + r <= 103 and count(p.sides, B2H) <= 60
        prev = p.sides
    assert count(prev, REP) == 10 and count(prev, B1) == 93 and count(prev, B2H) == 60
    assert all(prev[e] == REP for e in range(10))
    # converges_then_stops
    prev = top_k(c, 103)
    first = interleave_layer(c, prev, params())
    assert first.swaps > 0
    plan, n = first, 1
    while True:
        nxt = interleave_layer(c, plan.sides, params())
        if nxt.sides == plan.sides:
            break
        plan = nxt
        n += 1
        assert n < 20
    assert abs(plan.share1 - 0.60) <= 0.02
    again = interleave_layer(c, plan.sides, params())
    assert again.swaps == 0 and not again.b1_new and not again.b2_new
    # box1_reads_capped
    prev = top_k(c, 103)
    for _ in range(20):
        p = interleave_layer(c, prev, params())
        assert len(p.b1_new) <= 3
        prev = p.sides
    # jitter_is_ignored
    c2 = list(c)
    for i in range(10, 80, 2):
        c2[i], c2[i + 1] = c2[i + 1], c2[i]
    p = interleave_layer(c2, prev, params())
    assert p.swaps <= 1
    assert sum(1 for e in range(384) if p.sides[e] != prev[e]) <= 2
    # dead ids leave, new hot ids enter
    c3 = list(c)
    c3[50] = 0
    c3[300] = c[14]
    p = interleave_layer(c3, prev, params())
    assert p.sides[50] == B2 and p.sides[300] != B2
    # n1 shrinks safely
    pp = params()
    pp.n1 = 80
    plan = interleave_layer(c, prev, pp)
    assert count(plan.sides, B1) + count(plan.sides, REP) <= 80
    # legs
    lc = LegCosts()
    assert choose_legs(lc, 1, [(1, 1)], 1, 1, 0, 0) == [False]
    assert choose_legs(lc, 2, [(1, 2), (2, 1)], 0, 0, 0, 0) == [False, False]
    v = choose_legs(lc, 4, [(1, 3), (2, 2)], 20, 14, 4, 2)
    assert any(v)
    w = choose_legs(lc, 4, [(2, 2), (1, 3)], 20, 14, 4, 2)
    assert v[0] == w[1] and v[1] == w[0]
    return 'ok'


if __name__ == '__main__':
    print(selftest())
