#!/usr/bin/env python3
"""Replay engine: steps of the aligned cache through a split policy and the
step DES (des.py). Produces per-step predictions and per-(lanes, rows) tables.
"""
import math
import statistics as st
from collections import defaultdict

import des
import simlib

NL, NE = simlib.N_LAYER, simlib.N_EXPERT


class Box2Pool:
    """Box 2's pool: the policy's pinned ids never miss; the rest is an LRU of
    the remaining slots (capacity re-read from the policy, which may re-pin)."""

    def __init__(self, policy):
        self.p = policy
        from collections import OrderedDict
        self.lru = [OrderedDict() for _ in range(NL)]

    def touch(self, l, e, force_hit=False):
        """Use e on layer l; True on a demand miss. force_hit: the pick is a
        cache-prior substitute, resident by construction (admitted, no miss)."""
        if e in self.p.box2_pinned(l):
            return False
        d = self.lru[l]
        if e in d:
            d.move_to_end(e)
            return False
        cap = self.p.box2_lru_cap(l)
        d[e] = True
        while len(d) > max(0, cap):
            d.popitem(last=False)
        return not force_hit


def route_replicated(rep_ids, rep_picks, b, n1, d1, n2, d2, P, lanes):
    """Assign replicated distinct ids to the leg that finishes first.
    Legs (ms): iGPU = I(b, n1, d1); box 2 = link + srv(d2). With two lanes the
    iGPU also runs the other lane's MoE and box 2 the other lane's request, so a
    leg's cost is weighted by the resource's share (both legs alike: plain greedy
    on the per-lane legs). Returns (n1, d1, n2, d2)."""
    def ig(n, d):
        return P.ig0 + (P.ig_d * d + P.ig_n * n if d > 0 else 0.0) + P.ig_b * b

    def b2(n, d):
        if d == 0:
            return 0.0
        return P.link(b) + (P.s0 + P.s_d * d + P.s_b * b) * P.b2_scale / 1000.0

    # hottest replicated first (most picks)
    for e in sorted(rep_ids, key=lambda x: -rep_picks[x]):
        k = rep_picks[e]
        t1 = max(ig(n1 + k, d1 + 1), b2(n2, d2))
        t2 = max(ig(n1, d1), b2(n2 + k, d2 + 1))
        if t1 <= t2:
            n1, d1 = n1 + k, d1 + 1
        else:
            n2, d2 = n2 + k, d2 + 1
    return n1, d1, n2, d2


def _apply_split(split, step, li, l, n1, d1, n2, d2):
    """split_override (calibration): the hub's MEASURED box-2 picks / distinct for
    this lane-layer (none without a request); box 1 gets the rest."""
    mn, md = split.get((step, li, l), (0, 0))
    n, d = n1 + n2, d1 + d2
    mn, md = min(mn, n), min(md, d)
    return n - mn, d - md, mn, md


def keep_miss_fn(miss_keep):
    """The pool model's miss thinning (calibrated on today's measured n_paged)."""
    def keep(si, l, e):
        return miss_keep >= 1.0 or ((si * 2654435761 + l * 40503 + e * 97) % 10007) / 10007.0 < miss_keep
    return keep


def sim_classes(spec):
    """'1' -> live == 1 only (the 10-03 sim); '1,2' -> live 1 or 2; 'all' -> every step
    with <= 2 lanes; a callable is used as is (step dict -> bool)."""
    if callable(spec):
        return spec
    if spec in (None, '1'):
        return lambda st: st['live'] == 1
    if spec == 'all':
        return lambda st: st['lanes'] <= 2
    vals = {float(x) for x in str(spec).split(',')}
    return lambda st: st['live'] in vals


def run(cache, policy, P, clk, eval_from=0, picks='router', engram=True, observe=True, prefill_pollute=False,
        record=None, m2_override=None, b1miss_override=None, prior_hits=True, miss_keep=1.0, classes=None,
        split_override=None):
    """Replay all steps (pools warm from step 0); simulate steps >= eval_from.
    m2_override: {(step id, lane index, layer): measured box-2 paged experts}
    (calibration: the MEASURED misses instead of the pool model's).
    b1miss_override: {step id: measured box-1 misses} spread over the step's first lane-layers.
    classes: which steps are simulated (`sim_classes`; default live == 1).
    A policy with `classify` (policies_live) owns both boxes' residency and
    decides every lane-layer itself (its own box-2 pool, holder fallback, legs).
    Returns list of per-step result dicts (simulated steps only)."""
    live_pol = hasattr(policy, 'classify')
    pool2 = None if live_pol else Box2Pool(policy)
    keep_miss = keep_miss_fn(miss_keep)
    want = sim_classes(classes)
    out = []
    pf = cache.get('prefill', []) if prefill_pollute else []
    pfi = 0
    for si, stp in enumerate(cache['steps']):
        t_unix = clk.to_unix(stp['t_start']) if clk else 0.0
        policy.begin_step(si, t_unix)
        while prefill_pollute and pfi < len(pf) and pf[pfi][0] <= si:
            _, l, b, ids = pf[pfi]
            for e, c in ids:
                if live_pol:
                    policy.prefill_touch(l, e)
                elif not policy.owner1(l, e):
                    pool2.touch(l, e)
            pfi += 1
        lanes = int(stp['lanes'])
        simulate = si >= eval_from and want(stp)
        lane_ll = []
        tot = defaultdict(int)
        for li in range(lanes):
            lls = []
            for l in range(NL):
                rows = stp[picks][li][l]
                b = len(rows)
                p1, p2 = [], []
                rep = {}
                subst = set()
                if picks == 'ran' and prior_hits:
                    # picks the cache prior swapped IN (ran but not the router's own)
                    for rr, ro in zip(rows, stp['router'][li][l]):
                        if rr is not ro and rr != ro:
                            subst.update(set(rr) - set(ro))
                if live_pol:
                    n1, d1, n2, d2, m1, m2, nrep = policy.classify(si, l, rows, subst, b, keep_miss)
                    if split_override is not None:
                        n1, d1, n2, d2 = _apply_split(split_override, int(stp['step']), li, l, n1, d1, n2, d2)
                    if m2_override is not None:
                        m2 = m2_override.get((int(stp['step']), li, l), 0)
                    if b1miss_override is not None:
                        m1 = 0
                    tot['n1'] += n1
                    tot['n2'] += n2
                    tot['d1'] += d1
                    tot['d2'] += d2
                    tot['m1'] += m1
                    tot['m2'] += m2
                    tot['rep'] += nrep
                    lls.append(des.LL(b, n1, d1, n2, d2, m2, m1))
                    continue
                for row in rows:
                    for e in row:
                        if not (0 <= e < NE):
                            continue
                        if policy.replicated(l, e):
                            rep[e] = rep.get(e, 0) + 1
                        elif policy.owner1(l, e):
                            p1.append(e)
                        else:
                            p2.append(e)
                d1s = set(p1)
                d2s = set(p2)
                m1 = sum(1 for e in d1s if policy.box1_miss(l, e))
                m2 = 0
                for e in d2s:
                    if pool2.touch(l, e, force_hit=e in subst):
                        # miss_keep < 1: only that fraction of the pool model's misses is a
                        # demand read (calibrated on today's measured n_paged; the rest are
                        # covered by box 2's prefetch / pin ledger, which the LRU omits)
                        if miss_keep >= 1.0 or ((si * 2654435761 + l * 40503 + e * 97) % 10007) / 10007.0 < miss_keep:
                            m2 += 1
                if m2_override is not None:
                    m2 = m2_override.get((int(stp['step']), li, l), 0)
                if b1miss_override is not None:
                    m1 = 0
                n1, d1, n2, d2 = len(p1), len(d1s), len(p2), len(d2s)
                if rep:
                    n1, d1, n2, d2 = route_replicated(list(rep), rep, b, n1, d1, n2, d2, P, lanes)
                if split_override is not None:
                    n1, d1, n2, d2 = _apply_split(split_override, int(stp['step']), li, l, n1, d1, n2, d2)
                tot['n1'] += n1
                tot['n2'] += n2
                tot['d1'] += d1
                tot['d2'] += d2
                tot['m1'] += m1
                tot['m2'] += m2
                tot['rep'] += sum(rep.values())
                lls.append(des.LL(b, n1, d1, n2, d2, m2, m1))
            lane_ll.append(lls)
        if b1miss_override is not None:
            k = int(b1miss_override.get(int(stp['step']), 0) or 0)
            # measured box-1 misses: one per lane-layer from the top (where they are unknown)
            for li in range(lanes):
                for l in range(NL):
                    if k <= 0:
                        break
                    lane_ll[li][l].m1 += 1
                    tot['m1'] += 1
                    k -= 1
        if observe:
            for li in range(lanes):
                for l in range(NL):
                    for row in stp['router'][li][l]:
                        for e in row:
                            if 0 <= e < NE:
                                policy.observe(l, e)
        if simulate:
            ej = stp.get('lh_engram_join', 0.0) if engram else 0.0
            if ej is None or (isinstance(ej, float) and math.isnan(ej)):
                ej = 0.0
            if record == 'collect':
                r = {'ll': lane_ll, 'ej': ej}
            else:
                r = des.simulate(lane_ll, P, engram_join_ms=ej)
            r.update({k: v for k, v in tot.items()})
            if live_pol and policy.cur is not None:
                r['pol'] = dict(policy.cur)
            r['si'] = si
            # the step CLASS as a lane code (cells and T0 are keyed by it): plain
            # lanes for a lone stream (the 10-03 cells), 20 + lanes for live == 2
            # (two-stream DSpark / two plain streams), 30 + lanes for live >= 3
            lv = stp['live']
            r['lanes'] = lanes if not lv >= 2 else (20 if lv == 2 else 30) + lanes
            r['rows'] = int(stp['rows'])
            r['meas'] = stp
            out.append(r)
    return out


def simulate_collected(collected, P):
    """Re-run the DES over steps collected with record='collect'."""
    out = []
    for c in collected:
        r = des.simulate(c['ll'], P, engram_join_ms=c['ej'])
        for k, v in c.items():
            if k not in ('ll', 'ej'):
                r[k] = v
        out.append(r)
    return out


def _nz(x):
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else x


def cell_table(results, T0, keys=None):
    """Per (lanes, rows): medians of measured vs simulated."""
    g = defaultdict(list)
    for r in results:
        g[(r['lanes'], r['rows'])].append(r)
    rows = []
    for k in sorted(g):
        rs = g[k]
        m = rs[0]['meas']
        def med(v):
            v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
            return st.median(v) if v else float('nan')
        rows.append({
            'cell': k, 'n': len(rs),
            'fwd_meas': med([r['meas']['fwd_ms'] for r in rs]),
            'fwd_sim': med([r['fwd_core'] + T0.get(k[0], 0.0) for r in rs]),
            'igpu_meas': med([r['meas']['igpu_busy_ms'] for r in rs]),
            'igpu_sim': med([r['igpu_busy'] for r in rs]),
            'dgpu_meas': med([r['meas']['dgpu_busy_ms'] for r in rs]),
            'dgpu_sim': med([r['dgpu_busy'] + _nz(r['meas'].get('d_engram')) + _nz(r['meas'].get('d_head_batch')) for r in rs]),
            'rtt_meas': med([r['meas']['remote_rtt_ms'] for r in rs]),
            'rtt_sim': med([r['rtt_sum'] for r in rs]),
            'wait_meas': med([r['meas']['remote_wait_ms'] for r in rs]),
            'wait_sim': med([r['wait_exposed'] for r in rs]),
            'm2_sim': st.fmean([r['m2'] for r in rs]),
            'm1_sim': st.fmean([r['m1'] for r in rs]),
            'b1miss_meas': st.fmean([0.0 if math.isnan(r['meas']['b1_misses']) else r['meas']['b1_misses'] for r in rs]),
            'share1_sim': st.fmean([r['n1'] / max(1, r['n1'] + r['n2']) for r in rs]),
        })
    return rows
