#!/usr/bin/env python3
"""Validate (1) the trace <-> hub_step alignment and (2) the replica of today's
ownership: per (step, lane, layer) the box-2 picks / distinct the replica
predicts from the RAN picks vs hub_req's n_picks / n_distinct.

  check_ownership.py CACHE.pkl REQ.tsv REFRESH.txt EVT_HEADER_EVT
"""
import csv
import math
import sys
from collections import Counter, defaultdict

import simlib


def f(x):
    try:
        return float(x)
    except ValueError:
        return float('nan')


def main():
    cache_path, req_path, refresh_path, evt_path = sys.argv[1:5]
    sys.path.insert(0, '..')
    from evt2perfetto import header_of
    h = header_of(evt_path)[1]
    clk = simlib.Clock(h['t_mono_raw_at_open'], h['t_realtime_at_open'])
    refresh = simlib.load_refresh_times(refresh_path)
    cache = simlib.load_cache(cache_path)
    steps = cache['steps']
    # hub_req index
    req = {}
    lane_b = defaultdict(Counter)
    for r in csv.DictReader(open(req_path), delimiter='\t'):
        s = f(r['step'])
        if math.isnan(s):
            continue
        key = (int(s), int(f(r['lane'])), int(f(r['layer'])))
        req[key] = (int(f(r['n_picks'])), int(f(r['n_distinct'])), int(f(r['b'])))
        lane_b[int(s)][int(f(r['lane']))] += 1
    # lane small-id -> index: lane A has b = lane_rows[0]
    hs = simlib.HotSet(hash_milli=int(sys.argv[5]) if len(sys.argv) > 5 else 420)
    # a windowed cache starts from the replica's state before the window
    hs.load(cache.get('hs0'))
    ri = simlib.first_refresh_after(refresh, cache)
    rc = cache.get('meta', {}).get('refresh_check')
    if rc:
        ok = sum(1 for _, x, y in rc if x is not None and y is not None and x == y)
        near = sum(1 for _, x, y in rc if x is not None and y is not None and abs(x - y) <= 2)
        print(f'replica changed== log changed at {ok}/{len(rc)} refreshes ({ok / len(rc):.3f}); within +-2: {near / len(rc):.3f}')
    stats = Counter()
    mism_by_phase = Counter()
    n_ref = 0
    since = 10 ** 9
    by_since = Counter()
    diffs = Counter()
    shift = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    for st in steps:
        since += 1
        t_unix = clk.to_unix(st['t_start'])
        while ri < len(refresh) and refresh[ri] + shift <= t_unix:
            hs.refresh()
            n_ref += 1
            ri += 1
            since = 0
        sid = int(st['step'])
        lanes = int(st['lanes'])
        # Which hub_req lane id is lane A / B for this step: match by b
        ids = sorted(lane_b.get(sid, {}).keys())
        # predicted per lane-layer
        preds = []
        for li in range(lanes):
            for l in range(simlib.N_LAYER):
                rows = st['ran'][li][l]
                b2 = [e for row in rows for e in row if 0 <= e < simlib.N_EXPERT and hs.box2(l, e)]
                preds.append((li, l, len(b2), len(set(b2)), len(rows)))
        # assign hub lane ids: try both mappings, keep the better
        best = None
        import itertools
        for perm in itertools.permutations(ids, min(len(ids), lanes)) if ids else [()]:
            ok = tot = 0
            for (li, l, npk, nd, b) in preds:
                if li >= len(perm):
                    continue
                got = req.get((sid, perm[li], l))
                exp_none = npk == 0
                if got is None:
                    tot += 1
                    ok += exp_none
                else:
                    tot += 1
                    ok += (got[0] == npk and got[1] == nd)
            if best is None or ok > best[0]:
                best = (ok, tot, perm)
        if best is not None:
            stats['lane_layers'] += best[1]
            stats['match'] += best[0]
            phase = 'warm' if hs.warm else 'cold'
            mism_by_phase[phase + '_tot'] += best[1]
            mism_by_phase[phase + '_ok'] += best[0]
            if hs.warm:
                bucket = min(since, 600) // 50
                by_since[(bucket, 'tot')] += best[1]
                by_since[(bucket, 'ok')] += best[0]
                perm = best[2]
                for (li, l, npk, nd, b) in preds:
                    if li < len(perm):
                        got = req.get((sid, perm[li], l))
                        g = got[0] if got else 0
                        diffs[max(-3, min(3, npk - g))] += 1
        # count router picks for the hot set (b <= 8: every decode batch)
        for li in range(lanes):
            for l in range(simlib.N_LAYER):
                for row in st['router'][li][l]:
                    for e in row:
                        if 0 <= e < simlib.N_EXPERT:
                            hs.note(l, e)
    print('refreshes applied', n_ref)
    print('lane-layers', stats['lane_layers'], 'exact match', stats['match'],
          f"{stats['match'] / max(1, stats['lane_layers']):.4f}")
    print('warm match by steps since refresh (50-step buckets):',
          ' '.join(f"{k * 50}:{by_since[(k, 'ok')] / by_since[(k, 'tot')]:.3f}" for k in range(13) if by_since[(k, 'tot')]))
    print('warm pred-actual n_picks:', sorted(diffs.items()))
    for ph in ('cold', 'warm'):
        t = mism_by_phase[ph + '_tot']
        if t:
            print(ph, t, f"{mism_by_phase[ph + '_ok'] / t:.4f}")


if __name__ == '__main__':
    main()
