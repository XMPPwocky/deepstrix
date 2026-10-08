#!/usr/bin/env python3
"""Calibrate the step model on TODAY's policy: replay today's ownership over the
picks that RAN, simulate every warm single-stream step, compare per
(lanes, rows) with the measured hub_step medians.

  calibrate.py CACHE.pkl REQ.tsv REFRESH.txt HUB.evt [--params params.json] [--fit]

--fit: grid-search the host costs (h_r1+h_r2, h_p2) and the per-lane-count
constant T0 (step setup + final sync) on the cells' median fwd; prints the best.
"""
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict

import des
import policies
import replay
import simlib


def f(x):
    try:
        return float(x)
    except ValueError:
        return float('nan')


def load_params(path=None, costs=None):
    P = des.Params()
    if costs:
        c = json.load(open(costs))
        ig = c['igpu']['total']['beta_ms']
        P.ig0, P.ig_d, P.ig_n, P.ig_b = ig['per_lane_layer'], ig['per_distinct'], ig['per_pick'], ig['per_row']
        b2 = c['box2']['srv_us']
        P.s0, P.s_d, P.s_b = b2['s0'], b2['s_per_distinct'], b2['s_per_row']
        P.link_us = {int(k): float(v) for k, v in c['box2']['link_us_median_by_b'].items()}
        P.page_us = c['box2'].get('page_us_per_paged', P.page_us)
        dl = c['dgpu_lane_layer']
        P.chain0, P.chain_b = dl['chain']['per_lane_layer_ms'], dl['chain']['per_row_ms']
        # mid = shared expert + mix_ffn_late (+ the push, which runs on the xfer stream)
        P.mid0 = dl['mid']['per_lane_layer_ms'] - P.push0
        P.mid_b = dl['mid']['per_row_ms']
        P.comb_local = dl['post']['per_lane_layer_ms'] / 2
        P.comb_remote0 = dl['post']['per_lane_layer_ms'] / 2
        P.comb_b = dl['post']['per_row_ms']
    if path:
        for k, v in json.load(open(path)).items():
            if k == 'T0':
                continue
            setattr(P, k, v)
    return P


def measured_req(req_path):
    """Per step: [requests, paged experts]; and per (step, lane index, layer)
    the paged count (lane index: lane A = the lane that submits a common layer first)."""
    agg = defaultdict(lambda: [0, 0.0, 0])
    per = defaultdict(dict)  # step -> {(lane id, layer): (t_submit, paged)}
    for r in csv.DictReader(open(req_path), delimiter='\t'):
        s = f(r['step'])
        if math.isnan(s):
            continue
        a = agg[int(s)]
        a[0] += 1
        npg = f(r['n_paged'])
        npg = 0.0 if math.isnan(npg) else npg
        a[1] += npg
        per[int(s)][(int(f(r['lane'])), int(f(r['layer'])))] = (f(r['t_submit']), npg)
    return agg, _m2_from(per)


def measured_split(req_path):
    """{(step, lane index, layer): (box-2 picks, box-2 distinct)} from hub_req:
    the split the hub ACTUALLY made (a lane-layer without a request had none on
    box 2). Box 1's share is the rest of the lane-layer's picks. Lane index as
    in `_m2_from` (lane A submits a common layer first)."""
    per = defaultdict(dict)
    for r in csv.DictReader(open(req_path), delimiter='\t'):
        s = f(r['step'])
        if math.isnan(s):
            continue
        per[int(s)][(int(f(r['lane'])), int(f(r['layer'])))] = (f(r['t_submit']), int(f(r['n_picks'])), int(f(r['n_distinct'])))
    out = {}
    for s, d in per.items():
        ids = sorted({k[0] for k in d})
        order = ids
        if len(ids) == 2:
            a, b = ids
            common = sorted({k[1] for k in d if k[0] == a} & {k[1] for k in d if k[0] == b})
            if common:
                l0 = common[0]
                order = [a, b] if d[(a, l0)][0] <= d[(b, l0)][0] else [b, a]
        for (lid, layer), (_, npk, nd) in d.items():
            out[(s, order.index(lid), layer)] = (npk, nd)
    return out


def measured_cadence(req_path):
    """Per step: (periods, sub_to_wait, exit_to_next_sub) lists (ms)."""
    rows = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(req_path), delimiter='\t'):
        s = f(r['step'])
        if math.isnan(s):
            continue
        rows[int(s)][int(f(r['lane']))].append((int(f(r['layer'])), f(r['t_submit']), f(r['t_wait_enter']), f(r['t_wait_exit'])))
    out = {}
    for s, lanes in rows.items():
        per, s2w, ex2n = [], [], []
        for lid, v in lanes.items():
            v.sort()
            for x in v:
                s2w.append((x[2] - x[1]) / 1e6)
            for x, y in zip(v, v[1:]):
                if y[0] == x[0] + 1:
                    per.append((y[1] - x[1]) / 1e6)
                    ex2n.append((y[1] - x[3]) / 1e6)
        out[s] = (per, s2w, ex2n)
    return out


def _m2_from(per):
    m2 = {}
    for s, d in per.items():
        ids = sorted({k[0] for k in d})
        order = ids
        if len(ids) == 2:
            a, b = ids
            common = sorted({k[1] for k in d if k[0] == a} & {k[1] for k in d if k[0] == b})
            if common:
                l0 = common[0]
                order = [a, b] if d[(a, l0)][0] <= d[(b, l0)][0] else [b, a]
        for (lid, layer), (_, npg) in d.items():
            if npg:
                m2[(s, order.index(lid), layer)] = int(npg)
    return m2


def print_table(rows, req=None, cache=None, results=None):
    print(f"{'cell':>8} {'n':>5} | {'fwd meas':>8} {'sim':>7} {'err%':>6} | {'iGPU meas':>9} {'sim':>6} | "
          f"{'dGPU meas':>9} {'sim':>6} | {'rtt meas':>8} {'sim':>6} | {'wait meas':>9} {'sim':>6} | "
          f"{'b2miss meas':>11} {'sim':>5} | {'b1miss meas':>11} {'sim':>5} | share1")
    for r in rows:
        k = r['cell']
        m2m = float('nan')
        if req is not None and results is not None:
            v = [req.get(int(x['meas']['step']), [0, 0.0, 0])[1] for x in results if (x['lanes'], x['rows']) == k]
            m2m = st.fmean(v) if v else float('nan')
        err = 100 * (r['fwd_sim'] - r['fwd_meas']) / r['fwd_meas']
        print(f"{k[0]}L r{k[1]:<4} {r['n']:5d} | {r['fwd_meas']:8.1f} {r['fwd_sim']:7.1f} {err:+6.1f} | "
              f"{r['igpu_meas']:9.1f} {r['igpu_sim']:6.1f} | {r['dgpu_meas']:9.1f} {r['dgpu_sim']:6.1f} | "
              f"{r['rtt_meas']:8.1f} {r['rtt_sim']:6.1f} | {r['wait_meas']:9.1f} {r['wait_sim']:6.1f} | "
              f"{m2m:11.2f} {r['m2_sim']:5.2f} | {r['b1miss_meas']:11.2f} {r['m1_sim']:5.2f} | {r['share1_sim']:.3f}")


def grid_search(col, P0, spec, cad):
    """spec 'k1=a:b:c;k2=x:y' -> every combination; prints per combination the
    lane-count T0s and the per-cell fwd / period errors with ONE common T0."""
    import copy
    import itertools
    keys, vals = [], []
    for part in spec.split(';'):
        k, v = part.split('=')
        keys.append(k)
        vals.append([float(x) for x in v.split(':')])
    # measured period per cell (pooled)
    for combo in itertools.product(*vals):
        P = copy.copy(P0)
        for k, v in zip(keys, combo):
            setattr(P, k, v)
        res = replay.simulate_collected(col, P)
        by = defaultdict(list)
        for r in res:
            by[r['lanes']].append(r['meas']['fwd_ms'] - r['fwd_core'])
        T0 = {k: st.median(v) for k, v in by.items()}
        t_common = st.median([r['meas']['fwd_ms'] - r['fwd_core'] for r in res])
        g = defaultdict(list)
        for r in res:
            g[(r['lanes'], r['rows'])].append(r)
        errs, perr = [], []
        for k in sorted(g):
            rs = g[k]
            fm = st.median([r['meas']['fwd_ms'] for r in rs])
            fs = st.median([r['fwd_core'] + t_common for r in rs])
            errs.append((k, 100 * (fs - fm) / fm, len(rs)))
            mp, sp = [], []
            for r in rs:
                c = cad.get(int(r['meas']['step']))
                if c:
                    mp += c[0]
                sp += r['cad'][0]
            perr.append(st.median(sp) - st.median(mp) if mp and sp else float('nan'))
        n = sum(e[2] for e in errs)
        wabs = sum(abs(e[1]) * e[2] for e in errs) / n
        mx = max(abs(e[1]) for e in errs)
        print(' '.join(f'{k}={v}' for k, v in zip(keys, combo)),
              f"| T0 1L {T0.get(1, float('nan')):.2f} 2L {T0.get(2, float('nan')):.2f} common {t_common:.2f}"
              f" | w|err| {wabs:.2f}% max {mx:.1f}% |", ' '.join(f'{e[0][0]}L{e[0][1]}:{e[1]:+.1f}' for e in errs),
              '| dper', ' '.join(f'{x:+.2f}' for x in perr))


def main():
    cache_path, req_path, refresh_path, evt_path = sys.argv[1:5]
    params_path = sys.argv[sys.argv.index('--params') + 1] if '--params' in sys.argv else None
    costs_path = sys.argv[sys.argv.index('--costs') + 1] if '--costs' in sys.argv else None
    sys.path.insert(0, '..')
    from evt2perfetto import header_of
    h = header_of(evt_path)[1]
    clk = simlib.Clock(h['t_mono_raw_at_open'], h['t_realtime_at_open'])
    cache = simlib.load_cache(cache_path)
    refresh = simlib.load_refresh_times(refresh_path)
    P = load_params(params_path, costs_path)
    if '--set' in sys.argv:
        for kv in sys.argv[sys.argv.index('--set') + 1].split(','):
            k, v = kv.split('=')
            setattr(P, k, float(v))
    T0 = {}
    if params_path:
        T0 = {int(k): v for k, v in json.load(open(params_path)).get('T0', {}).items()}
    # first warm step: after the first refresh (a windowed cache starts warm)
    first_warm = 0
    snap = cache.get('hs0')
    r0 = simlib.first_refresh_after(refresh, cache)
    if not (snap and snap.get('warm')):
        for i, s in enumerate(cache['steps']):
            if clk.to_unix(s['t_start']) > refresh[0]:
                first_warm = i
                break
    if '--from-step' in sys.argv:
        first_warm = max(first_warm, int(sys.argv[sys.argv.index('--from-step') + 1]))
    req, m2_meas = measured_req(req_path)
    a = sys.argv
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    if opt('--policy', 'today') == 'live-top':
        import policies_live
        pol = policies_live.LiveSplit('top', refresh_times=refresh, snapshot=snap, refresh_start=r0,
                                      b1_slots=int(opt('--b1-slots', 126)), b2_slots=int(opt('--b2-slots', 119)),
                                      pin_total=int(opt('--pin-total', 105)))
    else:
        pol = policies.Today(refresh_times=refresh, snapshot=snap, refresh_start=r0,
                             pin2=int(opt('--pin2', 85)), b2_slots=int(opt('--b2-slots', 106)),
                             b1_slots=int(opt('--b1-slots', 135)))
    use_meas = '--pool-model' not in sys.argv
    b1m = {int(s['step']): (0 if math.isnan(s['b1_misses']) else s['b1_misses']) for s in cache['steps']}
    picks = sys.argv[sys.argv.index('--picks') + 1] if '--picks' in sys.argv else 'ran'
    split = measured_split(req_path) if '--measured-split' in a else None
    col = replay.run(cache, pol, P, clk, eval_from=first_warm, picks=picks, record='collect',
                     m2_override=m2_meas if use_meas else None, b1miss_override=b1m if use_meas else None,
                     miss_keep=float(sys.argv[sys.argv.index('--miss-keep') + 1]) if '--miss-keep' in sys.argv else 1.0,
                     classes=opt('--classes'), split_override=split)
    if split is not None:
        print('box-1 / box-2 split per lane-layer: MEASURED (hub_req n_picks / n_distinct; box 1 = the rest)')
    print('box-2 / box-1 misses:', 'MEASURED (hub_req n_paged, hub_step b1_misses)' if use_meas else 'pool model')
    if '--grid' in sys.argv:
        grid_search(col, P, sys.argv[sys.argv.index('--grid') + 1], measured_cadence(req_path))
        return
    res = replay.simulate_collected(col, P)
    if not T0:
        # T0 per lane count: median of (measured - simulated core)
        by = defaultdict(list)
        for r in res:
            by[r['lanes']].append(r['meas']['fwd_ms'] - r['fwd_core'])
        T0 = {k: st.median(v) for k, v in by.items()}
        print('T0 (median measured - simulated core, ms):', {k: round(v, 2) for k, v in T0.items()})
    rows = replay.cell_table(res, T0)
    print_table(rows, req, cache, res)
    tm = sum(req.get(int(r['meas']['step']), [0, 0.0, 0])[1] for r in res)
    ts = sum(r['m2'] for r in res)
    print(f'box-2 misses over {len(res)} steps: measured {tm:.0f} ({tm / len(res):.2f}/step), simulated {ts} '
          f'({ts / len(res):.2f}/step), ratio meas/sim {tm / max(1, ts):.3f}')
    # traffic-weighted
    n = sum(r['n'] for r in rows)
    wm = sum(r['n'] * r['fwd_meas'] for r in rows) / n
    ws = sum(r['n'] * r['fwd_sim'] for r in rows) / n
    print(f'traffic-weighted median fwd: meas {wm:.1f} sim {ws:.1f} ({100 * (ws - wm) / wm:+.1f}%)')
    # per-step correlation within cells (does the model track step-to-step variation?)
    num = den1 = den2 = 0.0
    g = defaultdict(list)
    for r in res:
        g[(r['lanes'], r['rows'])].append(r)
    for k, rs in g.items():
        if len(rs) < 30:
            continue
        mm = st.fmean(r['meas']['fwd_ms'] for r in rs)
        ms = st.fmean(r['fwd_core'] for r in rs)
        for r in rs:
            a = r['meas']['fwd_ms'] - mm
            b = r['fwd_core'] - ms
            num += a * b
            den1 += a * a
            den2 += b * b
    print(f'within-cell correlation of per-step fwd (meas vs sim): {num / math.sqrt(den1 * den2):.3f}; '
          f'slope meas-on-sim {num / den2:.2f}')
    # host cadence per cell (lane-layers with a box-2 request)
    cad = measured_cadence(req_path)
    print(f"{'cell':>8} | {'period meas':>11} {'sim':>6} | {'sub->wait meas':>14} {'sim':>6} | {'exit->next meas':>15} {'sim':>6}")
    for k in sorted(g):
        rs = g[k]
        mp, ms_, me = [], [], []
        sp, ss, se = [], [], []
        for r in rs:
            c = cad.get(int(r['meas']['step']))
            if c:
                mp += c[0]
                ms_ += c[1]
                me += c[2]
            sp += r['cad'][0]
            ss += r['cad'][1]
            se += r['cad'][2]
        md = lambda v: st.median(v) if v else float('nan')  # noqa: E731
        print(f"{k[0]}L r{k[1]:<4} | {md(mp):11.3f} {md(sp):6.3f} | {md(ms_):14.3f} {md(ss):6.3f} | {md(me):15.3f} {md(se):6.3f}")
    if '--dump' in sys.argv:
        out = sys.argv[sys.argv.index('--dump') + 1]
        with open(out, 'w') as fh:
            fh.write('si\tstep\tlanes\trows\tfwd_meas\tfwd_core\tigpu_meas\tigpu_sim\trtt_meas\trtt_sim\tm2\tm1\td1\td2\tn1\tn2\tb1miss\tb2page\n')
            for r in res:
                m = r['meas']
                fh.write(f"{r['si']}\t{int(m['step'])}\t{r['lanes']}\t{r['rows']}\t{m['fwd_ms']:.3f}\t{r['fwd_core']:.3f}\t"
                         f"{m['igpu_busy_ms']:.3f}\t{r['igpu_busy']:.3f}\t{m['remote_rtt_ms']:.3f}\t{r['rtt_sum']:.3f}\t"
                         f"{r['m2']}\t{r['m1']}\t{r['d1']}\t{r['d2']}\t{r['n1']}\t{r['n2']}\t{m['b1_misses']}\t{m['b2_page_ms']}\n")


if __name__ == '__main__':
    main()
