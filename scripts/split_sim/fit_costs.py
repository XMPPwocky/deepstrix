#!/usr/bin/env python3
"""Fit the per-box cost model from today's records.

  fit_costs.py CACHE.pkl REQ.tsv REFRESH.txt HUB.evt OUT.json

1. Replays today's ownership (simlib.HotSet, validated by check_ownership.py)
   over the RAN picks, giving per lane-layer (rows b, box-1 picks n1 / distinct d1,
   box-2 picks n2 / distinct d2).
2. Box-1 iGPU: per-step device time (i_pair_kwide, i_q2k_down, the rest) regressed
   on per-step sums of lane-layer features -> per lane-layer cost
   I(b, n1, d1) = c_ll + c_d * d1 + c_n * n1 (+ c_b * b).
3. dGPU: per lane-layer chain / shared / combine from the d_* stage sums,
   regressed on lanes and rows.
4. Box 2: srv(d, b) = s0 + s_d * d + s_b * b with paging separate; link(b) by b
   (rtt - srv on non-paging requests); page cost per paged expert.
Writes OUT.json with the coefficients and fit diagnostics.
"""
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict

import simlib


def f(x):
    try:
        return float(x)
    except ValueError:
        return float('nan')


def lstsq(X, y, w=None):
    """Ordinary least squares via normal equations (small k)."""
    k = len(X[0])
    A = [[0.0] * k for _ in range(k)]
    bv = [0.0] * k
    for i, xi in enumerate(X):
        wi = 1.0 if w is None else w[i]
        for a in range(k):
            bv[a] += wi * xi[a] * y[i]
            for c in range(k):
                A[a][c] += wi * xi[a] * xi[c]
    # Gaussian elimination
    M = [row[:] + [bv[i]] for i, row in enumerate(A)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        if abs(M[i][i]) < 1e-12:
            continue
        for r in range(k):
            if r != i:
                fac = M[r][i] / M[i][i]
                for c in range(i, k + 1):
                    M[r][c] -= fac * M[i][c]
    return [M[i][k] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(k)]


def r2(X, y, beta):
    pred = [sum(a * b for a, b in zip(x, beta)) for x in X]
    m = st.fmean(y)
    ss_t = sum((v - m) ** 2 for v in y)
    ss_r = sum((v - p) ** 2 for v, p in zip(y, pred))
    return 1 - ss_r / ss_t if ss_t else float('nan'), pred


def replay_today_features(cache, refresh, clk):
    """Per step: list of lane-layer feature tuples under today's ownership."""
    hs = simlib.HotSet()
    ri = 0
    out = []
    for stp in cache['steps']:
        t_unix = clk.to_unix(stp['t_start'])
        while ri < len(refresh) and refresh[ri] <= t_unix:
            hs.refresh()
            ri += 1
        lanes = int(stp['lanes'])
        ll = []
        for li in range(lanes):
            for l in range(simlib.N_LAYER):
                rows = stp['ran'][li][l]
                p1 = [e for row in rows for e in row if 0 <= e < simlib.N_EXPERT and not hs.box2(l, e)]
                p2 = [e for row in rows for e in row if 0 <= e < simlib.N_EXPERT and hs.box2(l, e)]
                ll.append((li, l, len(rows), len(p1), len(set(p1)), len(p2), len(set(p2))))
        out.append((stp, ll, hs.warm))
        for li in range(lanes):
            for l in range(simlib.N_LAYER):
                for row in stp['router'][li][l]:
                    for e in row:
                        if 0 <= e < simlib.N_EXPERT:
                            hs.note(l, e)
    return out


def main():
    cache_path, req_path, refresh_path, evt_path, out_path = sys.argv[1:6]
    sys.path.insert(0, '..')
    from evt2perfetto import header_of
    h = header_of(evt_path)[1]
    clk = simlib.Clock(h['t_mono_raw_at_open'], h['t_realtime_at_open'])
    cache = simlib.load_cache(cache_path)
    refresh = simlib.load_refresh_times(refresh_path)
    feats = replay_today_features(cache, refresh, clk)
    res = {}

    # ---------------- box-1 iGPU per lane-layer
    X, Yk, Yd, Yo, Yt, keys = [], [], [], [], [], []
    for stp, ll, warm in feats:
        if stp['live'] != 1 or not warm or not (stp['b1_misses'] == 0):
            continue
        if any(math.isnan(stp[k]) for k in ('i_pair_kwide', 'i_q2k_down', 'igpu_busy_ms')):
            continue
        n_ll = len(ll)
        sd1 = sum(x[4] for x in ll)
        sn1 = sum(x[3] for x in ll)
        sb = sum(x[2] for x in ll)
        X.append([n_ll, sd1, sn1, sb])
        Yk.append(stp['i_pair_kwide'])
        Yd.append(stp['i_q2k_down'])
        Yt.append(stp['igpu_busy_ms'])
        Yo.append(stp['igpu_busy_ms'] - stp['i_pair_kwide'] - stp['i_q2k_down'])
        keys.append((int(stp['lanes']), int(stp['rows'])))
    names = ['per_lane_layer', 'per_distinct', 'per_pick', 'per_row']
    fits = {}
    for nm, Y in (('kwide', Yk), ('down', Yd), ('other', Yo), ('total', Yt)):
        beta = lstsq(X, Y)
        rr, pred = r2(X, Y, beta)
        fits[nm] = {'beta_ms': dict(zip(names, beta)), 'r2': rr, 'n': len(Y)}
    # By-cell check of the total fit
    cell = defaultdict(lambda: [[], []])
    beta_t = [fits['total']['beta_ms'][n] for n in names]
    for x, y, k in zip(X, Yt, keys):
        cell[k][0].append(y)
        cell[k][1].append(sum(a * b for a, b in zip(x, beta_t)))
    fits['total']['by_cell'] = {f'{k[0]}L r{k[1]}': (st.median(v[0]), st.median(v[1]), len(v[0])) for k, v in sorted(cell.items())}
    res['igpu'] = fits
    # Mean features per cell (for the report)
    fc = defaultdict(list)
    for stp, ll, warm in feats:
        if stp['live'] != 1 or not warm:
            continue
        k = f"{int(stp['lanes'])}L r{int(stp['rows'])}"
        fc[k].append((sum(x[4] for x in ll), sum(x[3] for x in ll), sum(x[6] for x in ll), sum(x[5] for x in ll)))
    res['features_by_cell'] = {k: {'d1': st.fmean(a[0] for a in v), 'n1': st.fmean(a[1] for a in v),
                                   'd2': st.fmean(a[2] for a in v), 'n2': st.fmean(a[3] for a in v), 'n': len(v)}
                               for k, v in sorted(fc.items())}

    # ---------------- dGPU per lane-layer by stage group
    chain_f = ['d_mhc_pre_attn', 'd_q_chain', 'd_kv_chain', 'd_kv_append_compressor_serial', 'd_attn_compute',
               'd_output_proj', 'd_mhc_post_attn', 'd_mhc_pre_ffn', 'd_router', 'd_rb_pack', 'd_prefill_indexer',
               'd_prefill_indexer_reuse']
    mid_f = ['d_shared_expert', 'd_peer_push_ffn_input_norm', 'd_mhc_mix_ffn_late']
    post_f = ['d_ffn_combine_local', 'd_ffn_combine_remote']
    step_f = ['d_engram', 'd_head_batch']
    dg = defaultdict(lambda: defaultdict(list))
    for stp, ll, warm in feats:
        if stp['live'] != 1:
            continue
        k = (int(stp['lanes']), int(stp['rows']))
        def s(fs):
            v = [stp[x] for x in fs]
            return sum(0.0 if math.isnan(a) else a for a in v)
        dg[k]['chain'].append(s(chain_f))
        dg[k]['mid'].append(s(mid_f))
        dg[k]['post'].append(s(post_f))
        dg[k]['step'].append(s(step_f))
        dg[k]['busy'].append(stp['dgpu_busy_ms'])
        dg[k]['shared'].append(stp['d_shared_expert'])
    res['dgpu_by_cell_ms'] = {f'{k[0]}L r{k[1]}': {g: st.median(v) for g, v in d.items()} for k, d in sorted(dg.items())}
    # per lane-layer regression: chain/mid/post ~ a + c * b_lane ; X rows = (lanes*40, rows*40)
    for g in ('chain', 'mid', 'post'):
        Xg, Yg = [], []
        for (lanes, rows), d in dg.items():
            for v in d[g]:
                Xg.append([lanes * 40, rows * 40])
                Yg.append(v)
        beta = lstsq(Xg, Yg)
        res.setdefault('dgpu_lane_layer', {})[g] = {'per_lane_layer_ms': beta[0], 'per_row_ms': beta[1],
                                                     'r2': r2(Xg, Yg, beta)[0]}

    # ---------------- box 2 per request
    srv_x, srv_y = [], []
    link = defaultdict(list)
    page = []
    for r in csv.DictReader(open(req_path), delimiter='\t'):
        b = int(f(r['b']))
        if b > 8:
            continue
        nd, srv, rtt, pg = f(r['n_distinct']), f(r['srv_us']), f(r['rtt_us']), f(r['page_us'])
        npg = f(r['n_paged'])
        if math.isnan(nd) or math.isnan(srv):
            continue
        if pg == 0:
            srv_x.append([1.0, nd, b])
            srv_y.append(srv)
            link[b].append(rtt - srv)
        elif not math.isnan(npg) and npg > 0:
            page.append((npg, pg, srv, nd, b))
    beta = lstsq(srv_x, srv_y)
    res['box2'] = {'srv_us': {'s0': beta[0], 's_per_distinct': beta[1], 's_per_row': beta[2],
                              'r2': r2(srv_x, srv_y, beta)[0], 'n': len(srv_y)},
                   'link_us_median_by_b': {b: st.median(v) for b, v in sorted(link.items())},
                   'link_us_p90_by_b': {b: sorted(v)[int(0.9 * (len(v) - 1))] for b, v in sorted(link.items())}}
    if page:
        # page_us ~ per paged expert (slope through origin) and srv excess
        sx = sum(p[0] * p[1] for p in page)
        sxx = sum(p[0] * p[0] for p in page)
        res['box2']['page_us_per_paged'] = sx / sxx
        res['box2']['page_us_median_1'] = st.median([p[1] for p in page if p[0] == 1] or [float('nan')])
        res['box2']['n_paged_requests'] = len(page)
    with open(out_path, 'w') as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
