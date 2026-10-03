#!/usr/bin/env python3
"""Sensitivity of the policy ranking: re-run a short list of policies under
perturbed model assumptions and print mean fwd (held-out half) and the change
vs today's policy under the SAME assumptions.

  sensitivity.py CACHE.pkl REFRESH.txt HUB.evt --costs costs.json --params params.json [--out sens.json]
"""
import copy
import json
import statistics as st
import sys

import calibrate
import compare
import replay
import simlib

POLICIES = ['a_today_live', 'a_static133', 'c_share0.55', 'c_share0.60', 'd_rep10_0.60', 'd_rep25_0.60']

# (label, param overrides, replay kwargs)
CONFIGS = [
    ('base', {}, {}),
    ('link x2', {'link_scale': 2.0}, {}),
    ('link +150us/req', {'link_add_us': 150.0}, {}),
    ('link +400us/req', {'link_add_us': 400.0}, {}),
    ('box-2 srv x1.25', {'b2_scale': 1.25}, {}),
    ('box-2 srv x0.8', {'b2_scale': 0.8}, {}),
    ('box-2 per-expert +25%', {'s_d_mult': 1.25}, {}),
    ('box-1 iGPU per-expert +15%', {'ig_d_mult': 1.15}, {}),
    ('host poll 0.06 (T0 refit)', {'h_poll': 0.06}, {}),
    ('host poll 0.12 (T0 refit)', {'h_poll': 0.12}, {}),
    ('no H2D drain (async copy fix)', {'h2d_drain': 0}, {}),
    ('post gated on MoE event', {'post_wait_moe': 1}, {}),
    ('misses: raw LRU (keep 1.0)', {}, {'miss_keep': 1.0}),
    ('misses: page 4.4 ms', {'page_us': 4400.0}, {}),
    ('prefill pollutes box-2 LRU', {}, {'prefill_pollute': True}),
    ('router picks (no cache prior)', {}, {'picks': 'router'}),
]


def main():
    cache_path, refresh_path, evt_path = sys.argv[1:4]
    a = sys.argv
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    sys.path.insert(0, '..')
    from evt2perfetto import header_of
    h = header_of(evt_path)[1]
    clk = simlib.Clock(h['t_mono_raw_at_open'], h['t_realtime_at_open'])
    cache = simlib.load_cache(cache_path)
    refresh = simlib.load_refresh_times(refresh_path)
    P0 = calibrate.load_params(opt('--params'), opt('--costs'))
    T0 = {int(k): v for k, v in json.load(open(opt('--params'))).get('T0', {}).items()}
    n = len(cache['steps'])
    half = n // 2
    counts = compare.first_half_counts(cache, half)
    pols = {name: mk for name, mk, _ in compare.build_policies(['a', 'c', 'd'], counts, refresh,
                                                                {'targets': (0.55, 0.60), 'rep_targets': (0.60,)})}
    only = opt('--configs')
    out = {}
    print(f"{'config':32s} " + ' '.join(f'{p:>14s}' for p in POLICIES))
    for label, over, kw in CONFIGS:
        if only and label not in only.split(';'):
            continue
        P = copy.copy(P0)
        for k, v in over.items():
            if k == 's_d_mult':
                P.s_d = P0.s_d * v
            elif k == 'ig_d_mult':
                P.ig_d = P0.ig_d * v
            else:
                setattr(P, k, v)
        kw2 = {'miss_keep': float(opt('--miss-keep', '0.336')), 'picks': 'ran'}
        kw2.update(kw)
        res_by = {}
        for name in POLICIES:
            res = replay.run(cache, pols[name](), P, clk, eval_from=half, **kw2)
            res_by[name] = res
        # T0 refit when the host model changes: hold today's replay at its calibrated level
        t0 = dict(T0)
        if 'h_poll' in over or 'h2d_drain' in over:
            if 'h_poll' in over:
                # refit T0 so today's policy matches the measured mean on the held-out half
                base = res_by['a_today_live']
                for lanes in (1, 2):
                    v = [r['meas']['fwd_ms'] - r['fwd_core'] for r in base if r['lanes'] == lanes]
                    t0[lanes] = st.median(v)
        means = {}
        for name, res in res_by.items():
            means[name] = st.fmean(r['fwd_core'] + t0[r['lanes']] for r in res)
        b = means['a_today_live']
        out[label] = means
        print(f'{label:32s} ' + ' '.join(f'{means[p]:6.1f} ({100 * (means[p] - b) / b:+5.1f}%)' for p in POLICIES), flush=True)
    if opt('--out'):
        json.dump(out, open(opt('--out'), 'w'), indent=1)


if __name__ == '__main__':
    main()
