#!/usr/bin/env python3
"""Compare split policies on the held-out half of the trace.

  compare.py CACHE.pkl REFRESH.txt HUB.evt --costs costs.json --params params.json
             [--miss-keep 0.336] [--picks ran|router] [--policies a,b,c,d] [--set k=v,...]
             [--prefill-pollute] [--out results.json]

Ownership of every static policy is learned on the FIRST half (router decode
picks, counted per layer) and evaluated on the SECOND half. Today's policy is
replayed live (its hot set learns online from the start, refreshes at the
logged times) and is evaluated on the same second half. Pools (box-2 LRU) warm
through the first half for every policy. The step mix (lanes, rows) is today's.
"""
import json
import math
import statistics as st
import sys
from collections import defaultdict

import calibrate
import policies
import replay
import simlib

NL, NE = simlib.N_LAYER, simlib.N_EXPERT


def first_half_counts(cache, upto):
    counts = [[0] * NE for _ in range(NL)]
    for stp in cache['steps'][:upto]:
        for lane in stp['router']:
            for l, rows in enumerate(lane):
                for row in rows:
                    for e in row:
                        if 0 <= e < NE:
                            counts[l][e] += 1
    return counts


def mass_curve(counts, ks=(25, 50, 103, 133, 175, 245)):
    out = {}
    for k in ks:
        v = []
        for l in range(NL):
            c = sorted(counts[l], reverse=True)
            v.append(sum(c[:k]) / max(1, sum(c)))
        out[k] = st.fmean(v)
    return out


def summarize(name, res, T0, note=''):
    g = defaultdict(list)
    for r in res:
        g[(r['lanes'], r['rows'])].append(r)
    cells = {}
    for k in sorted(g):
        rs = g[k]
        cells[f'{k[0]}L r{k[1]}'] = {
            'n': len(rs),
            'fwd_med': st.median(r['fwd_core'] + T0[k[0]] for r in rs),
            'fwd_mean': st.fmean(r['fwd_core'] + T0[k[0]] for r in rs),
            'meas_med': st.median(r['meas']['fwd_ms'] for r in rs),
            'igpu': st.median(r['igpu_busy'] for r in rs),
            'rtt': st.median(r['rtt_sum'] for r in rs),
            'm2': st.fmean(r['m2'] for r in rs),
            'm1': st.fmean(r['m1'] for r in rs),
            'host': st.fmean(r['host_busy'] for r in rs),
            'b2busy': st.fmean(r['b2_busy'] for r in rs),
            'dgpu': st.fmean(r['dgpu_busy'] for r in rs),
            'attr': {k: st.fmean(r['attr'][k] for r in rs) for k in rs[0]['attr']},
        }
    n = len(res)
    return {
        'name': name, 'note': note, 'n': n,
        'fwd_mean': st.fmean(r['fwd_core'] + T0[r['lanes']] for r in res),
        'meas_mean': st.fmean(r['meas']['fwd_ms'] for r in res),
        'share1': sum(r['n1'] for r in res) / max(1, sum(r['n1'] + r['n2'] for r in res)),
        'm2_per_step': st.fmean(r['m2'] for r in res),
        'm1_per_step': st.fmean(r['m1'] for r in res),
        'd1_per_step': st.fmean(r['d1'] for r in res),
        'd2_per_step': st.fmean(r['d2'] for r in res),
        'req_per_step': st.fmean(r['n_req'] for r in res),
        'igpu_mean': st.fmean(r['igpu_busy'] for r in res),
        'dgpu_mean': st.fmean(r['dgpu_busy'] for r in res),
        'host_mean': st.fmean(r['host_busy'] for r in res),
        'b2_mean': st.fmean(r['b2_busy'] for r in res),
        'rtt_mean': st.fmean(r['rtt_sum'] for r in res),
        'cells': cells,
    }


def block_bootstrap(base, other, block=50, reps=400, seed=1):
    """95% interval of mean(other)/mean(base) - 1 over held-out steps, resampling
    blocks of consecutive steps (steps of one request are correlated)."""
    import random
    rnd = random.Random(seed)
    n = len(base)
    nb = max(1, n // block)
    vals = []
    for _ in range(reps):
        sb = so = 0.0
        for _ in range(nb):
            i = rnd.randrange(0, max(1, n - block + 1))
            sb += sum(base[i:i + block])
            so += sum(other[i:i + block])
        vals.append(so / sb - 1.0)
    vals.sort()
    return (vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1])


def build_policies(which, counts, refresh, args):
    pols = []
    if 'a' in which:
        pols.append(('a_today_live', lambda: policies.Today(refresh_times=refresh), 'live hot set K=103, logged refreshes'))
        pols.append(('a_static103', lambda: policies.policy_static_today(counts, 103, 85), 'top-103 static (first half)'))
        pols.append(('a_static133', lambda: policies.policy_static_today(counts, 133, 85), 'top-133 static: all box-1 slots'))
    if 'b' in which:
        for p2 in (90,):
            pols.append((f'b_swap_pin{p2}', lambda p2=p2: policies.policy_swap(counts, pin2=p2), f'box 2 pins top-{p2}, box 1 owns next 133'))
    if 'c' in which:
        for t in args.get('targets', (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)):
            pols.append((f'c_share{t:.2f}', lambda t=t: policies.policy_interleave(counts, t), f'interleave, box-1 mass target {t}'))
    if 'd' in which:
        for k in (10, 25, 40):
            for t in args.get('rep_targets', (0.55, 0.65)):
                pols.append((f'd_rep{k}_{t:.2f}', lambda k=k, t=t: policies.policy_interleave(counts, t, rep_k=k),
                             f'top-{k} on both, rest interleaved to box-1 target {t}'))
    return pols


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
    P = calibrate.load_params(opt('--params'), opt('--costs'))
    if opt('--set'):
        for kv in opt('--set').split(','):
            k, v = kv.split('=')
            setattr(P, k, float(v))
    T0 = {int(k): v for k, v in json.load(open(opt('--params'))).get('T0', {}).items()}
    miss_keep = float(opt('--miss-keep', '0.336'))
    picks = opt('--picks', 'ran')
    which = opt('--policies', 'a,b,c,d').split(',')
    n = len(cache['steps'])
    half = n // 2
    counts = first_half_counts(cache, half)
    if opt('--counts-from'):
        # learn the static placements on ANOTHER trace's counts (count_trace.py output),
        # e.g. the pre-move trace: a placement ~1 day stale
        import pickle
        segs = pickle.load(open(opt('--counts-from'), 'rb'))['segments']
        last = int(opt('--counts-last', '0'))
        if last:
            segs = segs[-last:]
        counts = [[sum(s[l][e] for s in segs) for e in range(NE)] for l in range(NL)]
        print(f"static placements learned on {opt('--counts-from')} ({len(segs)} segments)")
    print(f'steps {n}; learn on [0, {half}), evaluate on [{half}, {n}); picks={picks}; miss_keep={miss_keep}')
    print('first-half pick mass of top-K per layer (mean over layers):',
          {k: round(v, 3) for k, v in mass_curve(counts).items()})
    ev_counts = [[0] * NE for _ in range(NL)]
    for stp in cache['steps'][half:]:
        for lane in stp['router']:
            for l, rows in enumerate(lane):
                for row in rows:
                    for e in row:
                        if 0 <= e < NE:
                            ev_counts[l][e] += 1
    print('second-half pick mass of top-K per layer:', {k: round(v, 3) for k, v in mass_curve(ev_counts).items()})
    out = []
    args = {}
    if opt('--targets'):
        args['targets'] = tuple(float(x) for x in opt('--targets').split(':'))
    if opt('--rep-targets'):
        args['rep_targets'] = tuple(float(x) for x in opt('--rep-targets').split(':'))
    only = set(opt('--only').split(',')) if opt('--only') else None
    per_step = {}
    for name, mk, note in build_policies(which, counts, refresh, args):
        if only is not None and name not in only:
            continue
        pol = mk()
        res = replay.run(cache, pol, P, clk, eval_from=half, picks=picks, miss_keep=miss_keep,
                         prefill_pollute='--prefill-pollute' in a)
        per_step[name] = [r['fwd_core'] + T0[r['lanes']] for r in res]
        s = summarize(name, res, T0, note)
        if per_step and len(per_step) > 1:
            base_name = next(iter(per_step))
            s['ci95_vs_first'] = block_bootstrap(per_step[base_name], per_step[name])
        # realized held-out box-1 share of picks and per-layer sizes
        if hasattr(pol, 'own1'):
            s['own1_per_layer'] = st.fmean(len(x) for x in pol.own1)
            s['pin2_per_layer'] = st.fmean(len(x) for x in pol.pin2)
            s['rep_per_layer'] = st.fmean(len(x) for x in pol.rep)
        out.append(s)
        print(f"{name:18s} fwd mean {s['fwd_mean']:6.1f} (meas {s['meas_mean']:.1f}) share1 {s['share1']:.3f} "
              f"b2miss/step {s['m2_per_step']:5.2f} b1miss/step {s['m1_per_step']:.2f} busy: iGPU {s['igpu_mean']:5.1f} "
              f"b2 {s['b2_mean']:5.1f} dGPU {s['dgpu_mean']:5.1f} host {s['host_mean']:5.1f} | rtt {s['rtt_mean']:5.1f} "
              f"req/step {s['req_per_step']:5.1f}", flush=True)
    base = out[0]['fwd_mean'] if out else 1.0
    cells = sorted(out[0]['cells']) if out else []
    print('\npredicted median fwd (ms) by cell; last column = traffic-weighted mean over all held-out steps')
    print(f"{'policy':18s} " + ' '.join(f'{c:>8s}' for c in cells) + f" {'mean':>7s} {'vs a':>6s}")
    print(f"{'measured':18s} " + ' '.join(f"{out[0]['cells'][c]['meas_med']:8.1f}" for c in cells) + f" {out[0]['meas_mean']:7.1f}")
    for s in out:
        ci = s.get('ci95_vs_first')
        cis = f"  [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else ''
        print(f"{s['name']:18s} " + ' '.join(f"{s['cells'][c]['fwd_med']:8.1f}" for c in cells)
              + f" {s['fwd_mean']:7.1f} {100 * (s['fwd_mean'] - base) / base:+6.1f}%{cis}")
    print('(bracket: 95% block-bootstrap interval of the mean change over held-out steps -- sampling only, not model error)')
    print('\nn per cell: ' + ' '.join(f"{c}={out[0]['cells'][c]['n']}" for c in cells))
    for cell in (opt('--detail', '1L r1;2L r3;2L r4;2L r6')).split(';'):
        if cell not in cells:
            continue
        print(f'\n{cell}: means per step (ms): busy iGPU / box-2 / dGPU / host; box-2 rtt sum; misses; '
              f'lane waits: route_lag (host busy elsewhere) / moe_queue / b2_beyond_moe / drain_block')
        for s in out:
            c = s['cells'][cell]
            at = c['attr']
            print(f"  {s['name']:18s} fwd {c['fwd_mean']:6.1f} | iGPU {c['igpu']:5.1f} b2 {c['b2busy']:5.1f} dGPU {c['dgpu']:5.1f} "
                  f"host {c['host']:5.1f} | rtt {c['rtt']:5.1f} | m2 {c['m2']:4.2f} | {at['route_lag']:5.1f} "
                  f"{at['moe_queue']:5.1f} {at['b2_beyond_moe']:5.1f} {at['drain_block']:5.1f}")
    if opt('--out'):
        with open(opt('--out'), 'w') as fh:
            json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
