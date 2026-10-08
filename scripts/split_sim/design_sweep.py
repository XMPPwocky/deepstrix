#!/usr/bin/env python3
"""Price the hot-split DESIGN (docs/v41/HOT_SPLIT_DESIGN.md rev 3) on a
windowed cache: today's live hot set vs the live sticky interleave
(`policies_live.LiveSplit`, `hot_split.interleave_layer` refresh by refresh
from today's incumbents), plus the 10-03 static policies for continuity.

  design_sweep.py CACHE.pkl REFRESH.txt HUB.evt --costs costs.json --params params.json
        [--miss-keep K] [--classes 1|1,2|all] [--only name,name] [--set k=v,...]
        [--pin-total 105] [--prefill-pollute] [--out results.json]

Every policy replays every step from the window start (pools and the live
placements warm through the first half) and is evaluated on the second half
(the static ones are learned on the first half, as compare.py does). Names:
  today_live            policies.Today at today's sizes (old pool model, continuity)
  top                   LiveSplit('top'): today's refresh + explicit residency (the baseline)
  c0.55_s / c0.60_s     10-03 static interleave (133 box-1 / 106 box-2 slots)
  d10_0.60_s            10-03 static top-10 replicated + interleave
  il_N<n1>_P<p2>_T<target>_R<rep>[_O<open_ms>]   the design (live)
"""
import json
import statistics as st
import sys
import time

import calibrate
import compare
import hot_split as hsp
import policies
import policies_live
import replay
import simlib

NL, NE = simlib.N_LAYER, simlib.N_EXPERT

DESIGN_POINTS = [
    # (n1, p2, target, rep, open_ms) -- start at 103/60/0.60/0, vary one at a time
    (103, 60, 0.60, 0, 0.2),
    (115, 60, 0.60, 0, 0.2),
    (124, 60, 0.60, 0, 0.2),
    (103, 50, 0.60, 0, 0.2),
    (103, 75, 0.60, 0, 0.2),
    (103, 60, 0.55, 0, 0.2),
    (103, 60, 0.65, 0, 0.2),
    (103, 60, 0.60, 10, 0.2),
    (103, 60, 0.60, 10, 0.0),
]


def il_name(n1, p2, t, rep, op):
    s = f'il_N{n1}_P{p2}_T{t:.2f}_R{rep}'
    return s + (f'_O{op:g}' if rep else '')


def leg_costs(P, open_ms):
    """The leg-choice constants from TODAY's fit (what the live knobs would be set to)."""
    return hsp.LegCosts(ig0=P.ig0, ig_d=P.ig_d, ig_n=P.ig_n, ig_b=P.ig_b, s0_us=P.s0, s_d_us=P.s_d, s_b_us=P.s_b,
                        link_us=tuple(P.link_us.get(b, P.link_us[max(P.link_us)]) for b in range(1, 7)),
                        open_ms=open_ms)


def build(cache, refresh, counts, P, args):
    snap = cache.get('hs0')
    r0 = simlib.first_refresh_after(refresh, cache)
    pin_total = args['pin_total']
    pols = [
        ('top', lambda: policies_live.LiveSplit('top', refresh_times=refresh, snapshot=snap, refresh_start=r0,
                                                pin_total=pin_total)),
        ('today_live', lambda: policies.Today(refresh_times=refresh, snapshot=snap, refresh_start=r0,
                                              pin2=pin_total, b2_slots=119, b1_slots=126)),
        ('c0.55_s', lambda: policies.policy_interleave(counts, 0.55)),
        ('c0.60_s', lambda: policies.policy_interleave(counts, 0.60)),
        ('d10_0.60_s', lambda: policies.policy_interleave(counts, 0.60, rep_k=10)),
    ]
    for (n1, p2, t, rep, op) in args['points']:
        def mk(n1=n1, p2=p2, t=t, rep=rep, op=op):
            il = hsp.IlParams(n1=min(n1, 124), k=rep, p2=min(p2, 80), target=t, tol=args['tol'], moves=args['moves'],
                              hyst=args['hyst'], rep_hyst=5)
            return policies_live.LiveSplit('interleave', refresh_times=refresh, snapshot=snap, refresh_start=r0, il=il,
                                           pin_total=pin_total, legs=leg_costs(P, op) if rep else None,
                                           prewarm_step=args['prewarm_step'], keep_reads_step=args['keep_reads_step'],
                                           holder=args['holder'], name=il_name(n1, p2, t, rep, op))
        pols.append((il_name(n1, p2, t, rep, op), mk))
    return pols


def transition(pol, steps, half, target, tol):
    """Refresh-log digest of a live interleave: the share curve, refreshes until
    the mean share is within tolerance, newcomers per refresh during / after."""
    log = pol.log or []
    if not log or 'share' not in log[0]:
        return None
    within_at = None
    for i, x in enumerate(log):
        if abs(x['share'] - target) <= tol and within_at is None:
            within_at = i
    # per-layer convergence: first refresh at which >= 95% of layers are within
    lay95 = next((i for i, x in enumerate(log) if x['within'] >= 0.95 * NL), None)
    steady = [x for x in log if x['si'] >= half]
    trans = log[:(within_at + 1) if within_at is not None else len(log)]
    def mean(v, k):
        return st.fmean(x[k] for x in v) if v else float('nan')
    return {
        'n_refresh': len(log),
        'share_curve': [round(x['share'], 4) for x in log[:25]],
        'refresh_within_tol_mean_share': within_at,
        'refresh_95pct_layers_within': lay95,
        'first': {k: log[0][k] for k in ('share', 'swaps', 'b1_new', 'b2_new', 'keep', 'changed', 'prewarm_queued',
                                         'keep_queued')},
        'transition_mean': {k: mean(trans, k) for k in ('swaps', 'b1_new', 'b2_new', 'changed')},
        'steady_mean': {k: mean(steady, k) for k in ('share', 'swaps', 'b1_new', 'b2_new', 'changed', 'within')},
        'steady_n': len(steady),
    }


def main():
    a = sys.argv
    cache_path, refresh_path, evt_path = a[1:4]
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
    if opt('--s-d-mult'):
        P.s_d *= float(opt('--s-d-mult'))
    T0 = {int(k): v for k, v in json.load(open(opt('--params'))).get('T0', {}).items()}
    miss_keep = float(opt('--miss-keep', '0.31'))
    classes = opt('--classes', '1')
    n = len(cache['steps'])
    half = int(opt('--eval-from', n // 2))
    counts = compare.first_half_counts(cache, half)
    args = {'pin_total': int(opt('--pin-total', 105)), 'tol': float(opt('--tol', 0.02)), 'moves': int(opt('--moves', 3)),
            'hyst': int(opt('--il-hyst', 40)), 'prewarm_step': int(opt('--prewarm-step', 2)),
            'keep_reads_step': int(opt('--keep-reads-step', 30)), 'holder': opt('--holder', '1') == '1',
            'points': DESIGN_POINTS}
    if opt('--points'):
        args['points'] = [tuple(float(x) if i in (2, 4) else int(float(x)) for i, x in enumerate(p.split(':')))
                          for p in opt('--points').split(';')]
    pols = build(cache, refresh, counts, P, args)
    only = opt('--only')
    if only:
        keep_ = only.split(',')
        pols = [p for p in pols if p[0] in keep_]
    print(f'steps {n}; evaluate on [{half}, {n}); classes={classes}; miss_keep={miss_keep}; pin_total={args["pin_total"]}; '
          f'params {opt("--params")} set {opt("--set")}', flush=True)
    out = []
    per_step = {}
    for name, mk in pols:
        t0 = time.time()
        pol = mk()
        res = replay.run(cache, pol, P, clk, eval_from=half, picks='ran', miss_keep=miss_keep, classes=classes,
                         prefill_pollute='--prefill-pollute' in a)
        per_step[name] = [r['fwd_core'] + T0[r['lanes']] for r in res]
        s = compare.summarize(name, res, T0)
        if len(per_step) > 1:
            base_name = next(iter(per_step))
            s['ci95_vs_first'] = compare.block_bootstrap(per_step[base_name], per_step[name])
        if res and 'pol' in res[0]:
            for k in res[0]['pol']:
                s['pol_' + k] = st.fmean(r['pol'].get(k, 0) for r in res)
        s['rep_per_step'] = st.fmean(r.get('rep', 0) for r in res)
        if isinstance(pol, policies_live.LiveSplit) and pol.mode == 'interleave':
            s['transition'] = transition(pol, cache['steps'], half, pol.il.target, pol.il.tol)
        out.append(s)
        print(f"{name:26s} fwd mean {s['fwd_mean']:6.2f} (meas {s['meas_mean']:.1f}) share1 {s['share1']:.3f} "
              f"b2miss/step {s['m2_per_step']:5.2f} b1miss/step {s['m1_per_step']:.2f} busy: iGPU {s['igpu_mean']:5.1f} "
              f"b2 {s['b2_mean']:5.1f} dGPU {s['dgpu_mean']:5.1f} host {s['host_mean']:5.1f} | req/step {s['req_per_step']:5.1f}"
              f" [{time.time() - t0:.0f}s]", flush=True)
        if s.get('transition'):
            tr = s['transition']
            print(f"    transition: within tol (mean share) at refresh {tr['refresh_within_tol_mean_share']}, 95% layers at "
                  f"{tr['refresh_95pct_layers_within']}; first {tr['first']}; transition mean {tr['transition_mean']}; "
                  f"steady {tr['steady_mean']} (n {tr['steady_n']}); holder b1/b2 per step "
                  f"{s.get('pol_holder_b1', 0):.2f}/{s.get('pol_holder_b2', 0):.2f}, prewarm admits/step "
                  f"{s.get('pol_prewarm_admits', 0):.2f}, keep reads/step {s.get('pol_keep_reads', 0):.2f}", flush=True)
    if not out:
        return
    base = out[0]['fwd_mean']
    cells = sorted(out[0]['cells'], key=lambda c: (int(c.split('L')[0]), int(c.split('r')[1])))
    print('\npredicted median fwd (ms) by cell (lane code: 1/2 = lone stream, 2x = live 2, 3x = live >= 3); '
          'mean = traffic-weighted mean over held-out steps')
    print(f"{'policy':26s} " + ' '.join(f'{c:>8s}' for c in cells) + f" {'mean':>7s} {'vs first':>8s}")
    print(f"{'measured':26s} " + ' '.join(f"{out[0]['cells'][c]['meas_med']:8.1f}" for c in cells) + f" {out[0]['meas_mean']:7.1f}")
    for s in out:
        ci = s.get('ci95_vs_first')
        cis = f"  [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else ''
        print(f"{s['name']:26s} " + ' '.join(f"{s['cells'][c]['fwd_med']:8.1f}" if c in s['cells'] else f"{'':8s}" for c in cells)
              + f" {s['fwd_mean']:7.2f} {100 * (s['fwd_mean'] - base) / base:+7.1f}%{cis}")
    print('n per cell: ' + ' '.join(f"{c}={out[0]['cells'][c]['n']}" for c in cells))
    if opt('--out'):
        with open(opt('--out'), 'w') as fh:
            json.dump(out, fh, indent=1, default=str)


if __name__ == '__main__':
    main()
