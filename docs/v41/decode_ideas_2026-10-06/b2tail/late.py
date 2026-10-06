#!/usr/bin/env python3
"""Late-reply decomposition per lane-layer request (hub side), from h000/h001.json (extract40.py).
LATE = reply whole at the hub (t4) after the lane-layer's iGPU MoE end, reconstructed as in pipe/analyze.py
(FIFO over launches at t_submit + 60 us, dur = per-step iGPU compute busy / lane-layers).
RTT decomposition (us) from the 40 hub_req fields (remote_experts.rs: t1 = hub writer stamp before write(),
t2_b2 = box-2 frame-whole stamp, srv_us = t_ready - t_done (box 2), compute_us = dequeue -> d2h start,
t3_b2 = box-2 writer stamp before write(), t4 = hub frame-whole stamp; clock_offset_ns = add to box-1 stamp -> box-2):
  enc    = t_submit_end - t_submit        (encode + hand to writer thread)
  wq     = t1 - t_submit_end              (hub writer queue)
  up     = (t2_b2 - off) - t1             (uplink, offset-dependent)
  b2q    = srv_us - compute_us            (box-2 queue before dequeue + d2h/finish)
  comp   = compute_us                     (box-2 dequeue -> d2h start, includes paging)
  page   = page_us (within comp)
  wr     = (t3_b2 - t2_b2) - srv_us       (box-2 writer handoff: ready -> write(); clock-free)
  down   = t4 - (t3_b2 - off)             (downlink incl. write time, offset-dependent)
  link   = (t4 - t1) - (t3_b2 - t2_b2)    (up + down, clock-free)
  notice = t_wait_exit - max(t4, t_wait_enter) when blocked (host noticed the reply late)
Writes late.json (per-cell summaries) and seqs.json (seq -> [cell, late, lateness_us, b]) for the box-2 join."""
import json, math, sys, statistics as st
from collections import defaultdict, Counter
D_US = 60.0
OUT = '/home/claude-code/.claude/jobs/749c61d3/tmp/b2tail/'
files = sys.argv[1:] or [OUT + 'h000.json', OUT + 'h001.json']
WANT = ['lone_spec_r3', 'lone_spec_r4', 'lone_spec_r5', 'lone_spec_r6', 'plain_M3', 'two_spec_r7', 'plain_M8', 'plain_M1']
ENGRAM = {1, 14}; INDEXER = {2, 8, 14, 20, 24, 28, 32, 36}
COMPS = ['enc', 'wq', 'up', 'b2q', 'comp', 'wr', 'down', 'link', 'notice']

def med(v):
    v = [x for x in v if x == x]
    return st.median(v) if v else float('nan')
def p(v, q):
    v = sorted(x for x in v if x == x)
    if not v: return float('nan')
    i = (len(v) - 1) * q; lo = int(i); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)
def mean(v):
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float('nan')

recs = defaultdict(list)     # cell -> list of per-request dicts
stepinfo = defaultdict(list) # cell -> per-step dicts
seqmap = {}
B_FILL = {}
for fn in files:
    D = json.load(open(fn))
    rf = D['req_fields']; ri = {n: i for i, n in enumerate(rf)}
    sf = D['step_fields']; si = {n: i for i, n in enumerate(sf)}
    pf = D['phase_fields']; pi = {n: i for i, n in enumerate(pf)}
    for cell in WANT:
        steps = D['cells'].get(cell, [])
        # B_FILL as in analyze.py
        v = []
        for s in steps:
            reqs = s['reqs']
            for L in set(int(r[ri['lane']]) for r in reqs):
                rs = sorted([r for r in reqs if int(r[ri['lane']]) == L], key=lambda r: r[ri['layer']])
                for j in range(len(rs) - 1):
                    if int(rs[j + 1][ri['layer']]) == int(rs[j][ri['layer']]) + 1:
                        v.append((rs[j + 1][ri['t_submit']] - rs[j][ri['t_wait_exit']]) / 1e3)
        if v: B_FILL.setdefault(cell, med(v))
        for sidx, s in enumerate(steps):
            stv = s['step']; reqs = s['reqs']
            if not reqs: continue
            lanes = sorted(set(int(r[ri['lane']]) for r in reqs)); nl = len(lanes)
            by_lane = {L: sorted([r for r in reqs if int(r[ri['lane']]) == L], key=lambda r: r[ri['layer']]) for L in lanes}
            if any(len(x) < 30 for x in by_lane.values()): continue
            nll = 40 * nl
            igpu_busy = stv[si['igpu_busy_ms']]; i_push = stv[si['i_peer_push_ffn_moe']]; i_mtp = stv[si['i_mtp']]
            comp = igpu_busy - (i_push if i_push == i_push else 0) - (i_mtp if i_mtp == i_mtp else 0)
            if not comp == comp or comp <= 0: continue
            d_moe = comp * 1e3 / nll  # us
            all_launch = []
            for L in lanes:
                rs = by_lane[L]; present = {int(r[ri['layer']]): r for r in rs}; prevC = None
                for lj in range(40):
                    if lj in present:
                        r = present[lj]; S = r[ri['t_submit']]; C = r[ri['t_wait_exit']]
                        all_launch.append((S + D_US * 1e3, L, lj, r)); prevC = C
                    else:
                        if prevC is None: continue
                        S = prevC + B_FILL.get(cell, 700.0) * 1e3; C = S + 80e3
                        all_launch.append((S + D_US * 1e3, L, lj, None)); prevC = C
            all_launch.sort(key=lambda x: x[0])
            t = all_launch[0][0]; prev_end = None
            out_step = []
            t0 = stv[si['t_start']]
            # box-2 concurrency within the step: intervals [t2', t3'] of every request (hub clock)
            ivals = []
            for r in reqs:
                off = r[ri['clock_offset_ns']]
                if off == off and r[ri['t2_b2']] == r[ri['t2_b2']]:
                    ivals.append((r[ri['t2_b2']] - off, r[ri['t3_b2']] - off, r[ri['seq']]))
            for rank, (lt, L, lj, r) in enumerate(all_launch):
                start = max(lt, t); end = start + d_moe * 1e3
                t = end; prev_end = end
                if r is None: continue
                S = r[ri['t_submit']]; SE = r[ri['t_submit_end']]; T1 = r[ri['t1']]; T4 = r[ri['t4']]
                WE = r[ri['t_wait_enter']]; WX = r[ri['t_wait_exit']]; T2 = r[ri['t2_b2']]; T3 = r[ri['t3_b2']]
                off = r[ri['clock_offset_ns']]
                srv = r[ri['srv_us']]; cmp_ = r[ri['compute_us']]; page = r[ri['page_us']]
                late = T4 > end
                t2h = T2 - off if off == off else float('nan'); t3h = T3 - off if off == off else float('nan')
                b2_busy_others = sum(1 for (a, z, q) in ivals if q != r[ri['seq']] and a < t2h < z)
                b2_arrivals_ahead = sum(1 for (a, z, q) in ivals if q != r[ri['seq']] and a <= t2h and z > t2h)
                rec = {
                    'lane': L, 'layer': lj, 'b': int(r[ri['b']]), 'seq': int(r[ri['seq']]), 'late': late,
                    'lateness': (T4 - end) / 1e3, 'rtt': (T4 - S) / 1e3, 'rtt_wire': (T4 - T1) / 1e3, 'moe_us': d_moe,
                    'slack_moe': (end - T4) / 1e3,  # >0: reply before MoE end
                    'blocked': T4 >= WE, 'rank': rank, 'n_in_step': len(all_launch),
                    'enc': (SE - S) / 1e3, 'wq': (T1 - SE) / 1e3, 'up': (t2h - T1) / 1e3,
                    'b2q': srv - cmp_, 'comp': cmp_, 'page': page, 'srv': srv,
                    'wr': (T3 - T2) / 1e3 - srv, 'down': (T4 - t3h) / 1e3, 'link': (T4 - T1) / 1e3 - (T3 - T2) / 1e3,
                    'notice': (WX - max(T4, WE)) / 1e3 if T4 >= WE else 0.0,
                    'n_miss': r[ri['n_miss']], 'n_paged': r[ri['n_paged']], 'miss_bits': r[ri['miss_bits']],
                    'bytes_out': r[ri['bytes_out']], 'bytes_in': r[ri['bytes_in']], 'n_picks': r[ri['n_picks']], 'n_distinct': r[ri['n_distinct']],
                    'n_pred_miss': r[ri['n_pred_miss']], 'n_pred_incoming': r[ri['n_pred_incoming']], 'n_pred_pending': r[ri['n_pred_pending']],
                    'n_held': r[ri['n_held']], 'pinned': r[ri['pinned']], 'delay_ns': r[ri['clock_delay_ns']],
                    'b2_busy_others': b2_busy_others, 'flags': int(r[ri['flags']]),
                    't_in_step_ms': (S - t0) / 1e6, 'step_idx': (fn, cell, sidx),
                }
                out_step.append(rec)
            ph = s['phase']
            t_since_phase = (t0 - ph[pi['t']]) / 1e9 if ph else float('nan')
            ph_from_to = (int(ph[pi['from']]), int(ph[pi['to']])) if ph else None
            n_late = sum(1 for x in out_step if x['late'])
            stepinfo[cell].append({'n_late': n_late, 'exposed_ms': sum(x['lateness'] for x in out_step if x['late']) / 1e3,
                                   'n_req': len(out_step), 'step_ms': stv[si['step_ms']], 'fwd_ms': stv[si['fwd_ms']],
                                   'b2_page_ms': stv[si['b2_page_ms']], 'b2_misses': stv[si['b2_misses']], 'b2_paged_replies': stv[si['b2_paged_replies']],
                                   'remote_rtt_ms': stv[si['remote_rtt_ms']], 'remote_srv_ms': stv[si['remote_srv_ms']], 'lh_remote_wait': stv[si['lh_remote_wait']],
                                   'hop_blocked': stv[si['hop_blocked']], 'b2_pinned': stv[si['b2_pinned']], 'b2_pin_released': stv[si['b2_pin_released']],
                                   'd_moe_us': d_moe, 't_since_phase': t_since_phase, 'ph': ph_from_to, 'rows': stv[si['rows']],
                                   'sum_link': sum(x['link'] for x in out_step if x['link'] == x['link']) / 1e3,
                                   'sum_b2q': sum(x['b2q'] for x in out_step) / 1e3, 'sum_comp': sum(x['comp'] for x in out_step) / 1e3,
                                   'sum_wr': sum(x['wr'] for x in out_step if x['wr'] == x['wr']) / 1e3,
                                   'late_layers': [x['layer'] for x in out_step if x['late']],
                                   'late_ranks': sorted(x['rank'] for x in out_step if x['late'])})
            for x in out_step:
                x['t_since_phase'] = t_since_phase; x['ph'] = ph_from_to
                seqmap[x['seq']] = [cell, int(x['late']), round(x['lateness'], 1), x['b'], x['layer'], x['lane']]
            recs[cell].extend(out_step)
    del D

def summarize(cell, R, S):
    o = {'n_steps': len(S), 'n_req': len(R)}
    late = [x for x in R if x['late']]; ok = [x for x in R if not x['late']]
    o['n_late'] = len(late); o['late_share'] = len(late) / max(1, len(R))
    o['late_per_step'] = mean([s['n_late'] for s in S]); o['exposed_ms_per_step'] = mean([s['exposed_ms'] for s in S])
    o['exposed_p90'] = p([s['exposed_ms'] for s in S], .9)
    o['moe_us_med'] = med([x['moe_us'] for x in R]); o['rtt_med'] = med([x['rtt'] for x in R]); o['rtt_p90'] = p([x['rtt'] for x in R], .9)
    o['lateness_med'] = med([x['lateness'] for x in late]); o['lateness_p90'] = p([x['lateness'] for x in late], .9)
    o['blocked_share_all'] = mean([float(x['blocked']) for x in R]); o['blocked_share_late'] = mean([float(x['blocked']) for x in late])
    # component medians late vs on time, by b
    comp = {}
    for k in COMPS + ['rtt', 'rtt_wire', 'page', 'srv', 'bytes_in', 'bytes_out', 'n_distinct', 'b2_busy_others', 'n_miss', 'n_paged']:
        comp[k] = {'ok_med': med([x[k] for x in ok]), 'ok_p90': p([x[k] for x in ok], .9), 'late_med': med([x[k] for x in late]), 'late_p90': p([x[k] for x in late], .9)}
    o['comp'] = comp
    # excess attribution: dominant component of (late - on-time median of same b)
    okmed = {}
    for b in set(x['b'] for x in R):
        okb = [x for x in ok if x['b'] == b]
        okmed[b] = {k: med([x[k] for x in okb]) for k in COMPS}
    dom = Counter(); excess_sum = defaultdict(float); lateness_by_dom = defaultdict(float)
    cls = Counter()
    for x in late:
        m = okmed.get(x['b'], okmed[next(iter(okmed))])
        ex = {k: max(0.0, x[k] - m[k]) for k in ['enc', 'wq', 'b2q', 'comp', 'wr', 'link', 'notice'] if x[k] == x[k]}
        if not ex: continue
        d = max(ex, key=ex.get); dom[d] += 1; lateness_by_dom[d] += x['lateness']
        for k, v in ex.items(): excess_sum[k] += v
        paged = (x['page'] > 0) or (x['n_miss'] > 0) or (x['n_paged'] == x['n_paged'] and x['n_paged'] > 0)
        cls['paged' if paged else 'no_page'] += 1
        # finer: paged / box-2 service tail (comp or b2q) / link / hub-side
        if paged: cls['paged_dom_' + d] += 1
        else: cls['nopage_dom_' + d] += 1
    o['dominant'] = dict(dom); o['lateness_ms_by_dominant_per_step'] = {k: v / 1e3 / max(1, len(S)) for k, v in lateness_by_dom.items()}
    o['excess_ms_sum_per_step'] = {k: v / 1e3 / max(1, len(S)) for k, v in excess_sum.items()}
    o['classes'] = dict(cls)
    # late share by layer group
    def grp(l):
        return 'L0' if l == 0 else ('engram' if l in ENGRAM else ('indexer' if l in INDEXER else 'other'))
    byg = defaultdict(lambda: [0, 0])
    for x in R:
        g = grp(x['layer']); byg[g][1] += 1; byg[g][0] += int(x['late'])
    o['late_share_by_group'] = {g: (a / max(1, n), n) for g, (a, n) in byg.items()}
    byl = defaultdict(lambda: [0, 0])
    for x in R: byl[x['layer']][1] += 1; byl[x['layer']][0] += int(x['late'])
    o['late_share_by_layer'] = {l: round(a / max(1, n), 3) for l, (a, n) in sorted(byl.items())}
    # position in step (launch rank quintile) and time since phase switch
    byq = defaultdict(lambda: [0, 0])
    for x in R:
        q = min(4, int(5 * x['rank'] / max(1, x['n_in_step']))); byq[q][1] += 1; byq[q][0] += int(x['late'])
    o['late_share_by_rank_quintile'] = {q: round(a / max(1, n), 3) for q, (a, n) in sorted(byq.items())}
    first3 = [x for x in R if x['rank'] < 3]; o['late_share_first3_launches'] = mean([float(x['late']) for x in first3])
    recent = [s for s in S if s['t_since_phase'] == s['t_since_phase'] and s['t_since_phase'] < 2.0]
    later = [s for s in S if s['t_since_phase'] == s['t_since_phase'] and s['t_since_phase'] >= 2.0]
    o['late_per_step_within2s_of_phase'] = (mean([s['n_late'] for s in recent]), len(recent)); o['late_per_step_after2s'] = (mean([s['n_late'] for s in later]), len(later))
    o['phases_seen'] = dict(Counter(str(s['ph']) for s in S))
    # burstiness: runs of consecutive late launches vs Bernoulli null
    runs = []; n_pairs = 0; n_adj = 0
    for s in S:
        rk = s['late_ranks']; n = s['n_req']
        for i in range(len(rk) - 1):
            n_pairs += 1
            if rk[i + 1] == rk[i] + 1: n_adj += 1
    pl = o['late_share']
    o['adjacent_pairs_share'] = n_adj / max(1, n_pairs); o['adjacent_null'] = pl  # P(next launch late | this late) under independence ~ p
    # per-step lateness variance: dispersion index of n_late
    nl = [s['n_late'] for s in S]
    o['n_late_mean'] = mean(nl); o['n_late_var'] = st.pvariance(nl) if len(nl) > 1 else 0.0
    o['n_late_p90'] = p(nl, .9); o['steps_with_zero_late'] = sum(1 for v in nl if v == 0) / max(1, len(nl))
    # step-level correlations of n_late / exposed with box-2 stage fields
    def corr(a, b):
        pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y]
        if len(pairs) < 10: return float('nan')
        ma = mean([x for x, _ in pairs]); mb = mean([y for _, y in pairs])
        sa = math.sqrt(sum((x - ma) ** 2 for x, _ in pairs)); sb = math.sqrt(sum((y - mb) ** 2 for _, y in pairs))
        return sum((x - ma) * (y - mb) for x, y in pairs) / (sa * sb) if sa and sb else float('nan')
    ex = [s['exposed_ms'] for s in S]
    o['corr_exposed'] = {k: round(corr(ex, [s[k] for s in S]), 3) for k in ['b2_page_ms', 'b2_misses', 'b2_paged_replies', 'sum_link', 'sum_b2q', 'sum_comp', 'sum_wr', 'd_moe_us', 'remote_rtt_ms', 'b2_pin_released']}
    o['stage_med'] = {k: med([s[k] for s in S]) for k in ['b2_page_ms', 'b2_misses', 'b2_paged_replies', 'lh_remote_wait', 'remote_rtt_ms', 'hop_blocked', 'b2_pinned', 'd_moe_us', 'step_ms', 'sum_link', 'sum_b2q', 'sum_comp', 'sum_wr']}
    # how much would lateness drop if each component were capped at its on-time p90 (per b)?
    okp90 = {}
    for b in okmed:
        okb = [x for x in ok if x['b'] == b]
        okp90[b] = {k: p([x[k] for x in okb], .9) for k in COMPS}
    saved = defaultdict(float)
    for x in late:
        q = okp90.get(x['b'], okp90[next(iter(okp90))])
        for k in ['b2q', 'comp', 'wr', 'link', 'wq', 'enc', 'notice']:
            if x[k] == x[k] and x[k] > q[k]:
                saved[k] += min(x['lateness'], x[k] - q[k])
    o['lateness_removed_if_capped_at_ok_p90_ms_per_step'] = {k: round(v / 1e3 / max(1, len(S)), 3) for k, v in saved.items()}
    # rows = b distribution
    o['b_counts'] = dict(Counter(x['b'] for x in R))
    # paged share among ALL requests
    o['paged_share_all'] = mean([float((x['page'] > 0) or (x['n_miss'] > 0)) for x in R])
    o['late_share_given_paged'] = mean([float(x['late']) for x in R if (x['page'] > 0) or (x['n_miss'] > 0)])
    o['late_share_given_nopage'] = mean([float(x['late']) for x in R if not ((x['page'] > 0) or (x['n_miss'] > 0))])
    o['late_share_given_b2_busy'] = mean([float(x['late']) for x in R if x['b2_busy_others'] > 0]); o['n_b2_busy'] = sum(1 for x in R if x['b2_busy_others'] > 0)
    o['late_share_given_b2_free'] = mean([float(x['late']) for x in R if x['b2_busy_others'] == 0])
    return o

summ = {}
for cell in WANT:
    if recs.get(cell): summ[cell] = summarize(cell, recs[cell], stepinfo[cell])
json.dump(summ, open(OUT + 'late.json', 'w'), indent=1, default=str)
json.dump(seqmap, open(OUT + 'seqs.json', 'w'))
# compact per-request dump for ad-hoc queries (late + a matched on-time sample)
import random
rng = random.Random(3)
dump = {}
for cell in WANT:
    R = recs.get(cell, [])
    late = [x for x in R if x['late']]; ok = [x for x in R if not x['late']]
    rng.shuffle(ok)
    dump[cell] = late + ok[:len(late)]
for c in dump:
    for x in dump[c]: x.pop('step_idx', None)
json.dump(dump, open(OUT + 'reqs_dump.json', 'w'))

for cell, o in summ.items():
    c = o['comp']
    print(f"\n== {cell}: steps {o['n_steps']} req {o['n_req']} late {o['n_late']} ({100*o['late_share']:.1f}%) late/step {o['late_per_step']:.1f} exposed {o['exposed_ms_per_step']:.2f} ms/step (p90 {o['exposed_p90']:.1f}); moe {o['moe_us_med']:.0f} us rtt med {o['rtt_med']:.0f} p90 {o['rtt_p90']:.0f}; lateness med {o['lateness_med']:.0f} p90 {o['lateness_p90']:.0f} us; blocked all {100*o['blocked_share_all']:.1f}% late {100*o['blocked_share_late']:.1f}%")
    print('  comp      ok_med  ok_p90 | late_med late_p90')
    for k in COMPS + ['rtt', 'page', 'bytes_in', 'n_distinct', 'b2_busy_others']:
        v = c[k]; print(f"  {k:9s} {v['ok_med']:7.0f} {v['ok_p90']:7.0f} | {v['late_med']:8.0f} {v['late_p90']:8.0f}")
    print('  dominant excess:', o['dominant'], ' classes:', o['classes'])
    print('  lateness ms/step by dominant:', {k: round(v, 2) for k, v in o['lateness_ms_by_dominant_per_step'].items()})
    print('  removed if capped at ok p90:', o['lateness_removed_if_capped_at_ok_p90_ms_per_step'])
    print('  paged share all', round(o['paged_share_all'], 3), 'late|paged', round(o['late_share_given_paged'], 3), 'late|nopage', round(o['late_share_given_nopage'], 3), 'late|b2busy', round(o['late_share_given_b2_busy'], 3), f"(n {o['n_b2_busy']})", 'late|b2free', round(o['late_share_given_b2_free'], 3))
    print('  by group:', {g: (round(a, 3), n) for g, (a, n) in o['late_share_by_group'].items()}, ' first3', round(o['late_share_first3_launches'], 3), ' quintiles', o['late_share_by_rank_quintile'])
    print('  by layer:', o['late_share_by_layer'])
    print('  adjacent-late share', round(o['adjacent_pairs_share'], 3), 'null', round(o['adjacent_null'], 3), ' n_late mean/var/p90/zero', round(o['n_late_mean'], 2), round(o['n_late_var'], 2), o['n_late_p90'], round(o['steps_with_zero_late'], 3))
    print('  phase<2s', o['late_per_step_within2s_of_phase'], ' >=2s', o['late_per_step_after2s'], o['phases_seen'])
    print('  corr(exposed, .):', o['corr_exposed'])
    print('  stage med:', {k: round(v, 2) for k, v in o['stage_med'].items()})
