#!/usr/bin/env python3
"""Join box-2 b2_req / b2_write / b2_read / b2_ensure records (b2x.py output) with the hub's late classification
(seqs.json from late.py). Service decomposition on box 2's own clock (us):
  q      = t_dequeue - t_frame     (queue before the single serve loop takes it)
  pre    = t_run_start - t_dequeue (decode_request, merge, hints, ensure/admit incl. waiting for own certain reads)
  run    = t_run_end - t_run_start (GPU pass incl. demand reads landing inside the pass)
  d2h    = t_d2h_end - t_run_end
  fin    = t_ready - t_d2h_end     (pin block, residency map, record)
  wr     = t3 - t_ready            (writer thread handoff, from b2_write)
  write  = t_written - t3
"""
import json, sys, statistics as st
from collections import defaultdict, Counter
OUT = '/home/claude-code/.claude/jobs/749c61d3/tmp/b2tail/'
seqs = json.load(open(OUT + 'seqs.json'))
seqs = {int(k): v for k, v in seqs.items()}
files = [OUT + 'b2-20261006-025337-2698631-050.evt.json', OUT + 'b2-20261006-034704-2698631-051.evt.json']
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
R = defaultdict(list)   # cell -> list of joined dicts
reads = defaultdict(list); ens = {}
n_dup = 0
for fn in files:
    D = json.load(open(fn))
    ri = {n: i for i, n in enumerate(D['req_fields'])}; wi = {n: i for i, n in enumerate(D['write_fields'])}
    rdi = {n: i for i, n in enumerate(D['read_fields'])}; ei = {n: i for i, n in enumerate(D['ensure_fields'])}
    wr = {int(w[wi['seq']]): w for w in D['write']}
    for e in D['ensure']: ens[int(e[ei['seq']])] = e
    for rd in D['read']: reads[int(rd[rdi['seq']])].append(rd)
    for r in D['req']:
        seq = int(r[ri['seq']]); meta = seqs.get(seq)
        if meta is None: continue
        cell, late, lateness, b, layer, lane = meta
        if int(r[ri['layer']]) != layer or int(r[ri['b']]) != b: n_dup += 1; continue
        g = lambda k: r[ri[k]]
        w = wr.get(seq)
        rec = {'cell': cell, 'late': bool(late), 'lateness': lateness, 'b': b, 'layer': layer, 'seq': seq,
               'q': (g('t_dequeue') - g('t_frame')) / 1e3, 'pre': (g('t_run_start') - g('t_dequeue')) / 1e3,
               'merge': (g('t_merge_end') - g('t_dequeue')) / 1e3, 'hints': (g('t_hints_end') - g('t_merge_end')) / 1e3, 'ensure': (g('t_run_start') - g('t_hints_end')) / 1e3,
               'run': (g('t_run_end') - g('t_run_start')) / 1e3, 'd2h': (g('t_d2h_end') - g('t_run_end')) / 1e3, 'fin': (g('t_ready') - g('t_d2h_end')) / 1e3,
               'wr': (w[wi['t3']] - g('t_ready')) / 1e3 if w else float('nan'), 'write': (w[wi['t_written']] - w[wi['t3']]) / 1e3 if w else float('nan'),
               'srv': (g('t_ready') - g('t_frame')) / 1e3, 'hdr_to_frame': (g('t_frame') - g('t_hdr')) / 1e3,
               'depth': g('depth_on_take'), 'pending_after': g('pending_after'), 'idle_before': g('idle_before_us'), 'merged': g('merged'), 'served_under': g('served_under'),
               'n_sel': g('n_sel'), 'n_distinct': g('n_distinct'), 'n_hint_admit': g('n_hint_admit'), 'n_pf_words': g('n_prefetch_words'),
               'n_miss': g('n_miss'), 'page_us': g('page_us'), 'compute_us': g('compute_us'), 'server_us': g('server_us'),
               'd_misses': g('d_misses'), 'd_read': g('d_read_ns') / 1e3, 'd_h2d': g('d_h2d_ns') / 1e3, 'd_repack_gpu': g('d_repack_gpu_ns') / 1e3, 'd_pread': g('d_pread_ns') / 1e3,
               'd_repack_cpu': g('d_repack_cpu_ns') / 1e3, 'd_pf_wait': g('d_prefetch_wait_ns') / 1e3, 'park_wait': g('park_wait_ns') / 1e3, 'park_serve': g('park_serve_ns') / 1e3,
               'path_decode': g('path_decode'), 'two_pass': g('two_pass'), 'n_work_items': g('n_work_items'), 'n_missing': g('n_missing'), 'exec_h2d': g('exec_h2d_us'), 'exec_gpu': g('exec_gpu_us'),
               'pf_run_certain': g('pf_run_certain'), 'pf_run_spec': g('pf_run_spec'), 'pf_q_certain': g('pf_q_certain'), 'pf_q_spec': g('pf_q_spec'), 'pf_pending': g('pf_pending'),
               'pf_d_hinted': g('pf_d_hinted'), 'pf_d_admitted': g('pf_d_admitted'), 'pf_d_waited': g('pf_d_waited'), 'pf_d_promoted': g('pf_d_promoted'),
               'pool_resident': g('pool_resident'), 'pin_pinned': g('pin_pinned'), 'pin_evictions': g('pin_evictions'), 'pin_new': g('pin_new'), 'n_paged': g('n_paged'),
               'stage_claims': g('stage_claims'), 'stage_hits': g('stage_hits'), 'stage_spills': g('stage_spills'), 'flags': int(g('flags'))}
        e = ens.get(seq)
        if e:
            ge = lambda k: e[ei[k]]
            rec.update({'e_n_want': ge('n_want'), 'e_n_hits': ge('n_hits'), 'e_n_miss': ge('n_miss'), 'e_admit_wait': ge('admit_wait_ns') / 1e3, 'e_admit_landed': ge('admit_landed'),
                        'e_admit_blocking_recvs': ge('admit_blocking_recvs'), 'e_victim_scan': ge('victim_scan_ns') / 1e3, 'e_total': (ge('t_end') - ge('t_start')) / 1e3,
                        'e_admit': (ge('t_admit_end') - ge('t_start')) / 1e3, 'e_dirty': (ge('t_dirty_end') - ge('t_admit_end')) / 1e3, 'e_victims': (ge('t_victims_end') - ge('t_dirty_end')) / 1e3,
                        'e_reads': (ge('t_reads_end') - ge('t_victims_end')) / 1e3, 'e_tail': (ge('t_end') - ge('t_reads_end')) / 1e3, 'e_remap_upload': ge('remap_upload_ns') / 1e3, 'e_k_par': ge('k_par'), 'e_took_free': ge('took_free'), 'e_evicted_foreign': ge('evicted_foreign')})
        rds = reads.get(seq, [])
        rec['n_reads'] = len(rds)
        if rds:
            rec['read_src'] = Counter(int(x[rdi['src']]) for x in rds); rec['read_route'] = Counter(int(x[rdi['route']]) for x in rds)
            rec['read_wanted'] = sum(1 for x in rds if x[rdi['wanted']] == 1.0); rec['read_blocked_on'] = Counter(int(x[rdi['blocked_on']]) for x in rds)
            rec['read_dur_max'] = max((x[rdi['t_read_end']] - x[rdi['t_read_start']]) / 1e3 for x in rds)
            rec['read_pop_to_land_max'] = max((x[rdi['t_land_end']] - x[rdi['t_pop']]) / 1e3 for x in rds)
            rec['read_already_resident'] = sum(1 for x in rds if x[rdi['already_resident']] == 1.0)
        R[cell].append(rec)
    del D
print('joined', {c: len(v) for c, v in R.items()}, 'layer/b mismatches', n_dup, 'ensure recs', len(ens), 'read seqs', len(reads))
KEYS = ['q', 'pre', 'merge', 'hints', 'ensure', 'run', 'd2h', 'fin', 'wr', 'write', 'srv', 'compute_us', 'page_us', 'idle_before', 'depth', 'n_distinct', 'n_work_items', 'exec_gpu', 'exec_h2d',
        'd_pf_wait', 'd_read', 'd_h2d', 'd_repack_gpu', 'park_wait', 'n_miss', 'd_misses', 'n_paged', 'pf_run_certain', 'pf_run_spec', 'pf_q_certain', 'pf_q_spec', 'pf_pending', 'pf_d_waited', 'pf_d_admitted',
        'e_admit_wait', 'e_n_want', 'e_n_miss', 'e_admit_landed', 'e_admit_blocking_recvs', 'e_total', 'e_admit', 'e_victims', 'e_reads', 'e_victim_scan', 'n_reads', 'stage_hits', 'stage_claims']
summ = {}
for cell, recs in sorted(R.items()):
    late = [x for x in recs if x['late']]; ok = [x for x in recs if not x['late']]
    paged_late = [x for x in late if x['page_us'] > 0 or x['n_miss'] > 0 or x['n_paged'] > 0]
    nop_late = [x for x in late if not (x['page_us'] > 0 or x['n_miss'] > 0 or x['n_paged'] > 0)]
    s = {'n': len(recs), 'n_late': len(late), 'n_paged_late': len(paged_late), 'n_nopage_late': len(nop_late)}
    print(f"\n== {cell}: n {len(recs)} late {len(late)} (paged {len(paged_late)}, no-page {len(nop_late)}) ok {len(ok)}")
    print(f"  {'key':22s} {'ok_med':>8s} {'ok_p90':>8s} | {'late_med':>8s} {'late_p90':>8s} | {'pgL_med':>8s} {'pgL_p90':>8s} | {'npL_med':>8s} {'npL_p90':>8s}")
    for k in KEYS:
        row = {}
        for nm, grp in [('ok', ok), ('late', late), ('pgL', paged_late), ('npL', nop_late)]:
            v = [x.get(k, float('nan')) for x in grp]; row[nm] = (med(v), p(v, .9))
        s[k] = row
        print(f"  {k:22s} {row['ok'][0]:8.0f} {row['ok'][1]:8.0f} | {row['late'][0]:8.0f} {row['late'][1]:8.0f} | {row['pgL'][0]:8.0f} {row['pgL'][1]:8.0f} | {row['npL'][0]:8.0f} {row['npL'][1]:8.0f}")
    # excess attribution on box 2 for late requests: component minus on-time median (same b)
    okmed = {}
    for b in set(x['b'] for x in recs):
        okb = [x for x in ok if x['b'] == b]
        okmed[b] = {k: med([x[k] for x in okb]) for k in ['q', 'merge', 'hints', 'ensure', 'run', 'd2h', 'fin', 'wr', 'write']}
    dom = Counter(); exs = defaultdict(float)
    for x in late:
        m = okmed.get(x['b']) or okmed[next(iter(okmed))]
        ex = {k: max(0.0, x[k] - m[k]) for k in m if x[k] == x[k]}
        if ex:
            d = max(ex, key=ex.get); dom[d] += 1
            for k, v in ex.items(): exs[k] += v
    s['dominant'] = dict(dom); s['excess_us_mean_per_late'] = {k: round(v / max(1, len(late)), 1) for k, v in exs.items()}
    print('  dominant box-2 component (late):', dict(dom))
    print('  mean excess us per late reply:', s['excess_us_mean_per_late'])
    # no-page late: what is in `ensure`/`run`? reads with wanted / src
    if nop_late:
        print('  no-page late: n_reads>0', sum(1 for x in nop_late if x['n_reads'] > 0), 'pf_d_waited>0', sum(1 for x in nop_late if x['pf_d_waited'] > 0), 'd_pf_wait>0', sum(1 for x in nop_late if x['d_pf_wait'] > 0),
              'e_admit_wait>100us', sum(1 for x in nop_late if x.get('e_admit_wait', 0) > 100), 'merged', sum(1 for x in nop_late if x['merged'] > 0), 'two_pass', sum(1 for x in nop_late if x['two_pass'] > 0),
              'depth>0', sum(1 for x in nop_late if x['depth'] > 0), 'idle_before>2ms', sum(1 for x in nop_late if x['idle_before'] > 2000))
        print('  paged late: n_reads>0', sum(1 for x in paged_late if x['n_reads'] > 0), 'read src', sum((x.get('read_src', Counter()) for x in paged_late), Counter()), 'route', sum((x.get('read_route', Counter()) for x in paged_late), Counter()),
              'blocked_on', sum((x.get('read_blocked_on', Counter()) for x in paged_late), Counter()), 'read_dur_max med', med([x.get('read_dur_max', float('nan')) for x in paged_late]), 'p90', p([x.get('read_dur_max', float('nan')) for x in paged_late], .9),
              'pop_to_land med', med([x.get('read_pop_to_land_max', float('nan')) for x in paged_late]))
    # exec_gpu vs n_distinct slope (ok requests): per-expert cost
    pts = [(x['n_distinct'], x['exec_gpu']) for x in ok if x['exec_gpu'] == x['exec_gpu'] and x['n_distinct'] == x['n_distinct']]
    if len(pts) > 20:
        mx = mean([a for a, _ in pts]); my = mean([b for _, b in pts])
        sxx = sum((a - mx) ** 2 for a, _ in pts); sxy = sum((a - mx) * (b - my) for a, b in pts)
        slope = sxy / sxx if sxx else float('nan'); icpt = my - slope * mx
        s['exec_gpu_fit'] = (icpt, slope); print(f'  exec_gpu(ok) = {icpt:.0f} + {slope:.0f} us/expert (n {len(pts)})')
        resid_late = [x['exec_gpu'] - (icpt + slope * x['n_distinct']) for x in late if x['exec_gpu'] == x['exec_gpu']]
        resid_ok = [x['exec_gpu'] - (icpt + slope * x['n_distinct']) for x in ok if x['exec_gpu'] == x['exec_gpu']]
        print(f'  exec_gpu residual med/p90: ok {med(resid_ok):.0f}/{p(resid_ok, .9):.0f}  late {med(resid_late):.0f}/{p(resid_late, .9):.0f}')
    s['idle_before_share_gt2ms'] = {'ok': mean([float(x['idle_before'] > 2000) for x in ok]), 'late': mean([float(x['idle_before'] > 2000) for x in late])}
    s['depth_gt0'] = {'ok': mean([float(x['depth'] > 0) for x in ok]), 'late': mean([float(x['depth'] > 0) for x in late])}
    print('  idle_before>2ms share ok/late', s['idle_before_share_gt2ms'], ' depth>0 share ok/late', s['depth_gt0'], ' path_decode share', mean([x['path_decode'] for x in recs]), ' merged share', mean([float(x['merged'] > 0) for x in recs]))
    summ[cell] = s
json.dump(summ, open(OUT + 'b2join.json', 'w'), indent=1, default=str)
