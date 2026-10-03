#!/usr/bin/env python3
"""hub_req exploration: per (lanes, rows) of the step, per-request stats, and
box-2 service vs distinct experts / misses. Streams the TSV."""
import csv
import math
import statistics as st
import sys
from collections import defaultdict


def f(x):
    try:
        return float(x)
    except ValueError:
        return float('nan')


def q(v, p):
    v = sorted(x for x in v if not math.isnan(x))
    if not v:
        return float('nan')
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]


def main():
    step_path, req_path = sys.argv[1], sys.argv[2]
    steps = {}
    for r in csv.DictReader(open(step_path), delimiter='\t'):
        steps[int(f(r['step']))] = (int(f(r['live'])), int(f(r['lanes'])), int(f(r['rows'])))
    per = defaultdict(lambda: defaultdict(list))
    nreq = defaultdict(lambda: defaultdict(int))
    srv_by_d = defaultdict(list)
    link_by_b = defaultdict(list)
    srv_by_paged = defaultdict(list)
    for r in csv.DictReader(open(req_path), delimiter='\t'):
        s = f(r['step'])
        if math.isnan(s) or int(s) not in steps:
            continue
        k = steps[int(s)]
        if k[0] != 1:
            continue
        rtt, srv, page, comp = f(r['rtt_us']), f(r['srv_us']), f(r['page_us']), f(r['compute_us'])
        nd, npk, npg = f(r['n_distinct']), f(r['n_picks']), f(r['n_paged'])
        mb = f(r['miss_bits'])
        d = per[k]
        d['n_picks'].append(npk)
        d['n_distinct'].append(nd)
        d['rtt'].append(rtt)
        d['srv'].append(srv)
        d['link'].append(rtt - srv)
        d['page'].append(page)
        d['comp'].append(comp)
        d['paged'].append(npg)
        d['miss_bits'].append(mb)
        d['blocked'].append(f(r['blocked']))
        d['wait'].append((f(r['t_wait_exit']) - f(r['t_wait_enter'])) / 1e3)
        d['sub2wait'].append((f(r['t_wait_enter']) - f(r['t_submit'])) / 1e3)
        nreq[k][int(s)] += 1
        if page == 0 and not math.isnan(nd):
            srv_by_d[(int(nd), int(f(r['b'])))].append(srv)
            link_by_b[int(f(r['b']))].append(rtt - srv)
        if not math.isnan(npg):
            srv_by_paged[int(npg)].append((srv, page, comp))
    cols = ['n_picks', 'n_distinct', 'rtt', 'srv', 'link', 'page', 'comp', 'paged', 'miss_bits', 'blocked', 'wait', 'sub2wait']
    print('lanes rows nsteps req/step ' + ' '.join(f'{c:>9s}' for c in cols) + '   (medians; page/paged/miss_bits means)')
    for k in sorted(per):
        d = per[k]
        vals = []
        for c in cols:
            if c in ('page', 'paged', 'miss_bits', 'blocked'):
                v = [x for x in d[c] if not math.isnan(x)]
                vals.append(sum(v) / max(1, len(v)))
            else:
                vals.append(q(d[c], 0.5))
        ns = len(nreq[k])
        rps = sum(nreq[k].values()) / max(1, ns)
        print(f'{k[1]:5d} {k[2]:4d} {ns:6d} {rps:8.1f} ' + ' '.join(f'{v:9.3f}' for v in vals))
    print('\nsrv_us (no paging) by (n_distinct, b): n p25 p50 p75 mean')
    for k in sorted(srv_by_d):
        v = srv_by_d[k]
        if len(v) < 30:
            continue
        print(f'  d={k[0]:2d} b={k[1]} n={len(v):6d} {q(v, .25):7.0f} {q(v, .5):7.0f} {q(v, .75):7.0f} {st.fmean(v):7.0f}')
    print('\nlink_us (rtt - srv, no paging) by b: n p10 p50 p90 mean')
    for k in sorted(link_by_b):
        v = link_by_b[k]
        print(f'  b={k} n={len(v):6d} {q(v, .1):7.0f} {q(v, .5):7.0f} {q(v, .9):7.0f} {st.fmean(v):7.0f}')
    print('\nby n_paged: n srv p50 page p50 comp p50, page mean')
    for k in sorted(srv_by_paged):
        v = srv_by_paged[k]
        if len(v) < 10:
            continue
        print(f'  paged={k:2d} n={len(v):6d} srv={q([x[0] for x in v], .5):7.0f} page={q([x[1] for x in v], .5):7.0f} '
              f'comp={q([x[2] for x in v], .5):7.0f} page_mean={st.fmean([x[1] for x in v]):7.0f}')


if __name__ == '__main__':
    main()
