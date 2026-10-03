#!/usr/bin/env python3
"""Host-side per-lane-layer timeline of decode steps from hub_req (+hub_step).

  req_timeline.py STEP.tsv REQ.tsv --step N          one step's requests in time order
  req_timeline.py STEP.tsv REQ.tsv --cadence         per (lanes, rows): medians of the
        per-layer submit period, submit->wait, wait, exit->next submit (same lane)
"""
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


def load(step_path, req_path, want=None):
    steps = {}
    for r in csv.DictReader(open(step_path), delimiter='\t'):
        steps[int(f(r['step']))] = r
    reqs = defaultdict(list)
    for r in csv.DictReader(open(req_path), delimiter='\t'):
        s = f(r['step'])
        if math.isnan(s):
            continue
        s = int(s)
        if want is not None and s != want:
            continue
        reqs[s].append({k: f(v) for k, v in r.items()})
    return steps, reqs


def q(v, p=0.5):
    v = sorted(x for x in v if not math.isnan(x))
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float('nan')


def main():
    step_path, req_path = sys.argv[1], sys.argv[2]
    if '--step' in sys.argv:
        n = int(sys.argv[sys.argv.index('--step') + 1])
        steps, reqs = load(step_path, req_path, n)
        s = steps[n]
        t0 = f(s['t_start'])
        print({k: s[k] for k in ('rows', 'live', 'lanes', 'fwd_ms', 'dgpu_busy_ms', 'igpu_busy_ms', 'remote_rtt_ms')})
        lanes = sorted({r['lane'] for r in reqs[n]})
        for r in sorted(reqs[n], key=lambda r: r['t_submit']):
            li = lanes.index(r['lane'])
            print(f"lane{li} L{int(r['layer']):2d} b={int(r['b'])} sub={(r['t_submit'] - t0) / 1e6:8.3f} "
                  f"t1={(r['t1'] - t0) / 1e6:8.3f} t4={(r['t4'] - t0) / 1e6:8.3f} wait=[{(r['t_wait_enter'] - t0) / 1e6:8.3f},"
                  f"{(r['t_wait_exit'] - t0) / 1e6:8.3f}] rtt={r['rtt_us']:.0f} srv={r['srv_us']:.0f} d={r['n_distinct']:.0f} pg={r['page_us']:.0f}")
        print(f"t_end {(f(s['t_end']) - t0) / 1e6:.3f}")
        return
    steps, reqs = load(step_path, req_path)
    agg = defaultdict(lambda: defaultdict(list))
    for sid, rs in reqs.items():
        s = steps.get(sid)
        if s is None or f(s['live']) != 1:
            continue
        key = (int(f(s['lanes'])), int(f(s['rows'])))
        t0 = f(s['t_start'])
        by_lane = defaultdict(list)
        for r in rs:
            by_lane[r['lane']].append(r)
        a = agg[key]
        first = min(r['t_submit'] for r in rs)
        a['first_submit'].append((first - t0) / 1e6)
        a['fwd'].append(f(s['fwd_ms']))
        last_exit = max(r['t_wait_exit'] for r in rs)
        a['last_exit_to_end'].append((f(s['t_end']) - last_exit) / 1e6)
        for lane, lr in by_lane.items():
            lr.sort(key=lambda r: r['layer'])
            for x, y in zip(lr, lr[1:]):
                if y['layer'] == x['layer'] + 1:
                    a['period'].append((y['t_submit'] - x['t_submit']) / 1e6)
                    a['exit_to_next_sub'].append((y['t_submit'] - x['t_wait_exit']) / 1e6)
            for x in lr:
                a['sub_to_wait'].append((x['t_wait_enter'] - x['t_submit']) / 1e6)
                a['wait'].append((x['t_wait_exit'] - x['t_wait_enter']) / 1e6)
                a['submit_call'].append((x['t_submit_end'] - x['t_submit']) / 1e6)
                a['t1_minus_sub'].append((x['t1'] - x['t_submit']) / 1e6)
                a['exit_minus_t4'].append((x['t_wait_exit'] - x['t4']) / 1e6)
    cols = ['fwd', 'first_submit', 'period', 'sub_to_wait', 'wait', 'exit_to_next_sub', 'submit_call', 't1_minus_sub',
            'exit_minus_t4', 'last_exit_to_end']
    print('lanes rows ' + ' '.join(f'{c[:14]:>14s}' for c in cols))
    for k in sorted(agg):
        print(f'{k[0]:5d} {k[1]:4d} ' + ' '.join(f'{q(agg[k][c]):14.3f}' for c in cols))


if __name__ == '__main__':
    main()
