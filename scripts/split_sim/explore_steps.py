#!/usr/bin/env python3
"""Quick look at hub_step TSV: counts and medians by (live, lanes, rows)."""
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


def med(v):
    v = [x for x in v if not math.isnan(x)]
    return st.median(v) if v else float('nan')


def main():
    path = sys.argv[1]
    cols = sys.argv[2].split(',') if len(sys.argv) > 2 else [
        'fwd_ms', 'step_ms', 'dgpu_busy_ms', 'igpu_busy_ms', 'i_pair_kwide', 'i_q2k_down', 'remote_wait_ms',
        'remote_rtt_ms', 'remote_srv_ms', 'b2_page_ms', 'b2_misses', 'b1_misses', 'b1_read_ms']
    rows = list(csv.DictReader(open(path), delimiter='\t'))
    g = defaultdict(list)
    for r in rows:
        g[(int(f(r['live'])), int(f(r['lanes'])), int(f(r['rows'])))].append(r)
    print('live lanes rows   n  nprof ' + ' '.join(f'{c[:12]:>12s}' for c in cols))
    for k in sorted(g):
        rs = g[k]
        nprof = sum(1 for r in rs if f(r['profiled']) == 1)
        print(f'{k[0]:4d} {k[1]:5d} {k[2]:4d} {len(rs):5d} {nprof:5d} ' + ' '.join(
            f'{med([f(r[c]) for r in rs]):12.2f}' for c in cols))


if __name__ == '__main__':
    main()
