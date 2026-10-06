#!/usr/bin/env python3
import json, statistics as st
from collections import Counter, defaultdict
OUT = '/home/claude-code/.claude/jobs/749c61d3/tmp/b2tail/'
def med(v):
    v = [x for x in v if x == x]; return st.median(v) if v else float('nan')
tot = Counter(); acc = defaultdict(lambda: defaultdict(list)); gaps = defaultdict(list)
for fn in [OUT + 'h000.json', OUT + 'h001.json']:
    D = json.load(open(fn)); pf = D['phase_fields']; pi = {n: i for i, n in enumerate(pf)}
    ph = D['phases']; prev = None
    for p in ph:
        key = (int(p[pi['from']]), int(p[pi['to']])); tot[key] += 1
        for k in ['live', 'prefills', 'queued', 'burst_ms', 'next_budget_ms', 'b2_pin_released', 'b2_pin_restore', 'starved']:
            acc[key][k].append(p[pi[k]])
        if prev is not None: gaps[key].append((p[pi['t']] - prev) / 1e9)
        prev = p[pi['t']]
    print(fn, 'phases', len(ph), 'span h', (ph[-1][pi['t']] - ph[0][pi['t']]) / 3.6e12 if ph else 0)
    del D
print('fields', pf)
for key, n in tot.items():
    print(key, 'n', n, {k: round(med(v), 1) for k, v in acc[key].items()}, 'median s since previous phase record', round(med(gaps[key]), 1))
