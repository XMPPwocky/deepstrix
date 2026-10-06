#!/usr/bin/env python3
import json, statistics as st
from collections import defaultdict
OUT = '/home/claude-code/.claude/jobs/749c61d3/tmp/b2tail/'
def med(v):
    v = [x for x in v if x == x]; return st.median(v) if v else float('nan')
def mean(v):
    v = [x for x in v if x == x]; return sum(v) / len(v) if v else float('nan')
BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]
acc = defaultdict(lambda: defaultdict(list))
for fn in [OUT + 'h000.json', OUT + 'h001.json']:
    D = json.load(open(fn)); si = {n: i for i, n in enumerate(D['step_fields'])}; pi = {n: i for i, n in enumerate(D['phase_fields'])}
    for cell, steps in D['cells'].items():
        for s in steps:
            ph = s['phase']
            if ph is None or int(ph[pi['to']]) != 0: continue
            t = (s['step'][si['t_start']] - ph[pi['t']]) / 1e9
            for lo, hi in BINS:
                if lo <= t < hi:
                    a = acc[cell][(lo, hi)]
                    a.append((s['step'][si['b2_paged_replies']], s['step'][si['b2_page_ms']], s['step'][si['step_ms']], s['step'][si['sub_blocked']], s['step'][si['b2_pin_released']]))
                    break
    del D
print('cell / bin(s since decode phase start): n steps (share) | paged_replies mean | b2_page_ms mean | step_ms med | sub_blocked mean')
for cell in ['lone_spec_r3', 'lone_spec_r4', 'lone_spec_r5', 'lone_spec_r6', 'plain_M3', 'two_spec_r7', 'plain_M8']:
    tot = sum(len(v) for v in acc[cell].values())
    print(cell)
    for b in BINS:
        v = acc[cell].get(b, [])
        if not v: continue
        print(f"  {str(b):14s} n {len(v):4d} ({100*len(v)/tot:4.1f}%) | paged {mean([x[0] for x in v]):5.2f} | page_ms {mean([x[1] for x in v]):6.2f} | step_ms {med([x[2] for x in v]):6.1f} | blocked {mean([x[3] for x in v]):5.2f}")
