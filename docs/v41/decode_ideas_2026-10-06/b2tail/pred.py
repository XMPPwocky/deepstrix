#!/usr/bin/env python3
import json, statistics as st
from collections import Counter
OUT = '/home/claude-code/.claude/jobs/749c61d3/tmp/b2tail/'
dump = json.load(open(OUT + 'reqs_dump.json'))
def med(v):
    v = [x for x in v if x == x]; return st.median(v) if v else float('nan')
print('cell: ok sample pred_miss>0 share | paged share among ok pred_miss>0 | late share among pred_miss>0 (ok+late sample, biased) | n_pred_miss med of paged late')
for cell, R in dump.items():
    ok = [x for x in R if not x['late']]; late = [x for x in R if x['late']]
    okpm = [x for x in ok if x['n_pred_miss'] > 0]
    pg_late = [x for x in late if x['page'] > 0 or x['n_miss'] > 0]
    print(f"{cell:14s} ok pred_miss>0 {len(okpm)/max(1,len(ok)):.3f} (n {len(okpm)}/{len(ok)}); of those paged {sum(1 for x in okpm if x['page']>0 or x['n_miss']>0)/max(1,len(okpm)):.3f}; n_pred_miss med paged-late {med([x['n_pred_miss'] for x in pg_late])}; n_paged med {med([x['n_paged'] for x in pg_late])}; pred_miss dist paged-late {dict(sorted(Counter(int(x['n_pred_miss']) for x in pg_late).items()))}; pred_miss dist ok {dict(sorted(Counter(int(x['n_pred_miss']) for x in ok).items()))}")
del dump
# hub_step prior counters per cell
import math
SUB = ['sub_predicted_miss', 'sub_reads_avoided', 'sub_picks_swapped', 'sub_blocked', 'sub_plan_failed', 'sub_admits_queued', 'sub_incoming_covered', 'sub_admits_gated', 'b2_surprises', 'b2_held_picks', 'b2_pin_released', 'b2_pinned', 'b2_pin_budget', 'b2_pin_released_unused', 'b2_misses', 'b2_paged_replies', 'b2_page_ms', 'b1_misses', 'b1_read_ms', 'lh_pager_block']
acc = {}
for fn in [OUT + 'h000.json', OUT + 'h001.json']:
    D = json.load(open(fn)); si = {n: i for i, n in enumerate(D['step_fields'])}
    for cell, steps in D['cells'].items():
        a = acc.setdefault(cell, {k: [] for k in SUB})
        for s in steps:
            for k in SUB: a[k].append(s['step'][si[k]])
    del D
print('\nhub_step prior / pin counters (median per step):')
print('cell           ' + ' '.join(f'{k[-14:]:>14s}' for k in SUB))
for cell, a in acc.items():
    print(f'{cell:14s} ' + ' '.join(f'{med(a[k]):14.2f}' for k in SUB))
