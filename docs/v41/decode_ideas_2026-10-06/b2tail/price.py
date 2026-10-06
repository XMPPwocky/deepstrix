#!/usr/bin/env python3
"""Price levers from the late replies (reqs_dump.json = all late + equal on-time sample) and late.json."""
import json, statistics as st
from collections import Counter, defaultdict
OUT = '/home/claude-code/.claude/jobs/749c61d3/tmp/b2tail/'
dump = json.load(open(OUT + 'reqs_dump.json')); L = json.load(open(OUT + 'late.json'))
def med(v):
    v = [x for x in v if x == x]; return st.median(v) if v else float('nan')
def p(v, q):
    v = sorted(x for x in v if x == x)
    if not v: return float('nan')
    i = (len(v) - 1) * q; lo = int(i); hi = min(lo + 1, len(v) - 1); return v[lo] + (v[hi] - v[lo]) * (i - lo)
CUTS = [20, 50, 100, 200, 300, 500, 1000, 2000]
for cell in ['lone_spec_r3', 'lone_spec_r4', 'lone_spec_r5', 'lone_spec_r6', 'plain_M3', 'two_spec_r7', 'plain_M8', 'plain_M1']:
    R = dump.get(cell, [])
    if not R: continue
    n_steps = L[cell]['n_steps']
    late = [x for x in R if x['late']]; ok = [x for x in R if not x['late']]
    okmed_b2q = {b: med([x['b2q'] for x in ok if x['b'] == b]) for b in set(x['b'] for x in R)}
    okmed_comp = {b: med([x['comp'] for x in ok if x['b'] == b]) for b in set(x['b'] for x in R)}
    cls_ms = defaultdict(float); cls_n = Counter()
    for x in late:
        paged = x['page'] > 0 or x['n_miss'] > 0 or (x['n_paged'] == x['n_paged'] and x['n_paged'] > 0)
        exq = x['b2q'] - okmed_b2q.get(x['b'], 20); exc = x['comp'] - okmed_comp.get(x['b'], 300)
        if paged: c = 'paged'
        elif exq > exc and exq > 50: c = 'queue'
        elif x['link'] == x['link'] and x['link'] - L[cell]['comp']['link']['ok_med'] > max(exc, exq): c = 'link'
        else: c = 'compute_nopage'
        cls_ms[c] += x['lateness'] / 1e3 / n_steps; cls_n[c] += 1 / n_steps
    tot = sum(cls_ms.values())
    print(f"\n== {cell} (steps {n_steps}): exposed {tot:.2f} ms/step; by class ms/step: " + ', '.join(f"{k} {v:.2f} ({cls_n[k]:.1f}/step)" for k, v in sorted(cls_ms.items(), key=lambda kv: -kv[1])))
    # lateness removed if every reply arrived X us earlier (fixed-cost / link / earlier-issue levers)
    curve = {X: sum(min(x['lateness'], X) for x in late) / 1e3 / n_steps for X in CUTS}
    print('  ms/step removed if every reply were X us earlier:', {X: round(v, 2) for X, v in curve.items()})
    # same, but paged replies only vs no-page only
    for nm, sel in [('paged', lambda x: x['page'] > 0 or x['n_miss'] > 0), ('nopage', lambda x: not (x['page'] > 0 or x['n_miss'] > 0))]:
        sub = [x for x in late if sel(x)]
        print(f'   {nm}: n/step {len(sub)/n_steps:.1f} lateness med {med([x["lateness"] for x in sub]):.0f} p90 {p([x["lateness"] for x in sub], .9):.0f} us; removed if X earlier:', {X: round(sum(min(x['lateness'], X) for x in sub) / 1e3 / n_steps, 2) for X in [100, 300, 500, 1000, 2000]})
    # hub prediction state of paged late replies
    pg = [x for x in late if x['page'] > 0 or x['n_miss'] > 0]
    if pg:
        print('  paged late: n_pred_miss>0 share', round(sum(1 for x in pg if x['n_pred_miss'] > 0) / len(pg), 3), 'pred_incoming>0', round(sum(1 for x in pg if x['n_pred_incoming'] > 0) / len(pg), 3),
              'pred_pending>0', round(sum(1 for x in pg if x['n_pred_pending'] > 0) / len(pg), 3), 'n_paged med', med([x['n_paged'] for x in pg]), 'page_us med', med([x['page'] for x in pg]), 'p90', p([x['page'] for x in pg], .9),
              'n_held med', med([x['n_held'] for x in pg]), 'n_distinct med', med([x['n_distinct'] for x in pg]))
        allpg_ok = [x for x in ok if x['page'] > 0 or x['n_miss'] > 0]
        print('  paged ON-TIME replies in the ok sample:', len(allpg_ok), 'of', len(ok))
    npg = [x for x in late if not (x['page'] > 0 or x['n_miss'] > 0)]
    print('  no-page late: n_distinct dist', dict(sorted(Counter(int(x['n_distinct']) for x in npg).items())), ' ok n_distinct dist', dict(sorted(Counter(int(x['n_distinct']) for x in ok).items())))
    print('  no-page late: b2q>100us share', round(sum(1 for x in npg if x['b2q'] > 100) / max(1, len(npg)), 3), ' b2_busy_others>0 share', round(sum(1 for x in npg if x['b2_busy_others'] > 0) / max(1, len(npg)), 3), ' ok b2_busy share', round(sum(1 for x in ok if x['b2_busy_others'] > 0) / max(1, len(ok)), 3))
    # layer-0 late: what class?
    l0 = [x for x in late if x['layer'] == 0]
    print('  layer-0 late n/step', round(len(l0) / n_steps, 2), 'paged share', round(sum(1 for x in l0 if x['page'] > 0 or x['n_miss'] > 0) / max(1, len(l0)), 3), 'lateness med', med([x['lateness'] for x in l0]), 'comp med', med([x['comp'] for x in l0]), 'n_distinct med', med([x['n_distinct'] for x in l0]), 'rtt med', med([x['rtt'] for x in l0]))
    # time within step of late replies with phase<2s
    ph = [x for x in late if x['t_since_phase'] == x['t_since_phase'] and x['t_since_phase'] < 2.0]
    print('  late within 2 s of a phase switch: n', len(ph), 'paged share', round(sum(1 for x in ph if x['page'] > 0 or x['n_miss'] > 0) / max(1, len(ph)), 3))
