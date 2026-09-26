#!/usr/bin/env python3
"""Generate LEDGER.md from a workflow journal: every idea + reviewer verdict, ranked."""
import json
import sys
from collections import OrderedDict

journal = sys.argv[1]
out = sys.argv[2]

labels, families, verdicts = {}, OrderedDict(), {}
for line in open(journal):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get('type') == 'started':
        labels[d['agentId']] = d['label']
    elif d.get('type') in ('result', 'completed'):
        r = d.get('result')
        lab = labels.get(d.get('agentId'), '?')
        if isinstance(r, dict) and 'ideas' in r:
            families[r['family']] = r
        elif isinstance(r, dict) and 'verdict' in r:
            verdicts[r['id']] = r

def fmt(x, nd=2):
    return '—' if x is None else (f'{x:.{nd}f}' if isinstance(x, (int, float)) else str(x))

rows = []
for fam, r in families.items():
    for i in r['ideas']:
        v = verdicts.get(i['id'])
        rows.append((fam, i, v))

def rank_key(t):
    fam, i, v = t
    verdict = v['verdict'] if v else ('unreviewed' if i['status'] == 'measured_win' else i['status'])
    order = {'CONFIRMED': 0, 'unreviewed': 1, 'UNVERIFIABLE': 2, 'REFUTED': 3}.get(verdict, 4)
    lev = (i.get('est_ms_per_token') or 0) + 3 * (i.get('est_prefill_pct') or 0)
    return (order, -lev)

rows.sort(key=rank_key)

L = []
L.append('# Kernel-ideas sweep 2026-09-26 — LEDGER\n')
L.append('Branch `worktree-kernel-ideas-2026-09-26` (base 361d4f9 = production). Nothing here is merged; '
         'candidates live under `scratch/kernel_ideas/<family>/`, reviewer artefacts under `<family>/review/`. '
         'Numbers are kernel-level medians against the UNMODIFIED production code object built with the production '
         'flags, at production shapes; "reviewer x" is the independent re-run. est columns are engineer estimates '
         '(kernel delta x production call count), not e2e measurements.\n')
n_conf = sum(1 for _, i, v in rows if v and v['verdict'] == 'CONFIRMED')
n_ref = sum(1 for _, i, v in rows if v and v['verdict'] == 'REFUTED')
n_win = sum(1 for _, i, _ in rows if i['status'] == 'measured_win')
L.append(f'Totals: {len(rows)} ideas, {n_win} measured wins, {n_conf} CONFIRMED by review, {n_ref} REFUTED.\n')

L.append('## Ranked table\n')
L.append('| # | id | kernel | engineer x | reviewer x | verdict | bit-exact | est ms/step | est prefill % | integration | caveats |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|')
for n, (fam, i, v) in enumerate(rows, 1):
    verdict = v['verdict'] if v else ('unreviewed' if i['status'] == 'measured_win' else i['status'])
    cav = (v['issues'] if v else i.get('correctness', ''))
    cav = ' '.join(str(cav).split())[:220]
    L.append(f"| {n} | `{i['id']}` | {i['kernel'][:60]} | {fmt(i.get('speedup'))} | {fmt(v.get('reran_speedup')) if v else '—'} | "
             f"{verdict} | {'yes' if i.get('bit_exact') else 'NO'} | {fmt(i.get('est_ms_per_token'))} | {fmt(i.get('est_prefill_pct'))} | "
             f"{i.get('integration_cost')} | {cav} |")

L.append('\n## Details by family\n')
for fam, r in families.items():
    L.append(f'### {fam}\n')
    L.append('**Baselines measured**\n')
    L.append('| kernel | device | shape | regime | med us | roof us | % of roof | bottleneck |')
    L.append('|---|---|---|---|---|---|---|---|')
    for b in r['baseline']:
        L.append(f"| {b['kernel'][:50]} | {b['device']} | {b['shape'][:60]} | {b['regime'][:50]} | {fmt(b['med_us'])} | "
                 f"{fmt(b['roofline_us'])} | {fmt(b['pct_of_roof'], 0)} | {' '.join(str(b['bottleneck']).split())[:160]} |")
    L.append('')
    for i in r['ideas']:
        v = verdicts.get(i['id'])
        L.append(f"#### `{i['id']}` — {i['status']}" + (f" — review: {v['verdict']}" if v else ''))
        L.append(f"- kernel: `{i['kernel']}`")
        L.append(f"- idea: {' '.join(str(i['idea']).split())}")
        if i.get('speedup') is not None:
            L.append(f"- engineer: {fmt(i.get('baseline_med_us'))} -> {fmt(i.get('candidate_med_us'))} us, x{fmt(i.get('speedup'), 3)}, runs={i.get('runs')}")
        L.append(f"- correctness: bit_exact={i.get('bit_exact')}; {' '.join(str(i.get('correctness')).split())}")
        L.append(f"- leverage: est_ms_per_token={fmt(i.get('est_ms_per_token'))}, est_prefill_pct={fmt(i.get('est_prefill_pct'))}; integration_cost={i.get('integration_cost')}")
        L.append(f"- files: {i.get('files')}")
        L.append(f"- repro: `{' '.join(str(i.get('repro')).split())[:400]}`")
        if v:
            L.append(f"- REVIEW ({v['verdict']}): reran x{fmt(v.get('reran_speedup'), 3)} runs={v.get('reran_runs')} correctness_ok={v.get('correctness_ok')} "
                     f"baseline_matches_production={v.get('baseline_matches_production')}")
            L.append(f"  - issues: {' '.join(str(v.get('issues')).split())}")
            L.append(f"  - merge notes: {' '.join(str(v.get('merge_notes')).split())}")
        L.append('')
    L.append(f"**Dead ends ({fam}):** {' '.join(str(r.get('dead_ends')).split())}\n")
    L.append(f"**Notes ({fam}):** {' '.join(str(r.get('notes')).split())}\n")

open(out, 'w').write('\n'.join(L) + '\n')
print(f'{out}: {len(rows)} ideas, {n_conf} confirmed, {n_ref} refuted, families={list(families)}')
