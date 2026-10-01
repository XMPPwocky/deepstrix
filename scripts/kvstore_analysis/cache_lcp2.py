#!/usr/bin/env python3
"""How much KV the current prefix cache leaves on the table.

For every snapshot on disk (in creation order) compute, against the snapshots
created BEFORE it (a lower bound on what was cached at the time, since evicted
ones are gone):
  lcp_any  = longest common token prefix with any older snapshot
             (what a token/chunk-granular cache could have reused)
  restorable = the longest older snapshot that is a COMPLETE prefix of it
             (roughly what today's whole-snapshot lookup can restore; it also
              needs the boundary token, so this is an upper bound)
"""
import json, os
import numpy as np
ROOT = '/home/claude-code/.cache/deepstrix/snapshots-v41'
snaps = []
for d in os.listdir(ROOT):
    try:
        meta = json.load(open(os.path.join(ROOT, d, 'meta.json')))
        toks = np.fromfile(os.path.join(ROOT, d, 'tokens.bin'), dtype='<i4')
    except Exception:
        continue
    snaps.append((meta.get('created_at_unix', 0), d, toks, meta.get('disk_bytes', 0)))
snaps.sort(key=lambda s: s[0])
CK = {int(x) for x in open('/home/claude-code/.claude/jobs/16b63e08/tmp/ckpt_counts.txt').read().split()}
ck = [s for s in snaps if len(s[2]) in CK]
print(f'mid-prefill checkpoints on disk: {len(ck)}, {sum(s[3] for s in ck) / 1e9:.1f} GB (unreachable: wrong key, non-boundary length)')
snaps = [s for s in snaps if len(s[2]) not in CK]

def lcp(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    ne = np.nonzero(a[:n] != b[:n])[0]
    return int(ne[0]) if len(ne) else n

rows = []
for i, (t, d, toks, nbytes) in enumerate(snaps):
    best_any = best_full = 0
    for (_, _, o, _) in snaps[:i]:
        l = lcp(toks, o)
        best_any = max(best_any, l)
        if l == len(o):
            best_full = max(best_full, l)
    rows.append((len(toks), best_any, best_full, nbytes))

n = len(rows)
tot = sum(r[0] for r in rows)
print(f'{n} snapshots, {sum(r[3] for r in rows) / 1e9:.1f} GB, {tot:,} tokens')
print(f'reusable by token-granular LCP: {sum(r[1] for r in rows):,} ({sum(r[1] for r in rows) / tot:.1%})')
print(f'reusable by whole-snapshot prefix: {sum(r[2] for r in rows):,} ({sum(r[2] for r in rows) / tot:.1%})')
gap = [(r[0], r[1], r[2]) for r in rows if r[1] - r[2] > 4096]
print(f'snapshots where LCP beats whole-snapshot reuse by > 4096 tokens: {len(gap)}; extra reusable tokens {sum(a - c for _, a, c in gap):,}')
for tok, a, c in sorted(gap, key=lambda g: -(g[1] - g[2]))[:15]:
    print(f'  len {tok:>7,}  lcp_any {a:>7,}  whole-snapshot {c:>7,}  gap {a - c:>7,}')
cold = [r for r in rows if r[2] < 1000 and r[0] > 20000]
print(f'large snapshots (>20K) with no whole-snapshot prefix: {len(cold)}; their best LCP: '
      + ', '.join(f'{r[1]:,}/{r[0]:,}' for r in sorted(cold, key=lambda r: -r[0])[:15]))
# How much of the disk is duplicated prefix (sum over snapshots of the part shared with an older one)
print(f'disk: share of tokens duplicated from an older snapshot: {sum(r[1] for r in rows) / tot:.1%}')
