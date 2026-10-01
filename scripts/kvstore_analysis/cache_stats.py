#!/usr/bin/env python3
"""Prefix-cache effectiveness from hub logs: every multistream admission
(prompt, restored, prefill_ms) across the current + archived logs."""
import glob, re, os
ANSI = re.compile(r'\x1b\[[0-9;]*m')
PAT = re.compile(r'^(\S+) .*stream admitted .*prompt=(\d+) restored=(\d+) prefill_ms=(\d+)')
files = sorted(glob.glob('/home/claude-code/logs/v41-server.log.2026*')) + ['/home/claude-code/logs/v41-server.log']
rows = []
for f in files:
    for line in open(f, errors='replace'):
        if 'stream admitted' not in line:
            continue
        m = PAT.match(ANSI.sub('', line))
        if m:
            rows.append((m.group(1)[:19], int(m.group(2)), int(m.group(3)), int(m.group(4))))
rows.sort()
print(f'{len(rows)} admissions, {rows[0][0]} .. {rows[-1][0]}')
tot_p = sum(r[1] for r in rows); tot_r = sum(r[2] for r in rows); tot_ms = sum(r[3] for r in rows)
print(f'prompt tokens {tot_p:,}  restored {tot_r:,} ({tot_r / tot_p:.1%})  prefilled {tot_p - tot_r:,}  prefill wall {tot_ms / 3.6e6:.2f} h')
cold = [r for r in rows if r[2] == 0]
big_cold = [r for r in cold if r[1] >= 20000]
print(f'restored=0: {len(cold)} admissions, {sum(r[3] for r in cold) / 3.6e6:.2f} h of prefill;'
      f' of those prompt>=20K: {len(big_cold)}, {sum(r[3] for r in big_cold) / 3.6e6:.2f} h')
warm = [r for r in rows if r[2] > 0]
if warm:
    sfx = sorted(r[1] - r[2] for r in warm)
    print(f'restored>0: {len(warm)} admissions, suffix p50 {sfx[len(sfx) // 2]:,} p90 {sfx[int(len(sfx) * .9)]:,} tokens,'
          f' {sum(r[3] for r in warm) / 3.6e6:.2f} h of prefill')
print('largest cold prefills:')
for r in sorted(big_cold, key=lambda r: -r[3])[:12]:
    print(f'  {r[0]}  prompt {r[1]:>7,}  prefill {r[3] / 1000:6.0f} s')
