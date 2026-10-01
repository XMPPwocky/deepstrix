import glob, re, statistics as st
ANSI = re.compile(r'\x1b\[[0-9;]*m')
files = sorted(glob.glob('/home/claude-code/logs/v41-server.log.2026*')) + ['/home/claude-code/logs/v41-server.log', '/home/claude-code/scratch-ms/b2_hang_2026-09-27/hub_logs/v41-server.log.pre-8cap-2354']
rows = []
for f in files:
    for line in open(f, errors='replace'):
        if 'checkpoint saved' not in line and 'partial snapshot saved' not in line:
            continue
        l = ANSI.sub('', line)
        m = re.search(r'tokens=(\d+) done=(\d+) total=(\d+) ms=(\d+)', l)
        if m:
            rows.append(tuple(map(int, m.groups())))
kvpos = [t - tot for t, d, tot, ms in rows]
rate = sorted(p * 2760 / 1e9 / (r[3] / 1e3) for p, r in zip(kvpos, rows) if r[3] > 0)
print(len(rows), 'checkpoints; true GB/s (KV positions = tokens - total) p50 %.2f p10 %.2f p90 %.2f' % (st.median(rate), rate[len(rate) // 10], rate[9 * len(rate) // 10]))
print('naive (tokens*2760) GB/s p50 %.2f' % st.median([r[0] * 2760 / 1e9 / (r[3] / 1e3) for r in rows if r[3] > 0]))
print('examples (kv positions, ms):', [(p, r[3]) for p, r in zip(kvpos, rows)][:8])
