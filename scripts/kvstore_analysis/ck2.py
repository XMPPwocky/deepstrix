import glob, re, statistics as st, os
ANSI = re.compile(r'\x1b\[[0-9;]*m')
files = sorted(glob.glob('/home/claude-code/scratch-ms/b2_hang_2026-09-27/hub_logs/v41-server.log*')) + sorted(glob.glob('/home/claude-code/logs/v41-server.log.2026*'))
for f in files:
    ck = []; rs = []
    for line in open(f, errors='replace'):
        if 'checkpoint saved' in line or 'partial snapshot saved' in line:
            m = re.search(r'tokens=(\d+) done=(\d+) total=(\d+) ms=(\d+)', ANSI.sub('', line))
            if m:
                t, d, tot, ms = map(int, m.groups())
                if ms > 0:
                    ck.append((t - tot) * 2760 / 1e9 / (ms / 1e3))
        elif 'snapshot restored' in line:
            m = re.search(r'restored=(\d+) total=(\d+) ms=(\d+)', ANSI.sub('', line))
            if m:
                r, tot, ms = map(int, m.groups())
                if ms > 0 and r > 20000:
                    rs.append(r * 2760 / 1e9 / (ms / 1e3))
    if ck or rs:
        print(os.path.basename(f), 'ckpt n=%d GB/s p50 %s' % (len(ck), '%.2f' % st.median(ck) if ck else '-'), ' restore(>20K) n=%d GB/s p50 %s' % (len(rs), '%.2f' % st.median(rs) if rs else '-'))
