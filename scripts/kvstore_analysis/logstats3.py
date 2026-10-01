import glob, re
ANSI = re.compile(r'\x1b\[[0-9;]*m')
files = sorted(glob.glob('/home/claude-code/logs/v41-server.log.2026*')) + ['/home/claude-code/logs/v41-server.log']
rest = []; adm = []; ck = []
t0 = '2026-09-28T16:00'; t1 = '2026-10-01T09:00'
for f in files:
    for line in open(f, errors='replace'):
        if 'multistream' not in line: continue
        l = ANSI.sub('', line)
        ts = l[:16]
        if not (t0 <= ts <= t1): continue
        m = re.search(r'snapshot restored restored=(\d+) total=(\d+) ms=(\d+)', l)
        if m: rest.append(tuple(map(int, m.groups()))); continue
        m = re.search(r'stream admitted .*prompt=(\d+) restored=(\d+) prefill_ms=(\d+)', l)
        if m: adm.append(tuple(map(int, m.groups()))); continue
        m = re.search(r'checkpoint saved tokens=(\d+) done=(\d+) total=(\d+) ms=(\d+)', l)
        if m: ck.append(tuple(map(int, m.groups())))
print('window', t0, t1, 'admissions', len(adm), 'restores', len(rest))
print('restore ms total %.1f min, mean %.0f ms' % (sum(r[2] for r in rest) / 6e4, sum(r[2] for r in rest) / len(rest)))
tot_p = sum(a[0] for a in adm); tot_r = sum(a[1] for a in adm)
print('prompt tokens %d restored %d (%.2f%%) prefilled %d' % (tot_p, tot_r, 100 * tot_r / tot_p, tot_p - tot_r))
print('save bytes if every admission saves its prompt: %.1f GB; at 1.61 GB/s = %.1f min' % (tot_p * 2760 / 1e9, tot_p * 2760 / 1.61e9 / 60))
print('checkpoints', len(ck), 'ms total %.1f s' % (sum(c[3] for c in ck) / 1e3))
sfx = sorted(a[0] - a[1] for a in adm if a[1] > 0)
le128 = sum(1 for s in sfx if s <= 128)
print('warm', len(sfx), 'suffix<=128:', le128, '(%.1f%%)' % (100 * le128 / len(sfx)))
gen = None
