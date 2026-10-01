import glob, re, statistics as st
ANSI = re.compile(r'\x1b\[[0-9;]*m')
files = sorted(glob.glob('/home/claude-code/logs/v41-server.log.2026*')) + ['/home/claude-code/logs/v41-server.log']
adm=[]; ev=[]; t0=None; t1=None
for f in files:
    for line in open(f, errors='replace'):
        if 'stream admitted' not in line and 'evicted snapshot' not in line: continue
        l = ANSI.sub('', line)
        ts = l[:19]
        m = re.search(r'stream admitted .*prompt=(\d+) restored=(\d+) prefill_ms=(\d+)', l)
        if m:
            adm.append((ts,)+tuple(map(int,m.groups())))
            t0 = ts if t0 is None or ts < t0 else t0; t1 = ts if t1 is None or ts > t1 else t1
            continue
        m = re.search(r'evicted snapshot hash=\S+ bytes=(\d+)', l)
        if m: ev.append((ts, int(m.group(1))))
print('admissions', len(adm), t0, '..', t1)
evw = [e for e in ev if t0 <= e[0] <= t1]
print('evictions in window', len(evw), '%.1f GB' % (sum(e[1] for e in evw)/1e9))
# bytes saved per admission estimate: prompt tokens*2760
saved = sum(p*2760 for _,p,r,ms in adm)
print('if every admission saved its whole prompt: %.1f GB' % (saved/1e9))
# linear fit of warm prefill ms on suffix, suffix in [129, 16384]
w=[(p-r, ms) for _,p,r,ms in adm if r>0 and 129 <= p-r <= 16384]
n=len(w); mx=sum(x for x,_ in w)/n; my=sum(y for _,y in w)/n
b=sum((x-mx)*(y-my) for x,y in w)/sum((x-mx)**2 for x,_ in w); a=my-b*mx
print('warm 129..16384 fit: prefill_ms = %.0f + %.2f * suffix  (n=%d)' % (a,b,n))
# median-based: bucket medians
c=[(p, ms) for _,p,r,ms in adm if r==0 and p>20000]
n=len(c); mx=sum(x for x,_ in c)/n; my=sum(y for _,y in c)/n
b=sum((x-mx)*(y-my) for x,y in c)/sum((x-mx)**2 for x,_ in c); a=my-b*mx
print('cold >20K fit: prefill_ms = %.0f + %.2f * prompt  (n=%d)' % (a,b,n))
