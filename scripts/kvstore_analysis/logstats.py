import glob, re, statistics as st
from collections import Counter
ANSI = re.compile(r'\x1b\[[0-9;]*m')
files = sorted(glob.glob('/home/claude-code/logs/v41-server.log.2026*')) + ['/home/claude-code/logs/v41-server.log']
rest=[]; ck=[]; ev=[]; adm=[]; rep=[]; notpref=0; restfail=0
for f in files:
    for line in open(f, errors='replace'):
        if 'multistream' not in line and 'evicted snapshot' not in line and 'prefill_job_replay' not in line: continue
        l = ANSI.sub('', line)
        m = re.search(r'snapshot restored restored=(\d+) total=(\d+) ms=(\d+)', l)
        if m: rest.append(tuple(map(int,m.groups()))); continue
        m = re.search(r'(prefill checkpoint saved|partial snapshot saved) tokens=(\d+) done=(\d+) total=(\d+) ms=(\d+)', l)
        if m: ck.append((m.group(1), int(m.group(2)), int(m.group(5)))); continue
        m = re.search(r'evicted snapshot hash=\S+ bytes=(\d+) reason=(.*)$', l)
        if m: ev.append((int(m.group(1)), m.group(2).strip())); continue
        m = re.search(r'stream admitted .*prompt=(\d+) restored=(\d+) prefill_ms=(\d+)', l)
        if m: adm.append(tuple(map(int,m.groups()))); continue
        m = re.search(r'prefill_job_replay replay_tokens=(\d+) seg_pos0=(\d+) elapsed_s="([\d.]+)"', l)
        if m: rep.append(float(m.group(3))); continue
        if 'not a prefix of the request' in l: notpref+=1
        if 'snapshot restore failed' in l: restfail+=1
print('files', len(files))
print('restores', len(rest))
if rest:
    gb = [r[0]*2760/1e9 for r in rest]; ms=[r[2] for r in rest]
    rate = sorted(g/(m/1000) for g,m in zip(gb,ms) if m>0)
    print(' restore ms p50 %d p90 %d max %d; GB/s p50 %.2f p10 %.2f' % (st.median(ms), sorted(ms)[int(.9*len(ms))], max(ms), st.median(rate), rate[int(.1*len(rate))]))
    big=[(r[0],r[2]) for r in rest if r[0]>100000]
    print(' restores >100K tokens:', len(big), 'ms p50', st.median([b[1] for b in big]) if big else None, 'p90', sorted([b[1] for b in big])[int(.9*len(big))] if big else None)
print('checkpoint saves', len(ck))
for kind in set(c[0] for c in ck):
    cc=[c for c in ck if c[0]==kind]
    print(' ',kind, len(cc), 'ms p50', st.median([c[2] for c in cc]), 'max', max(c[2] for c in cc), 'tokens p50', st.median([c[1] for c in cc]))
    print('   GB/s p50 %.2f' % st.median([c[1]*2760/1e9/(c[2]/1000) for c in cc if c[2]>0]))
print('evictions', len(ev), 'bytes %.1f GB' % (sum(e[0] for e in ev)/1e9))
print(' reasons', Counter(e[1] for e in ev).most_common(6))
if rep: print('replays', len(rep), 'p50 %.2f p90 %.2f' % (st.median(rep), sorted(rep)[int(.9*len(rep))]))
print('not-a-prefix', notpref, 'restore failed', restfail)
warm=[(p-r, ms) for p,r,ms in adm if r>0]
for lo,hi in [(1,128),(129,512),(513,2048),(2049,8192),(8193,32768),(32769,10**7)]:
    w=[x for x in warm if lo<=x[0]<=hi]
    if w: print(' suffix %6d..%-8d n=%4d prefill_ms p50 %6d  ms/token p50 %.1f' % (lo,hi,len(w),st.median([x[1] for x in w]), st.median([x[1]/x[0] for x in w])))
cold=[(p,ms) for p,r,ms in adm if r==0]
for lo,hi in [(1,2048),(2049,20000),(20001,10**7)]:
    w=[x for x in cold if lo<=x[0]<=hi]
    if w: print(' cold %6d..%-8d n=%4d prefill_ms p50 %6d  ms/token p50 %.2f' % (lo,hi,len(w),st.median([x[1] for x in w]), st.median([x[1]/x[0] for x in w])))
