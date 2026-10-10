"""Build windows file from server log(s): per LM job (tokens_total>=4096 or multi-window), list windows + replay.
Marks windows overlapping a Prefill->Decode phase (decode burst inside) with suffix 'D'."""
import re, sys
ansi = re.compile(r'\x1b\[[0-9;]*m')
ev = []
for fn in sys.argv[1:]:
    for line in open(fn, errors='replace'):
        line = ansi.sub('', line)
        if 'prefill_lm_window' in line or 'prefill_job_replay' in line or 'ms.phase' in line:
            ts = line[11:23]
            ev.append((line[:26], ts, line))
ev.sort()
jobs = []
cur = None
phase = []
for full, ts, l in ev:
    if 'ms.phase' in l:
        phase.append((ts, 'to=Decode' in l))
        continue
    m = re.search(r'tokens_total=(\d+)', l)
    if 'window open' in l:
        rows = int(re.search(r'rows=(\d+)', l).group(1))
        td = int(re.search(r'tokens_done=(\d+)', l).group(1))
        tt = int(m.group(1))
        if td == 0 or cur is None:
            cur = {'tt': tt, 'w': [], 'open': ts}
            jobs.append(cur)
        cur['w'].append([ts, None, rows])
    elif 'window done' in l and cur and cur['w'] and cur['w'][-1][1] is None:
        cur['w'][-1][1] = ts
    elif 'prefill_job_replay' in l and cur is not None and 'replay' not in cur and cur['w'] and cur['w'][-1][1]:
        el = float(re.search(r'elapsed_s="([\d.]+)"', l).group(1))
        tot = float(re.search(r'total_s="([\d.]+)"', l).group(1))
        cur['replay'] = (cur['w'][-1][1], ts, el, tot)
for j, job in enumerate(jobs):
    if job['tt'] < 4096:
        continue
    print(f"# job tt={job['tt']} open {job['open']} windows {len(job['w'])} replay {job.get('replay')}")
    for i, (a, b, rows) in enumerate(job['w']):
        if b is None:
            continue
        dec = any(a <= t <= b and to_dec for t, to_dec in phase)
        print(f"J{job['open'][:5].replace(':','')}w{i+1}{'D' if dec else ''} {a} {b} {rows}")
    if 'replay' in job:
        print(f"J{job['open'][:5].replace(':','')}R {job['replay'][0]} {job['replay'][1]} 128")
