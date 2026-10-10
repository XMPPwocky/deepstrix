import sys, datetime, re
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *
hdr, R = load(sys.argv[1], {'hub_phase'})
r0, e0 = hdr['t_mono_raw_at_open'], hdr['t_realtime_at_open']
print('hub_phase n', len(R['hub_phase']), 'hdr pid', hdr['pid'], 'e0', e0)
ph = [(p['t'] - r0 + e0) / 1e9 for p in R['hub_phase']]
print('first/last evt utc', [datetime.datetime.fromtimestamp(x, datetime.UTC).strftime('%H:%M:%S') for x in (ph[0], ph[-1])] if ph else None)
logp = []
for line in open(sys.argv[2], errors='replace'):
    if 'ms.phase' in line:
        line = ansi.sub('', line)
        logp.append(datetime.datetime.fromisoformat(line[:26] + '+00:00').timestamp())
print('log phase n', len(logp), datetime.datetime.fromtimestamp(logp[0], datetime.UTC), datetime.datetime.fromtimestamp(logp[-1], datetime.UTC))
for x in ph[:5]:
    b = min(logp, key=lambda y: abs(y - x))
    print(x, b - x)
