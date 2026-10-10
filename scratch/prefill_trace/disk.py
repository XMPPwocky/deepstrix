"""Box-2 disk read rate per window (sys records), plus window mean n_missing / park / run.
usage: disk.py HUBEXTRACT LOG WINFILE SPLITJSON B2FILES"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
S = {r['lab']: r for r in json.load(open(sys.argv[4]))}
B, SY = {}, []
for fn in sys.argv[5].split(','):
    _, RB = load(fn, {'b2_req', 'sys'})
    for b in RB['b2_req']:
        B[(int(b['seq']), int(b['t_frame']))] = b
    SY += RB['sys']
SY.sort(key=lambda r: r['t'])
SY = [r for i, r in enumerate(SY) if i == 0 or r['t'] > SY[i - 1]['t']]
ts = [r['t'] for r in SY]
print('sys samples', len(SY), 'dt median ms', pct([(b - a) / 1e6 for a, b in zip(ts, ts[1:])], .5))


def at(t):
    i = bisect.bisect_left(ts, t)
    return SY[min(max(i, 0), len(SY) - 1)]


out = []
print(f"{'win':<10}{'wall':>6}{'nmiss(hub)':>11}{'n_missing':>10}{'park ms':>8}{'run ms':>7} | disk rd MB/s: nvme0 nvme1 sum | util% nvme0 nvme1 | b2 reads MB/s(sect)")
for lab, a, b, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W':
        continue
    lo, hi = H.raw(a), H.raw(b)
    rs = [r for r in H.pre if lo <= r['t_submit'] <= hi]
    o = H.off(lo)
    s0, s1 = at(lo + o), at(hi + o)
    dt = (s1['t'] - s0['t']) / 1e9
    nm, pk, rn, n = 0, 0, 0, 0
    for h in rs:
        bb = B.get((int(h['seq']), int(h['t2_b2'])))
        if bb:
            nm += bb['n_missing']
            pk += bb['park_wait_ns'] / 1e6
            rn += (bb['t_run_end'] - bb['t_run_start']) / 1e6
            n += 1
    r0 = (s1['d0_rsect'] - s0['d0_rsect']) * 512 / dt / 1e6
    r1 = (s1['d1_rsect'] - s0['d1_rsect']) * 512 / dt / 1e6
    u0 = (s1['d0_io_ms'] - s0['d0_io_ms']) / dt / 10
    u1 = (s1['d1_io_ms'] - s0['d1_io_ms']) / dt / 10
    out.append((lab, s['wall'], s['nmiss'], nm / max(n, 1), pk / max(n, 1), rn / max(n, 1), r0, r1, u0, u1, s['ndec']))
    print(f"{lab:<10}{s['wall']:6.2f}{s['nmiss']:11.0f}{nm / max(n, 1):10.2f}{pk / max(n, 1):8.1f}{rn / max(n, 1):7.1f} | {r0:8.0f}{r1:8.0f}{r0 + r1:8.0f} | {u0:6.0f}{u1:6.0f}   dt={dt:.1f}")
json.dump(out, open(sys.argv[6] if len(sys.argv) > 6 else 'disk_out.json', 'w'))
