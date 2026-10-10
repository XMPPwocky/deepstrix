"""Box-2 read concurrency (b2_read intervals) inside steady windows. usage: reads2.py HUB LOG WIN SPLIT B2FILES MAXS"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
S = {r['lab']: r for r in json.load(open(sys.argv[4]))}
MAXS = int(sys.argv[6])
RD = []
for fn in sys.argv[5].split(','):
    _, RB = load(fn, {'b2_read'})
    RD += RB['b2_read']
RD.sort(key=lambda r: r['t_read_start'])
H0 = defaultdict(float)
tot = 0.0
nw = 0
for lab, a, b, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W' or s['ndec'] > 0 or rows != 4096 or s['nmiss'] > MAXS:
        continue
    lo, hi = H.raw(a), H.raw(b)
    o = H.off(lo)
    blo, bhi = lo + o, hi + o
    ev = []
    for r in RD:
        if r['t_read_end'] < blo or r['t_read_start'] > bhi:
            continue
        ev.append((max(r['t_read_start'], blo), 1))
        ev.append((min(r['t_read_end'], bhi), -1))
    ev.sort()
    c, prev = 0, blo
    for t, d in ev:
        if t > prev:
            H0[min(c, 6)] += (t - prev) / 1e9
        prev = t
        c += d
    if bhi > prev:
        H0[min(c, 6)] += (bhi - prev) / 1e9
    tot += (bhi - blo) / 1e9
    nw += 1
print('windows', nw, 'total s', tot)
print('time share by number of reads in flight (role-thread read intervals):', {k: f"{100 * v / tot:.1f}%" for k, v in sorted(H0.items())})
