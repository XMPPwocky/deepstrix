"""Per-window split: wall, exposed wait, hub busy in units, G0 prep gaps, other gaps, head, tail, misses.
usage: split3.py HUBEXTRACT LOG WINFILE OUTJSON"""
import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
print('drift pts', [(round(a, 1), round(b, 3)) for a, b in H.drift_pts][:8], 'slope', H.slope, 'icpt', H.icpt)
rows_out = []
for lab, a, b, rows in windows(sys.argv[3]):
    lo, hi = H.raw(a), H.raw(b)
    rs = [r for r in H.pre if lo <= r['t_submit'] <= hi]
    wall = (hi - lo) / 1e9
    if not rs:
        print('no reqs', lab)
        continue
    expo = union([(r['t_wait_enter'], r['t_wait_exit']) for r in rs]) / 1e9
    nmiss = sum(r['n_miss'] for r in rs if not isnan(r['n_miss']))
    npaged = sum(r['n_paged'] for r in rs if not isnan(r['n_paged']))
    pg = sum(r['page_us'] for r in rs if not isnan(r['page_us'])) / 1e6
    if lab.endswith('R'):
        rows_out.append(dict(lab=lab, kind='R', wall=wall, expo=expo, n=len(rs), rows=rows, nmiss=nmiss, pg=pg))
        continue
    us = units_of(rs)
    span = sum((max(r['t_wait_exit'] for r in u) - min(r['t_submit'] for r in u)) / 1e9 for u in us)
    head = (rs[0]['t_submit'] - lo) / 1e9
    tail = (hi - max(r['t_wait_exit'] for r in rs)) / 1e9
    prep = 0.0
    other = 0.0
    gl = defaultdict(float)
    for ua, ub in zip(us, us[1:]):
        g = (min(r['t_submit'] for r in ub) - max(r['t_wait_exit'] for r in ua)) / 1e9
        key = f'{gid(ua[0])}->{gid(ub[0])}'
        gl[key] += g
        if gid(ua[0]) == 0 and gid(ub[0]) == 0:
            prep += g
        else:
            other += g
    ndec = sum(1 for r in H.dec if lo <= r['t_submit'] <= hi)
    rows_out.append(dict(lab=lab, kind='W', wall=wall, expo=expo, busy=span - expo, prep=prep, other=other, head=head,
                         tail=tail, n=len(rs), nunits=len(us), rows=rows, gl=dict(gl), ndec=ndec, nmiss=nmiss, pg=pg, npaged=npaged))
import json
json.dump(rows_out, open(sys.argv[4], 'w'))
print(f"{'win':<10}{'rows':>5}{'wall':>7}{'expo':>7}{'busy':>7}{'prepG0':>7}{'other':>7}{'head':>6}{'tail':>6}{'nreq':>5}{'nU':>4}{'ndec':>5}{'miss':>6}{'pgs':>6}  rows/s")
for r in rows_out:
    if r['kind'] == 'R':
        print(f"{r['lab']:<10}{r['rows']:5d}{r['wall']:7.2f}{r['expo']:7.2f}   (replay) nreq {r['n']} miss {r['nmiss']:.0f}")
    else:
        print(f"{r['lab']:<10}{r['rows']:5d}{r['wall']:7.2f}{r['expo']:7.2f}{r['busy']:7.2f}{r['prep']:7.2f}{r['other']:7.2f}{r['head']:6.2f}{r['tail']:6.2f}{r['n']:5d}{r['nunits']:4d}{r['ndec']:5d}{r['nmiss']:6.0f}{r['pg']:6.2f}  {r['rows'] / r['wall']:6.0f}")
