"""Hub-side per-lane-layer work: submit_end -> wait_enter ; wait_exit -> next submit (same lane); by lane/group.
usage: hubwork.py HUBEXTRACT LOG WINFILE SPLITJSON"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
S = {r['lab']: r for r in json.load(open(sys.argv[4]))}
P = defaultdict(list)
for lab, a, b, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W' or s['ndec'] > 0 or rows != 4096:
        continue
    lo, hi = H.raw(a), H.raw(b)
    rs = [r for r in H.pre if lo <= r['t_submit'] <= hi]
    for u in units_of(rs):
        for ln in (0, 1):
            v = [r for r in u if int(r['lane']) == ln]
            for p, q in zip(v, v[1:]):
                if q['layer'] == p['layer'] + 1:
                    g = gid(q)
                    P[('turn', ln, g)].append((q['t_submit'] - p['t_wait_exit']) / 1e6)
                    P[('turn', ln, 'all')].append((q['t_submit'] - p['t_wait_exit']) / 1e6)
            for r in v:
                P[('sub2wait', ln, gid(r))].append((r['t_wait_enter'] - r['t_submit_end']) / 1e6)
                P[('sub2wait', ln, 'all')].append((r['t_wait_enter'] - r['t_submit_end']) / 1e6)
                P[('waitlen', ln, 'all')].append((r['t_wait_exit'] - r['t_wait_enter']) / 1e6)
        # first layer of unit: time since previous unit end for lane0
    for ua, ub in zip(units_of(rs), units_of(rs)[1:]):
        pass
for k in sorted(P, key=str):
    v = P[k]
    print(f"{str(k):<26} n={len(v):5d} p10 {pct(v, .1):7.2f} p50 {pct(v, .5):7.2f} mean {sum(v) / len(v):7.2f} p90 {pct(v, .9):7.2f}")
