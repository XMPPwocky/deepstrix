"""Condition per-request box-2 quantities on n_missing (experts not landed at pass start), steady+mid windows (no decode).
usage: cond.py HUBEXTRACT LOG WINFILE SPLITJSON B2FILES"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
S = {r['lab']: r for r in json.load(open(sys.argv[4]))}
B = {}
for fn in sys.argv[5].split(','):
    _, RB = load(fn, {'b2_req'})
    for b in RB['b2_req']:
        B[(int(b['seq']), int(b['t_frame']))] = b
rec = []
for lab, a, b, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W' or s['ndec'] > 0 or rows != 4096:
        continue
    lo, hi = H.raw(a), H.raw(b)
    for h in H.pre:
        if lo <= h['t_submit'] <= hi:
            bb = B.get((int(h['seq']), int(h['t2_b2'])))
            if bb:
                rec.append((h, bb))
print('N', len(rec))
bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 999)]
for ln in (0, 1):
    print('lane', ln)
    for lo_, hi_ in bins:
        sel = [(h, bb) for h, bb in rec if int(h['lane']) == ln and lo_ <= bb['n_missing'] <= hi_]
        if not sel:
            continue
        run = [(bb['t_run_end'] - bb['t_run_start']) / 1e6 for h, bb in sel]
        park = [bb['park_wait_ns'] / 1e6 for h, bb in sel]
        net = [r - p for r, p in zip(run, park)]
        pg = [bb['page_us'] / 1e3 for h, bb in sel]
        print(f"  n_missing {lo_}-{hi_}: n={len(sel):5d}  run p50 {pct(run, .5):6.1f} mean {sum(run) / len(run):6.1f} | park p50 {pct(park, .5):6.1f} | run-park p10/p50/p90 {pct(net, .1):5.1f}/{pct(net, .5):5.1f}/{pct(net, .9):5.1f} | page_us p50 {pct(pg, .5):6.1f}")
allr = [(bb['t_run_end'] - bb['t_run_start']) / 1e6 - bb['park_wait_ns'] / 1e6 for h, bb in rec]
print('all: run-park p10/p50/mean/p90', pct(allr, .1), pct(allr, .5), sum(allr) / len(allr), pct(allr, .9))
print('share of requests with n_missing==0:', sum(1 for h, bb in rec if bb['n_missing'] == 0) / len(rec))
print('mean n_missing', sum(bb['n_missing'] for h, bb in rec) / len(rec), 'mean n_distinct', sum(bb['n_distinct'] for h, bb in rec) / len(rec))
