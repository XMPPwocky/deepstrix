"""Median b2_req fields over steady (no decode, 4096 row) windows. usage: cmpf.py HUB LOG WIN SPLIT B2FILES"""
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
for f in ['pool_resident', 'pin_on', 'pin_pinned', 'pin_budget', 'pin_new', 'pin_denied', 'nursery_occupied', 'nursery_lands', 'pf_free_sets', 'pf_pending', 'pf_run_certain', 'pf_run_spec', 'pf_q_certain', 'n_prefetch_words', 'n_distinct', 'n_sel', 'stage_claims', 'pf_d_hinted', 'pf_d_admitted', 'pf_d_dropped', 'pf_d_waited', 'n_paged', 'n_missing', 'd_misses', 'd_read_ns', 'pf_d_promoted']:
    v = [bb[f] for h, bb in rec if f in bb and not isnan(bb[f])]
    if v:
        print(f"  {f:<18} p10 {pct(v, .1):10.1f} p50 {pct(v, .5):10.1f} mean {sum(v) / len(v):10.1f} p90 {pct(v, .9):10.1f}")
print('hub-side n_hints/n_pf_words/n_distinct:')
for f in ['n_hints', 'n_pf_words', 'n_distinct', 'n_pred_miss', 'n_held', 'n_paged', 'pinned']:
    v = [h[f] for h, bb in rec if not isnan(h[f])]
    if v:
        print(f"  {f:<18} p10 {pct(v, .1):10.1f} p50 {pct(v, .5):10.1f} mean {sum(v) / len(v):10.1f} p90 {pct(v, .9):10.1f}")
