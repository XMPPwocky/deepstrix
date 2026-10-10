"""Why does box-2 run vary? tabulate b2_run by lane/group/layer and vs b2_req fields (steady windows)."""
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
MAXS = int(sys.argv[6])
recs = []
for lab, a, b, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W' or s['ndec'] > 0 or rows != 4096 or s['nmiss'] > MAXS:
        continue
    lo, hi = H.raw(a), H.raw(b)
    for h in H.pre:
        if lo <= h['t_submit'] <= hi:
            bb = B.get((int(h['seq']), int(h['t2_b2'])))
            if bb:
                recs.append((h, bb))
print('N', len(recs))
fields = ['n_sel', 'n_distinct', 'n_work_items', 'two_pass', 'path_decode', 'n_miss', 'n_missing', 'pool_resident', 'exec_gpu_us', 'exec_h2d_us', 'pf_run_certain', 'pf_run_spec', 'pf_run_likely', 'pf_pending',
          'd_repack_gpu_ns', 'd_h2d_ns', 'd_read_ns', 'n_prefetch_words', 'n_hint_admit', 'park_wait_ns', 'partial_lands', 'partial_slots', 'nursery_lands', 'stage_claims', 'n_paged', 'idle_before_us', 'depth_on_take']
for ln in (0, 1):
    for g in range(5):
        v = [(bb['t_run_end'] - bb['t_run_start']) / 1e6 for h, bb in recs if int(h['lane']) == ln and gid(h) == g]
        q = [(bb['t_dequeue'] - bb['t_frame']) / 1e6 for h, bb in recs if int(h['lane']) == ln and gid(h) == g]
        if v:
            print(f"lane {ln} G{g}: n={len(v)} run p10/p50/p90/mean {pct(v, .1):.1f}/{pct(v, .5):.1f}/{pct(v, .9):.1f}/{sum(v) / len(v):.1f}  queue p50/mean {pct(q, .5):.1f}/{sum(q) / len(q):.1f}")
print('per layer (lane0 | lane1) run p50:')
for L in range(20):
    a = [(bb['t_run_end'] - bb['t_run_start']) / 1e6 for h, bb in recs if int(h['lane']) == 0 and int(h['layer']) == L]
    b = [(bb['t_run_end'] - bb['t_run_start']) / 1e6 for h, bb in recs if int(h['lane']) == 1 and int(h['layer']) == L]
    print(f"  L{L:2d} lane0 {pct(a, .5):6.1f} (p90 {pct(a, .9):6.1f})   lane1 {pct(b, .5):6.1f} (p90 {pct(b, .9):6.1f})")
print('medians of b2 fields:')
for f in fields:
    v = [bb[f] for h, bb in recs if not isnan(bb[f])]
    print(f"  {f:<20} p10 {pct(v, .1):12.1f} p50 {pct(v, .5):12.1f} p90 {pct(v, .9):12.1f}")


def corr(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((a - my) ** 2 for a in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float('nan')


run = [(bb['t_run_end'] - bb['t_run_start']) / 1e6 for h, bb in recs]
print('corr(run, field):')
for f in fields:
    xs = [bb[f] for h, bb in recs]
    ok = [(x, r) for x, r in zip(xs, run) if not isnan(x)]
    if ok:
        print(f"  {f:<20} {corr([a for a, b in ok], [b for a, b in ok]):6.2f}")
# run vs time since previous request's run end on the b2 (gap), and vs previous request's run
bs = sorted((bb for h, bb in recs), key=lambda b: b['t_run_start'])
