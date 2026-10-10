"""Run ON box 2: prefill b2_req run time by n_missing, before vs after a knob change.
usage: python3 -I b2split.py EVTRACE_PY EVT_FILE CHANGE_WALL_UTC_ISO [PRE_MINUTES]
CHANGE_WALL_UTC_ISO e.g. 2026-10-10T18:21:56  (wall clock UTC); maps to CLOCK_MONOTONIC_RAW via 'now'."""
import sys, time, datetime, importlib.util, math

spec = importlib.util.spec_from_file_location('evtrace', sys.argv[1])
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
path, iso = sys.argv[2], sys.argv[3]
pre_min = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0

now_raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
now_wall = datetime.datetime.now(datetime.timezone.utc).timestamp()
chg_wall = datetime.datetime.fromisoformat(iso).replace(tzinfo=datetime.timezone.utc).timestamp()
t_chg = now_raw - (now_wall - chg_wall) * 1e9
t_lo = t_chg - pre_min * 60e9

hdr, recs = ev.read_file(path)
rows = recs.get('b2_req', [])
print('b2_req records', len(rows))
if rows:
    print('fields', sorted(rows[0].keys()))
rowf = next((f for f in ('b', 'rows', 'n_rows', 'nb') if rows and f in rows[0]), None)
print('rows field', rowf)


def pct(v, p):
    v = sorted(x for x in v if x == x)
    if not v:
        return float('nan')
    i = (len(v) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def bucket(n):
    return '0-2' if n <= 2 else '3-6' if n <= 6 else '7-12' if n <= 12 else '13+'


groups = {'pre': {}, 'post': {}}
tmin = tmax = None
for r in rows:
    if rowf and r[rowf] <= 16:
        continue
    t = r['t_frame']
    which = 'pre' if t_lo <= t < t_chg else 'post' if t >= t_chg else None
    if which is None:
        continue
    run = (r['t_run_end'] - r['t_run_start']) / 1e6
    q = (r['t_dequeue'] - r['t_frame']) / 1e6
    nm = r.get('n_missing', float('nan'))
    if nm != nm:
        continue
    for key in (bucket(nm), 'ALL'):
        groups[which].setdefault(key, []).append((run, q, nm, r.get('park_wait_ns', float('nan')) / 1e6))
print('t_chg raw', t_chg, 'pre window min', pre_min)
for key in ('0-2', '3-6', '7-12', '13+', 'ALL'):
    for which in ('pre', 'post'):
        g = groups[which].get(key, [])
        if not g:
            print(f'{key:>5} {which:>4}: n=0')
            continue
        run = [x[0] for x in g]
        q = [x[1] for x in g]
        nm = [x[2] for x in g]
        pw = [x[3] for x in g]
        print(f'{key:>5} {which:>4}: n={len(g):5d} run p50 {pct(run, .5):6.1f} mean {sum(run) / len(run):6.1f} p90 {pct(run, .9):6.1f} | '
              f'queue p50 {pct(q, .5):5.1f} | n_missing mean {sum(nm) / len(nm):4.1f} | park_wait p50 {pct(pw, .5):5.1f}')
