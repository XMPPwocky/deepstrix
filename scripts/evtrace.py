#!/usr/bin/env python3
"""Reader / analysis for `het::evtrace` files (*.evt; stdlib only).

  evtrace.py summary  FILE...
  evtrace.py hist     FILE... -k KIND -e EXPR [-w COND] [--by EXPR] [--bins N]
  evtrace.py corr     FILE... -k KIND -e EXPR [-w COND] [--top N]
  evtrace.py report   FILE...
  evtrace.py join     FILE...            (hub + b2 files together)
  evtrace.py csv      FILE... -k KIND [-w COND]

EXPR / COND are Python over a record's fields (e.g. `(t_read_end - t_read_start)/1e6`,
`src == 0 and layer >= 20`). Timestamps (`t_*`) are CLOCK_MONOTONIC_RAW ns of the
emitting box; divide by 1e6 for ms. NaN = not measured. Files of several runs /
roles can be mixed; records keep their file's header.
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import struct
import sys
from collections import defaultdict

NAN = float('nan')


# ------------------------------------------------------------------ reading

def read_file(path):
    """-> (header dict, {kind name: [record dict]})."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'EVT1':
        raise ValueError(f'{path}: not an EVT1 file')
    hlen = struct.unpack_from('<I', data, 4)[0]
    header = json.loads(data[8:8 + hlen])
    kinds = {k['id']: (k['name'], k['fields']) for k in header['kinds']}
    out = defaultdict(list)
    off, n_data = 8 + hlen, len(data)
    cache = {}
    while off + 4 <= n_data:
        kid, n = struct.unpack_from('<HH', data, off)
        off += 4
        if off + 8 * n > n_data:
            break  # truncated tail (file still being written)
        s = cache.get(n)
        if s is None:
            s = cache[n] = struct.Struct(f'<{n}d')
        vals = s.unpack_from(data, off)
        off += 8 * n
        name, fields = kinds.get(kid, (f'kind{kid}', [f'f{i}' for i in range(n)]))
        rec = dict(zip(fields, vals))
        rec['_role'] = header.get('role', '?')
        out[name].append(rec)
    return header, out


def expand(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, '*.evt')))
        else:
            files += sorted(glob.glob(p)) or [p]
    return files


def load(paths):
    headers, recs = [], defaultdict(list)
    for p in expand(paths):
        h, r = read_file(p)
        h['_path'] = p
        headers.append(h)
        for k, v in r.items():
            recs[k].extend(v)
    return headers, recs


# ------------------------------------------------------------------ stats

def finite(v):
    return [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]


def pct(sorted_v, p):
    if not sorted_v:
        return NAN
    i = (len(sorted_v) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (i - lo)


def describe(v):
    v = sorted(finite(v))
    if not v:
        return 'n=0'
    ps = ' '.join(f'p{int(p * 1000) / 10:g}={pct(v, p):.4g}' for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999))
    return f'n={len(v)} mean={st.fmean(v):.4g} min={v[0]:.4g} {ps} max={v[-1]:.4g}'


def text_hist(v, bins=24, width=50):
    v = sorted(finite(v))
    if len(v) < 2:
        return ''
    lo, hi = v[0], v[-1]
    log = lo > 0 and hi / lo > 20
    edges = []
    for i in range(bins + 1):
        f = i / bins
        edges.append(lo * (hi / lo) ** f if log else lo + (hi - lo) * f)
    counts = [0] * bins
    j = 0
    for x in v:
        while j < bins - 1 and x > edges[j + 1]:
            j += 1
        counts[j] += 1
    m = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        if c:
            lines.append(f'  {edges[i]:>10.4g} .. {edges[i + 1]:<10.4g} {c:>7} {"#" * max(1, round(width * c / m))}')
    return '\n'.join(lines) + ('\n  (log-spaced bins)' if log else '')


def pearson(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return NAN
    xs, ys = zip(*pairs)
    mx, my = st.fmean(xs), st.fmean(ys)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx == 0 or sy == 0:
        return NAN
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def ev(expr, rec):
    try:
        return float(eval(expr, {'math': math, 'nan': NAN}, rec))
    except (ZeroDivisionError, TypeError, ValueError, KeyError, NameError):
        return NAN


def select(recs, cond):
    if not cond:
        return recs
    code = compile(cond, '<where>', 'eval')
    out = []
    for r in recs:
        try:
            if eval(code, {'math': math, 'nan': NAN}, r):
                out.append(r)
        except (ZeroDivisionError, TypeError, ValueError, KeyError, NameError):
            pass
    return out


def gkey(x):
    """Group key: every NaN is one group."""
    return 'nan' if isinstance(x, float) and math.isnan(x) else x


def key_order(k):
    return (k == 'nan', 0 if k == 'nan' else k)


def show(title, v, bins=24, hist=True):
    print(f'{title}: {describe(v)}')
    if hist:
        h = text_hist(v, bins)
        if h:
            print(h)


# ------------------------------------------------------------------ commands

def cmd_summary(a):
    headers, recs = load(a.files)
    for h in headers:
        print(f"{h['_path']}: role={h.get('role')} pid={h.get('pid')} devices={h.get('sys_devices')}")
    for k in sorted(recs):
        # Per role: each box's stamps are its own RAW clock (different epochs).
        by_role = defaultdict(list)
        for r in recs[k]:
            by_role[r['_role']].append(r)
        for role, v in sorted(by_role.items()):
            ts = finite([r.get('t') or r.get('t_start') or r.get('t_submit') or r.get('t_dequeue') or r.get('t_mono_raw') or r.get('t_hint') or r.get('t_written') or NAN for r in v])
            span = (max(ts) - min(ts)) / 1e9 if len(ts) > 1 else 0
            print(f'  {k:<10} {role:<4} {len(v):>9} records over {span:8.1f} s')
    for m in recs.get('meta', [])[-1:]:
        print(f"  last meta: written={m['written']:.0f} dropped={m['dropped']:.0f} files={m['files']:.0f}")


def cmd_hist(a):
    _, recs = load(a.files)
    rs = select(recs.get(a.kind, []), a.where)
    if a.by:
        groups = defaultdict(list)
        for r in rs:
            groups[gkey(ev(a.by, r))].append(ev(a.expr, r))
        for g in sorted(groups, key=key_order):
            show(f'[{a.by} = {g}] {a.expr}', groups[g], a.bins, hist=not a.no_hist)
    else:
        show(a.expr, [ev(a.expr, r) for r in rs], a.bins)


def cmd_corr(a):
    _, recs = load(a.files)
    rs = select(recs.get(a.kind, []), a.where)
    y = [ev(a.expr, r) for r in rs]
    fields = [f for f in (rs[0].keys() if rs else []) if not f.startswith('_')]
    out = []
    for f in fields:
        x = [r.get(f, NAN) for r in rs]
        c = pearson(x, y)
        if not math.isnan(c):
            out.append((abs(c), c, f, st.median(finite(x)) if finite(x) else NAN))
    out.sort(reverse=True)
    print(f'{len(rs)} records; r(field, {a.expr}):')
    for _, c, f, med in out[:a.top]:
        print(f'  {f:<32} r={c:+.3f}  median={med:.4g}')


def cmd_csv(a):
    _, recs = load(a.files)
    rs = select(recs.get(a.kind, []), a.where)
    if not rs:
        return
    fields = [f for f in rs[0] if not f.startswith('_')]
    print(','.join(fields))
    for r in rs:
        print(','.join(f'{r[f]:.17g}' for f in fields))


def ms(a, b):
    return (b - a) / 1e6


def cmd_report(a):
    _, recs = load(a.files)
    bins = a.bins
    steps = recs.get('hub_step', [])
    if steps:
        print(f'\n===== hub_step ({len(steps)} steps)')
        by = defaultdict(list)
        for s in steps:
            by[s['rows']].append(s)
        for rows in sorted(by):
            g = by[rows]
            if len(g) < 5:
                continue
            print(f'\n-- rows={rows:g} ({len(g)} steps)')
            show('step_ms', [s['step_ms'] for s in g], bins)
            for f in ('fwd_ms', 'b2_page_ms', 'remote_wait_ms', 'b2_misses', 'lh_pager_block', 'dgpu_busy_ms', 'igpu_busy_ms'):
                show(f, [s.get(f, NAN) for s in g], hist=False)
            y = [s['fwd_ms'] for s in g]
            cs = []
            for f in g[0]:
                if f.startswith('_') or f in ('fwd_ms', 'fwd_all_ms', 'step_ms', 't_start', 't_end', 'step'):
                    continue
                c = pearson([s.get(f, NAN) for s in g], y)
                if not math.isnan(c):
                    cs.append((abs(c), c, f))
            cs.sort(reverse=True)
            print('  r(field, fwd_ms): ' + ', '.join(f'{f} {c:+.2f}' for _, c, f in cs[:10]))
    hr = recs.get('hub_req', [])
    if hr:
        print(f'\n===== hub_req ({len(hr)} box-2 requests)')
        show('rtt ms (t4 - t_submit)', [ms(r['t_submit'], r['t4']) for r in hr], bins)
        show('exposed wait ms', [ms(r['t_wait_enter'], r['t_wait_exit']) for r in hr], bins)
        show('blocked', [r['blocked'] for r in hr], hist=False)
        show('box-2 server ms (srv_us)', [r['srv_us'] / 1e3 for r in hr], hist=False)
        show('box-2 page ms (page_us)', [r['page_us'] / 1e3 for r in hr], hist=False)
        show('link ms (rtt_us - srv_us)', [(r['rtt_us'] - r['srv_us']) / 1e3 for r in hr], hist=False)
        show('clock delay us', [r['clock_delay_ns'] / 1e3 for r in hr], hist=False)
    br = recs.get('b2_req', [])
    if br:
        n_merged = sum(1 for r in br if r['merged'] == 2)
        n_parked = sum(1 for r in br if not math.isnan(r['served_under']))
        print(f'\n===== b2_req ({len(br)} requests on box 2; {n_merged} merged partners, {n_parked} served under a park)')
        # Per-REQUEST rows use every record; per-PASS rows skip merged partners
        # (merged == 2), whose pass is the carrier's.
        passes = [r for r in br if r['merged'] != 2]
        for name, f, rows in [
            ('queue (frame -> dequeue)', lambda r: ms(r['t_frame'], r['t_dequeue']), br),
            ('merge', lambda r: ms(r['t_dequeue'], r['t_merge_end']), passes),
            ('hints', lambda r: ms(r['t_merge_end'], r['t_hints_end']), passes),
            ('run_path (paging + kernels)', lambda r: ms(r['t_run_start'], r['t_run_end']), passes),
            ('d2h', lambda r: ms(r['t_run_end'], r['t_d2h_end']), passes),
            ('reply build (residency, partner, len)', lambda r: ms(r['t_d2h_end'], r['t_ready']), passes),
            ('server (frame -> ready)', lambda r: ms(r['t_frame'], r['t_ready']), br),
            ('page_us reported (ms)', lambda r: r['page_us'] / 1e3, passes),
            ('d_read_ns (ms; incl. h2d + blocked prefetch wait)', lambda r: r['d_read_ns'] / 1e6, passes),
            ('d_prefetch_wait_ns (ms)', lambda r: r['d_prefetch_wait_ns'] / 1e6, passes),
        ]:
            show(name, [f(r) for r in rows], bins, hist=name.startswith(('queue', 'run_path', 'server')))
        by = defaultdict(list)
        for r in passes:
            k = r['n_miss']
            by[gkey(k if math.isnan(k) else int(min(k, 6)))].append(ms(r['t_run_start'], r['t_run_end']))
        print('  run_path ms by n_miss (6 = 6+; merged partners excluded):')
        for k in sorted(by, key=key_order):
            v = sorted(finite(by[k]))
            print(f'    n_miss={k}: n={len(v)} p50={pct(v, .5):.3g} p90={pct(v, .9):.3g} p99={pct(v, .99):.3g}')
    en = recs.get('b2_ensure', [])
    if en:
        print(f'\n===== b2_ensure ({len(en)} paging calls)')
        show('admit (land background reads) ms', [ms(r['t_start'], r['t_admit_end']) for r in en], hist=False)
        show('admit blocked wait ms', [r['admit_wait_ns'] / 1e6 for r in en], hist=False)
        show('victim scan ms', [r['victim_scan_ns'] / 1e6 for r in en], hist=False)
        m = [r for r in en if r['n_miss'] > 0]
        show('demand reads ms (calls with misses)', [ms(r['t_victims_end'], r['t_reads_end']) for r in m], bins)
        show('demand reads ms per miss', [ms(r['t_victims_end'], r['t_reads_end']) / r['n_miss'] for r in m], bins)
        show('remap upload ms', [r['remap_upload_ns'] / 1e6 for r in en], hist=False)
    rd = recs.get('b2_read', [])
    if rd:
        names = {0: 'demand', 1: 'bg certain', 2: 'bg spec->certain', 3: 'bg speculative'}
        print(f'\n===== b2_read ({len(rd)} expert reads)')
        by = defaultdict(list)
        for r in rd:
            by[r['src']].append(r)
        for src in sorted(by):
            g = by[src]
            print(f'\n-- src={src:g} {names.get(int(src), "?")} ({len(g)})')
            show('read ms (read_start -> read_end)', [ms(r['t_read_start'], r['t_read_end']) for r in g], bins)
            show('slowest role ms', [max(ms(r[f'r{i}_start'], r[f'r{i}_end']) for i in range(3)) for r in g], hist=False)
            show('queue ms (hint -> pop)', [ms(r['t_hint'], r['t_pop']) for r in g], hist=False)
            show('yield ms', [r['yield_ns'] / 1e6 for r in g], hist=False)
            show('pause ms', [r['pause_ns'] / 1e6 for r in g], hist=False)
            show('landed-after ms (read_end -> recv)', [ms(r['t_read_end'], r['t_recv']) for r in g], hist=False)
            show('repack ms', [r['repack_ns'] / 1e6 for r in g], hist=False)
            show('wanted', [r['wanted'] for r in g], hist=False)
            conc = defaultdict(list)
            for r in g:
                # demand_reads_at_start is NaN on demand reads (see the kind's doc)
                d = r['demand_reads_at_start']
                c = (0 if math.isnan(d) else d) + r['run_certain_at_start'] + r['run_spec_at_start']
                if r['src'] == 0 and not math.isnan(r['chunk_n']):
                    c += r['chunk_n'] - 1  # the other demand reads of its chunk
                conc[gkey(c if math.isnan(c) else int(min(c, 6)))].append(ms(r['t_read_start'], r['t_read_end']))
            print('  read ms by concurrency at read start (demand + bg running; 6 = 6+):')
            for k in sorted(conc, key=key_order):
                v = sorted(finite(conc[k]))
                if v:
                    print(f'    {k}: n={len(v)} p50={pct(v, .5):.3g} p90={pct(v, .9):.3g} p99={pct(v, .99):.3g}')
    sy = [r for r in recs.get('sys', [])]
    if len(sy) > 2:
        by_role = defaultdict(list)
        for r in sy:
            by_role[r['_role']].append(r)
        for role, g in by_role.items():
            g.sort(key=lambda r: r['t'])
            print(f'\n===== sys ({role}, {len(g)} samples)')
            for d in range(3):
                mbps, util = [], []
                for p, q in zip(g, g[1:]):
                    dt = (q['t'] - p['t']) / 1e9
                    if dt <= 0 or math.isnan(q[f'd{d}_rsect']):
                        continue
                    mbps.append((q[f'd{d}_rsect'] - p[f'd{d}_rsect']) * 512 / dt / 1e6)
                    util.append((q[f'd{d}_io_ms'] - p[f'd{d}_io_ms']) / 1e3 / dt)
                if mbps:
                    show(f'  d{d} read MB/s', mbps, hist=False)
                    show(f'  d{d} busy fraction', util, hist=False)
            for f in ('psi_io_some', 'psi_io_full', 'psi_cpu_some'):
                rate = [(q[f] - p[f]) / ((q['t'] - p['t']) / 1e3) for p, q in zip(g, g[1:]) if q['t'] > p['t']]
                show(f'  {f} (fraction stalled)', rate, hist=False)
    meta = recs.get('meta', [])
    if meta:
        print(f"\nmeta: dropped={max(m['dropped'] for m in meta):.0f} written={max(m['written'] for m in meta):.0f}")


def smoothed_offset(hub, bucket_ns=2e9):
    """Box-2-minus-hub clock offset as a function of hub time: in each 2 s bucket
    the sample with the smallest link delay (least queueing, so the most
    symmetric path), linearly interpolated between buckets and extrapolated
    along the end segments (the RAW clocks drift apart by tens of ppm)."""
    best = {}
    for h in hub:
        o, d, t = h['clock_offset_ns'], h['clock_delay_ns'], h['t1']
        if math.isnan(o) or math.isnan(d) or math.isnan(t):
            continue
        k = int(t // bucket_ns)
        if k not in best or d < best[k][1]:
            best[k] = (t, d, o)
    pts = sorted((t, o) for t, _, o in best.values())
    if not pts:
        return lambda t: NAN

    def at(t):
        # Linear between bucket points, and EXTRAPOLATED along the end
        # segments (the two RAW clocks drift by tens of ppm: a flat edge
        # would be off by hundreds of us after a few seconds).
        if math.isnan(t):
            return NAN
        if len(pts) == 1:
            return pts[0][1]
        if t <= pts[0][0]:
            lo, hi = 0, 1
        elif t >= pts[-1][0]:
            lo, hi = len(pts) - 2, len(pts) - 1
        else:
            lo, hi = 0, len(pts) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if pts[mid][0] <= t:
                    lo = mid
                else:
                    hi = mid
        (t0, o0), (t1, o1) = pts[lo], pts[hi]
        return o0 if t1 == t0 else o0 + (o1 - o0) * (t - t0) / (t1 - t0)
    return at


def cmd_join(a):
    _, recs = load(a.files)
    hub = recs.get('hub_req', [])
    b2 = {}
    for r in recs.get('b2_req', []):
        b2[(int(r['seq']), int(r['t_frame']))] = r
    reads = defaultdict(list)
    for r in recs.get('b2_read', []):
        if r['src'] == 0 and not math.isnan(r['seq']):
            reads[int(r['seq'])].append(r)
    joined = []
    for h in hub:
        if math.isnan(h['t2_b2']):
            continue
        b = b2.get((int(h['seq']), int(h['t2_b2'])))
        if b:
            joined.append((h, b))
    n_under = sum(1 for _, b in joined if not math.isnan(b['served_under']))
    n_partner = sum(1 for _, b in joined if b['merged'] == 2)
    print(f'{len(hub)} hub_req, {len(b2)} b2_req, {len(joined)} joined on (seq, t2) '
          f'({n_partner} merged partners, {n_under} served under a park)')
    if not joined:
        return
    offset = smoothed_offset(hub)
    parts = defaultdict(list)
    for h, b in joined:
        # box-2 intervals are box-2 clock differences; the hub's are hub-clock.
        # The wire legs need the clock offset: a per-request estimate makes
        # both legs equal to its own delay by construction, so use the
        # smoothed one (min-delay sample per 30 s, interpolated).
        off = offset(h['t1'])
        parts['hub submit call'].append(ms(h['t_submit'], h['t_submit_end']))
        parts['hub queue->write (t1 - t_submit)'].append(ms(h['t_submit'], h['t1']))
        parts['wire out (t2 - t1 - offset)'].append((h['t2_b2'] - h['t1'] - off) / 1e6)
        parts['b2 queue'].append(ms(b['t_frame'], b['t_dequeue']))
        parts['b2 merge+hints'].append(ms(b['t_dequeue'], b['t_hints_end']))
        parts['b2 run_path'].append(ms(b['t_run_start'], b['t_run_end']))
        parts['b2 d2h+reply'].append(ms(b['t_run_end'], b['t_ready']))
        parts['b2 ready->t3 (writer)'].append(ms(b['t_ready'], h['t3_b2']))
        parts['wire back (t4 - t3 + offset)'].append((h['t4'] - h['t3_b2'] + off) / 1e6)
        parts['hub exposed wait'].append(ms(h['t_wait_enter'], h['t_wait_exit']))
        parts['rtt'].append(ms(h['t_submit'], h['t4']))
    for k, v in parts.items():
        show(k, v, hist=False)
    # per step: box-2 demand-read time under the step's requests vs the step
    per_step = defaultdict(lambda: {'n_req': 0, 'n_miss': 0, 'run_ms': 0.0, 'read_ms': 0.0, 'max_read_ms': 0.0, 'wait_ms': 0.0})
    for h, b in joined:
        if math.isnan(h['step']):
            continue  # a prefill request (no decode step)
        s = per_step[int(h['step'])]
        s['n_req'] += 1
        s['n_miss'] += b['n_miss']
        s['run_ms'] += ms(b['t_run_start'], b['t_run_end'])
        s['wait_ms'] += ms(h['t_wait_enter'], h['t_wait_exit'])
        for r in reads.get(int(b['seq']), []):
            if b['t_dequeue'] <= r['t_read_start'] <= b['t_ready']:
                d = ms(r['t_read_start'], r['t_read_end'])
                s['read_ms'] += d
                s['max_read_ms'] = max(s['max_read_ms'], d)
    steps = {int(s['step']): s for s in recs.get('hub_step', [])}
    rows = [(steps[k], v) for k, v in per_step.items() if k in steps]
    if rows:
        print(f'\n{len(rows)} steps with joined requests:')
        y = [s['fwd_ms'] for s, _ in rows]
        for f in ('n_miss', 'run_ms', 'read_ms', 'max_read_ms', 'wait_ms'):
            print(f'  r({f}, fwd_ms) = {pearson([v[f] for _, v in rows], y):+.3f}   median {st.median(v[f] for _, v in rows):.3g}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    for name in ('summary', 'hist', 'corr', 'report', 'join', 'csv'):
        p = sp.add_parser(name)
        p.add_argument('files', nargs='+')
        p.add_argument('-k', '--kind')
        p.add_argument('-e', '--expr')
        p.add_argument('-w', '--where')
        p.add_argument('--by')
        p.add_argument('--bins', type=int, default=24)
        p.add_argument('--top', type=int, default=25)
        p.add_argument('--no-hist', action='store_true')
    a = ap.parse_args()
    {'summary': cmd_summary, 'hist': cmd_hist, 'corr': cmd_corr, 'report': cmd_report, 'join': cmd_join, 'csv': cmd_csv}[a.cmd](a)


if __name__ == '__main__':
    main()
