#!/usr/bin/env python3
"""evtrace (*.evt) -> a perfetto timeline (Chrome JSON; open in ui.perfetto.dev).
Both boxes on the hub's clock; the hub's device stage intervals (calibrated to
host time by Tier B, `evtrace_dev`) when a hub Tier B dump is given.

  evt2perfetto.py window HUB.evt --last S [--lone]   hub + box-2 RAW windows of the last S s (JSON);
                                                     --lone: inside the latest run of lone-stream
                                                     (DSpark) steps lasting >= S (else the longest)
  evt2perfetto.py cut FILE... --from T --to T -o PREFIX   one PREFIX.<i>.evt per file holding records
                                                     in [T, T] (run on box 2; prints the paths)
  evt2perfetto.py trace FILE... --from T --to T -o OUT.json[.gz]
                                                     FILEs by header role: the hub's Tier A file,
                                                     its Tier B dumps (device intervals), box-2 cuts

Tracks: hub = decode steps (rows / lanes / live; a lone stream with rows > 1 is a
DSpark block), scheduler phases, per lane the box-2 requests (submit -> reply)
and the hub's exposed waits, counters per step (box-2 wait / page ms, box-1
pager misses / read ms, b2 misses); box 2 = requests (frame -> ready), run_path,
ensure (paging), SSD reads by class (demand / certain / speculative; args: the
drive route), reply writes; hub devices = one track per device and stream
(`dgpu compute`, `igpu xfer`, ...) with each stage's step / layer / lane and
`queued_us` (start - the host's record), calibration counters. The causality
of the device intervals is checked (stderr): (a) none starts before the host
recorded it, (b) none on the dGPU compute stream ends after its step's forward
sync returned (`hub_step.t_fwd_sync`). Box-2 stamps are moved onto the hub's
CLOCK_MONOTONIC_RAW by the hub's measured offset (`Offsets`). Overlapping
slices of one kind go on numbered sub-tracks. The evt headers' env / knob
tables are args of the `knobs` instants. Stdlib only.

These run next to production: files are STREAMED (64 MB chunks, dropped from
the page cache once parsed) and only per-bucket clock samples plus the
window's own records are kept (a hub file holds millions of requests; the hub
can have ~3 GB free). A record whose first time field is NaN is skipped.
"""
import argparse
import gzip
import json
import math
import os
import re
import struct
import sys
from collections import defaultdict

NAN = float('nan')
TFIELD = re.compile(r'^t(_|\d|$)')
CHUNK = 64 << 20
BUCKET = 2e9
# Records are written close to time order: a file is abandoned this far past
# the window's end (box-2 background reads are recorded when LANDED, seconds
# after their read).
STOP_MARGIN = 30e9


def header_of(path):
    """(header length, header) -- read to its exact length: a format_rev 2
    header carries the string table (up to ~43 MB)."""
    with open(path, 'rb') as f:
        head = f.read(8)
        if head[:4] != b'EVT1':
            raise ValueError(f'{path}: not an EVT1 file')
        hlen = struct.unpack_from('<I', head, 4)[0]
        return hlen, json.loads(f.read(hlen))


def decode_str(vals):
    """A `str` record (format_rev 2): id, len, then the UTF-8 bytes six per f64
    slot as an exact integer."""
    sid, ln = int(vals[0]), int(vals[1])
    b = bytearray()
    for x in vals[2:]:
        b += int(x).to_bytes(6, 'little')
    return sid, bytes(b[:ln]).decode('utf-8', 'replace')


def records(path, lo=-math.inf, hi=math.inf, raw=False, stop=math.inf, strings=None):
    """Yield (t, kind name, fields, values[, raw bytes]) of the records whose
    first finite t-field `t` is in [lo, hi]; stop at the first record past
    `stop`. Streams the file (64 MB chunks, POSIX_FADV_DONTNEED after each).
    With `strings` (a dict), every `str` record read (whatever its position)
    is decoded into it: `str` records have no time, so a window never keeps
    them. A Tier B dump (`"tier": "B"`) is in BATCH order, not time order:
    `stop` is ignored there (a newer thread's batch can precede an older one)."""
    hlen, header = header_of(path)
    if header.get('tier') == 'B':
        stop = math.inf
    kinds = {k['id']: (k['name'], k['fields']) for k in header['kinds']}
    tidx = {kid: [i for i, n in enumerate(fs) if TFIELD.match(n)] for kid, (_, fs) in kinds.items()}
    fd = os.open(path, os.O_RDONLY)
    try:
        cache, pos, buf = {}, 8 + hlen, b''
        while True:
            chunk = os.pread(fd, CHUNK, pos)
            if hasattr(os, 'posix_fadvise') and chunk:
                os.posix_fadvise(fd, pos, len(chunk), os.POSIX_FADV_DONTNEED)
            pos += len(chunk)
            data = buf + chunk
            off, n_data = 0, len(data)
            while off + 4 <= n_data:
                kid, n = struct.unpack_from('<HH', data, off)
                if off + 4 + 8 * n > n_data:
                    break  # continues in the next chunk (or a file still being written)
                s = cache.get(n) or cache.setdefault(n, struct.Struct(f'<{n}d'))
                vals = s.unpack_from(data, off + 4)
                if strings is not None and kinds.get(kid, ('',))[0] == 'str':
                    sid, text = decode_str(vals)
                    strings[sid] = text
                t = next((vals[i] for i in tidx.get(kid, []) if i < n and not math.isnan(vals[i])), NAN)
                if t > stop:
                    return
                if lo <= t <= hi:
                    name, fields = kinds.get(kid, (f'kind{kid}', [f'f{i}' for i in range(n)]))
                    if raw:
                        yield t, name, fields, vals, data[off:off + 4 + 8 * n]
                    else:
                        yield t, name, fields, vals
                off += 4 + 8 * n
            buf = data[off:]
            if not chunk:
                return
    finally:
        os.close(fd)


class Offsets:
    """Box-2-minus-hub clock offset vs hub time, from the hub's per-request
    estimates (`hub_req` clock_offset_ns / clock_delay_ns / t1): per 2 s bucket
    the sample with the smallest link delay. `fn` keeps the buckets whose best
    delay is near the link's floor -- the 5th percentile of the bucket minima
    (robust to one bogus sample); keep <= 1.5 x floor + 20 us -- because a
    bucket inside a link-saturating burst (10-01: delays of 14 ms, the offset
    skewed ~4 ms by asymmetric queueing) would bend the interpolation by
    hundreds of us. Linear between kept points; the end segments extrapolate
    with the slope clamped to +-200 ppm (the RAW clocks drift ~100 ppm)."""

    def __init__(self):
        self.best = {}

    def add(self, r):
        o, d, t = r['clock_offset_ns'], r['clock_delay_ns'], r['t1']
        if math.isnan(o) or math.isnan(d) or math.isnan(t) or d <= 0:
            return
        k = int(t // BUCKET)
        if k not in self.best or d < self.best[k][1]:
            self.best[k] = (t, d, o)

    def fn(self, lo=-math.inf, hi=math.inf):
        pts = [v for v in self.best.values() if lo <= v[0] <= hi]
        if not pts:
            print('offsets: no clock samples; box 2 left unaligned', file=sys.stderr)
            return lambda t: NAN
        ds = sorted(d for _, d, _ in pts)
        floor = ds[int(0.05 * (len(ds) - 1))]
        kept = sorted((t, o) for t, d, o in pts if d <= 1.5 * floor + 20e3)
        print(f'offsets: link-delay floor {floor / 1e3:.0f} us, {len(kept)}/{len(pts)} buckets kept', file=sys.stderr)
        if floor > 1e6:
            print('offsets: WARNING the link was saturated across the whole range: box 2 may be shifted by '
                  'up to ~half the floor', file=sys.stderr)
        clamp = 200e-6

        def at(t):
            if math.isnan(t):
                return NAN
            if len(kept) == 1:
                return kept[0][1]
            if t <= kept[0][0] or t >= kept[-1][0]:
                (t0, o0), (t1, o1) = (kept[0], kept[1]) if t <= kept[0][0] else (kept[-2], kept[-1])
                slope = 0.0 if t1 == t0 else max(-clamp, min(clamp, (o1 - o0) / (t1 - t0)))
                anchor = (t0, o0) if t <= kept[0][0] else (t1, o1)
                return anchor[1] + slope * (t - anchor[0])
            lo_i, hi_i = 0, len(kept) - 1
            while hi_i - lo_i > 1:
                mid = (lo_i + hi_i) // 2
                if kept[mid][0] <= t:
                    lo_i = mid
                else:
                    hi_i = mid
            (t0, o0), (t1, o1) = kept[lo_i], kept[hi_i]
            return o0 if t1 == t0 else o0 + (o1 - o0) * (t - t0) / (t1 - t0)
        return at


def cmd_window(a):
    """The last `--last` s of the hub file (by its last step; `--lone`: inside a
    lone-stream run), and the matching box-2 RAW window (+-2 s for drift).
    One streaming pass; keeps only clock samples and lone-run bounds."""
    last, offs, runs, run = -math.inf, Offsets(), [], None
    for _, name, fields, vals in records(a.hub):
        if name == 'hub_step':
            r = dict(zip(fields, vals))
            last = max(last, r['t_end'])
            if r['live'] == 1:
                run = [r['t_start'], r['t_end']] if run is None else [run[0], r['t_end']]
            elif run is not None:
                runs.append(run)
                run = None
        elif name == 'hub_req':
            offs.add(dict(zip(fields, vals)))
    if run is not None:
        runs.append(run)
    if not math.isfinite(last):
        sys.exit('window: no hub_step records')
    if a.lone:
        if not runs:
            sys.exit('window: no lone-stream steps in this file')
        long_enough = [r for r in runs if r[1] - r[0] >= a.last * 1e9]
        pick = long_enough[-1] if long_enough else max(runs, key=lambda r: r[1] - r[0])
        last = pick[1]
        print(f'lone-stream run {(pick[1] - pick[0]) / 1e9:.1f} s ({len(runs)} runs in the file)', file=sys.stderr)
    lo = last - a.last * 1e9
    off = offs.fn(lo - 300e9, last + 300e9)(last)
    print(json.dumps({'hub_from': lo, 'hub_to': last, 'offset': off, 'b2_from': lo + off - 2e9, 'b2_to': last + off + 2e9}))


def first_t(path):
    """The first record's time (a 1 MB read): where a rotated file starts. (The
    header's `t_mono_raw_at_open` is the PROCESS's open, shared by its files.)"""
    hlen, header = header_of(path)
    kinds = {k['id']: [i for i, n in enumerate(k['fields']) if TFIELD.match(n)] for k in header['kinds']}
    with open(path, 'rb') as f:
        f.seek(8 + hlen)
        data = f.read(1 << 20)
    off = 0
    while off + 4 <= len(data):
        kid, n = struct.unpack_from('<HH', data, off)
        if off + 4 + 8 * n > len(data):
            break
        vals = struct.unpack_from(f'<{n}d', data, off + 4)
        t = next((vals[i] for i in kinds.get(kid, []) if i < n and not math.isnan(vals[i])), NAN)
        if not math.isnan(t):
            return t
        off += 4 + 8 * n
    return -math.inf  # none in the first MB: span it from the start (scanned, not skipped)


def boot_of(path):
    """The box's boot (realtime at RAW zero, s): RAW times of files from
    different boots do not compare."""
    h = header_of(path)[1]
    return (h['t_realtime_at_open'] - h['t_mono_raw_at_open']) / 1e9


def boot_now():
    """This box's current boot, from its own clocks (`cut` runs on the box that
    wrote the files; a header's realtime can be skewed or stepped)."""
    import time
    return time.time() - time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def cmd_cut(a):
    """One cut per input file whose time span can hold the window (a file spans
    from its first record to the next file's), each with ITS OWN header: a
    window across a daemon restart may span two binaries' kind tables. Files of
    an earlier boot are skipped (their RAW clock restarted)."""
    now = boot_now()
    same_boot = [p for p in a.files if abs(boot_of(p) - now) < 60]
    # Tier A files span from their first record to the next file's; a Tier B
    # dump (batch order: its first record is not its earliest) spans the
    # window its header names.
    spans = []
    tier_a = sorted((first_t(p), p) for p in same_boot if header_of(p)[1].get('tier') != 'B')
    for i, (t_open, p) in enumerate(tier_a):
        spans.append((t_open, tier_a[i + 1][0] if i + 1 < len(tier_a) else math.inf, p))
    for p in same_boot:
        h = header_of(p)[1]
        if h.get('tier') == 'B':
            d = h.get('dump', {})
            lo = d.get('t_from_raw')
            spans.append((-math.inf if lo is None else lo, d.get('t_to_raw', math.inf), p))
    written = []
    for i, (t_open, t_next, p) in enumerate(spans):
        if t_open > a.t_to or t_next < a.t_from:
            continue
        # The cut's header carries every string defined up to the cut's end
        # (format_rev 2): the window filter drops `str` records (no time).
        _, header = header_of(p)
        strings = dict(enumerate(header.get('strings', [])))
        body = [rawb for _, _, _, _, rawb in records(p, a.t_from, a.t_to, raw=True, stop=a.t_to + STOP_MARGIN, strings=strings)]
        if not body:
            continue
        if strings:
            header['strings'] = [strings.get(k, '') for k in range(max(strings) + 1)]
        hb = json.dumps(header).encode()
        path = f'{a.o}.{i}.evt'
        with open(path, 'wb') as out:
            out.write(b'EVT1' + struct.pack('<I', len(hb)) + hb)
            for rawb in body:
                out.write(rawb)
        written.append(path)
    if not written:
        sys.exit('cut: no records in the window')
    print(' '.join(written))


# String-id fields of a record, decoded as `<field>_s` (format_rev 2).
STR_FIELDS = {'knob': ('name', 'value'), 'dev': ('name', 'device', 'stream'), 'step_dev': ('device',), 'cal': ('device',)}
# `step_dev` -> `hub_step` (as evtrace.py's `merge_step_dev`).
STEP_DEV_SUM = ('pairs', 'deferred', 'dropped', 'viol_a', 'viol_b', 'checked_b', 'tierb_us')
STEP_DEV_MAX = ('q_us_max', 'lag_ms')
STEP_DEV_IDENT = ('t_start', 'step', 'device')


def merge_step_dev(steps, parts):
    """SUM a step's `step_dev` records (per device; its drafter rides the
    previous step's buffer) into its `hub_step` by field name where the step
    has no value of its own; counters as `dev_<name>`."""
    acc = defaultdict(dict)
    for r in parts:
        if r.get('step', NAN) != r.get('step', NAN):
            continue
        d = acc[(r.get('_pid'), int(r['step']))]
        for k, v in r.items():
            if k.startswith('_') or k in STEP_DEV_IDENT or k.endswith('_s') or v != v:
                continue
            d[k] = max(d.get(k, v), v) if k in STEP_DEV_MAX else d.get(k, 0.0) + v
    for s in steps:
        if s.get('step', NAN) != s.get('step', NAN):
            continue
        for k, v in acc.get((s.get('_pid'), int(s['step'])), {}).items():
            if k in STEP_DEV_SUM or k in STEP_DEV_MAX:
                s['dev_' + k] = v
            elif s.get(k, NAN) != s.get(k, NAN):
                s[k] = v


def cmd_trace(a):
    hub_files, b2_files = [], []
    for p in a.files:
        (b2_files if header_of(p)[1].get('role') == 'b2' else hub_files).append(p)
    if not hub_files:
        sys.exit('trace: no hub file')
    # The Tier A file first (its header is the run's; Tier B dumps add intervals).
    hub_files.sort(key=lambda p: header_of(p)[1].get('tier') == 'B')
    hub_hdr = header_of(hub_files[0])[1]
    offs, hub = Offsets(), defaultdict(list)
    keep_lo, keep_hi = a.t_from - 5e9, a.t_to + 5e9
    # String ids are PER PROCESS: decode each file's against its own table
    # (a box-2 window can span a daemon restart: two interners).
    for p in hub_files:
        h = header_of(p)[1]
        f_str = dict(enumerate(h.get('strings', [])))
        # A Tier B dump is in batch order: never stop early in one.
        stop = math.inf if h.get('tier') == 'B' else a.t_to + 300e9 + STOP_MARGIN
        for t, name, fields, vals in records(p, a.t_from - 300e9, a.t_to + 300e9, stop=stop, strings=f_str):
            r = dict(zip(fields, vals))
            if name == 'hub_req':
                offs.add(r)
            if not keep_lo <= t <= keep_hi:
                continue
            for k in STR_FIELDS.get(name, ()):
                r[k + '_s'] = f_str.get(int(r[k]), '?') if r[k] == r[k] else '?'
            r['_pid'] = h.get('pid')
            hub[name].append(r)
    merge_step_dev(hub.get('hub_step', []), hub.get('step_dev', []))
    b2_hdr, b2 = None, defaultdict(list)
    for p in b2_files:
        b2_hdr = header_of(p)[1]
        f_str = dict(enumerate(b2_hdr.get('strings', [])))
        for _, name, fields, vals in records(p, strings=f_str):
            r = dict(zip(fields, vals))
            for k in STR_FIELDS.get(name, ()):
                r[k + '_s'] = f_str.get(int(r[k]), '?') if r[k] == r[k] else '?'
            b2[name].append(r)
    tr = Tracks(a.t_from)
    tr.ev.append({'ph': 'M', 'name': 'process_name', 'pid': 1, 'args': {'name': 'hub (box 1)'}})
    tr.ev.append({'ph': 'M', 'name': 'process_name', 'pid': 2, 'args': {'name': 'box 2 (expertd), on hub clock'}})
    inside = lambda t: a.t_from <= t <= a.t_to
    ex = hub_hdr.get('extras', {})
    tr.instant(1, 'knobs', 'knobs (hub)', a.t_from, {**ex.get('env', {}), **{f'knob {k}': v for k, v in ex.get('knobs', {}).items()}})
    for s in hub.get('hub_step', []):
        if not inside(s['t_start']):
            continue
        rows, live, lanes = int(s['rows']), int(s['live']), s.get('lanes', NAN)
        kind = f'DSpark block K={rows - 1}' if live == 1 and rows > 1 else f'step {rows} rows'
        lanes_s = '' if math.isnan(lanes) else f' {int(lanes)}L'
        # + the step's device busy time by stage (`d_*` dGPU, `i_*` iGPU; ms
        # summed over the step, V41_MS_PROFILE): what, not when.
        stages = {k: s[k] for k in s if k[:2] in ('d_', 'i_') and not math.isnan(s[k]) and s[k] > 0}
        tr.slice(1, 'decode steps', f'{kind}{lanes_s}', s['t_start'], s['t_end'],
                 {**fin(s, 'step', 'rows', 'live', 'lanes', 'step_ms', 'fwd_ms', 'remote_wait_ms', 'b2_page_ms', 'b2_misses',
                        'b1_misses', 'b1_read_ms', 'rf_chain_waits', 'dgpu_busy_ms', 'igpu_busy_ms', 'pos_max',
                        'dev_pairs', 'dev_dropped', 'dev_skipped', 'dev_lag_ms'), **stages})
        for c in ('remote_wait_ms', 'b2_page_ms', 'b2_misses', 'b1_misses', 'b1_read_ms', 'step_ms'):
            tr.counter(1, c, s['t_start'], s.get(c, NAN))
    # Live knob changes (format_rev 2 `knob` records).
    src_name = {0: 'default', 1: 'env', 2: 'legacy file', 3: 'knob file', 4: 'set'}
    # Knob changes are recorded in BOTH tiers: a Tier A file and a Tier B dump
    # of one window hold each change twice -- drawn once per (box, t, name).
    knobs_drawn = set()
    for k in hub.get('knob', []):  # (box 2's: below, on the hub clock)
        nm, val = k['name_s'], k['value_s']
        if inside(k['t']) and (1, k['t'], nm) not in knobs_drawn:
            knobs_drawn.add((1, k['t'], nm))
            tr.instant(1, 'knobs', f'knob {nm}={val}', k['t'], {'source': src_name.get(int(k['source']), '?')})
    for p in hub.get('hub_phase', []):
        if inside(p['t']):
            tr.instant(1, 'phases', 'prefill' if p['to'] == 1 else 'decode', p['t'], fin(p, 'live', 'prefills', 'queued', 'burst_ms', 'starved'))
    for r in hub.get('hub_req', []):
        if not inside(r['t_submit']):
            continue
        lane = 'AB'[int(r['lane'])] if not math.isnan(r['lane']) and r['lane'] < 2 else '?'
        name = f"L{int(r['layer'])} b={int(r['b'])}" if not math.isnan(r['layer']) else 'req'
        args = fin(r, 'step', 'seq', 'rtt_us', 'srv_us', 'page_us', 'compute_us', 'n_miss', 'n_distinct', 'blocked', 'n_surprise')
        tr.slice(1, f'lane {lane}: box-2 request', name, r['t_submit'], r['t4'], args)
        if r['t_wait_exit'] > r['t_wait_enter']:
            tr.slice(1, f'lane {lane}: hub waits', f'wait {name}', r['t_wait_enter'], r['t_wait_exit'], args)
    draw_devices(tr, hub, inside)
    if b2:
        off = offs.fn(a.t_from - 300e9, a.t_to + 300e9)
        to_hub = lambda t: t - off(t - off(a.t_to)) if not math.isnan(t) else NAN
        for k in b2.get('knob', []):
            t = to_hub(k['t'])
            nm, val = k['name_s'], k['value_s']
            if inside(t) and (2, k['t'], nm) not in knobs_drawn:
                knobs_drawn.add((2, k['t'], nm))
                tr.instant(2, 'knobs', f'knob {nm}={val}', t, {'source': src_name.get(int(k['source']), '?')})
        b2ex = (b2_hdr or {}).get('extras', {})
        tr.instant(2, 'knobs', 'knobs (box 2)', a.t_from, {**b2ex.get('env', {}), **{f'knob {k}': v for k, v in b2ex.get('knobs', {}).items()}})
        for r in b2.get('b2_req', []):
            t = to_hub(r['t_frame'])
            if not inside(t):
                continue
            name = f"L{int(r['layer'])} b={int(r['b'])}"
            args = fin(r, 'seq', 'n_miss', 'page_us', 'compute_us', 'server_us', 'merged', 'served_under', 'n_paged', 'depth_on_take')
            tr.slice(2, 'requests (frame -> ready)', name, t, to_hub(r['t_ready']), args)
            tr.slice(2, 'run_path (paging + kernels)', name, to_hub(r['t_run_start']), to_hub(r['t_run_end']), args)
        for e in b2.get('b2_ensure', []):
            t = to_hub(e['t_start'])
            if inside(t) and e['n_miss'] > 0:
                tr.slice(2, 'ensure (paging)', f"L{int(e['layer'])} miss={int(e['n_miss'])}", t, to_hub(e['t_end']),
                         fin(e, 'seq', 'n_want', 'n_hits', 'n_miss', 'admit_wait_ns', 'victim_scan_ns', 'k_par'))
        cls = {0: 'SSD demand reads', 1: 'SSD certain reads', 2: 'SSD certain reads', 3: 'SSD speculative reads'}
        route = {0: 'split', 1: 'mirror (SN5000)', 2: 'primary (E100)'}
        for r in b2.get('b2_read', []):
            t = to_hub(r['t_read_start'])
            if not inside(t):
                continue
            args = fin(r, 'seq', 'layer', 'expert', 'wanted', 'blocked_on', 'yield_ns', 'pause_ns')
            if not math.isnan(r.get('route', NAN)):
                args['route'] = route.get(int(r['route']), '?')
            tr.slice(2, cls.get(int(r['src']), 'SSD reads'), f"L{int(r['layer'])} E{int(r['expert'])}", t, to_hub(r['t_read_end']), args)
        for w in b2.get('b2_write', []):
            t = to_hub(w['t3'])
            if inside(t):
                tr.slice(2, 'reply writes', f"seq {int(w['seq'])}", t, to_hub(w['t_written']), fin(w, 'bytes'))
    tr.finish()
    out = {'traceEvents': tr.ev, 'displayTimeUnit': 'ms',
           'otherData': {'t0_hub_mono_raw_ns': a.t_from, 'window_s': (a.t_to - a.t_from) / 1e9,
                         'hub_realtime_at_open': hub_hdr.get('t_realtime_at_open'), 'hub_mono_raw_at_open': hub_hdr.get('t_mono_raw_at_open')}}
    opener = gzip.open if a.o.endswith('.gz') else open
    with opener(a.o, 'wt') as f:
        json.dump(out, f)
    print(a.o, len(tr.ev), 'events')


def draw_devices(tr, hub, inside):
    """Tier B device intervals (pid 3), calibration counters, and the
    causality check of the intervals against the host."""
    devs = hub.get('dev', [])
    if not devs:
        return
    tr.ev.append({'ph': 'M', 'name': 'process_name', 'pid': 3, 'args': {'name': 'hub devices (calibrated to host time)'}})
    fwd_sync = {(s.get('_pid'), int(s['step'])): s['t_fwd_sync'] for s in hub.get('hub_step', [])
                if s.get('t_fwd_sync', NAN) == s.get('t_fwd_sync', NAN) and s['step'] == s['step']}
    n = viol_a = viol_b = checked_b = 0
    for d in devs:
        if not (inside(d['t_start']) or inside(d['t_end'])):
            continue
        n += 1
        q = d['q_us'] * 1e3
        args = fin(d, 'step', 'unit', 'layer', 'lane', 'q_us')
        if d['t_host'] == d['t_host']:
            args['queued_us'] = round((d['t_start'] - d['t_host']) / 1e3, 1)
            viol_a += d['t_start'] < d['t_host'] - q
        tr.slice(3, f"{d['device_s']} {d['stream_s']}", d['name_s'], d['t_start'], d['t_end'], args)
        if d['device_s'] == 'dgpu' and d['stream_s'] == 'compute' and d['step'] == d['step']:
            ts = fwd_sync.get((d.get('_pid'), int(d['step'])))
            if ts is not None and d['t_host'] < ts:
                checked_b += 1
                viol_b += d['t_end'] > ts + q
    for c in hub.get('cal', []):
        if not inside(c['t']):
            continue
        if c['ok'] == 1:
            tr.counter(3, f"cal {c['device_s']} resid_us", c['t'], c['resid_us'])
            tr.counter(3, f"cal {c['device_s']} q_us", c['t'], c['q_us'])
        else:
            tr.instant(3, f"cal {c['device_s']}", 'anchor discarded', c['t'], fin(c, 'spin_us'))
    print(f'device intervals: {n}; causality (a) start before the host recorded it: {viol_a}; '
          f'(b) after the forward sync: {viol_b} of {checked_b}', file=sys.stderr)


class Tracks:
    """Chrome JSON events. Slices are buffered per track and, at `finish`,
    sorted by start and spread over numbered sub-tracks so that no two overlap
    on one (records arrive in completion order: lanes interleaved, replies out
    of order, parked requests served inside others)."""

    def __init__(self, t0):
        self.t0, self.ev, self.pending, self.tids = t0, [], defaultdict(list), {}

    def tid(self, pid, name):
        key = (pid, name)
        if key not in self.tids:
            self.tids[key] = len(self.tids) + 1
            self.ev.append({'ph': 'M', 'name': 'thread_name', 'pid': pid, 'tid': self.tids[key], 'args': {'name': name}})
            self.ev.append({'ph': 'M', 'name': 'thread_sort_index', 'pid': pid, 'tid': self.tids[key], 'args': {'sort_index': self.tids[key]}})
        return self.tids[key]

    def us(self, t):
        return (t - self.t0) / 1e3

    def slice(self, pid, track, name, a, b, args=None):
        if math.isnan(a) or math.isnan(b) or b < a:
            return
        self.pending[(pid, track)].append((a, b, name, args))

    def finish(self):
        for (pid, track), sl in self.pending.items():
            sl.sort(key=lambda x: (x[0], x[1]))
            ends = []
            for a, b, name, args in sl:
                for i, e in enumerate(ends):
                    if e <= a:
                        ends[i] = b
                        break
                else:
                    i = len(ends)
                    ends.append(b)
                label = track if i == 0 else f'{track} #{i + 1}'
                ev = {'ph': 'X', 'pid': pid, 'tid': self.tid(pid, label), 'name': name, 'ts': self.us(a), 'dur': max((b - a) / 1e3, 0.001)}
                if args:
                    ev['args'] = args
                self.ev.append(ev)
        self.pending.clear()

    def instant(self, pid, track, name, t, args=None):
        if math.isnan(t):
            return
        self.ev.append({'ph': 'i', 's': 't', 'pid': pid, 'tid': self.tid(pid, track), 'name': name, 'ts': self.us(t), 'args': args or {}})

    def counter(self, pid, name, t, value):
        if math.isnan(t) or value is None or math.isnan(value):
            return
        self.ev.append({'ph': 'C', 'pid': pid, 'name': name, 'ts': self.us(t), 'args': {'v': value}})


def fin(r, *keys):
    return {k: r[k] for k in keys if k in r and not math.isnan(r[k])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('window')
    w.add_argument('hub')
    w.add_argument('--last', type=float, default=30.0)
    w.add_argument('--lone', action='store_true', help='a window inside a run of lone-stream (DSpark) steps')
    c = sub.add_parser('cut')
    c.add_argument('files', nargs='+')
    c.add_argument('--from', dest='t_from', type=float, required=True)
    c.add_argument('--to', dest='t_to', type=float, required=True)
    c.add_argument('-o', required=True, help='output prefix: writes PREFIX.<i>.evt per contributing file')
    t = sub.add_parser('trace')
    t.add_argument('files', nargs='+', help='the hub Tier A file, hub Tier B dumps, box-2 files (cuts); by header role')
    t.add_argument('--from', dest='t_from', type=float, required=True)
    t.add_argument('--to', dest='t_to', type=float, required=True)
    t.add_argument('-o', required=True)
    a = ap.parse_args()
    {'window': cmd_window, 'cut': cmd_cut, 'trace': cmd_trace}[a.cmd](a)


if __name__ == '__main__':
    main()
