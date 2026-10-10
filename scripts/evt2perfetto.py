#!/usr/bin/env python3
"""evtrace (*.evt) -> a perfetto timeline (Chrome JSON; open in ui.perfetto.dev).
Both boxes on the hub's clock; the hub's device stage intervals (calibrated to
host time by Tier B, `evtrace_dev`) when a hub Tier B dump is given.

  evt2perfetto.py window HUB.evt --last S [--lone|--two]   hub + box-2 RAW windows of the last S s (JSON);
                                                     --lone: inside the latest run of lone-stream
                                                     (DSpark) steps lasting >= S (else the longest);
                                                     --two: the same for a run of 2-stream DSpark
                                                     steps (live == 2 and spec_streams >= 2; files
                                                     without spec_streams: rows >= 4), tolerating
                                                     short interruptions (stats on stderr)
  evt2perfetto.py cut FILE... --from T --to T -o PREFIX   one PREFIX.<i>.evt per file holding records
                                                     in [T, T] (run on box 2; prints the paths); `sys`
                                                     records also from 5 s before T (delta predecessor)
  evt2perfetto.py trace FILE... --from T --to T -o OUT.json[.gz] [--disk-models-hub M] [--disk-models-b2 M]
                         [--flows auto|on|off] [--max-flows N]
                                                     FILEs by header role: the hub's Tier A file,
                                                     its Tier B dumps (device intervals), box-2 cuts.
                                                     M = a JSON object / NAME=MODEL list / `lsblk -d
                                                     -n -o NAME,MODEL` text, inline or a file path

Tracks: hub = decode steps (rows / lanes / live; a lone stream with rows > 1 is a
DSpark block), scheduler phases, per lane the box-2 requests (submit -> reply)
and the hub's exposed waits, counters of the profiled steps (box-2 wait / page ms,
b2 misses: true per-step values, reset to 0 at the step's end so they do not hold
across the ~3 s of unprofiled steps); box 2 = requests (frame -> ready), run_path,
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

System tracks (the `sys` kind, sampled every V41_EVTRACE_SYS_MS; deltas between
consecutive samples, drawn over the interval they cover): "box 1 disks + system"
and "box 2 disks + system" -- per NVMe disk read / write MB/s and IOPS, average
read / write latency, in-flight, average queue depth, utilisation; PSI io / cpu /
mem, procs running / blocked, dirty + writeback, this process's own I/O; box 1's
pager misses / read ms (the hub records these on PROFILED steps only, about every
3 s, as the delta of a lifetime counter since the previous profiled step, prefill
misses included: drawn as an average rate over that whole interval, not per step);
the box-1 pager's SSD reads (`b1_read`) and box
2's SSD reads (`b2_read`, by drive -- the `route` code mapped through the drive
the expert dir / mirror dir sit on -- and class demand / certain / speculative).
"Thunderbolt link (box 1 <-> box 2)": per request the wire legs (hub send t1 ->
box-2 receive t2, box-2 send t3 -> hub receive t4, the box-2 stamps moved onto the
hub clock: legs are ~30 us against an offset bound of tens of us, so a leg that
comes out shorter than 2 us or negative is drawn 2 us wide with
`clock_clamped=1` and counted on stderr), link MB/s per direction from the
requests' bytes (10 ms buckets), each box's NIC MB/s and rx_missed from the
`sys` tb_* fields, and flow arrows hub submit -> wire -> box-2 request -> wire ->
hub wait. Fields a file lacks (older binaries) skip their tracks; every skip is
listed on stderr.

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


class TwoRuns:
    """Streaming finder of runs of 2-stream DSpark steps. A step QUALIFIES when
    live == 2 and (spec_streams >= 2; a file without that field: rows >= 4, the
    two streams' blocks). A run survives short interruptions: a non-qualifying
    stretch of <= 2 steps, or of < 1 s, between qualifying steps (a qualifying
    step also joins when it starts < 3 s after the run's end with 1-2 bad steps
    in between; an idle gap with NO steps in between joins only when < 1 s, so
    a scheduler stall of 1 s or more splits the run). A run's bounds are its first / last qualifying step's."""
    GAP_NS = 1e9
    GAP_STEPS = 2
    GAP_STEPS_NS = 3e9

    def __init__(self):
        self.runs, self.run, self.pend = [], None, 0
        self.n_steps = self.n_live2 = self.n_qual = self.n_spec_field = 0

    @staticmethod
    def qualifies(live, rows, spec):
        if live != 2:
            return False
        return spec >= 2 if spec == spec else rows >= 4

    def add(self, t0, t1, live, rows, spec):
        self.n_steps += 1
        self.n_live2 += live == 2
        self.n_spec_field += spec == spec
        q = self.qualifies(live, rows, spec)
        self.n_qual += q
        run = self.run
        if run is not None:
            if q:
                gap = t0 - run['end']
                if gap < self.GAP_NS or (0 < self.pend <= self.GAP_STEPS and gap < self.GAP_STEPS_NS):
                    run['end'] = t1
                    run['n_qual'] += 1
                    run['n_total'] += 1 + self.pend
                    self.pend = 0
                    return
                self.close()
            else:
                self.pend += 1
                if self.pend > self.GAP_STEPS and t1 - run['end'] >= self.GAP_NS:
                    self.close()
                return
        if q:
            self.run = {'start': t0, 'end': t1, 'n_qual': 1, 'n_total': 1}
            self.pend = 0

    def close(self):
        if self.run is not None:
            self.runs.append(self.run)
        self.run, self.pend = None, 0

    def finish(self):
        self.close()
        return self.runs


def cmd_window(a):
    """The last `--last` s of the hub file (by its last step; `--lone`: inside a
    lone-stream run; `--two`: inside a 2-stream DSpark run), and the matching
    box-2 RAW window (+-2 s for drift). One streaming pass; keeps only clock
    samples and run bounds."""
    if a.lone and a.two:
        sys.exit('window: --lone and --two are exclusive')
    last, offs, runs, run, two = -math.inf, Offsets(), [], None, TwoRuns()
    for _, name, fields, vals in records(a.hub):
        if name == 'hub_step':
            r = dict(zip(fields, vals))
            last = max(last, r['t_end'])
            if a.two:
                two.add(r['t_start'], r['t_end'], r['live'], r.get('rows', NAN), r.get('spec_streams', NAN))
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
        if a.full and not long_enough:
            sys.exit(f'window: no lone-stream run >= {a.last:g} s in this file (--full)')
        pick = long_enough[-1] if long_enough else max(runs, key=lambda r: r[1] - r[0])
        last = pick[1]
        print(f'lone-stream run {(pick[1] - pick[0]) / 1e9:.1f} s ({len(runs)} runs in the file)', file=sys.stderr)
    if a.two:
        truns = two.finish()
        basis = 'spec_streams >= 2' if two.n_spec_field else 'rows >= 4, the file has no spec_streams field'
        print(f'2-stream steps: {two.n_qual} qualifying of {two.n_live2} live==2 of {two.n_steps} steps '
              f'({two.n_qual / max(two.n_steps, 1):.1%} of all; qualifying = live == 2 and {basis})', file=sys.stderr)
        if not truns:
            sys.exit('window: no 2-stream DSpark steps in this file')
        long_enough = [r for r in truns if r['end'] - r['start'] >= a.last * 1e9]
        if a.full and not long_enough:
            sys.exit(f'window: no 2-stream run >= {a.last:g} s in this file (--full)')
        pick = long_enough[-1] if long_enough else max(truns, key=lambda r: r['end'] - r['start'])
        last = pick['end']
        print(f"2-stream run {(pick['end'] - pick['start']) / 1e9:.1f} s, {pick['n_total']} steps "
              f"({pick['n_qual'] / pick['n_total']:.1%} qualifying); {len(truns)} runs in the file, "
              f"{len(long_enough)} >= {a.last:g} s" + ('' if long_enough else f' -> the longest, shorter than --last {a.last:g} s'),
              file=sys.stderr)
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


SYS_PAD = 5e9


def whole_disk(src):
    """`/dev/nvme0n1p3` -> `nvme0n1` (via sysfs; None when src is no block device)."""
    name = os.path.basename(src)
    p = os.path.realpath(f'/sys/class/block/{name}')
    if not os.path.isdir(p):
        return None
    return os.path.basename(os.path.dirname(p)) if os.path.exists(os.path.join(p, 'partition')) else os.path.basename(p)


def disk_of_path(path):
    """The whole disk the file system holding `path` is on (from
    /proc/self/mountinfo: the longest mount point containing it), or None."""
    try:
        path = os.path.realpath(path)
        best = None
        with open('/proc/self/mountinfo') as f:
            for line in f:
                head, _, tail = line.partition(' - ')
                h, t = head.split(), tail.split()
                mp = h[4].replace('\\040', ' ')
                if (path == mp or path.startswith(mp.rstrip('/') + '/')) and (best is None or len(mp) > len(best[0])):
                    best = (mp, t[1] if len(t) > 1 else '')
        return whole_disk(best[1]) if best and best[1].startswith('/dev/') else None
    except (OSError, IndexError, ValueError):
        return None


def route_devices(header):
    """{'primary': disk, 'mirror': disk} of a box-2 header (the daemon's
    `--model` dir and `V41_EXPERT_MIRROR_DIR`), run ON that box; {} if unknown."""
    ex = header.get('extras', {})
    argv = ex.get('argv', [])
    model = argv[argv.index('--model') + 1] if '--model' in argv[:-1] else None
    mirror = ex.get('env', {}).get('V41_EXPERT_MIRROR_DIR')
    out = {}
    if model and (d := disk_of_path(model)):
        out['primary'] = d
    if mirror and (d := disk_of_path(mirror)):
        out['mirror'] = d
    return out if len(out) == 2 else {}


def disk_model(dev):
    try:
        with open(f'/sys/block/{dev}/device/model') as f:
            return f.read().strip()
    except OSError:
        return None


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
        # `sys` samples also from SYS_PAD before the window: the first in-window
        # sample's delta needs its predecessor (the window's 2 s drift margin
        # covers a 100 ms sampler; this covers a slow one).
        body = [rawb for t, name, _, _, rawb in records(p, a.t_from - SYS_PAD, a.t_to, raw=True, stop=a.t_to + STOP_MARGIN, strings=strings)
                if t >= a.t_from or name == 'sys']
        if not body:
            continue
        if strings:
            header['strings'] = [strings.get(k, '') for k in range(max(strings) + 1)]
        # Box-side facts the converter cannot learn on the hub: which drive the
        # expert dir (primary) and the mirror dir sit on, and the drives' models.
        extras = header.setdefault('extras', {})
        rd = route_devices(header)
        if rd:
            extras['route_devices'] = rd
        dm = {d: m for d in header.get('sys_devices', []) if (m := disk_model(d))}
        if dm:
            extras['disk_models'] = dm
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
STEP_DEV_MIN = ('t_start',)  # the step's first device stage start
STEP_DEV_IDENT = ('step', 'device')


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
            d[k] = max(d.get(k, v), v) if k in STEP_DEV_MAX else min(d.get(k, v), v) if k in STEP_DEV_MIN else d.get(k, 0.0) + v
    for s in steps:
        if s.get('step', NAN) != s.get('step', NAN):
            continue
        for k, v in acc.get((s.get('_pid'), int(s['step'])), {}).items():
            if k in STEP_DEV_SUM or k in STEP_DEV_MAX or k in STEP_DEV_MIN:
                s['dev_' + k] = v
            elif s.get(k, NAN) != s.get(k, NAN):
                s[k] = v


PID_HUB, PID_B2, PID_DEV, PID_D1, PID_D2, PID_TB = 1, 2, 3, 4, 5, 6
PAGER_RATE_M = 'pager misses/s (avg since prev profiled step, incl. prefill)'
PAGER_RATE_R = 'pager read ms per s (avg since prev profiled step, incl. prefill)'
MIN_LEG_NS = 2000.0     # a wire leg is drawn at least this wide (see the module doc)
LINK_BUCKET_NS = 10e6   # derived link MB/s buckets
TB_OUT = '-> box 2 (requests)'
TB_IN = '<- box 1 (replies)'


def parse_models(spec):
    """{disk: model} from a JSON object, `NAME=MODEL` items (`;`, newline or, on
    one line, `,` separated) or `lsblk -d -n -o NAME,MODEL` lines; `spec` is the
    text or the path of a file holding it."""
    if not spec:
        return {}
    text = open(spec).read() if os.path.isfile(spec) else spec
    text = text.strip()
    if text.startswith('{'):
        return {k: str(v).strip() for k, v in json.loads(text).items()}
    items = text.split(',') if '\n' not in text and ';' not in text and '=' in text else re.split(r'[\n;]+', text)
    out = {}
    for it in items:
        it = it.strip()
        if not it:
            continue
        name, model = it.split('=', 1) if '=' in it else (it.split(None, 1) + [''])[:2]
        if model.strip():
            out[name.strip()] = model.strip()
    return out


def disk_label(dev, models):
    m = models.get(dev)
    return f'{dev} ({m})' if m else dev


def tb_legs(r, to_hub):
    """(t1, t2, t3, t4) of a `hub_req` on the HUB clock: t1 the hub writer's
    stamp before write(), t2 box 2's stamp when the whole frame was in, t3 box
    2's stamp just before the reply's write(), t4 the hub reader's stamp on
    receipt (t2 / t3 moved by the clock offset); None when a stamp is missing."""
    t1, t4 = r.get('t1', NAN), r.get('t4', NAN)
    t2, t3 = to_hub(r.get('t2_b2', NAN)), to_hub(r.get('t3_b2', NAN))
    if any(math.isnan(x) for x in (t1, t2, t3, t4)):
        return None
    return t1, t2, t3, t4


def req_key(r):
    s, l = r.get('seq', NAN), r.get('layer', NAN)
    return None if math.isnan(s) or math.isnan(l) else (int(s), int(l))


STEP_GAP_MAX_NS = 1e9   # a longer gap between decode steps is idle, not draft + sched


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
    skipped = []  # what the input lacks (older binaries / no records): listed on stderr
    hub_sys, b2_sys = [], []  # one series per file: {'devices', 'net', 'link', 'recs'}
    keep_lo, keep_hi = a.t_from - 5e9, a.t_to + 5e9
    # String ids are PER PROCESS: decode each file's against its own table
    # (a box-2 window can span a daemon restart: two interners).
    for p in hub_files:
        h = header_of(p)[1]
        f_str = dict(enumerate(h.get('strings', [])))
        tier_b = h.get('tier') == 'B'
        # A Tier B dump is in batch order: never stop early in one.
        stop = math.inf if tier_b else a.t_to + 300e9 + STOP_MARGIN
        ser = {'devices': h.get('sys_devices', []), 'net': h.get('sys_net'), 'link': h.get('sys_link'), 'recs': []}
        for t, name, fields, vals in records(p, a.t_from - 300e9, a.t_to + 300e9, stop=stop, strings=f_str):
            r = dict(zip(fields, vals))
            if name == 'hub_req':
                offs.add(r)
            if not keep_lo <= t <= keep_hi:
                continue
            if tier_b and name in ('sys', 'b1_read'):
                continue  # Tier A's records (a dump repeats some kinds)
            for k in STR_FIELDS.get(name, ()):
                r[k + '_s'] = f_str.get(int(r[k]), '?') if r[k] == r[k] else '?'
            r['_pid'] = h.get('pid')
            (ser['recs'] if name == 'sys' else hub[name]).append(r)
        if ser['recs']:
            hub_sys.append(ser)
        if tier_b:
            hub['_tier_b_files'].append(p)
    merge_step_dev(hub.get('hub_step', []), hub.get('step_dev', []))
    hub_kinds = {k['name']: k['fields'] for k in hub_hdr.get('kinds', [])}
    # b2_hdr: the FIRST Tier A box-2 header (a Tier B dump's header may lack
    # argv / env / route_devices / disk_models); the last header if all are B.
    b2_hdr, b2_hdr_last, b2, b2_tier_b = None, None, defaultdict(list), False
    for p in b2_files:
        h2 = header_of(p)[1]
        b2_hdr_last = h2
        if b2_hdr is None and h2.get('tier') != 'B':
            b2_hdr = h2
        b2_tier_b |= h2.get('tier') == 'B'

        f_str = dict(enumerate(h2.get('strings', [])))
        ser = {'devices': h2.get('sys_devices', []), 'net': h2.get('sys_net'), 'link': h2.get('sys_link'), 'recs': []}
        for _, name, fields, vals in records(p, strings=f_str):
            r = dict(zip(fields, vals))
            for k in STR_FIELDS.get(name, ()):
                r[k + '_s'] = f_str.get(int(r[k]), '?') if r[k] == r[k] else '?'
            (ser['recs'] if name == 'sys' else b2[name]).append(r)
        if ser['recs']:
            b2_sys.append(ser)
    if b2_hdr is None:
        b2_hdr = b2_hdr_last
    tr = Tracks(a.t_from)
    tr.declare(PID_HUB, 'hub (box 1)', 1)
    tr.declare(PID_B2, 'box 2 (expertd), on hub clock', 2)
    tr.declare(PID_TB, 'Thunderbolt link (box 1 <-> box 2)', 3)
    tr.declare(PID_D1, 'box 1 disks + system', 4)
    tr.declare(PID_D2, 'box 2 disks + system', 5)
    tr.declare(PID_DEV, 'hub devices (calibrated to host time)', 6)
    tr.used.update((PID_HUB, PID_B2))
    inside = lambda t: a.t_from <= t <= a.t_to
    window_s = (a.t_to - a.t_from) / 1e9
    models1, models2 = parse_models(a.disk_models_hub), parse_models(a.disk_models_b2)
    if b2_hdr:
        for k, v in b2_hdr.get('extras', {}).get('disk_models', {}).items():
            models2.setdefault(k, v)
    # Box 2's clock on the hub's: needed by every box-2 record and by the wire
    # legs (the NTP stamps t2 / t3 are box 2's).
    hub_reqs = [r for r in hub.get('hub_req', []) if inside(r['t_submit'])]
    has_stamps = any(not math.isnan(r.get('t2_b2', NAN)) for r in hub_reqs)
    off = offs.fn(a.t_from - 300e9, a.t_to + 300e9) if (b2_files or has_stamps) else None
    to_hub = (lambda t: t - off(t - off(a.t_to)) if not math.isnan(t) else NAN) if off else (lambda t: NAN)
    ex = hub_hdr.get('extras', {})
    tr.instant(1, 'knobs', 'knobs (hub)', a.t_from, {**ex.get('env', {}), **{f'knob {k}': v for k, v in ex.get('knobs', {}).items()}})
    if 'spec_streams' not in hub_kinds.get('hub_step', []):
        skipped.append("hub_step.spec_streams: not in this file (2-stream steps are not labelled; the step's rows / live are)")
    prev_prof_end, pager_last_end, pager_first, prev_step_end = None, None, 0, None
    for s in hub.get('hub_step', []):
        if not inside(s['t_start']):
            continue
        rows, live, lanes = int(s['rows']), int(s['live']), s.get('lanes', NAN)
        spec = s.get('spec_streams', NAN)
        # `live` is counted AFTER finished streams leave, so a 2-stream step in
        # which one stream finished says live == 1: trust spec_streams first.
        if spec == spec and spec >= 2:
            kind = f'2-stream DSpark {rows} rows'
        elif live == 1 and rows > 1:
            kind = f'DSpark block K={rows - 1}'
        else:
            kind = f'step {rows} rows'
        lanes_s = '' if math.isnan(lanes) else f' {int(lanes)}L'
        # + the step's device busy time by stage (`d_*` dGPU, `i_*` iGPU; ms
        # summed over the step, V41_MS_PROFILE): what, not when.
        stages = {k: s[k] for k in s if k[:2] in ('d_', 'i_') and not math.isnan(s[k]) and s[k] > 0}
        # b1_misses / b1_read_ms are NOT per step: the hub sets them on profiled
        # steps only, as the delta of the pager's lifetime counters since the
        # previous profiled step (prefill misses included). Named so in the args.
        args = fin(s, 'step', 'rows', 'live', 'lanes', 'spec_streams', 'step_ms', 'fwd_ms', 'remote_wait_ms', 'b2_page_ms', 'b2_misses',
                   'rf_chain_waits', 'dgpu_busy_ms', 'igpu_busy_ms', 'pos_max',
                   'dev_pairs', 'dev_dropped', 'dev_skipped', 'dev_lag_ms')
        dm, dr = s.get('b1_misses', NAN), s.get('b1_read_ms', NAN)
        if dm == dm:
            args['pager misses since prev profiled step'] = dm
        if dr == dr:
            args['pager read ms since prev profiled step (incl. prefill)'] = dr
        for c in ('b1_pf_queued', 'b1_pf_admitted', 'b1_pf_dropped_full', 'b1_pf_admit_ms'):
            v = s.get(c, NAN)
            if v == v:
                args[f'{c} since prev profiled step'] = v
        tr.slice(1, 'decode steps', f'{kind}{lanes_s}', s['t_start'], s['t_end'], {**args, **stages})
        # hub_step.t_start is stamped AFTER the DSpark draft (multistream.rs), so
        # the drafter and the scheduler sit in the gap before each step.
        if prev_step_end is not None and 0 < s['t_start'] - prev_step_end < STEP_GAP_MAX_NS:
            tr.slice(1, 'between steps (draft + sched, inferred)', 'draft + sched', prev_step_end, s['t_start'],
                     {'gap_ms': round((s['t_start'] - prev_step_end) / 1e6, 3)})
        prev_step_end = s['t_end']
        # Every step has these two.
        for c in ('step_ms', 'spec_streams'):
            tr.counter(1, c, s['t_start'], s.get(c, NAN))
        # True per-profiled-step values: drawn over the step only (0 at its
        # end), or the counter would hold across the ~3 s of unprofiled steps.
        for c in ('remote_wait_ms', 'b2_page_ms', 'b2_misses'):
            v = s.get(c, NAN)
            if v == v:
                tr.counter(1, c, s['t_start'], v)
                tr.counter(1, c, s['t_end'], 0.0)
        # The pager deltas cover [previous profiled step's end, this step's
        # end]: an average rate over that interval, in the box-1 disk group.
        prof = s.get('profiled', NAN)
        if (prof == prof and prof > 0) or dm == dm or dr == dr or s.get('remote_wait_ms', NAN) == s.get('remote_wait_ms', NAN):
            if dm == dm or dr == dr:
                if prev_prof_end is not None and s['t_end'] > prev_prof_end:
                    iv_s = (s['t_end'] - prev_prof_end) / 1e9
                    pargs = {'interval_s': round(iv_s, 3)}
                    label = []
                    if dm == dm:
                        pargs['misses'] = dm
                        label.append(f'{int(dm)} misses')
                        tr.counter(PID_D1, PAGER_RATE_M, prev_prof_end, dm / iv_s)
                    if dr == dr:
                        pargs['read_ms'] = dr
                        label.append(f'{dr:.0f} ms read')
                        tr.counter(PID_D1, PAGER_RATE_R, prev_prof_end, dr / iv_s)
                    tr.slice(PID_D1, 'pager: since prev profiled step', ', '.join(label), prev_prof_end, s['t_end'], pargs)
                    pager_last_end = s['t_end']
                else:
                    pager_first += 1  # no predecessor in the window: no interval to spread the delta over
            prev_prof_end = s['t_end']
    if pager_last_end is not None:  # do not hold the last rate past its interval
        tr.counter(PID_D1, PAGER_RATE_M, pager_last_end, 0.0)
        tr.counter(PID_D1, PAGER_RATE_R, pager_last_end, 0.0)
    if pager_first:
        skipped.append("pager misses / read ms: the window's first profiled step has no predecessor in it, so its delta is not drawn")
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
    # Flows (hub submit -> wire -> box-2 request -> wire -> hub wait): one per
    # request with wire stamps, thinned to --max-flows (a stride), and only in
    # windows <= 60 s unless --flows on.
    flow_id = {}
    cand = [r for r in hub_reqs if tb_legs(r, to_hub) is not None and req_key(r) is not None]
    if a.flows == 'off' or (a.flows == 'auto' and window_s > 60):
        if cand:
            print(f'flows: none drawn ({"--flows off" if a.flows == "off" else f"window {window_s:.0f} s > 60 s; --flows on to force"}; {len(cand)} requests could carry one)', file=sys.stderr)
    else:
        stride = max(1, -(-len(cand) // a.max_flows))
        for i, r in enumerate(cand[::stride]):
            flow_id[req_key(r)] = i + 1
        if stride > 1:
            print(f'flows: capped at --max-flows {a.max_flows}: every {stride}th of {len(cand)} requests drawn ({len(flow_id)} flows)', file=sys.stderr)
        else:
            print(f'flows: {len(flow_id)} of {len(hub_reqs)} requests', file=sys.stderr)
    tb_stats = defaultdict(int)
    link = {TB_OUT: defaultdict(float), TB_IN: defaultdict(float)}  # track -> bucket -> bytes
    for r in hub_reqs:
        lane = 'AB'[int(r['lane'])] if not math.isnan(r['lane']) and r['lane'] < 2 else '?'
        name = f"L{int(r['layer'])} b={int(r['b'])}" if not math.isnan(r['layer']) else 'req'
        lname = f"L{int(r['layer'])}" if not math.isnan(r['layer']) else 'req'
        args = fin(r, 'step', 'seq', 'rtt_us', 'srv_us', 'page_us', 'compute_us', 'n_miss', 'n_distinct', 'blocked', 'n_surprise')
        fid = flow_id.get(req_key(r))
        tr.slice(1, f'lane {lane}: box-2 request', name, r['t_submit'], r['t4'], args, flow=(fid, 0) if fid else None)
        if r['t_wait_exit'] > r['t_wait_enter']:
            tr.slice(1, f'lane {lane}: hub waits', f'wait {name}', r['t_wait_enter'], r['t_wait_exit'], args, flow=(fid, 4) if fid else None)
        legs = tb_legs(r, to_hub)
        tb_stats['requests'] += 1
        if legs is None:
            tb_stats['no_stamps'] += 1
            continue
        t1, t2, t3, t4 = legs
        base = {**fin(r, 'seq', 'step', 'layer'), 'lane': lane}
        for track, ta, tb, nbytes, node in ((TB_OUT, t1, t2, r.get('bytes_out', NAN), 1), (TB_IN, t3, t4, r.get('bytes_in', NAN), 3)):
            raw = tb - ta
            clamped = not raw >= MIN_LEG_NS
            largs = {**base, 'leg_us': round(raw / 1e3, 1)}
            if not math.isnan(nbytes):
                largs['bytes'] = int(nbytes)
            if clamped:
                largs['clock_clamped'] = 1
                tb_stats['clamped'] += 1
                tb_stats['negative'] += raw < 0
            tb_stats['legs'] += 1
            tr.slice(PID_TB, track, f"{lname}/{lane}", ta, ta + MIN_LEG_NS if clamped else tb, largs,
                     flow=(fid, node) if fid else None)
            if not math.isnan(nbytes):
                link[track][ta if track == TB_OUT else tb] += nbytes
    draw_link_rate(tr, link, a.t_from, a.t_to, skipped)
    if tb_stats['requests']:
        print(f"Thunderbolt wire legs: {tb_stats['legs']} on {tb_stats['requests'] - tb_stats['no_stamps']} of {tb_stats['requests']} requests "
              f"({tb_stats['no_stamps']} without NTP stamps / offset); {tb_stats['clamped']} clamped to {MIN_LEG_NS / 1e3:g} us "
              f"(clock_clamped=1; {tb_stats['negative']} of them negative)", file=sys.stderr)
    else:
        skipped.append('Thunderbolt link legs: no hub_req in the window')
    draw_devices(tr, hub, inside)
    # ---- box 1 disks + system (the hub's `sys` kind) and the pager's SSD reads
    if 'sys' not in hub_kinds:
        skipped.append("box 1 disks + system: no 'sys' kind in the hub file")
    elif not hub_sys:
        skipped.append("box 1 disks + system: no 'sys' records in the window")
    else:
        draw_system(tr, PID_D1, 'box 1', hub_sys, lambda t: t, models1, a.t_from, a.t_to, skipped)
    draw_nic(tr, 'box 1', hub_sys, lambda t: t, a.t_from, a.t_to, skipped)
    if 'b1_read' not in hub_kinds:
        skipped.append("box 1 pager SSD reads: no 'b1_read' kind in the hub file")
    elif not hub.get('b1_read'):
        skipped.append("box 1 pager SSD reads: no 'b1_read' records in the window")
    else:
        draw_b1_reads(tr, hub['b1_read'], hub_sys[0]['devices'] if hub_sys else hub_hdr.get('sys_devices', []), models1, inside)
    if b2:
        knob_b2 = lambda k: to_hub(k['t'])
        for k in b2.get('knob', []):
            t = knob_b2(k)
            nm, val = k['name_s'], k['value_s']
            if inside(t) and (2, k['t'], nm) not in knobs_drawn:
                knobs_drawn.add((2, k['t'], nm))
                tr.instant(2, 'knobs', f'knob {nm}={val}', t, {'source': src_name.get(int(k['source']), '?')})
        b2ex = (b2_hdr or {}).get('extras', {})
        tr.instant(2, 'knobs', 'knobs (box 2)', a.t_from, {**b2ex.get('env', {}), **{f'knob {k}': v for k, v in b2ex.get('knobs', {}).items()}})
        joined = 0
        for r in b2.get('b2_req', []):
            t = to_hub(r['t_frame'])
            if not inside(t):
                continue
            name = f"L{int(r['layer'])} b={int(r['b'])}"
            args = fin(r, 'seq', 'n_miss', 'page_us', 'compute_us', 'server_us', 'merged', 'served_under', 'n_paged', 'depth_on_take')
            fid = flow_id.get((int(r['seq']), int(r['layer'])))
            joined += fid is not None
            tr.slice(2, 'requests (frame -> ready)', name, t, to_hub(r['t_ready']), args, flow=(fid, 2) if fid else None)
            tr.slice(2, 'run_path (paging + kernels)', name, to_hub(r['t_run_start']), to_hub(r['t_run_end']), args)
        if flow_id:
            print(f'flows: {joined} of {len(flow_id)} also pass through a box-2 b2_req slice (joined on seq + layer)', file=sys.stderr)
        for e in b2.get('b2_ensure', []):
            t = to_hub(e['t_start'])
            if inside(t) and e['n_miss'] > 0:
                tr.slice(2, 'ensure (paging)', f"L{int(e['layer'])} miss={int(e['n_miss'])}", t, to_hub(e['t_end']),
                         fin(e, 'seq', 'n_want', 'n_hits', 'n_miss', 'admit_wait_ns', 'victim_scan_ns', 'k_par'))
        cls = {0: 'SSD demand reads', 1: 'SSD certain reads', 2: 'SSD certain reads', 3: 'SSD speculative reads',
               4: 'SSD likely (predicted-miss) reads'}
        route = {0: 'split', 1: 'mirror', 2: 'primary'}
        reads = []
        for r in b2.get('b2_read', []):
            t = to_hub(r['t_read_start'])
            if not inside(t):
                continue
            args = fin(r, 'seq', 'layer', 'expert', 'wanted', 'blocked_on', 'yield_ns', 'pause_ns')
            if not math.isnan(r.get('route', NAN)):
                args['route'] = route.get(int(r['route']), '?')
            t_end = to_hub(r['t_read_end'])
            tr.slice(2, cls.get(int(r['src']), 'SSD reads'), f"L{int(r['layer'])} E{int(r['expert'])}", t, t_end, args)
            reads.append((t, t_end, r, args))
        for w in b2.get('b2_write', []):
            t = to_hub(w['t3'])
            if inside(t):
                tr.slice(2, 'reply writes', f"seq {int(w['seq'])}", t, to_hub(w['t_written']), fin(w, 'bytes'))
        # Box-2 device intervals (`V41_B2_EVTRACE_DEV`, its Tier B dumps):
        # each request's GPU span, `unit` = its seq; one track per stream (a
        # parked request's `compute` and the requests served inside it on the
        # parking executor's `park.compute`).
        if b2_tier_b and not any(inside(to_hub(d['t_start'])) for d in b2.get('dev', [])):
            print('box-2 device intervals: none in the window (a box-2 Tier B dump was given: older than its ring, or V41_B2_EVTRACE_DEV off)', file=sys.stderr)
        for d in b2.get('dev', []):
            t = to_hub(d['t_start'])
            if not inside(t):
                continue
            args = fin(d, 'q_us')
            if d['unit'] == d['unit']:
                args['seq'] = int(d['unit'])
            if d['t_host'] == d['t_host']:
                args['queued_us'] = round((d['t_start'] - d['t_host']) / 1e3, 1)
            tr.slice(2, f"device {d['device_s']} {d['stream_s']}", d['name_s'], t, to_hub(d['t_end']), args)
        # ---- box 2 disks + system
        b2_devs = (b2_hdr or {}).get('sys_devices', [])
        if not b2_sys:
            skipped.append("box 2 disks + system: no 'sys' records in the box-2 files")
        elif off is None:
            skipped.append('box 2 disks + system: no clock offset')
        else:
            draw_system(tr, PID_D2, 'box 2', b2_sys, to_hub, models2, a.t_from, a.t_to, skipped)
        draw_nic(tr, 'box 2', b2_sys, to_hub, a.t_from, a.t_to, skipped)
        draw_b2_drive_reads(tr, reads, b2_devs, b2ex.get('route_devices'), models2, b2ex.get('env', {}).get('V41_EXPERT_MIRROR_FRAC'), skipped)
    else:
        if b2_files:
            skipped.append('box 2: no records in the box-2 files')
        else:
            skipped.append('box 2 tracks (requests, paging, SSD reads, disks + system, NIC): no box-2 file given')
    # The NIC link state, once, on the wire group.
    for label, series in (('box 1', hub_sys), ('box 2', b2_sys)):
        for s in series[-1:]:
            if s.get('link') or s.get('net'):
                tr.instant(PID_TB, 'link', f"{label} {s.get('net') or '?'}", a.t_from, {k: v for k, v in (s.get('link') or {}).items()})
    tr.finish()
    for m in skipped:
        print(f'skipped: {m}', file=sys.stderr)
    out = {'traceEvents': tr.ev, 'displayTimeUnit': 'ms',
           'otherData': {'t0_hub_mono_raw_ns': a.t_from, 'window_s': window_s,
                         'hub_realtime_at_open': hub_hdr.get('t_realtime_at_open'), 'hub_mono_raw_at_open': hub_hdr.get('t_mono_raw_at_open')}}
    opener = gzip.open if a.o.endswith('.gz') else open
    with opener(a.o, 'wt') as f:
        json.dump(out, f)
    print(a.o, len(tr.ev), 'events')


def draw_link_rate(tr, link, t_from, t_to, skipped):
    """Derived link MB/s per direction (the requests' `bytes_out` -> box 2 at
    the hub's send stamp, `bytes_in` <- box 2 at the hub's receive stamp), in
    10 ms buckets (wider for a window > 120 s: <= 12000 buckets), zeros included."""
    width = max(LINK_BUCKET_NS, (t_to - t_from) / 12000)
    if not any(link.values()):
        skipped.append('Thunderbolt link MB/s (derived from hub_req bytes): no bytes in the window')
        return
    n = int((t_to - t_from) // width) + 1
    for track, label in ((TB_OUT, '-> box 2 MB/s (derived from requests)'), (TB_IN, '<- box 1 MB/s (derived from replies)')):
        acc = defaultdict(float)
        for t, b in link[track].items():
            acc[int((t - t_from) // width)] += b
        for k in range(n):
            tr.counter(PID_TB, f'link {label}', t_from + k * width, acc.get(k, 0.0) / (width / 1e9) / 1e6)


def draw_system(tr, pid, box, series, to_t, models, t_from, t_to, skipped):
    """Per-disk and system counters from consecutive `sys` samples: the value
    of a sample pair is drawn from the earlier sample's time (a counter holds
    until the next); point values (in-flight, procs, memory) at the later
    sample's own time. Counters that went backwards (a reset) or a gap >
    max(5 x the sampling period, 2 s) skip that pair; util % is clamped to 100."""
    for s in series:
        rs = sorted(s['recs'], key=lambda r: r['t'])
        if len(rs) < 2:
            skipped.append(f"{box} disks + system: fewer than 2 'sys' samples in the window")
            continue
        keys = set(rs[0])
        devs = s['devices']
        dts = sorted(q['t'] - p['t'] for p, q in zip(rs, rs[1:]))
        nominal, gaps = dts[len(dts) // 2], 0
        no_disk = [k for k in range(len(devs), 3) if any(r.get(f'd{k}_reads', NAN) == r.get(f'd{k}_reads', NAN) for r in rs)]
        for k in no_disk:
            skipped.append(f'{box} disk d{k}: not named in the header sys_devices')
        for p, q in zip(rs, rs[1:]):
            dt = q['t'] - p['t']
            if dt <= 0:
                continue
            if dt > max(5 * nominal, 2e9):
                gaps += 1
                continue
            ta, tb = to_t(p['t']), to_t(q['t'])
            if math.isnan(ta) or tb < t_from or ta > t_to:
                continue
            ts, sec = max(ta, t_from), dt / 1e9
            tq = tb if t_from <= tb <= t_to else None   # point values belong to q's own time

            def d(key):
                if key not in p or key not in q:
                    return NAN
                v = q[key] - p[key]
                return v if v == v and v >= 0 else NAN

            for k, dev in enumerate(devs[:3]):
                if f'd{k}_reads' not in keys:
                    continue
                lab = disk_label(dev, models)
                reads, writes = d(f'd{k}_reads'), d(f'd{k}_writes')
                tr.counter(pid, f'{lab} read MB/s', ts, d(f'd{k}_rsect') * 512 / sec / 1e6)
                tr.counter(pid, f'{lab} write MB/s', ts, d(f'd{k}_wsect') * 512 / sec / 1e6)
                tr.counter(pid, f'{lab} read IOPS', ts, reads / sec)
                tr.counter(pid, f'{lab} write IOPS', ts, writes / sec)
                tr.counter(pid, f'{lab} avg read latency ms', ts, d(f'd{k}_rms') / reads if reads > 0 else (0.0 if reads == 0 else NAN))
                tr.counter(pid, f'{lab} avg write latency ms', ts, d(f'd{k}_wms') / writes if writes > 0 else (0.0 if writes == 0 else NAN))
                if tq is not None:
                    tr.counter(pid, f'{lab} in-flight', tq, q.get(f'd{k}_inflight', NAN))
                tr.counter(pid, f'{lab} avg queue depth', ts, d(f'd{k}_wtd_ms') / (sec * 1e3))
                # io_ticks accounting overshoots on 100 ms samples (seen 135%): clamp
                tr.counter(pid, f'{lab} util %', ts, min(d(f'd{k}_io_ms') / (sec * 1e3) * 100, 100.0))
            for key, label in (('psi_io_some', 'PSI io some %'), ('psi_io_full', 'PSI io full %'), ('psi_cpu_some', 'PSI cpu some %'),
                               ('psi_mem_some', 'PSI mem some %'), ('psi_mem_full', 'PSI mem full %')):
                tr.counter(pid, label, ts, d(key) / (dt / 1e3) * 100)  # cumulative us / sample interval us
            if tq is not None:
                tr.counter(pid, 'procs_running', tq, q.get('procs_running', NAN))
                tr.counter(pid, 'procs_blocked', tq, q.get('procs_blocked', NAN))
                tr.counter(pid, 'dirty + writeback MB', tq, (q.get('dirty_kb', NAN) + q.get('writeback_kb', NAN)) / 1024)
                tr.counter(pid, 'mem available GB', tq, q.get('mem_avail_kb', NAN) / 1048576)
            tr.counter(pid, 'this process: read MB/s (self io)', ts, d('self_read_bytes') / sec / 1e6)
            tr.counter(pid, 'this process: write MB/s (self io)', ts, d('self_write_bytes') / sec / 1e6)
        if gaps:
            skipped.append(f'{box} sys: {gaps} sample gaps > max(5 x the {nominal / 1e6:.0f} ms period, 2 s) left blank')


def draw_nic(tr, box, series, to_t, t_from, t_to, skipped):
    """The box's thunderbolt NIC counters (`tb_*` of the `sys` kind), deltas per
    sample pair, on the wire group."""
    if not series:
        skipped.append(f"{box} NIC (tb_* counters): no 'sys' records")
        return
    drawn = False
    for s in series:
        rs = sorted(s['recs'], key=lambda r: r['t'])
        if 'tb_rx_bytes' not in rs[0]:
            skipped.append(f"{box} NIC (tb_* counters): the 'sys' kind of this file has no tb_* fields (older binary)")
            continue
        if all(math.isnan(r['tb_rx_bytes']) for r in rs):
            skipped.append(f'{box} NIC (tb_* counters): no thunderbolt interface (sys_net null)')
            continue
        net = s.get('net') or 'thunderbolt'
        for p, q in zip(rs, rs[1:]):
            dt = q['t'] - p['t']
            ta, tb = to_t(p['t']), to_t(q['t'])
            if dt <= 0 or dt > 2e9 or math.isnan(ta) or tb < t_from or ta > t_to:
                continue
            ts, sec = max(ta, t_from), dt / 1e9

            def d(key):
                v = q[key] - p[key]
                return v if v == v and v >= 0 else NAN

            tr.counter(PID_TB, f'{box} {net} rx MB/s', ts, d('tb_rx_bytes') / sec / 1e6)
            tr.counter(PID_TB, f'{box} {net} tx MB/s', ts, d('tb_tx_bytes') / sec / 1e6)
            tr.counter(PID_TB, f'{box} {net} rx packets/s', ts, d('tb_rx_packets') / sec)
            tr.counter(PID_TB, f'{box} {net} tx packets/s', ts, d('tb_tx_packets') / sec)
            tr.counter(PID_TB, f'{box} {net} rx_missed (per sample)', ts, d('tb_rx_missed'))
            tr.counter(PID_TB, f'{box} {net} rx_dropped (per sample)', ts, d('tb_rx_dropped'))
            drawn = True
    return drawn


def draw_b1_reads(tr, reads, devs, models, inside):
    """The box-1 pager's SSD reads (`b1_read`) as slices per drive and class."""
    cls = {0: 'demand', 1: 'prefetch'}
    for r in reads:
        if not inside(r['t_read_start']):
            continue
        dev = r.get('dev', NAN)
        lab = disk_label(devs[int(dev)], models) if dev == dev and 0 <= int(dev) < len(devs) else 'pager (drive unknown)'
        args = fin(r, 'layer', 'expert', 'bytes')
        tr.slice(PID_D1, f"{lab} {cls.get(int(r['src']), 'src ' + str(int(r['src'])))} reads", f"L{int(r['layer'])} E{int(r['expert'])}",
                 r['t_read_start'], r['t_read_end'], args)


def draw_b2_drive_reads(tr, reads, devs, route_devs, models, mirror_frac, skipped):
    """Box 2's SSD reads on per-drive tracks. `route` (0 split over both
    drives, 1 mirror only, 2 primary only) -> the drive: `route_devices`
    (the cut recorded which disks the expert dir / mirror dir sit on) else the
    first / second `sys_devices` (stderr says so)."""
    if not reads:
        return
    if route_devs and 'primary' in route_devs and 'mirror' in route_devs:
        drive = dict(route_devs)
    elif len(devs) >= 2:
        drive = {'primary': devs[0], 'mirror': devs[1]}
        skipped.append(f"box 2 per-drive SSD reads: the cut has no route_devices; ASSUMED primary={devs[0]}, mirror={devs[1]} (sys_devices order)")
    else:
        skipped.append('box 2 per-drive SSD reads: no route_devices and fewer than 2 sys_devices')
        return
    cls = {0: 'demand', 1: 'certain', 2: 'certain', 3: 'speculative', 4: 'likely (predicted-miss)'}
    n_noroute = 0
    for t, t_end, r, args in reads:
        route = r.get('route', NAN)
        if math.isnan(route):
            n_noroute += 1
            continue
        route = int(route)
        roles = ('mirror', 'primary') if route == 0 else ('mirror',) if route == 1 else ('primary',) if route == 2 else ()
        a2 = {**args, 'class_src': int(r['src'])}
        if route == 0 and mirror_frac:
            a2['mirror_frac_env_seed'] = mirror_frac   # startup env; the live knob can differ
        if route == 0:
            # Split over both drives: one slice on its own track (drawing it on
            # both would double each drive's apparent busy time); the drive
            # MB/s counters show the real per-drive volume.
            tr.slice(PID_D2, f"split reads (both drives) {cls.get(int(r['src']), 'other')}",
                     f"L{int(r['layer'])} E{int(r['expert'])}", t, t_end, a2)
            continue
        for role in roles:
            tr.slice(PID_D2, f"{disk_label(drive[role], models)} [{role}] {cls.get(int(r['src']), 'other')} reads",
                     f"L{int(r['layer'])} E{int(r['expert'])}", t, t_end, a2)
    if n_noroute:
        skipped.append(f'box 2 per-drive SSD reads: {n_noroute} b2_read records have no route (older expertd); not drawn per drive')


def draw_devices(tr, hub, inside):
    """Tier B device intervals (pid 3), calibration counters, and the
    causality check of the intervals against the host."""
    devs = hub.get('dev', [])
    if not devs:
        if hub.get('_tier_b_files'):
            print('device intervals: none in the window (a hub Tier B dump was given: the window may be older than the ring holds)', file=sys.stderr)
        return
    tr.used.add(PID_DEV)
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
        self.procs, self.used, self.flows = {}, set(), defaultdict(list)

    def declare(self, pid, name, sort):
        """A process group (name + sort order), emitted only if something is drawn in it."""
        self.procs[pid] = (name, sort)

    def tid(self, pid, name):
        key = (pid, name)
        if key not in self.tids:
            self.tids[key] = len(self.tids) + 1
            self.ev.append({'ph': 'M', 'name': 'thread_name', 'pid': pid, 'tid': self.tids[key], 'args': {'name': name}})
            self.ev.append({'ph': 'M', 'name': 'thread_sort_index', 'pid': pid, 'tid': self.tids[key], 'args': {'sort_index': self.tids[key]}})
        return self.tids[key]

    def us(self, t):
        return (t - self.t0) / 1e3

    def slice(self, pid, track, name, a, b, args=None, flow=None):
        """`flow` = (flow id, step): the slice is node `step` of that flow
        (Chrome flow events, drawn at `finish` once the slice has its track)."""
        if math.isnan(a) or math.isnan(b) or b < a:
            return
        self.pending[(pid, track)].append((a, b, name, args, flow))

    def finish_flows(self):
        """One s / t... / f chain per flow id, steps in order, each event inside
        its slice (the start of the first, the end of the last, the middle of the
        others) and never before the previous event."""
        n_bad = n_skip = 0
        for fid, nodes in self.flows.items():
            nodes.sort(key=lambda x: x[0])
            # A node that cannot sit inside its slice at or after the previous
            # node (box-2 clock uncertainty) is skipped: perfetto drops or
            # splits a chain whose events go backwards.
            # The first / last event hug their slice's edge (<= 1 us in): the
            # first slice (hub submit -> reply) spans ms while the next (the
            # ~50 us wire leg) starts right after it, so a fractional offset
            # would land past the wire leg's middle.
            prev, kept = -math.inf, []
            for i, (_, a, b, pid, tid) in enumerate(nodes):
                edge = min((b - a) * 0.01, 1e3)
                t0 = a + edge if i == 0 else b - edge if i == len(nodes) - 1 else a + (b - a) * 0.5
                ts = max(t0, prev)
                if ts > b:
                    n_skip += 1
                    continue
                prev = ts
                kept.append((ts, pid, tid))
            if len(kept) < 2:
                n_bad += 1
                continue
            for i, (ts, pid, tid) in enumerate(kept):
                ev = {'ph': 's' if i == 0 else 'f' if i == len(kept) - 1 else 't', 'id': fid, 'cat': 'req', 'name': 'req',
                      'pid': pid, 'tid': tid, 'ts': self.us(ts)}
                if i:
                    ev['bp'] = 'e'
                self.ev.append(ev)
        self.flows.clear()
        if n_bad:
            print(f'flows: {n_bad} had fewer than 2 drawable slices; not drawn', file=sys.stderr)
        if n_skip:
            print(f'flows: {n_skip} nodes skipped (would go backwards in time; clock uncertainty)', file=sys.stderr)

    def finish(self):
        self.used.update(pid for pid, _ in self.pending)
        for pid in sorted(self.used):
            if pid in self.procs:
                name, sort = self.procs[pid]
                self.ev.append({'ph': 'M', 'name': 'process_name', 'pid': pid, 'args': {'name': name}})
                self.ev.append({'ph': 'M', 'name': 'process_sort_index', 'pid': pid, 'args': {'sort_index': sort}})
        for (pid, track), sl in self.pending.items():
            sl.sort(key=lambda x: (x[0], x[1]))
            ends = []
            for a, b, name, args, flow in sl:
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
                if flow and flow[0]:
                    self.flows[flow[0]].append((flow[1], a, b, pid, ev['tid']))
        self.pending.clear()
        self.finish_flows()

    def instant(self, pid, track, name, t, args=None):
        if math.isnan(t):
            return
        self.used.add(pid)
        self.ev.append({'ph': 'i', 's': 't', 'pid': pid, 'tid': self.tid(pid, track), 'name': name, 'ts': self.us(t), 'args': args or {}})

    def counter(self, pid, name, t, value):
        if math.isnan(t) or value is None or math.isnan(value):
            return
        self.used.add(pid)
        self.ev.append({'ph': 'C','pid': pid, 'name': name, 'ts': self.us(t), 'args': {'v': value}})


def fin(r, *keys):
    return {k: r[k] for k in keys if k in r and not math.isnan(r[k])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('window')
    w.add_argument('hub')
    w.add_argument('--last', type=float, default=30.0)
    w.add_argument('--lone', action='store_true', help='a window inside a run of lone-stream (DSpark) steps')
    w.add_argument('--two', action='store_true', help='a window inside a run of 2-stream DSpark steps')
    w.add_argument('--full', action='store_true', help='with --lone / --two: fail unless a run lasts >= --last')
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
    t.add_argument('--disk-models-hub', help="box 1's disk models: JSON object, NAME=MODEL list or `lsblk -d -n -o NAME,MODEL` text, or a file holding it")
    t.add_argument('--disk-models-b2', help="box 2's disk models (same forms; default: the box-2 cut's own)")
    t.add_argument('--flows', choices=['auto', 'on', 'off'], default='auto', help='request flow arrows (auto: windows <= 60 s)')
    t.add_argument('--max-flows', type=int, default=10000, help='thin the flows to about this many (a stride over the requests)')
    a = ap.parse_args()
    {'window': cmd_window, 'cut': cmd_cut, 'trace': cmd_trace}[a.cmd](a)


if __name__ == '__main__':
    main()
