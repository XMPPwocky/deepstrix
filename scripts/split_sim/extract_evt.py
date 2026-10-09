#!/usr/bin/env python3
"""Stream hub .evt file(s) and write compact TSVs of the kinds the split
simulator needs (hub_step: every field; hub_req: the decode-relevant fields).

  extract_evt.py HUB.evt [HUB-001.evt ...] OUTDIR [--req-max-b 16] [--stem NAME]
                 [--req-from UNIX_S] [--req-to UNIX_S] [--step-to UNIX_S]

Several files (a rotated evtrace of ONE process, in order) go into one pair of
TSVs named after the first file (or --stem). --req-from/--req-to keep only the
hub_req records whose t_submit falls in that unix-time window (the hub_req
stream is ~25 MB per 1k steps); hub_step is always kept whole (the pick-trace
alignment walks every step from the trace's start) up to --step-to.

Never loads a file: records are scanned in 64 MB chunks and kinds other than
hub_step / hub_req are skipped without unpacking.
Output: OUTDIR/<stem>.hub_step.tsv, OUTDIR/<stem>.hub_req.tsv."""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evt2perfetto import header_of  # noqa: E402

REQ_FIELDS = ['t_submit', 't_submit_end', 't1', 't4', 't_wait_enter', 't_wait_exit', 't2_b2', 't3_b2', 'step', 'lane', 'layer', 'b',
              'seq', 'flags', 'partner', 'unmasked', 'n_picks', 'n_distinct', 'n_pred_miss', 'n_pred_incoming',
              'n_pred_pending', 'rtt_us', 'srv_us', 'page_us', 'compute_us', 'n_miss', 'miss_bits', 'blocked', 'step_rows',
              'n_held', 'n_surprise', 'n_paged', 'pinned', 'bytes_out', 'bytes_in', 'n_hints', 'n_pf_words']
CHUNK = 64 << 20


def fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return 'nan'
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f'{v:.6g}'
    return str(v)


def scan(path, want):
    """Yield (kind name, fields, values) of the kinds in `want`, streamed."""
    hlen, h = header_of(path)
    kinds = {k['id']: (k['name'], k['fields']) for k in h['kinds']}
    wid = {kid for kid, (nm, _) in kinds.items() if nm in want}
    hh = struct.Struct('<HH')
    structs = {}
    fd = os.open(path, os.O_RDONLY)
    try:
        pos, buf = 8 + hlen, b''
        while True:
            chunk = os.pread(fd, CHUNK, pos)
            if chunk and hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fd, pos, len(chunk), os.POSIX_FADV_DONTNEED)
            pos += len(chunk)
            data = buf + chunk
            off, nd = 0, len(data)
            while off + 4 <= nd:
                kid, n = hh.unpack_from(data, off)
                if off + 4 + 8 * n > nd:
                    break
                if kid in wid:
                    s = structs.get(n) or structs.setdefault(n, struct.Struct(f'<{n}d'))
                    yield kinds[kid][0], kinds[kid][1], s.unpack_from(data, off + 4)
                off += 4 + 8 * n
            buf = data[off:]
            if not chunk:
                return
    finally:
        os.close(fd)


def main():
    a = sys.argv[1:]
    opts = {}
    pos = []
    i = 0
    while i < len(a):
        if a[i].startswith('--'):
            opts[a[i]] = a[i + 1]
            i += 2
        else:
            pos.append(a[i])
            i += 1
    paths, outdir = pos[:-1], pos[-1]
    req_max_b = int(opts.get('--req-max-b', 16))
    req_from = float(opts.get('--req-from', '-inf'))
    req_to = float(opts.get('--req-to', 'inf'))
    step_to = float(opts.get('--step-to', 'inf'))
    os.makedirs(outdir, exist_ok=True)
    stem = opts.get('--stem') or os.path.basename(paths[0]).rsplit('.', 1)[0]
    fs = open(os.path.join(outdir, stem + '.hub_step.tsv'), 'w')
    fr = open(os.path.join(outdir, stem + '.hub_req.tsv'), 'w')
    fr.write('\t'.join(REQ_FIELDS) + '\n')
    step_fields = None
    ns = nr = 0
    # one mono -> unix mapping for the whole process: the FIRST file's header
    # (the scripts' Clock uses the same file)
    h0 = header_of(paths[0])[1]
    m0, r0 = h0['t_mono_raw_at_open'], h0['t_realtime_at_open']

    def unix(t):
        return (r0 + (t - m0)) / 1e9

    done = False
    for path in paths:
        for name, fields, vals in scan(path, ('hub_step', 'hub_req')):
            if name == 'hub_step':
                d = dict(zip(fields, vals))
                if unix(d['t_start']) > step_to:
                    done = True
                    break
                if step_fields is None:
                    step_fields = list(fields)
                    fs.write('\t'.join(step_fields) + '\n')
                fs.write('\t'.join(fmt(d.get(k, float('nan'))) for k in step_fields) + '\n')
                ns += 1
            else:
                d = dict(zip(fields, vals))
                b = d.get('b', float('nan'))
                if b == b and b > req_max_b:
                    continue
                t = d.get('t_submit', float('nan'))
                if t == t and not (req_from <= unix(t) <= req_to):
                    continue
                fr.write('\t'.join(fmt(d.get(k, float('nan'))) for k in REQ_FIELDS) + '\n')
                nr += 1
        print(f'{os.path.basename(path)}: hub_step {ns}, hub_req {nr} (cumulative)', file=sys.stderr, flush=True)
        if done:
            break
    fs.close()
    fr.close()
    print(f'{stem}: hub_step {ns}, hub_req {nr}', file=sys.stderr)


if __name__ == '__main__':
    main()
