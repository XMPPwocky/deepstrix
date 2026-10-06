#!/usr/bin/env python3
"""Stream a box-2 EVT1 trace from stdin (ssh cat); keep b2_req / b2_write / b2_read records whose seq is in the
hub sample (h000/h001.json from extract40.py), plus a reservoir of decode-path b2_req records for the general
service distribution. Usage: ssh ... cat FILE | b2x.py OUT.json h000.json h001.json"""
import sys, json, struct, random
from collections import defaultdict
rng = random.Random(99)
out = sys.argv[1]
seqs = {}
for fn in sys.argv[2:]:
    D = json.load(open(fn))
    ri = {n: i for i, n in enumerate(D['req_fields'])}
    for cell, steps in D['cells'].items():
        for s in steps:
            for r in s['reqs']:
                seqs[int(r[ri['seq']])] = (int(r[ri['layer']]), int(r[ri['b']]))
    del D
f = sys.stdin.buffer
d = f.read(8)
hlen = struct.unpack_from('<I', d, 4)[0]
header = json.loads(f.read(hlen))
kinds = {k['id']: (k['name'], k['fields']) for k in header['kinds']}
byname = {v[0]: (k, v[1]) for k, v in kinds.items()}
REQ, rf = byname['b2_req']; WR, wf = byname['b2_write']; RD, rdf = byname['b2_read']; ENS, ef = byname['b2_ensure']
ri = {n: i for i, n in enumerate(rf)}; wi = {n: i for i, n in enumerate(wf)}; rdi = {n: i for i, n in enumerate(rdf)}; ei = {n: i for i, n in enumerate(ef)}
keep_req = []; keep_wr = []; keep_rd = []; keep_ens = []
res = []; n_res = 0; CAP = 40000
counts = defaultdict(int)
buf = b''; CH = 32 << 20
t_first = None; t_last = None
while True:
    chunk = f.read(CH)
    if not chunk: break
    buf += chunk
    off = 0; n = len(buf)
    while off + 4 <= n:
        kid, cnt = struct.unpack_from('<HH', buf, off)
        if off + 4 + 8 * cnt > n: break
        if kid == REQ:
            vals = struct.unpack_from(f'<{cnt}d', buf, off + 4)
            counts['req'] += 1
            seq = int(vals[ri['seq']])
            if t_first is None: t_first = vals[ri['t_frame']]
            t_last = vals[ri['t_frame']]
            if seq in seqs and seqs[seq] == (int(vals[ri['layer']]), int(vals[ri['b']])):
                keep_req.append(list(vals)); counts['req_kept'] += 1
            if vals[ri['path_decode']] == 1.0:
                n_res += 1
                if len(res) < CAP: res.append(list(vals))
                else:
                    j = rng.randrange(n_res)
                    if j < CAP: res[j] = list(vals)
        elif kid == WR:
            vals = struct.unpack_from(f'<{cnt}d', buf, off + 4)
            counts['wr'] += 1
            if int(vals[wi['seq']]) in seqs:
                keep_wr.append(list(vals))
        elif kid == RD:
            vals = struct.unpack_from(f'<{cnt}d', buf, off + 4)
            counts['rd'] += 1
            if int(vals[rdi['seq']]) in seqs:
                keep_rd.append(list(vals))
        elif kid == ENS:
            vals = struct.unpack_from(f'<{cnt}d', buf, off + 4)
            counts['ens'] += 1
            if int(vals[ei['seq']]) in seqs:
                keep_ens.append(list(vals))
        off += 4 + 8 * cnt
    buf = buf[off:]
json.dump({'req_fields': rf, 'write_fields': wf, 'read_fields': rdf, 'ensure_fields': ef,
           'req': keep_req, 'write': keep_wr, 'read': keep_rd, 'ensure': keep_ens, 'reservoir_decode': res, 'n_decode_total': n_res,
           'counts': dict(counts), 't_first': t_first, 't_last': t_last,
           'hdr': {k: v for k, v in header.items() if k not in ('kinds', 'extras', 'knobs_at_open')}}, open(out, 'w'))
print(out, dict(counts), 'decode reqs', n_res, 'kept', len(keep_req), len(keep_wr), len(keep_rd), len(keep_ens), file=sys.stderr)
