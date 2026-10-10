#!/usr/bin/env python3
"""Stream-filter an EVT1 file by time ranges and kinds, with byte-offset
bisection (+ resync) so only the needed parts of a big file are read.

  evtfilt.py FILE LO:HI[,LO:HI...] KIND[,KIND...] [--hdr]

LO/HI are the file's own CLOCK_MONOTONIC_RAW ns. Output: '#F kind fields...'
lines, then one line per record 'kind v1 v2 ...'. A record is kept when its
time field is inside any range (records with a NaN time field are kept only
while scanning a range).
"""
import json, os, struct, sys

TFIELD = {'sys': 't', 'hub_req': 't_submit', 'hub_step': 't_start', 'hub_phase': 't',
          'b2_req': 't_frame', 'b2_read': 't_read_start', 'b2_ensure': 't_start',
          'b2_write': 't3', 'meta': 't_mono_raw'}


def main():
    path = sys.argv[1]
    ranges = [tuple(float(x) for x in r.split(':')) for r in sys.argv[2].split(',')]
    kinds = sys.argv[3].split(',')
    f = open(path, 'rb')
    size = os.fstat(f.fileno()).st_size
    head = f.read(8)
    if head[:4] != b'EVT1':
        sys.exit('not EVT1')
    hlen = struct.unpack_from('<I', head, 4)[0]
    header = json.loads(f.read(hlen))
    data0 = 8 + hlen
    kd = {k['id']: (k['name'], k['fields']) for k in header['kinds']}
    nf = {kid: len(v[1]) for kid, v in kd.items()}
    tidx = {kid: (v[1].index(TFIELD[v[0]]) if TFIELD.get(v[0]) in v[1] else None) for kid, v in kd.items()}
    if '--hdr' in sys.argv:
        print('#H ' + json.dumps({k: v for k, v in header.items() if k != 'kinds'}))
    for kid, (name, fields) in kd.items():
        if name in kinds:
            print('#F ' + name + ' ' + ' '.join(fields))

    def valid_at(buf, p, need=12):
        q = p
        for _ in range(need):
            if q + 4 > len(buf):
                return False
            kid, n = struct.unpack_from('<HH', buf, q)
            if kid not in nf or nf[kid] != n:
                return False
            q += 4 + 8 * n
        return True

    def sync(off):
        """-> (aligned offset >= off, time of first timed record) or (None, None)."""
        if off <= data0:
            return data0, None
        f.seek(off)
        buf = f.read(1 << 18)
        for p in range(0, min(len(buf), 1 << 16)):
            if valid_at(buf, p):
                q = p
                for _ in range(500):
                    if q + 4 > len(buf):
                        break
                    kid, n = struct.unpack_from('<HH', buf, q)
                    ti = tidx.get(kid)
                    if ti is not None and q + 4 + 8 * n <= len(buf):
                        t = struct.unpack_from('<d', buf, q + 4 + 8 * ti)[0]
                        if t == t and t > 0:
                            return off + p, t
                    q += 4 + 8 * n
                return off + p, None
        return None, None

    want = {kid for kid, (name, _) in kd.items() if name in kinds}
    out = sys.stdout
    cache = {}
    CH = 1 << 22
    for t_lo, t_hi in ranges:
        lo, hi = data0, size
        target = t_lo - 2e9
        for _ in range(48):
            if hi - lo < 1 << 20:
                break
            mid = (lo + hi) // 2
            p, t = sync(mid)
            if p is None or t is None:
                hi = mid
                continue
            if t < target:
                lo = p
            else:
                hi = mid
        start, _ = sync(lo)
        if start is None:
            continue
        f.seek(start)
        buf, pos, past = b'', 0, 0
        while True:
            if len(buf) - pos < 1 << 16:
                more = f.read(CH)
                buf = buf[pos:] + more
                pos = 0
                if not more and len(buf) < 4:
                    break
            if pos + 4 > len(buf):
                break
            kid, n = struct.unpack_from('<HH', buf, pos)
            if kid not in nf or nf[kid] != n:
                break  # torn tail
            if pos + 4 + 8 * n > len(buf):
                more = f.read(CH)
                if not more:
                    break
                buf = buf[pos:] + more
                pos = 0
                continue
            ti = tidx.get(kid)
            t = float('nan')
            if ti is not None:
                t = struct.unpack_from('<d', buf, pos + 4 + 8 * ti)[0]
                if t == t and t > t_hi + 2e9:
                    past += 1
                    if past > 2000:
                        break
            if kid in want and (t != t or t_lo <= t <= t_hi):
                s = cache.get(n)
                if s is None:
                    s = cache[n] = struct.Struct(f'<{n}d')
                out.write(kd[kid][0] + ' ' + ' '.join(repr(v) for v in s.unpack_from(buf, pos + 4)) + '\n')
            pos += 4 + 8 * n


main()
