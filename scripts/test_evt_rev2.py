#!/usr/bin/env python3
"""Self-test of the format_rev 2 readers (scripts only; stdlib): a synthetic
hub file with header strings, a late `str` record and a `knob` record that
uses it; `evtrace.read_file` decodes it, `evt2perfetto cut` keeps the late
string in the cut's header, and `trace` draws the knob change by name.

  python3 scripts/test_evt_rev2.py"""
import gzip
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evtrace  # noqa: E402

KINDS = [
    {'id': 0, 'name': 'meta', 'fields': ['t_mono_raw', 't_realtime', 'written', 'dropped', 'files', 'queue_cap']},
    {'id': 3, 'name': 'str', 'fields': ['id', 'len']},
    {'id': 5, 'name': 'knob', 'fields': ['t', 'name', 'value', 'source']},
    {'id': 11, 'name': 'hub_step', 'fields': ['t_start', 't_end', 'step', 'rows', 'live', 'lanes', 'step_ms']},
]


def rec(kid, vals):
    return struct.pack('<HH', kid, len(vals)) + struct.pack(f'<{len(vals)}d', *vals)


def str_rec(sid, s):
    b = s.encode()
    slots = [int.from_bytes(b[i:i + 6].ljust(6, b'\0'), 'little') for i in range(0, len(b), 6)]
    return rec(3, [sid, len(b)] + slots)


def write(path, header, body):
    hb = json.dumps(header).encode()
    with open(path, 'wb') as f:
        f.write(b'EVT1' + struct.pack('<I', len(hb)) + hb + body)


def main():
    d = tempfile.mkdtemp(prefix='evt-rev2-')
    hub = os.path.join(d, 'hub-test.evt')
    # This box's clock pair: `cut` keeps only files of the current boot.
    import time
    t0 = float(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
    header = {'format': 'EVT1', 'format_rev': 2, 'role': 'hub', 't_mono_raw_at_open': t0, 't_realtime_at_open': time.time_ns(),
              'kinds': KINDS, 'strings': ['<overflow>', 'V41_MS_SPEC_LANES'], 'extras': {}}
    body = b''
    for i in range(10):
        ts = t0 + i * 100e6
        body += rec(11, [ts, ts + 90e6, i, 1, 1, 1, 90.0])
    body += str_rec(2, '0')                       # interned after the file opened
    body += rec(5, [t0 + 450e6, 1, 2, 3])         # V41_MS_SPEC_LANES=0 from the knob file
    write(hub, header, body)

    h, recs = evtrace.read_file(hub)
    k = recs['knob'][0]
    assert (k['name_s'], k['value_s']) == ('V41_MS_SPEC_LANES', '0'), k
    assert 'str' not in recs, 'str records are the table, not data'

    cut = os.path.join(d, 'cut')
    py = sys.executable
    out = subprocess.run([py, os.path.join(HERE, 'evt2perfetto.py'), 'cut', hub, '--from', str(t0 + 400e6), '--to', str(t0 + 600e6), '-o', cut],
                         check=True, capture_output=True, text=True).stdout.split()
    hc, rc = evtrace.read_file(out[0])
    assert hc['strings'][2] == '0', hc['strings']
    assert rc['knob'][0]['value_s'] == '0'

    tj = os.path.join(d, 't.json.gz')
    subprocess.run([py, os.path.join(HERE, 'evt2perfetto.py'), 'trace', hub, '--from', str(t0), '--to', str(t0 + 1e9), '-o', tj],
                   check=True, capture_output=True, text=True)
    ev = json.load(gzip.open(tj, 'rt'))['traceEvents']
    assert any(e.get('name') == 'knob V41_MS_SPEC_LANES=0' for e in ev), [e.get('name') for e in ev if e.get('ph') == 'i']
    import shutil
    shutil.rmtree(d)
    print('ok')


if __name__ == '__main__':
    main()
