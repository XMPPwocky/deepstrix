#!/usr/bin/env python3
"""Self-test of the format_rev 2 readers (scripts only; stdlib): a synthetic
hub file with header strings, a late `str` record and a `knob` record that
uses it; `evtrace.read_file` decodes it, `evt2perfetto cut` keeps the late
string in the cut's header, and `trace` draws the knob change by name.
Tier B device intervals (P3): partial `step_dev` records merge into their
`hub_step` (summed per step), a hub Tier B dump's `dev` records become device
tracks, and the converter counts causality violations (a) and (b).

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
    {'id': 6, 'name': 'dev', 'fields': ['t_start', 't_end', 'name', 'device', 'stream', 'step', 'unit', 'layer', 'lane', 't_host', 'q_us']},
    {'id': 7, 'name': 'cal', 'fields': ['t', 'device', 'ok', 'spin_us', 'q_us', 'resid_us', 'tol_us', 'slope_ppm', 'slope_q_ppm', 'link_ms', 'anchors']},
    {'id': 11, 'name': 'hub_step', 'fields': ['t_start', 't_end', 'step', 'rows', 'live', 'lanes', 'step_ms',
                                               'dgpu_busy_ms', 'igpu_busy_ms', 'd_q_chain', 'd_mtp', 't_fwd_sync']},
    {'id': 13, 'name': 'step_dev', 'fields': ['t_start', 'step', 'device', 'pairs', 'dgpu_busy_ms', 'igpu_busy_ms', 'd_q_chain', 'd_mtp', 'lag_ms']},
]
N = float('nan')


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
    header = {'format': 'EVT1', 'format_rev': 2, 'role': 'hub', 'pid': 42, 't_mono_raw_at_open': t0, 't_realtime_at_open': time.time_ns(),
              'kinds': KINDS, 'strings': ['<overflow>', 'V41_MS_SPEC_LANES', 'dgpu', 'igpu'], 'extras': {}}
    body = b''
    for i in range(10):
        ts = t0 + i * 100e6
        # Tier B on: the device fields are NaN here; step 3's forward sync at +80 ms.
        body += rec(11, [ts, ts + 90e6, i, 1, 1, 1, 90.0, N, N, N, N, ts + 80e6 if i == 3 else N])
    body += str_rec(4, '0')                       # interned after the file opened
    body += rec(5, [t0 + 450e6, 1, 4, 3])         # V41_MS_SPEC_LANES=0 from the knob file
    # Step 3 in three parts: dGPU forward, dGPU drafter (the next buffer), iGPU.
    body += rec(13, [t0 + 301e6, 3, 2, 2, 2.0, N, 2.0, N, 1.5])
    body += rec(13, [t0 + 300e6, 3, 2, 1, N, N, N, 4.0, N])
    body += rec(13, [t0 + 305e6, 3, 3, 1, N, 5.0, N, N, 2.5])
    body += rec(7, [t0 + 200e6, 2, 1, 15.0, 3.0, 0.5, 6.0, -12.0, 2.0, 200.0, 9])
    write(hub, header, body)
    # The hub's Tier B dump: three device intervals of step 3 (own string table).
    dump = os.path.join(d, 'hub-dump-test.evt')
    dh = dict(header, tier='B', strings=['<overflow>', 'dgpu.q_chain', 'dgpu', 'compute'],
              dump={'t_from_raw': t0, 't_to_raw': t0 + 1e9})
    dbody = b''
    ok = [t0 + 302e6, t0 + 310e6, 1, 2, 3, 3, N, 5, 0, t0 + 301e6, 1.0]
    late = [t0 + 312e6, t0 + 385e6, 1, 2, 3, 3, N, 6, 0, t0 + 311e6, 1.0]    # ends after the sync: (b)
    early = [t0 + 315e6, t0 + 330e6, 1, 2, 3, 3, N, 7, 1, t0 + 320e6, 1.0]   # starts before its record: (a)
    for r in (late, ok, early):  # batch order, not time order
        dbody += rec(6, r)
    write(dump, dh, dbody)

    h, recs = evtrace.read_file(hub)
    k = recs['knob'][0]
    assert (k['name_s'], k['value_s']) == ('V41_MS_SPEC_LANES', '0'), k
    assert 'str' not in recs, 'str records are the table, not data'
    assert recs['cal'][0]['device_s'] == 'dgpu'

    # step_dev parts merge into their hub_step by (pid, step), by field name.
    _, recs = evtrace.load([hub, dump])
    s3 = next(s for s in recs['hub_step'] if s['step'] == 3)
    assert (s3['dgpu_busy_ms'], s3['d_q_chain'], s3['d_mtp'], s3['igpu_busy_ms']) == (2.0, 2.0, 4.0, 5.0), s3
    assert (s3['dev_pairs'], s3['dev_lag_ms'], s3['dev_t_start']) == (4.0, 2.5, t0 + 300e6), s3
    s4 = next(s for s in recs['hub_step'] if s['step'] == 4)
    assert s4['dgpu_busy_ms'] != s4['dgpu_busy_ms'], 'a step without step_dev stays NaN'
    dv = recs['dev'][0]
    assert (dv['name_s'], dv['device_s'], dv['stream_s']) == ('dgpu.q_chain', 'dgpu', 'compute'), dv

    cut = os.path.join(d, 'cut')
    py = sys.executable
    out = subprocess.run([py, os.path.join(HERE, 'evt2perfetto.py'), 'cut', hub, '--from', str(t0 + 400e6), '--to', str(t0 + 600e6), '-o', cut],
                         check=True, capture_output=True, text=True).stdout.split()
    hc, rc = evtrace.read_file(out[0])
    assert hc['strings'][4] == '0', hc['strings']
    assert rc['knob'][0]['value_s'] == '0'

    tj = os.path.join(d, 't.json.gz')
    # The dump first: files are told apart by their headers.
    r = subprocess.run([py, os.path.join(HERE, 'evt2perfetto.py'), 'trace', dump, hub, '--from', str(t0), '--to', str(t0 + 1e9), '-o', tj],
                       check=True, capture_output=True, text=True)
    ev = json.load(gzip.open(tj, 'rt'))['traceEvents']
    assert any(e.get('name') == 'knob V41_MS_SPEC_LANES=0' for e in ev), [e.get('name') for e in ev if e.get('ph') == 'i']
    assert 'causality (a) start before the host recorded it: 1; (b) after the forward sync: 1 of 3' in r.stderr, r.stderr
    tracks = {e['tid']: e['args']['name'] for e in ev if e.get('ph') == 'M' and e.get('name') == 'thread_name' and e['pid'] == 3}
    devs = [e for e in ev if e.get('ph') == 'X' and e['pid'] == 3]
    assert len(devs) == 3 and all(tracks[e['tid']].startswith('dgpu compute') for e in devs), (tracks, devs)
    assert {e['args']['queued_us'] for e in devs} == {1000.0, -5000.0}, devs
    step = next(e for e in ev if e.get('ph') == 'X' and e['pid'] == 1 and e.get('args', {}).get('step') == 3)
    assert step['args']['dgpu_busy_ms'] == 2.0 and step['args']['d_mtp'] == 4.0, step['args']
    import shutil
    shutil.rmtree(d)
    print('ok')


if __name__ == '__main__':
    main()
