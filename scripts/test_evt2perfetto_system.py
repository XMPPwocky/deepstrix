#!/usr/bin/env python3
"""Self-test of the evt2perfetto system / disk / Thunderbolt tracks (stdlib):
synthetic hub + box-2 files WITH the new fields (sys tb_*, hub_step.spec_streams,
b1_read, b2_read.route) and the same WITHOUT them (today's production files).

  * `window --two`: the run finder (interruptions tolerated, runs split by a
    long gap, the rows >= 4 heuristic when a file has no spec_streams) and its
    stderr stats;
  * `cut` keeps `sys` samples from before the window (the delta predecessor);
  * `trace`: "box 1 / box 2 disks + system" counters (MB/s, IOPS, latency,
    queue depth, util, PSI, NIC), the Thunderbolt wire legs (clamped legs
    flagged and counted), link MB/s, flows, box-2 reads on per-drive tracks,
    b1_read slices; a file without the new fields converts and lists the skips.

  python3 scripts/test_evt2perfetto_system.py"""
import gzip
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evt2perfetto as e2p  # noqa: E402

N = float('nan')
OFF = 1e9  # box 2's clock is 1 s ahead of the hub's

SYS_BASE = (['t'] + [f'd{k}_{f}' for k in range(3) for f in ('reads', 'rsect', 'rms', 'writes', 'wsect', 'wms', 'inflight', 'io_ms', 'wtd_ms')]
            + ['psi_io_some', 'psi_io_full', 'psi_cpu_some', 'psi_mem_some', 'psi_mem_full', 'load1', 'procs_running', 'procs_blocked',
               'mem_avail_kb', 'cached_kb', 'dirty_kb', 'writeback_kb', 'self_read_bytes', 'self_write_bytes', 'self_rchar', 'self_threads',
               'self_vol_ctxsw', 'self_invol_ctxsw', 'ctxt_total', 'sample_us'])
TB = ['tb_rx_bytes', 'tb_tx_bytes', 'tb_rx_packets', 'tb_tx_packets', 'tb_rx_dropped', 'tb_rx_missed']
HUB_REQ = ['t_submit', 't_submit_end', 't1', 't4', 't_wait_enter', 't_wait_exit', 't2_b2', 't3_b2', 'step', 'lane', 'layer', 'b', 'seq',
           'bytes_out', 'bytes_in', 'blocked', 'clock_offset_ns', 'clock_delay_ns', 'rtt_us']
HUB_STEP = ['t_start', 't_end', 'step', 'rows', 'live', 'lanes', 'step_ms', 'remote_wait_ms', 'b1_misses', 'b1_read_ms']
B2_REQ = ['seq', 'layer', 'b', 't_frame', 't_run_start', 't_run_end', 't_ready']
B2_READ = ['src', 'seq', 'layer', 'expert', 't_read_start', 't_read_end', 'route']
B1_READ = ['t_read_start', 't_read_end', 'layer', 'expert', 'bytes', 'src', 'dev']


def kinds(new):
    return [
        {'id': 1, 'name': 'sys', 'fields': SYS_BASE + (TB if new else [])},
        {'id': 11, 'name': 'hub_step', 'fields': HUB_STEP + (['spec_streams'] if new else [])},
        {'id': 10, 'name': 'hub_req', 'fields': HUB_REQ},
        {'id': 20, 'name': 'b2_req', 'fields': B2_REQ},
        {'id': 21, 'name': 'b2_read', 'fields': B2_READ},
    ] + ([{'id': 16, 'name': 'b1_read', 'fields': B1_READ}] if new else [])


def rec(kid, fields, **kw):
    vals = [float(kw.get(f, N)) for f in fields]
    return struct.pack('<HH', kid, len(vals)) + struct.pack(f'<{len(vals)}d', *vals)


def write(path, header, body):
    hb = json.dumps(header).encode()
    with open(path, 'wb') as f:
        f.write(b'EVT1' + struct.pack('<I', len(hb)) + hb + body)


def sys_rec(fields, t, i, with_tb, disks):
    kw = {'t': t, 'psi_io_some': i * 10000, 'psi_io_full': i * 5000, 'psi_cpu_some': i * 1000, 'psi_mem_some': 0, 'psi_mem_full': 0,
          'procs_running': 3, 'procs_blocked': 1, 'mem_avail_kb': 4 * 1048576, 'dirty_kb': 2048, 'writeback_kb': 1024,
          'self_read_bytes': i * 5e5, 'self_write_bytes': 0}
    for k in range(disks):
        kw.update({f'd{k}_reads': i * 100, f'd{k}_rsect': i * 2048 * (k + 1), f'd{k}_rms': i * 50, f'd{k}_writes': i * 10, f'd{k}_wsect': i * 100,
                   f'd{k}_wms': i * 20, f'd{k}_inflight': 3, f'd{k}_io_ms': i * 50, f'd{k}_wtd_ms': i * 200})
    if with_tb:
        kw.update({'tb_rx_bytes': i * 1e6, 'tb_tx_bytes': i * 2e6, 'tb_rx_packets': i * 700, 'tb_tx_packets': i * 900,
                   'tb_rx_dropped': 0, 'tb_rx_missed': i * 2})
    return rec(1, fields, **kw)


def make(d, new, t0, nan_layer=False):
    """Hub file + box-2 file around hub window [t0, t0 + 3 s]."""
    ks = kinds(new)
    hdr = {'format': 'EVT1', 'format_rev': 2, 'role': 'hub', 'pid': 7, 't_mono_raw_at_open': t0, 't_realtime_at_open': time.time_ns(),
           'kinds': ks, 'strings': ['<overflow>'], 'extras': {}, 'sys_devices': ['nvme0n1', 'nvme1n1']}
    h2 = dict(hdr, role='b2', pid=8, sys_devices=['nvme0n1', 'nvme1n1'],
              extras={'route_devices': {'primary': 'nvme0n1', 'mirror': 'nvme1n1'}, 'env': {'V41_EXPERT_MIRROR_FRAC': '0.70'}})
    if new:
        hdr.update(sys_net='thunderbolt0', sys_link={'rx_speed': '20.0 Gb/s', 'tx_speed': '20.0 Gb/s', 'rx_lanes': '2', 'tx_lanes': '2'})
        h2.update(sys_net='thunderbolt0', sys_link=hdr['sys_link'])
    kf = {k['name']: k['fields'] for k in ks}
    body = b''
    # box-1 sys: every 100 ms from 0.5 s before the window to 0.5 s after it
    for i in range(0, 41):
        body += sys_rec(kf['sys'], t0 - 0.5e9 + i * 1e8, i, new, 2)
    for i in range(40):  # 100 ms steps: 2-stream except step 10 (1 spec stream) and 20..31 (one live stream, 1.2 s)
        live = 1 if 20 <= i < 32 else 2
        spec = 1 if i == 10 or live == 1 else 2
        kw = dict(t_start=t0 + i * 1e8, t_end=t0 + i * 1e8 + 9e7, step=i, rows=8 if live == 2 else 1, live=live, lanes=2, step_ms=90)
        if i % 10 == 5:  # PROFILED steps (every 1 s here; every ~3 s in production) carry the per-step b2 wait
            kw.update(remote_wait_ms=7, b1_misses=10 * i, b1_read_ms=5.0 * i)  # pager: delta since the previous profiled step
        if new:
            kw['spec_streams'] = spec
        body += rec(11, kf['hub_step'], **kw)
    b2body = b''
    for i in range(0, 41):
        b2body += sys_rec(kf['sys'], t0 + OFF - 0.5e9 + i * 1e8, i, new, 2)
    for i in range(40):
        T = t0 + 1e8 + i * 5e7
        lat_out, lat_back = 20e3, 20e3
        t2, t3 = T + lat_out + OFF, T + lat_out + 300e3 + OFF
        if i == 7:  # a leg the clock bound turns negative
            t2 = T - 15e3 + OFF
        body += rec(10, kf['hub_req'], t_submit=T - 2e3, t_submit_end=T, t1=T, t4=T + 340e3, t_wait_enter=T + 100e3, t_wait_exit=T + 350e3,
                    t2_b2=t2, t3_b2=t3, step=i // 8, lane=i % 2, layer=N if (nan_layer and i == 39) else i % 43, b=8, seq=i, bytes_out=40000, bytes_in=60000, blocked=1,
                    clock_offset_ns=OFF, clock_delay_ns=40e3, rtt_us=340)
        b2body += rec(20, kf['b2_req'], seq=i, layer=i % 43, b=8, t_frame=T + lat_out + OFF + 2e3, t_run_start=T + 40e3 + OFF,
                      t_run_end=T + 290e3 + OFF, t_ready=T + lat_out + 300e3 + OFF - 3e3)
    for j, (src, route) in enumerate(((0, 1), (3, 2), (1, 0), (0, 1))):
        t = t0 + OFF + 5e8 + j * 2e8
        b2body += rec(21, kf['b2_read'], src=src, seq=j, layer=j, expert=j + 10, t_read_start=t, t_read_end=t + 1e8, route=route if new else N)
    if new:
        for j, (src, dev) in enumerate(((0, 0), (1, 1), (0, N), (1, 0))):
            t = t0 + 4e8 + j * 3e8
            body += rec(16, kf['b1_read'], t_read_start=t, t_read_end=t + 5e7, layer=j, expert=j, bytes=3e6, src=src, dev=dev)
    hub = os.path.join(d, ('hub-nan.evt' if nan_layer else 'hub-new.evt') if new else 'hub-old.evt')
    b2 = os.path.join(d, 'b2-new.evt' if new else 'b2-old.evt')
    write(hub, hdr, body)
    write(b2, h2, b2body)
    return hub, b2


def run(args, ok=True):
    r = subprocess.run([sys.executable, os.path.join(HERE, 'evt2perfetto.py')] + args, capture_output=True, text=True)
    assert (r.returncode == 0) == ok, (args, r.stdout, r.stderr)
    return r


def two_runs_unit():
    def feed(seq, spec_field=True):
        f = e2p.TwoRuns()
        t = 0.0
        for live, rows, spec in seq:
            f.add(t, t + 1.9e8, live, rows, spec if spec_field else N)
            t += 2e8
        return f, f.finish()
    q, bad1, bad_live1 = (2, 8, 2), (2, 8, 1), (1, 1, 1)
    f, runs = feed([q] * 10 + [bad1] + [q] * 10)
    assert len(runs) == 1 and runs[0]['n_total'] == 21 and runs[0]['n_qual'] == 20, runs
    # 6 interrupted steps = 1.2 s: the run ends, a second starts
    f, runs = feed([q] * 10 + [bad_live1] * 6 + [q] * 10)
    assert len(runs) == 2 and [r['n_qual'] for r in runs] == [10, 10], runs
    # 4 bad steps = 0.8 s < 1 s: tolerated by the time rule
    f, runs = feed([q] * 10 + [bad_live1] * 4 + [q] * 10)
    assert len(runs) == 1 and runs[0]['n_total'] == 24, runs
    # no spec_streams field: rows >= 4 with live == 2 qualifies, rows 1 does not
    f, runs = feed([(2, 8, N)] * 5 + [(2, 1, N)] * 6 + [(2, 8, N)] * 5, spec_field=False)
    assert len(runs) == 2 and f.n_spec_field == 0 and f.n_qual == 10, (runs, f.n_qual)
    # an idle gap with no steps at all (minutes) splits a run even with 0 bad steps
    f = e2p.TwoRuns()
    f.add(0, 1.9e8, 2, 8, 2)
    f.add(60e9, 60.19e9, 2, 8, 2)
    assert len(f.finish()) == 2
    # ... and so does a 2.5 s stall with NO steps in between (a scheduler stall is not "100% qualifying")
    f = e2p.TwoRuns()
    f.add(0, 1.9e8, 2, 8, 2)
    f.add(2.5e9, 2.69e9, 2, 8, 2)
    assert len(f.finish()) == 2
    # a gap < 1 s with no steps in between is still bridged
    f = e2p.TwoRuns()
    f.add(0, 1.9e8, 2, 8, 2)
    f.add(9e8, 1.09e9, 2, 8, 2)
    assert len(f.finish()) == 1
    # one bad step then a 2.5 s hole: the 3 s rule (1-2 bad steps between) still joins
    f = e2p.TwoRuns()
    f.add(0, 1.9e8, 2, 8, 2)
    f.add(2e8, 3.9e8, 2, 8, 1)
    f.add(2.5e9, 2.69e9, 2, 8, 2)
    runs = f.finish()
    assert len(runs) == 1 and runs[0]['n_qual'] == 2 and runs[0]['n_total'] == 3, runs


def main():
    two_runs_unit()
    d = tempfile.mkdtemp(prefix='evt-sys-')
    t0 = float(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)) - 20e9
    py_to = t0 + 3e9
    hub, b2 = make(d, True, t0)
    hub_old, b2_old = make(d, False, t0)

    # ---- window --two
    # steps 0..19 (step 10 interrupted: tolerated) = run A, 2.0 s; 20..31 one live stream = 1.2 s (splits);
    # steps 32..39 = run B, 0.8 s
    r = run(['window', hub, '--last', '1', '--two'])
    w = json.loads(r.stdout)
    assert abs(w['hub_to'] - (t0 + 19 * 1e8 + 9e7)) < 1, w     # the latest run >= 1 s is A
    assert '2-stream run 2.0 s, 20 steps (95.0% qualifying)' in r.stderr and '2 runs in the file, 1 >= 1 s' in r.stderr, r.stderr
    r = run(['window', hub, '--last', '0.5', '--two'])
    assert abs(json.loads(r.stdout)['hub_to'] - (t0 + 39 * 1e8 + 9e7)) < 1, r.stdout     # both are long enough: the latest
    r = run(['window', hub, '--last', '5', '--two'])
    assert abs(json.loads(r.stdout)['hub_to'] - (t0 + 19 * 1e8 + 9e7)) < 1 and 'the longest, shorter than --last 5 s' in r.stderr, r.stderr
    r = run(['window', hub_old, '--last', '1.5', '--two'])   # no spec_streams: rows >= 4 (step 10 then qualifies)
    assert 'rows >= 4, the file has no spec_streams field' in r.stderr, r.stderr
    assert '2-stream run 2.0 s, 20 steps (100.0% qualifying)' in r.stderr, r.stderr
    assert abs(json.loads(r.stdout)['hub_to'] - (t0 + 19 * 1e8 + 9e7)) < 1, r.stdout
    run(['window', hub, '--last', '5', '--two', '--lone'], ok=False)
    r = run(['window', hub, '--last', '1.5'])   # unchanged without --two
    assert '2-stream' not in r.stderr, r.stderr

    # ---- cut keeps the sys predecessor
    cut = os.path.join(d, 'cut')
    out = run(['cut', b2, '--from', str(t0 + OFF + 1e9), '--to', str(t0 + OFF + 2e9), '-o', cut]).stdout.split()
    import evtrace
    _, rc = evtrace.read_file(out[0])
    first = min(r['t'] for r in rc['sys'])
    assert first < t0 + OFF + 1e9 and first >= t0 + OFF + 1e9 - 5e9, (first - t0 - OFF - 1e9)
    assert min(r['t_read_start'] for r in rc['b2_read']) >= t0 + OFF + 1e9, 'non-sys records stay inside the window'

    # ---- trace WITH the new fields
    models = os.path.join(d, 'models.hub')
    open(models, 'w').write('nvme0n1 CT2000TEST\nnvme1n1 YMTC TEST 2TB\nzram0 \n')
    tj = os.path.join(d, 'new.json.gz')
    r = run(['trace', hub, b2, '--from', str(t0), '--to', str(py_to), '-o', tj, '--disk-models-hub', models,
             '--disk-models-b2', 'nvme0n1=SN5000TEST,nvme1n1=990PROTEST'])
    ev = json.load(gzip.open(tj, 'rt'))['traceEvents']
    procs = {e['pid']: e['args']['name'] for e in ev if e.get('name') == 'process_name'}
    assert procs[4] == 'box 1 disks + system' and procs[5] == 'box 2 disks + system', procs
    assert 'Thunderbolt link (box 1 <-> box 2)' in procs.values(), procs
    assert any(e.get('name') == 'process_sort_index' for e in ev)
    counters = {}
    for e in ev:
        if e.get('ph') == 'C':
            counters.setdefault((e['pid'], e['name']), []).append(e['args']['v'])

    one_n = lambda pid, name: counters[(pid, name)]

    def one(pid, name):
        vs = counters[(pid, name)]
        assert len(vs) >= 20, (name, len(vs))
        return vs

    d1 = 'nvme0n1 (CT2000TEST)'
    d2 = 'nvme1n1 (YMTC TEST 2TB)'
    near = lambda vs, x: all(abs(v - x) < 1e-6 * max(1, abs(x)) for v in vs)
    assert near(one(4, f'{d1} read MB/s'), 2048 * 512 / 0.1 / 1e6), one(4, f'{d1} read MB/s')[:3]
    assert near(one(4, f'{d2} read MB/s'), 2 * 2048 * 512 / 0.1 / 1e6)
    assert near(one(4, f'{d1} read IOPS'), 1000) and near(one(4, f'{d1} write IOPS'), 100)
    assert near(one(4, f'{d1} avg read latency ms'), 0.5) and near(one(4, f'{d1} avg write latency ms'), 2.0)
    assert near(one(4, f'{d1} in-flight'), 3) and near(one(4, f'{d1} avg queue depth'), 2.0) and near(one(4, f'{d1} util %'), 50)
    assert near(one(4, 'PSI io some %'), 10) and near(one(4, 'PSI io full %'), 5) and near(one(4, 'PSI cpu some %'), 1)
    assert near(one(4, 'procs_blocked'), 1) and near(one(4, 'dirty + writeback MB'), 3)
    assert near(one(5, 'nvme0n1 (SN5000TEST) read MB/s'), 2048 * 512 / 0.1 / 1e6) and near(one(5, 'nvme1n1 (990PROTEST) util %'), 50)
    # pager deltas: only profiled steps carry them (5, 15, 25 in the 3 s window; step i ends at t0 + i*0.1 s + 90 ms), spread
    # over [previous profiled end, this end] = 1 s; the first one in the window has no predecessor (skipped, listed)
    assert one_n(4, 'pager misses/s (avg since prev profiled step, incl. prefill)') == [150.0, 250.0, 0.0]
    assert one_n(4, 'pager read ms per s (avg since prev profiled step, incl. prefill)') == [75.0, 125.0, 0.0]
    assert (4, 'pager b1_misses / step') not in counters and (1, 'b1_misses') not in counters and (1, 'b1_read_ms') not in counters
    assert "the window's first profiled step has no predecessor" in r.stderr, r.stderr
    # per-profiled-step values are reset at the step's end (no hold across the unprofiled steps)
    assert counters[(1, 'remote_wait_ms')] == [7, 0.0] * 3, counters[(1, 'remote_wait_ms')]
    pg = sorted((e for e in ev if e.get('ph') == 'X' and e['pid'] == 4 and 'misses' in e['name']), key=lambda e: e['ts'])
    assert [e['args']['misses'] for e in pg] == [150, 250] and all(abs(e['dur'] - 1e6) < 1 for e in pg), pg
    step_args = [e['args'] for e in ev if e.get('ph') == 'X' and e['pid'] == 1 and 'pager misses since prev profiled step' in e.get('args', {})]
    assert len(step_args) == 3 and not any('b1_misses' in a for a in step_args)
    nic = [k for k in counters if k[0] == 6 and 'NIC' not in k[1] and 'rx MB/s' in k[1]]
    assert near(one(6, 'box 1 thunderbolt0 rx MB/s'), 10) and near(one(6, 'box 2 thunderbolt0 tx MB/s'), 20), nic
    assert near(one(6, 'box 1 thunderbolt0 rx_missed (per sample)'), 2)
    # derived link rate: 10 ms buckets; 40 requests x 40000 B out in the window
    out_mb = sum(counters[(6, 'link -> box 2 MB/s (derived from requests)')]) * 0.01 * 1e6
    assert abs(out_mb - 40 * 40000) < 1, out_mb
    in_mb = sum(counters[(6, 'link <- box 1 MB/s (derived from replies)')]) * 0.01 * 1e6
    assert abs(in_mb - 40 * 60000) < 1, in_mb
    # wire legs
    tid_names = {(e['pid'], e['tid']): e['args']['name'] for e in ev if e.get('name') == 'thread_name'}
    sl = [e for e in ev if e.get('ph') == 'X']
    track = lambda e: tid_names[(e['pid'], e['tid'])]
    outs = [e for e in sl if e['pid'] == 6 and track(e).startswith('-> box 2 (requests)')]
    backs = [e for e in sl if e['pid'] == 6 and track(e).startswith('<- box 1 (replies)')]
    assert len(outs) == 40 and len(backs) == 40, (len(outs), len(backs))
    clamped = [e for e in outs + backs if e['args'].get('clock_clamped')]
    assert len(clamped) == 1 and clamped[0]['args']['seq'] == 7 and clamped[0]['args']['leg_us'] < 0 and abs(clamped[0]['dur'] - 2.0) < 1e-9, clamped
    ok_out = next(e for e in outs if e['args']['seq'] == 3)
    assert abs(ok_out['args']['leg_us'] - 20.0) < 0.2 and ok_out['args']['bytes'] == 40000 and ok_out['name'] == f"L3/{'AB'[1]}", ok_out
    assert 'clock_clamped' not in ok_out['args']
    assert '1 clamped to 2 us' in r.stderr and '1 of them negative' in r.stderr, r.stderr
    # flows: every request, s ... f through hub req -> out -> b2 -> back -> wait
    fl = [e for e in ev if e.get('ph') in ('s', 't', 'f')]
    assert sum(e['ph'] == 's' for e in fl) == 40 and sum(e['ph'] == 'f' for e in fl) == 40 and sum(e['ph'] == 't' for e in fl) == 40 * 3, \
        (len(fl), r.stderr)
    one_flow = sorted((e for e in fl if e['id'] == fl[0]['id']), key=lambda e: e['ts'])
    assert [e['ph'] for e in one_flow] == ['s', 't', 't', 't', 'f'] and {e['pid'] for e in one_flow} == {1, 2, 6}, one_flow
    assert 'flows: 40 of 40 requests' in r.stderr and '40 of 40 also pass through a box-2' in r.stderr, r.stderr
    # box-2 reads: per-drive tracks; route 1 -> mirror (nvme1n1), 2 -> primary (nvme0n1), 0 -> ONE split track
    d5 = [(track(e), e['name']) for e in sl if e['pid'] == 5]
    assert ('nvme1n1 (990PROTEST) [mirror] demand reads', 'L0 E10') in d5, d5
    assert ('nvme0n1 (SN5000TEST) [primary] speculative reads', 'L1 E11') in d5, d5
    assert ('split reads (both drives) certain', 'L2 E12') in d5, d5
    assert sum(1 for _, n in d5 if n == 'L2 E12') == 1, 'a split read is drawn once, not on both drives'
    assert any(track(e).startswith('SSD demand reads') for e in sl if e['pid'] == 2), 'the box-2 tracks still exist'
    # b1_read slices per drive and class
    b1 = [(track(e), e['name']) for e in sl if e['pid'] == 4]
    assert (f'{d1} demand reads', 'L0 E0') in b1 and (f'{d2} prefetch reads', 'L1 E1') in b1 and ('pager (drive unknown) demand reads', 'L2 E2') in b1, b1
    # 2-stream steps are named
    names = {e['name'] for e in sl if e['pid'] == 1}
    assert '2-stream DSpark 8 rows 2L' in names and 'step 8 rows 2L' in names, names
    assert not any('skipped: box 1 NIC' in l for l in r.stderr.splitlines())
    # flows are capped when asked for
    r = run(['trace', hub, b2, '--from', str(t0), '--to', str(py_to), '-o', tj, '--max-flows', '10'])
    ev = json.load(gzip.open(tj, 'rt'))['traceEvents']
    assert sum(e.get('ph') == 's' for e in ev) == 10 and 'capped at --max-flows 10: every 4th of 40' in r.stderr, r.stderr
    r = run(['trace', hub, b2, '--from', str(t0), '--to', str(py_to), '-o', tj, '--flows', 'off'])
    ev = json.load(gzip.open(tj, 'rt'))['traceEvents']
    assert not any(e.get('ph') in ('s', 'f') for e in ev) and 'flows: none drawn' in r.stderr, r.stderr

    # ---- trace WITHOUT the new fields (today's files): no crash, skips listed
    tj2 = os.path.join(d, 'old.json.gz')
    r = run(['trace', hub_old, b2_old, '--from', str(t0), '--to', str(py_to), '-o', tj2])
    ev = json.load(gzip.open(tj2, 'rt'))['traceEvents']
    for needle in ('hub_step.spec_streams', "box 1 NIC (tb_* counters)", "box 2 NIC (tb_* counters)", "'b1_read' kind", 'b2_read records have no route'):
        assert needle in r.stderr, (needle, r.stderr)
    counters = {(e['pid'], e['name']) for e in ev if e.get('ph') == 'C'}
    assert (4, 'nvme0n1 read MB/s') in counters and (5, 'nvme0n1 util %') in counters, 'disk counters work without the new fields'
    assert not any('rx MB/s' in n and 'link' not in n for p, n in counters), 'no NIC counters'
    tn = {e['args']['name'] for e in ev if e.get('name') == 'thread_name' and e['pid'] == 6}
    assert any(t.startswith('-> box 2 (requests)') for t in tn), 'the wire legs need only hub_req stamps'
    # hub alone (no box-2 file): box-2 tracks skipped, wire legs still drawn from the hub_req stamps
    r = run(['trace', hub_old, '--from', str(t0), '--to', str(py_to), '-o', tj2])
    assert 'no box-2 file given' in r.stderr, r.stderr
    # a hub_req with stamps but a NaN layer: no crash, wire legs named 'req/<lane>'
    hub_nan, _ = make(d, True, t0, nan_layer=True)
    tj3 = os.path.join(d, 'nan.json.gz')
    run(['trace', hub_nan, b2, '--from', str(t0), '--to', str(py_to), '-o', tj3])
    ev = json.load(gzip.open(tj3, 'rt'))['traceEvents']
    assert any(e.get('ph') == 'X' and e['pid'] == 6 and e['name'].startswith('req/') for e in ev), 'NaN-layer wire leg'
    # a box-2 Tier B dump after the Tier A file (its header lacks extras / has another sys_devices order):
    # the drive tracks still come from the Tier A header
    with open(b2, 'rb') as f:
        raw = f.read()
    hl = struct.unpack('<I', raw[4:8])[0]
    hb = json.loads(raw[8:8 + hl])
    hb.update(tier='B', extras={}, sys_devices=['nvme1n1', 'nvme0n1'])
    b2b = os.path.join(d, 'b2-tierb.evt')
    write(b2b, hb, b'')
    tj4 = os.path.join(d, 'tierb.json.gz')
    r = run(['trace', hub, b2, b2b, '--from', str(t0), '--to', str(py_to), '-o', tj4,
             '--disk-models-b2', 'nvme0n1=SN5000TEST,nvme1n1=990PROTEST'])
    ev = json.load(gzip.open(tj4, 'rt'))['traceEvents']
    tn4 = {(e['pid'], e['tid']): e['args']['name'] for e in ev if e.get('name') == 'thread_name'}
    d5 = [(tn4[(e['pid'], e['tid'])], e['name']) for e in ev if e.get('ph') == 'X' and e['pid'] == 5]
    assert ('nvme1n1 (990PROTEST) [mirror] demand reads', 'L0 E10') in d5, d5
    assert 'ASSUMED' not in r.stderr, r.stderr
    # evtrace.py report over an old hub file (no tb_* keys) plus a new one (same role): no KeyError
    ev_py = os.path.join(HERE, 'evtrace.py')
    # (sys-only files: `report` needs the full hub_step field set otherwise)
    syso, sysn = os.path.join(d, 'sys-old.evt'), os.path.join(d, 'sys-new.evt')
    for path, flag in ((syso, False), (sysn, True)):
        ks = [k for k in kinds(flag) if k['name'] == 'sys']
        hh = {'format': 'EVT1', 'format_rev': 2, 'role': 'hub', 'pid': 7, 't_mono_raw_at_open': t0, 't_realtime_at_open': time.time_ns(),
              'kinds': ks, 'strings': ['<overflow>'], 'extras': {}, 'sys_devices': ['nvme0n1', 'nvme1n1']}
        write(path, hh, b''.join(sys_rec(ks[0]['fields'], t0 + (i + (0 if not flag else 10)) * 1e8, i, flag, 2) for i in range(10)))
    r = subprocess.run([sys.executable, ev_py, 'report', syso, sysn], capture_output=True, text=True)
    assert r.returncode == 0 and 'Traceback' not in r.stderr and 'thunderbolt rx MB/s' in r.stdout, (r.stdout[-500:], r.stderr)
    shutil.rmtree(d)
    print('ok')


if __name__ == '__main__':
    main()
