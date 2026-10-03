#!/usr/bin/env python3
"""Per-step device/host timeline from a perfetto JSON trace that carries
device stages (evtrace Tier B -> evt2perfetto). Used to study the two-lane
dependency structure for the split simulator.

  pf_timeline.py TRACE.json.gz            list the steps
  pf_timeline.py TRACE.json.gz STEP_IDX   dump that step's events (ms from step start)
  pf_timeline.py TRACE.json.gz --gaps     per two-lane step: device busy / idle / overlap summary
"""
import gzip
import json
import sys


def load(path):
    with gzip.open(path, 'rt') as fh:
        d = json.load(fh)
    ev = d['traceEvents'] if isinstance(d, dict) else d
    names = {}
    for e in ev:
        if e.get('ph') == 'M' and e['name'] == 'thread_name':
            names[(e['pid'], e.get('tid'))] = e['args']['name']
    return ev, names


def steps_of(ev, names):
    out = []
    for e in ev:
        if e.get('ph') == 'X' and names.get((e.get('pid'), e.get('tid'))) == 'decode steps':
            out.append(e)
    out.sort(key=lambda e: e['ts'])
    return out


def window(ev, names, t0, t1):
    rows = []
    for e in ev:
        if e.get('ph') != 'X':
            continue
        ts = e.get('ts')
        if ts is None or ts + e.get('dur', 0) < t0 or ts > t1:
            continue
        tn = names.get((e.get('pid'), e.get('tid')), str(e.get('tid')))
        rows.append((ts, ts + e.get('dur', 0), tn, e['name'], e.get('args', {})))
    rows.sort()
    return rows


def busy_union(iv):
    iv = sorted(iv)
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def main():
    ev, names = load(sys.argv[1])
    st = steps_of(ev, names)
    if len(sys.argv) == 2:
        for i, s in enumerate(st):
            a = s['args']
            print(i, s['name'], f"rows={a.get('rows')} lanes={a.get('lanes')} fwd={a.get('fwd_ms', 0):.1f} "
                  f"dgpu={a.get('dgpu_busy_ms', 0):.1f} igpu={a.get('igpu_busy_ms', 0):.1f} b1_miss={a.get('b1_misses')} "
                  f"b1_read={a.get('b1_read_ms', 0):.1f} b2_page={a.get('b2_page_ms', 0):.1f} rwait={a.get('remote_wait_ms', 0):.1f}")
        return
    if sys.argv[2] == '--gaps':
        for i, s in enumerate(st):
            a = s['args']
            t0, t1 = s['ts'], s['ts'] + s['dur']
            w = window(ev, names, t0, t1)
            dg = [(x[0], x[1]) for x in w if x[2] == 'dgpu compute']
            ig = [(x[0], x[1]) for x in w if x[2] == 'igpu compute' and not x[3].startswith('mtp.')]
            both = busy_union(dg + ig)
            print(f"{i:3d} rows={a.get('rows')} lanes={a.get('lanes')} fwd={a.get('fwd_ms', 0):6.1f} dgpu_u={busy_union(dg) / 1e3:6.1f} "
                  f"igpu_u={busy_union(ig) / 1e3:6.1f} either={both / 1e3:6.1f} b1_read={a.get('b1_read_ms', 0):.1f}")
        return
    s = st[int(sys.argv[2])]
    t0 = s['ts']
    print(s['name'], json.dumps(s['args'])[:600])
    for (a, b, tn, nm, args) in window(ev, names, t0, t0 + s['dur']):
        if tn in ('decode steps',):
            continue
        extra = ''
        if 'rtt_us' in args:
            extra = f" rtt={args['rtt_us']:.0f} srv={args.get('srv_us', 0):.0f} page={args.get('page_us', 0):.0f}"
        print(f'{(a - t0) / 1e3:8.3f} {(b - t0) / 1e3:8.3f} {(b - a) / 1e3:7.3f}  {tn[:28]:28s} {nm[:40]}{extra}')


if __name__ == '__main__':
    main()
