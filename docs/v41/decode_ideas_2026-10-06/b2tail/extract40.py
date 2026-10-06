#!/usr/bin/env python3
"""Stream an EVT1 hub trace (reader from pipe/extract.py); keep ALL 40 hub_req fields +
all hub_step fields for a reservoir sample of steps per regime cell, plus the phase record
preceding each sampled step. Usage: extract40.py FILE OUT.json CAP"""
import sys, json, struct, math, random
from collections import defaultdict
rng = random.Random(777)
path, out = sys.argv[1], sys.argv[2]
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 350
f = open(path, 'rb')
d = f.read(8)
hlen = struct.unpack_from('<I', d, 4)[0]
header = json.loads(f.read(hlen))
kinds = {k['id']: (k['name'], k['fields']) for k in header['kinds']}
REQ = next(k for k, v in kinds.items() if v[0] == 'hub_req')
STEP = next(k for k, v in kinds.items() if v[0] == 'hub_step')
PHASE = next(k for k, v in kinds.items() if v[0] == 'hub_phase')
rf = kinds[REQ][1]; sf = kinds[STEP][1]; pf = kinds[PHASE][1]
ri = {n: i for i, n in enumerate(rf)}
si = {n: i for i, n in enumerate(sf)}
cur_step = None; cur_reqs = []
per_cell = defaultdict(list); nstep_cell = defaultdict(int)
n_steps = 0; n_req = 0
buf = b''; CH = 32 << 20
last_phase = None; n_phase = 0
phases = []
WANT = {'lone_spec_r3', 'lone_spec_r4', 'lone_spec_r5', 'lone_spec_r6', 'plain_M3', 'two_spec_r7', 'plain_M1', 'plain_M8'}

def regime(rec):
    rows = rec[si['rows']]; live = rec[si['live']]; lanes = rec[si['lanes']]
    if any(math.isnan(x) for x in (rows, live, lanes)):
        return None
    rows, live, lanes = int(rows), int(live), int(lanes)
    if live == 1 and rows > 1: return f'lone_spec_r{rows}'
    if live == 1 and rows == 1: return 'plain_M1'
    if live == rows: return f'plain_M{rows}'
    if live == 2 and rows > 2: return f'two_spec_r{rows}'
    return f'other_live{live}_r{rows}'

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
            st = vals[ri['step']]
            if cur_step is None or st != cur_step:
                cur_step = st; cur_reqs = []
            cur_reqs.append(list(vals)); n_req += 1
        elif kid == STEP:
            vals = struct.unpack_from(f'<{cnt}d', buf, off + 4)
            n_steps += 1
            if t_first is None: t_first = vals[si['t_start']]
            t_last = vals[si['t_end']]
            reg = regime(vals); st = vals[si['step']]
            if reg is not None and reg in WANT:
                reqs = cur_reqs if (cur_step == st) else []
                rec = {'step': list(vals), 'reqs': reqs, 'phase': last_phase, 'n_phase': n_phase}
                nstep_cell[reg] += 1
                if len(per_cell[reg]) < CAP:
                    per_cell[reg].append(rec)
                else:
                    j = rng.randrange(nstep_cell[reg])
                    if j < CAP: per_cell[reg][j] = rec
            elif reg is not None:
                nstep_cell[reg] += 1
            cur_reqs = []; cur_step = None
        elif kid == PHASE:
            vals = struct.unpack_from(f'<{cnt}d', buf, off + 4)
            last_phase = list(vals); n_phase += 1
            phases.append(last_phase)
        off += 4 + 8 * cnt
    buf = buf[off:]

json.dump({'req_fields': rf, 'step_fields': sf, 'phase_fields': pf, 'phases': phases,
           'cells': per_cell, 'n_steps_seen': n_steps, 'n_req_seen': n_req, 'counts': dict(nstep_cell),
           't_first': t_first, 't_last': t_last,
           't_mono_raw_at_open': header.get('t_mono_raw_at_open'), 't_realtime_at_open': header.get('t_realtime_at_open')},
          open(out, 'w'))
print(path, 'steps', n_steps, 'reqs', n_req, 'phases', n_phase, {k: v for k, v in nstep_cell.items() if k in WANT})
