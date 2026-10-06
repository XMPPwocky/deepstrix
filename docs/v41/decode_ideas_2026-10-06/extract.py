#!/usr/bin/env python3
"""Stream an EVT1 hub trace; keep hub_req + hub_step for a bounded sample of steps per regime cell.
Regime from hub_step: live==1 & rows>1 -> lone_spec_r{rows}; live==rows -> plain_M{rows};
live==2 & rows>2 -> two_spec_r{rows}. Output: JSON with field names + per-step records.
Usage: extract.py FILE OUT.json [max_steps_per_cell]"""
import sys, json, struct, math, random
rng = random.Random(12345)
from collections import defaultdict

path, out = sys.argv[1], sys.argv[2]
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 300
f = open(path, 'rb')
d = f.read(8)
hlen = struct.unpack_from('<I', d, 4)[0]
header = json.loads(f.read(hlen))
kinds = {k['id']: (k['name'], k['fields']) for k in header['kinds']}
REQ = next(k for k, v in kinds.items() if v[0] == 'hub_req')
STEP = next(k for k, v in kinds.items() if v[0] == 'hub_step')
PHASE = next(k for k, v in kinds.items() if v[0] == 'hub_phase')
rf = kinds[REQ][1]; sf = kinds[STEP][1]
ri = {n: i for i, n in enumerate(rf)}
si = {n: i for i, n in enumerate(sf)}
KEEP_REQ = ['t_submit', 't_submit_end', 't1', 't4', 't_wait_enter', 't_wait_exit', 'lane', 'layer', 'b', 'seq', 'rtt_us', 'srv_us', 'page_us', 'blocked', 'n_picks', 'n_distinct', 'flags']
kr = [ri[k] for k in KEEP_REQ]
cur_step = None
cur_reqs = []
per_cell = defaultdict(list)
n_steps = 0; n_req = 0
buf = b''
CH = 32 << 20
phases = []
nstep_cell = defaultdict(int)

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

while True:
    chunk = f.read(CH)
    if not chunk:
        break
    buf += chunk
    off = 0; n = len(buf)
    while off + 4 <= n:
        kid, cnt = struct.unpack_from('<HH', buf, off)
        if off + 4 + 8 * cnt > n:
            break
        if kid == REQ:
            vals = struct.unpack_from(f'<{cnt}d', buf, off + 4)
            st = vals[ri['step']]
            if cur_step is None or st != cur_step:
                # a new step's requests begin (previous step's hub_step should have come already)
                cur_step = st; cur_reqs = []
            cur_reqs.append([vals[i] for i in kr])
            n_req += 1
        elif kid == STEP:
            vals = struct.unpack_from(f'<{cnt}d', buf, off + 4)
            n_steps += 1
            reg = regime(vals)
            st = vals[si['step']]
            if reg is not None:
                reqs = cur_reqs if (cur_step == st) else []
                rec = {'step': list(vals), 'reqs': reqs}
                nstep_cell[reg] += 1
                # reservoir sampling: unbiased CAP-sized sample over the whole file
                if len(per_cell[reg]) < CAP:
                    per_cell[reg].append(rec)
                else:
                    j = rng.randrange(nstep_cell[reg])
                    if j < CAP:
                        per_cell[reg][j] = rec
            cur_reqs = []; cur_step = None
        elif kid == PHASE:
            vals = struct.unpack_from(f'<{cnt}d', buf, off + 4)
            phases.append(list(vals))
        off += 4 + 8 * cnt
    buf = buf[off:]
    # stop early once every interesting cell is full
    want = ['lone_spec_r3', 'lone_spec_r4', 'lone_spec_r5', 'lone_spec_r6', 'plain_M3', 'plain_M5', 'plain_M8', 'two_spec_r7', 'plain_M1', 'plain_M2', 'plain_M4']
    pass

json.dump({'req_fields': KEEP_REQ, 'step_fields': sf, 'phase_fields': kinds[PHASE][1], 'phases': phases,
           'cells': per_cell, 'n_steps_seen': n_steps, 'n_req_seen': n_req, 'counts': dict(nstep_cell)}, open(out, 'w'))
print(path, 'steps seen', n_steps, 'reqs', n_req, dict(nstep_cell))
