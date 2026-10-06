#!/usr/bin/env python3
import json, sys, copy
sys.path.insert(0, '/home/claude-code/.claude/jobs/749c61d3/tmp/pipe')
from sim import run
from calib import params, CELLS, DMAP, FLOOR_CHAIN, FLOOR_SHARED, HOST

# regime shares of decode step wall (caller): lone 0.483 split by digest n_blocks; two-stream 0.123 -> r7; plain M3..M8
SH = {'lone_spec_r3': 0.483 * 331 / 2082, 'lone_spec_r4': 0.483 * 1019 / 2082, 'lone_spec_r5': 0.483 * 225 / 2082, 'lone_spec_r6': 0.483 * 507 / 2082,
      'two_spec_r7': 0.123, 'plain_M3': 0.141, 'plain_M4': 0.063, 'plain_M5': 0.078, 'plain_M8': 0.013}
# M6/M7 (not simulated) ride on M5/M8
EXTRA = {'plain_M6': (0.038, ['plain_M5', 'plain_M8']), 'plain_M7': (0.043, ['plain_M8'])}

def lever(P, name):
    P = copy.deepcopy(P)
    hostk = ['h_post', 'h_chain_enq', 'event_lat', 'poll', 'h_route', 'h_launch']
    if name == 'launch-30':
        P['D_chain'] -= 0.3 * (P['D_chain'] - FLOOR_CHAIN - 75); P['D_shared'] -= 0.3 * (P['D_shared'] - FLOOR_SHARED)
        P['D_combine'] *= 0.7; P['D_push'] *= 0.7
    elif name == 'bytes-50':
        P['D_chain'] -= 0.5 * FLOOR_CHAIN; P['D_shared'] -= 0.5 * FLOOR_SHARED
    elif name == 'handoff-50':
        for k in hostk: P[k] *= 0.5
        P['h_prep'] = 0.5 * HOST['h_prep'] + P['pager_block'] / (40 * P['n_lanes'])
    elif name == 'moe-45':
        P['D_hot_dgpu'] = 0.45 * P['D_moe'] * 214 / 600; P['D_moe'] *= 0.55
    elif name == 'engram-join':
        P['engram_join'] = 0.0
    elif name == 'engram-stage':
        P['engram_stage'] = 0.0
    elif name == 'sel_sync':
        P['h_route'] -= 34.0
    elif name == 'pager':
        P['h_prep'] -= P['pager_block'] / (40 * P['n_lanes']); P['pager_block'] = 0.0
    elif name == 'lanes3':
        n = 3; rows = sum(P['lane_rows'])
        if rows < 3: return None
        rpl_old = rows / P['n_lanes']; rpl_new = rows / n
        P['n_lanes'] = n; P['lane_rows'] = [rows // n] * n; P['moe_scale'] = [1.0] * n
        P['D_moe'] = P['D_moe'] * P['moe_scale'][0] * (2 / 3) * 1.05
        f = (270 + 100 * rpl_new) / (270 + 100 * rpl_old)
        P['rtt_samples'] = [x * f for x in P['rtt_samples']]
        P['h_prep'] = HOST['h_prep'] + P['pager_block'] / (40 * n)
    elif name == 'chain-early':
        # issue next layer's chain without waiting for the box-2 reply: Post gates only the combine. Approximate by
        # removing the reply wait from the chain's start (combine still waits): rtt no longer delays the chain.
        P['rtt_samples'] = [50.0 for _ in P['rtt_samples']]
    else:
        raise ValueError(name)
    return P

LEVERS = ['launch-30', 'bytes-50', 'handoff-50', 'moe-45', 'engram-join', 'engram-stage', 'sel_sync', 'pager', 'lanes3', 'chain-early']
BUNDLES = {'launch-30+handoff-50': ['launch-30', 'handoff-50'], 'bytes-50+handoff-50': ['bytes-50', 'handoff-50'],
           'dgpu-all (launch+bytes+handoff)': ['launch-30', 'bytes-50', 'handoff-50'],
           'engram both': ['engram-join', 'engram-stage'],
           'moe-45+handoff-50+engram': ['moe-45', 'handoff-50', 'engram-join', 'engram-stage'],
           'lanes3+handoff-50': ['lanes3', 'handoff-50'],
           'everything (no lanes3)': ['launch-30', 'bytes-50', 'handoff-50', 'moe-45', 'engram-join', 'engram-stage', 'sel_sync', 'pager']}

base = {}
for c in CELLS:
    P = params(c, DMAP.get(c)); base[c] = (P, run(P, n=30))

def evaluate(names):
    res = {}
    for c in CELLS:
        P = base[c][0]
        for nm in names:
            P = lever(P, nm)
            if P is None: break
        if P is None:
            res[c] = None; continue
        r = run(P, n=30)
        res[c] = (base[c][1]['fwd'] - r['fwd'], r['fwd'], r)
    return res

def weighted(res):
    tot_ms = 0.0; tot_pct = 0.0; w = 0.0
    for c, sh in SH.items():
        if res.get(c) is None: continue
        d, f, _ = res[c]; b = base[c][1]['fwd']
        tot_ms += sh * d; tot_pct += sh * d / b; w += sh
    for c, (sh, src) in EXTRA.items():
        ds = [res[s] for s in src if res.get(s) is not None]
        if not ds: continue
        pct = sum(d / base[s][1]['fwd'] for (d, f, _), s in zip(ds, src)) / len(ds)
        ms = sum(d for (d, f, _) in ds) / len(ds)
        tot_ms += sh * ms; tot_pct += sh * pct; w += sh
    return tot_ms / w, 100 * tot_pct / w

hdr = 'what-if                          | ' + ' '.join(f'{c[-7:]:>8s}' for c in CELLS) + ' | wtd ms  wtd %'
print(hdr)
print('baseline sim fwd (ms)            | ' + ' '.join(f"{base[c][1]['fwd']:8.1f}" for c in CELLS))
allres = {}
for nm in LEVERS + list(BUNDLES):
    names = BUNDLES.get(nm, [nm])
    res = evaluate(names); allres[nm] = res
    wm, wp = weighted(res)
    print(f'{nm:32s} | ' + ' '.join((f"{res[c][0]:+8.1f}" if res[c] else '       -') for c in CELLS) + f' | {wm:+6.1f} {wp:+6.1f}%')
print()
print('iGPU byte floor (fwd if every non-MoE term were hidden) = 40 x lanes x D_moe:')
for c in CELLS:
    P, r = base[c]
    fl = 40 * P['n_lanes'] * P['D_moe'] / 1e3
    print(f"  {c:15s} floor {fl:6.1f} ms vs sim fwd {r['fwd']:6.1f} / meas {P['_meas']['fwd']:6.1f}  -> non-MoE exposed {P['_meas']['fwd']-fl:5.1f} ms ({100*(P['_meas']['fwd']-fl)/P['_meas']['fwd']:.0f}%)")
json.dump({k: {c: (v[c][0] if v[c] else None) for c in CELLS} for k, v in allres.items()}, open('/home/claude-code/.claude/jobs/749c61d3/tmp/pipe/whatif.json', 'w'), indent=1)
