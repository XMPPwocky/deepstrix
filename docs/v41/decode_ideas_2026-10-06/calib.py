#!/usr/bin/env python3
"""Build per-cell simulator parameters from analysis.json (host stamps) + stage_digest.md (device stage us/call),
calibrate against measured fwd / igpu_busy / dgpu_busy / in-loop idle, then run the what-ifs."""
import json, re, sys, statistics as st, random
sys.path.insert(0, '/home/claude-code/.claude/jobs/749c61d3/tmp/pipe')
from sim import run

A = json.load(open('/home/claude-code/.claude/jobs/749c61d3/tmp/pipe/analysis.json'))
dig = open('/home/claude-code/.claude/jobs/749c61d3/tmp/stage_digest.md').read()
# parse digest cells -> {stage: us/call}
D = {}
cur = None
for line in dig.splitlines():
    m = re.match(r'## (\S+) rows=(\d+)', line)
    if m:
        kind, rows = m.group(1), int(m.group(2))
        if kind == 'lone_spec': cur = f'lone_spec_r{rows}'
        elif kind == 'two_spec': cur = f'two_spec_r{rows}'
        else: cur = f'plain_M{rows}'
        D[cur] = {}
        continue
    m = re.match(r'\s+(\S+)\s+([\d.]+) ms/step\s+([\d.]+) calls/step\s+(\d+) us/call', line)
    if m and cur:
        D[cur][m.group(1)] = (float(m.group(2)), float(m.group(3)), int(m.group(4)))

def us(cell, stage, default=0.0):
    return D.get(cell, {}).get(stage, (0, 0, default))[2]

BW = 600e9  # achievable dGPU GB/s for the byte floor
Q8 = 34 / 32  # bytes per param, Q8_0
def mb(params, bpp=Q8): return params * bpp
BYTES_CHAIN = mb(5120 * 1280 + 1280 * 32768) + mb(8 * 4096 * 1024 + 8192 * 5120) + 5120 * 384 * 2 + mb(5120 * 576) + 3e6  # q, wo, router f16, kv_a, mhc/misc
BYTES_SHARED = mb(3 * 5120 * 2304)
FLOOR_CHAIN = BYTES_CHAIN / BW * 1e6; FLOOR_SHARED = BYTES_SHARED / BW * 1e6
ATTN = 75.0

HOST = dict(h_post=20.0, h_chain_enq=70.0, event_lat=15.0, poll=10.0, h_route=60.0, h_prep=70.0, h_launch=60.0, h2d_partial=10.0)

def params(cell, dcell=None):
    a = A[cell]; dcell = dcell or cell
    n = int(round(a['n_lanes_med']))
    rows = int(a['rows_med'])
    d_chain = sum(us(dcell, s) for s in ['dgpu.mhc_pre_attn', 'dgpu.q_chain', 'dgpu.kv_chain', 'dgpu.kv_append_compressor_serial', 'dgpu.attn_compute', 'dgpu.output_proj', 'dgpu.mhc_post_attn', 'dgpu.mhc_pre_ffn', 'dgpu.router', 'dgpu.rb_pack']) + 12
    P = dict(HOST)
    P.update(dict(
        n_lanes=n, lane_rows=[rows // n] * n, ordered=('spec' in cell),
        D_chain=d_chain, D_indexer=us(dcell, 'dgpu.prefill_indexer') - 12, D_engram=us(dcell, 'dgpu.engram'),
        D_shared=us(dcell, 'dgpu.shared_expert') + us(dcell, 'dgpu.mhc_mix_ffn_late'),
        D_combine=us(dcell, 'dgpu.ffn_combine.local') + us(dcell, 'dgpu.ffn_combine.remote'),
        D_push=us(dcell, 'dgpu.peer_push_ffn_input_norm'),
        D_moe=a['d_moe_us']['med'], moe_scale=[1.0] * n, D_pushback=us(dcell, 'igpu.peer_push_ffn_moe'),
        D_head=us(dcell, 'dgpu.head_batch'), epilogue_host=300.0, prologue=550.0,
        engram_join=a['lh_engram_join']['mean'] * 1e3, engram_stage=250.0,
        rtt_samples=[max(50, x) for x in a['rtt_samples_us']], p_box2_idle=a['p_box2_idle'],
        pager_block=a['lh_pager_block']['mean'] * 1e3,
    ))
    # pager block: host stall inside prep, spread per lane-layer
    P['h_prep'] += P['pager_block'] / (40 * n)
    P['_meas'] = dict(fwd=a['fwd']['med'], igpu=a['igpu_busy']['med'], dgpu=a['dgpu_busy']['med'], inloop=a['gap_inloop']['mean'], step=a['step_ms']['med'], fwd_all=a['fwd_all']['med'])
    return P

CELLS = ['plain_M3', 'plain_M4', 'plain_M5', 'plain_M8', 'lone_spec_r3', 'lone_spec_r4', 'lone_spec_r5', 'lone_spec_r6', 'two_spec_r7']
DMAP = {'plain_M3': 'plain_M3', 'plain_M4': 'plain_M4', 'plain_M5': 'plain_M5', 'plain_M8': 'plain_M8'}

if __name__ == '__main__':
    print(f'byte floors (us @600GB/s): chain {FLOOR_CHAIN:.0f} (bytes {BYTES_CHAIN/1e6:.1f} MB), shared {FLOOR_SHARED:.0f} ({BYTES_SHARED/1e6:.1f} MB)')
    print('cell            lanes D_chain D_shared D_moe | meas fwd  sim fwd  err% | meas igpu sim igpu | meas dgpu sim dgpu | inloop meas sim')
    base = {}
    for c in CELLS:
        P = params(c, DMAP.get(c))
        r = run(P, n=30)
        m = P['_meas']
        base[c] = (P, r)
        print(f"{c:15s} {P['n_lanes']}  {P['D_chain']:6.0f} {P['D_shared']:6.0f} {P['D_moe']:6.0f} | {m['fwd']:7.1f} {r['fwd']:7.1f} {100*(r['fwd']/m['fwd']-1):+5.1f} | {m['igpu']:7.1f} {r['igpu_busy']:7.1f} | {m['dgpu']:7.1f} {r['dgpu_busy']:7.1f} | {m['inloop']:5.1f} {r['igpu_idle_inloop']:5.1f}")
    json.dump({c: {k: v for k, v in base[c][0].items() if k != 'rtt_samples'} for c in base}, open('/home/claude-code/.claude/jobs/749c61d3/tmp/pipe/params.json', 'w'), indent=1)
