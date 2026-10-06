#!/usr/bin/env python3
"""Per-regime timeline analysis of hub_req host stamps (ev000/ev001.json from extract.py).
Per lane-layer: A = wait_exit - submit (submit -> Post start), B = next submit (same lane) - wait_exit
(Post host + combine/chain enqueue + wait selected_ready + route/prep), rtt = t4 - submit, slack = wait_enter - t4.
iGPU FIFO reconstruction: launch_k = submit_k + DELTA, dur = per-step igpu compute busy / n lane-layers.
Idle attribution per gap: prologue / epilogue / box2 (next lane still waiting for its reply) / chain (handoff: chain+readback+host).
"""
import json, math, sys, statistics as st
from collections import defaultdict

DELTA_US = 60.0  # submit -> iGPU MoE launch (prep remainder; host enqueue)
files = sys.argv[1:] or ['ev000.json', 'ev001.json']
cells = defaultdict(list)
for fn in files:
    D = json.load(open(fn))
    sf = D['step_fields']; si = {n: i for i, n in enumerate(sf)}
    for k, v in D['cells'].items():
        cells[k].extend(v)
rf = D['req_fields']; ri = {n: i for i, n in enumerate(rf)}

def med(v):
    v = [x for x in v if x == x]
    return st.median(v) if v else float('nan')
def p(v, q):
    v = sorted(x for x in v if x == x)
    if not v: return float('nan')
    i = (len(v) - 1) * q; lo = int(i); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)
def mean(v):
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float('nan')

RTT = defaultdict(list); NLANES = defaultdict(list); ROWS = defaultdict(list)
B_FILL = {}
WANT = ['plain_M1', 'plain_M2', 'plain_M3', 'plain_M4', 'plain_M5', 'plain_M8', 'lone_spec_r3', 'lone_spec_r4', 'lone_spec_r5', 'lone_spec_r6', 'two_spec_r7']
# first pass: B_FILL = median B over consecutive present layers (MoE surely done or not; use all)
for cell in WANT:
    v = []
    for s in cells.get(cell, []):
        reqs = s['reqs']
        for L in set(int(r[ri['lane']]) for r in reqs):
            rs = sorted([r for r in reqs if int(r[ri['lane']]) == L], key=lambda r: r[ri['layer']])
            for j in range(len(rs) - 1):
                if int(rs[j + 1][ri['layer']]) == int(rs[j][ri['layer']]) + 1:
                    v.append((rs[j + 1][ri['t_submit']] - rs[j][ri['t_wait_exit']]) / 1e3)
    if v: B_FILL[cell] = med(v)
out = {}
for cell in WANT:
    steps = cells.get(cell, [])
    agg = defaultdict(list)
    per_layer_B = defaultdict(list)
    per_layer_gap = defaultdict(list)
    n_used = 0
    for s in steps:
        stv = s['step']; reqs = s['reqs']
        if not reqs: continue
        t0 = stv[si['t_start']]; t1 = stv[si['t_end']]
        fwd = stv[si['fwd_ms']]; fwd_all = stv[si['fwd_all_ms']]
        lanes = sorted(set(int(r[ri['lane']]) for r in reqs))
        nl = len(lanes)
        # requests by lane, sorted by layer
        by_lane = {L: sorted([r for r in reqs if int(r[ri['lane']]) == L], key=lambda r: r[ri['layer']]) for L in lanes}
        if any(len(v) < 30 for v in by_lane.values()):
            continue
        n_used += 1
        nll = 40 * nl
        n_missing = nll - len(reqs)
        agg['n_missing'].append(n_missing)
        igpu_busy = stv[si['igpu_busy_ms']]; dgpu_busy = stv[si['dgpu_busy_ms']]
        i_push = stv[si['i_peer_push_ffn_moe']]
        i_mtp = stv[si['i_mtp']]
        comp = igpu_busy - (i_push if i_push == i_push else 0) - (i_mtp if i_mtp == i_mtp else 0)
        d_moe_us = comp * 1e3 / nll
        agg['fwd'].append(fwd); agg['fwd_all'].append(fwd_all); agg['step_ms'].append(stv[si['step_ms']])
        agg['igpu_busy'].append(igpu_busy); agg['dgpu_busy'].append(dgpu_busy); agg['d_moe_us'].append(d_moe_us)
        agg['head'].append(fwd_all - fwd)
        for k in ['lh_engram_join', 'lh_pager_block', 'lh_sel_sync', 'lh_remote_wait', 'remote_rtt_ms', 'b2_page_ms', 'hop_slack_us', 'rf_chain_wait_us', 'i_pair_kwide', 'i_q2k_down', 'd_output_proj', 'd_q_chain', 'd_shared_expert', 'd_attn_compute', 'd_peer_push_ffn_input_norm', 'd_mhc_pre_attn', 'd_router', 'd_kv_chain', 'd_mhc_mix_ffn_late', 'd_kv_append_compressor_serial', 'd_head_batch', 'd_engram', 'd_prefill_indexer', 'd_prefill_indexer_reuse', 'd_mhc_pre_ffn', 'd_mhc_post_attn', 'd_ffn_combine_local', 'd_ffn_combine_remote', 'd_rb_pack', 'd_other', 'i_other', 'd_mtp', 'i_mtp', 'engram_ms', 'sample_ms']:
            agg[k].append(stv[si[k]])
        # per lane-layer intervals (us)
        all_launch = []
        for L in lanes:
            rs = by_lane[L]
            for j, r in enumerate(rs):
                S = r[ri['t_submit']]; C = r[ri['t_wait_exit']]; T4 = r[ri['t4']]; WE = r[ri['t_wait_enter']]
                agg['A'].append((C - S) / 1e3); agg['rtt'].append((T4 - S) / 1e3); agg['slack'].append((WE - T4) / 1e3)
                agg['post_host'].append((C - WE) / 1e3)
                agg['submit_host'].append((r[ri['t_submit_end']] - S) / 1e3)
                lj = int(r[ri['layer']])
                if j + 1 < len(rs) and int(rs[j + 1][ri['layer']]) == lj + 1:
                    S2 = rs[j + 1][ri['t_submit']]
                    B = (S2 - C) / 1e3
                    agg['B'].append(B); per_layer_B[lj + 1].append(B)
                    agg['P'].append((S2 - S) / 1e3)
            # full 40-layer launch list with synthetic entries for layers that sent nothing to box 2
            present = {int(r[ri['layer']]): r for r in rs}
            prevC = None
            for lj in range(40):
                if lj in present:
                    r = present[lj]; S = r[ri['t_submit']]; C = r[ri['t_wait_exit']]; T4 = r[ri['t4']]
                    all_launch.append((S + DELTA_US * 1e3, L, lj, C, T4, prevC, 0))
                    prevC = C
                else:
                    if prevC is None:
                        continue
                    S = prevC + B_FILL.get(cell, 700.0) * 1e3; C = S + 80e3
                    all_launch.append((S + DELTA_US * 1e3, L, lj, C, None, prevC, 1))
                    prevC = C
        all_launch.sort()
        # iGPU FIFO reconstruction
        t = all_launch[0][0]
        busy = 0.0; gaps = defaultdict(float)
        first_launch = all_launch[0][0]
        gaps['prologue'] = (first_launch - t0) / 1e6
        moe_end = {}
        prev_end = None
        # alternation period: consecutive launches
        for k in range(1, len(all_launch)):
            agg['alt_period'].append((all_launch[k][0] - all_launch[k - 1][0]) / 1e6)
        b2_exposed_step = 0.0
        for (lt, L, j, C, T4, prevC, synth) in all_launch:
            start = max(lt, t)
            if prev_end is not None and lt > prev_end:
                gap = (lt - prev_end) / 1e6
                if prevC is None:
                    gaps['layer0_sync'] += gap
                elif prevC > prev_end:
                    b2 = min(prevC, lt) - prev_end
                    gaps['box2'] += b2 / 1e6; gaps['chain'] += (lt - prevC) / 1e6
                else:
                    gaps['chain'] += gap
                if j in (1, 14):
                    gaps['at_engram_layers'] += gap
                per_layer_gap[j].append(gap)
            end = start + d_moe_us * 1e3
            moe_end[(L, j)] = end
            busy += d_moe_us
            t = end; prev_end = end
            # box-2 exposure: reply after this lane-layer's MoE end
            if T4 is not None and T4 > end:
                b2_exposed_step += (T4 - end) / 1e6
        gaps['epilogue'] = (t1 - prev_end) / 1e6
        last_C = max(r[ri['t_wait_exit']] for r in reqs)
        agg['tail_after_last_post'].append((t1 - last_C) / 1e6)
        agg['b2_reply_after_moe_end'].append(b2_exposed_step)
        agg['n_b2_late'].append(sum(1 for (lt, L, j, C, T4, pc, sy) in all_launch if T4 is not None and T4 > moe_end[(L, j)]))
        RTT[cell].extend(int(r[ri['t4']] - r[ri['t_submit']]) // 1000 for r in reqs)
        NLANES[cell].append(nl); ROWS[cell].append(int(stv[si['rows']]))
        for k, v in gaps.items(): agg['gap_' + k].append(v)
        agg['gap_inloop'].append(gaps['box2'] + gaps['chain'] + gaps['layer0_sync'])
        agg['igpu_idle_total'].append(stv[si['step_ms']] - comp)
        # B conditioned: lane-layers where the reply came late enough that MoE was surely done at Post:
        for L in lanes:
            rs = by_lane[L]
            for j in range(len(rs) - 1):
                lj = int(rs[j][ri['layer']])
                if int(rs[j + 1][ri['layer']]) != lj + 1: continue
                C = rs[j][ri['t_wait_exit']]; S2 = rs[j + 1][ri['t_submit']]
                me = moe_end[(L, lj)]
                if C >= me:
                    agg['B_moe_done'].append((S2 - C) / 1e3)
                else:
                    agg['B_moe_pending'].append((S2 - C) / 1e3)
                    agg['B_minus_moe_remaining'].append((S2 - me) / 1e3)
    if not n_used:
        continue
    summ = {'n_steps': n_used}
    for k, v in agg.items():
        summ[k] = {'med': med(v), 'mean': mean(v), 'p10': p(v, .1), 'p90': p(v, .9), 'n': len([x for x in v if x == x])}
    rr = RTT[cell]; import random; random.Random(1).shuffle(rr)
    summ['rtt_samples_us'] = rr[:4000]
    summ['p_box2_idle'] = mean(agg['n_missing']) / (40 * med(NLANES[cell]))
    summ['n_lanes_med'] = med(NLANES[cell]); summ['rows_med'] = med(ROWS[cell])
    summ['B_by_layer'] = {j: round(med(v), 3) for j, v in sorted(per_layer_B.items())}
    summ['gap_by_layer'] = {j: round(mean(v), 3) for j, v in sorted(per_layer_gap.items())}
    out[cell] = summ

json.dump(out, open('analysis.json', 'w'), indent=1)

def row(cell, keys, fmt='{:7.2f}'):
    s = out[cell]
    return ' '.join(fmt.format(s[k]['med']) if k in s else '      -' for k in keys)

print('cell            n   fwd  igpu_b dgpu_b d_moe_us |  A_med rtt_med slack  B_med  P_med altper | Bdone Bpend')
for cell in out:
    s = out[cell]
    print(f"{cell:15s} {s['n_steps']:3d} " + row(cell, ['fwd', 'igpu_busy', 'dgpu_busy']) + f" {s['d_moe_us']['med']:7.0f} | " + row(cell, ['A', 'rtt', 'slack', 'B', 'P', 'alt_period'], '{:6.3f}') + ' | ' + row(cell, ['B_moe_done', 'B_moe_pending'], '{:6.3f}'))
print()
print('iGPU idle (ms/step, MEAN): prologue layer0 box2 chain engramL epilogue inloop | total(step-comp)  tail_after_last_post head  engram_join pager_block  b2_late_ms n_late')
for cell in out:
    s = out[cell]
    g = lambda k: s[k]['mean'] if k in s else float('nan')
    print(f"{cell:15s} {g('gap_prologue'):6.2f} {g('gap_layer0_sync'):6.2f} {g('gap_box2'):6.2f} {g('gap_chain'):6.2f} {g('gap_at_engram_layers'):6.2f} {g('gap_epilogue'):6.2f} {g('gap_inloop'):6.2f} | {g('igpu_idle_total'):6.2f}  {g('tail_after_last_post'):6.2f} {g('head'):5.2f}  {g('lh_engram_join'):5.2f} {g('lh_pager_block'):5.2f}  {g('b2_reply_after_moe_end'):6.2f} {g('n_b2_late'):5.1f}")
print()
for cell in ['lone_spec_r4', 'plain_M3', 'plain_M8', 'two_spec_r7']:
    if cell in out:
        print(cell, 'B by layer (med ms):', out[cell]['B_by_layer'])
        print(cell, 'gap by layer (mean ms):', out[cell]['gap_by_layer'])
