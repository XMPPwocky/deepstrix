#!/usr/bin/env python3
"""Multi-stream decode step model, rev 1 (after the 2026-09-19 review).
Usage: multistream_plan_model.py SIM_JSON [E_dspark=2.9] [prefill=y|n]

A STEP runs R = S*K rows through 40 layers in lockstep. Per layer:

    dense(R) = d0 + d1*R      dGPU attention/router/shared chain, launch-count bound
                              (d0 MEASURED 22 ms/token; d1 ESTIMATED 8 us/row/layer)
    attn(R)  = a1*R           per-row KV work batching cannot amortise
                              (ESTIMATED 15 us/row/layer at ~100K; upper bound 58)
    leg_b    per box b in {1: hub iGPU, 2: box 2 (+link)}:
        compute_b = 105 + 20*R + (18.8 MB / bw_moe) * U_b            [box-2 fit, one
                    kernel family on both gfx1151 parts; bw_moe ESTIMATED 200 GB/s]
        local extras (box 1 only): 0.07 ms + 0.002*R  (ensure hit path, pushes)
        read_b    = lat_b + k_b * 18.8 MB / ssd_b     when the layer has a miss;
                    k_b = E[misses | layer stalled] = (m_b/40)/P_b   [SIMULATED]
        hits-first: leg_b = compute_b + P_b*(max(0, read_b - compute_b) + launch2)
        lockstep  : leg_b = compute_b + P_b*read_b                     (today's code)
        link (box 2 only) = 90 us + bytes/link_bw [+ 270 us hold when the reply
                    exceeds one 65,520-B segment and the busy-poll hold is unfixed]
    layer = dense + attn + max(leg_1, leg_2)
    step  = 40*layer + engram(R) + glue(R)
        engram: rows_for_chunk ESTIMATED 0.6 ms/row exposed in v1; 0 once issued at
                sampling time for the next step (Engram layers are 1 and 14)
        glue:   5 ms + 0.1 ms/row (sampling, demux, HTTP)

Union and miss statistics come from scratchpad simms2.py (distinct-request streams,
prefill rows replayed through the same pools), keyed by (S, K, prefill).
"""
import sys, json

N_LAYER = 40
EXP_MB = 18.8

BASE = dict(d0=0.55, d1=0.008, a1=0.015, bw_moe=200.0, link_fixed=90.0, link_bw=1.1e3, link_hold=270.0,
            f16=False, ssd1=2.4e3, lat1=1.0, ssd2=4.47e3, lat2=0.5, hits_first=False, launch2=0.15,
            engram_row=0.6, glue0=5.0, glue1=0.1, three_way=False, zero_miss=False, exp_mb=EXP_MB)

def leg(compute, P, m, ssd, lat, hw):
    if P <= 0 or hw['zero_miss']: return compute, 0.0
    k = (m / N_LAYER) / P
    read = lat + k * hw['exp_mb'] / ssd * 1000.0
    if hw['hits_first']:
        exposed = P * (max(0.0, read - compute) + hw['launch2'])
    else:
        exposed = P * read
    return compute + exposed, exposed

def step_ms(r, hw):
    S, K, R = r['S'], r['K'], r['rows']
    U1, U2, m1, m2, P1, P2 = r['u1'], r['u2'], r['m1step'], r['m2step'], r['p1'], r['p2']
    if hw['three_way']:
        U = U1 + U2; U1, U2 = 0.29 * U, 0.355 * U   # slot-proportional: 4,454 / (4,454+6,160+6,160) = 0.27-0.29 for the hub
    c_exp = hw['exp_mb'] / hw['bw_moe']            # ms per distinct expert
    comp1 = (105 + 20 * R) / 1000.0 + c_exp * U1 + 0.07 + 0.002 * R
    comp2 = (105 + 20 * R) / 1000.0 + c_exp * U2
    leg1, ex1 = leg(comp1, P1, m1, hw['ssd1'], hw['lat1'], hw)
    leg2, ex2 = leg(comp2, P2, m2, hw['ssd2'], hw['lat2'], hw)
    resp = R * (10240 if hw['f16'] else 20480)
    link = hw['link_fixed'] / 1000.0 + (R * 5840 + resp) / (hw['link_bw'] * 1e3)
    if hw['link_hold'] > 0 and resp > 65520: link += hw['link_hold'] / 1000.0
    leg2 += link
    dense = hw['d0'] + hw['d1'] * R
    attn = hw['a1'] * R
    layer = dense + attn + max(leg1, leg2)
    total = N_LAYER * layer + hw['engram_row'] * R + hw['glue0'] + hw['glue1'] * R
    return total, dict(dense=dense * N_LAYER, attn=attn * N_LAYER, leg1=leg1 * N_LAYER, leg2=leg2 * N_LAYER,
                       link=link * N_LAYER, ex1=ex1 * N_LAYER, ex2=ex2 * N_LAYER, engram=hw['engram_row'] * R)

SCEN = [
 ("A0 today's code on today's hardware: lockstep reads, busy-poll hold, engram per row", dict()),
 ("A1 A0 + HITS-FIRST MoE (software; validate on the daemon first)",                     dict(hits_first=True)),
 ("A2 A1 + engram issued at sample time + multi-segment hold fixed",                     dict(hits_first=True, engram_row=0.0, link_hold=0.0)),
 ("B  A2 + 7 GB/s plaintext NVMe in BOTH boxes + concurrent daemon reads",               dict(hits_first=True, engram_row=0.0, link_hold=0.0, ssd1=6.5e3, lat1=0.5, ssd2=6.5e3, lat2=0.4)),
 ("B2 B + two NVMe per box (RAID0, 13 GB/s)",                                            dict(hits_first=True, engram_row=0.0, link_hold=0.0, ssd1=13e3, lat1=0.4, ssd2=13e3, lat2=0.3)),
 ("C  B + f16 partials + 2nd USB4 link (2.2 GB/s)",                                      dict(hits_first=True, engram_row=0.0, link_hold=0.0, ssd1=6.5e3, lat1=0.5, ssd2=6.5e3, lat2=0.4, f16=True, link_bw=2.2e3)),
 ("B1 A2 + 7 GB/s plaintext NVMe on BOX 2 ONLY + concurrent daemon reads",                 dict(hits_first=True, engram_row=0.0, link_hold=0.0, ssd2=6.5e3, lat2=0.4)),
 ("E  A2 + THIRD 128 GB box: zero misses, hub 29% / remotes 35.5%+35.5% of the union",   dict(hits_first=True, engram_row=0.0, link_hold=0.0, zero_miss=True, three_way=True)),
 ("F  A2 + IQ2_S requant on 2 boxes: zero misses, 11.06 MB/expert",                      dict(hits_first=True, engram_row=0.0, link_hold=0.0, zero_miss=True, exp_mb=11.06)),
 ("S  B + dGPU chain 22 -> 12 ms (software, not built)",                                 dict(hits_first=True, engram_row=0.0, link_hold=0.0, ssd1=6.5e3, lat1=0.5, ssd2=6.5e3, lat2=0.4, d0=0.30)),
]

def price_files(files, scen_names=("A1", "B", "B2")):
    """Price extra simulation files (share / slot-count variants) under a few scenarios."""
    for f in files:
        rows = [r for r in json.load(open(f)) if r['prefill']]
        print(f"\n## {f}")
        for name, over in SCEN:
            if not any(name.startswith(n + " ") for n in scen_names): continue
            hw = dict(BASE); hw.update(over)
            out = []
            for r in rows:
                ms, p = step_ms(r, hw)
                out.append(f"S={r['S']}: {ms:.0f} ms, leg1 {p['leg1']:.0f} leg2 {p['leg2']:.0f} -> {r['S']*1000/ms:.1f} tok/s")
            print(f"  {name[:2]}: " + " | ".join(out))

def main():
    if sys.argv[1] == '--rows':
        price_files(sys.argv[2:]); return
    sim = json.load(open(sys.argv[1]))
    E = float(sys.argv[2]) if len(sys.argv) > 2 else 2.9
    pf = (sys.argv[3] if len(sys.argv) > 3 else 'y') == 'y'
    rows = [r for r in sim if r['prefill'] == pf]
    PF_ROWS = 5.1      # MEASURED on the trace: 97,559 encoder rows + 1,186 replay rows per 19,139 decode tokens
    CHUNK_RATE = 500.0 # ESTIMATED rows/s for a chunk run as its own forward (prefill measures 400-700 tok/s)
    print(f"prefill rows in the pools: {'YES' if pf else 'no'};  DSpark E = {E} accepted tokens per K=5 block")
    print(f"'aggr+pf' = aggregate when each decode token also carries {PF_ROWS} prefill rows at {CHUNK_RATE:.0f} rows/s in SEPARATE forwards (the trace's agent workload)")
    for name, over in SCEN:
        hw = dict(BASE); hw.update(over)
        print(f"\n== {name}")
        print(f"{'S':>3} {'K':>2} {'rows':>4} | {'step':>5} | {'dense':>5} {'attn':>4} {'leg1':>5} {'leg2':>5} {'(link':>5} {'ex1':>5} {'ex2)':>5} {'engr':>4} | {'/stream':>7} {'aggr':>6} | {'aggr+pf':>7} {'/str+pf':>7}")
        for r in rows:
            ms, p = step_ms(r, hw)
            tok_per_step = (1.0 if r['K'] == 1 else E)
            per = tok_per_step / ms * 1000
            # prefill serialized with decode: every decode token drags PF_ROWS rows of chunk work
            ms_pf = ms + r['S'] * tok_per_step * PF_ROWS / CHUNK_RATE * 1000
            per_pf = tok_per_step / ms_pf * 1000
            print(f"{r['S']:>3} {r['K']:>2} {r['rows']:>4} | {ms:>5.0f} | {p['dense']:>5.1f} {p['attn']:>4.1f} {p['leg1']:>5.1f} {p['leg2']:>5.1f} {p['link']:>5.1f} {p['ex1']:>5.1f} {p['ex2']:>5.1f} {p['engram']:>4.1f} | {per:>7.1f} {per*r['S']:>6.1f} | {per_pf*r['S']:>7.1f} {per_pf:>7.1f}")
    print("\n(share/slot variants are SIMULATED, not scaled: price them with --rows simms2_base.json simms2_s47.json simms2_swap.json simms2_swap47.json)")

if __name__ == "__main__":
    main()
