#!/usr/bin/env python3
# Itemised 1024-row prefill set from batch_scratch.rs formulas (HEAD c53b2a45) + FFD placement.
import math
MB = 1e6
rows = 1024
N_EMBD, HC_DIM, HC_MIX, Q_FLAT, N_LORA_Q, OUT_LOW = 5120, 20480, 24, 32768, 1280, 8192
N_FF_SHARED, N_FF_EXP, N_EXPERT, NU, N_HEAD, HD = 2304, 2304, 384, 6, 64, 512
ENGRAM_IN, ENGRAM_OUT, TOP_K, IDX_HD = 6144, 25600, 512, 32 * 128
BQ8K_GATE, BQ8K_DOWN, Q8K_BYTES = N_EMBD // 256, N_FF_EXP // 256, 292
MTP_CAP_ROWS, MTP_SRC_LAYERS, ROUTER_MAX_ALT, SWA = 128, 3, 4, 128
VISION_MAX = 512  # IMAGE_RAW_WINDOW_MAX = SWA + VISION_MAX_N_TOKEN (assumed 512)
MAX_KEYS = 368_640 + SWA + VISION_MAX
ATTN_SCORES_STRIDE = 3072
CAND_BLOCK = 8
engram_chunk = 64

def align256(b): return (b + 255) // 256 * 256
def topk_levels(n, k, sort_n=4096):
    if n <= sort_n: return []
    gs = max(sort_n // k, 1) * k
    lv = [math.ceil(n / sort_n) * k]
    while lv[-1] > sort_n:
        nx = math.ceil(lv[-1] / gs) * k
        if nx >= lv[-1]: break
        lv.append(nx)
    return lv

lane = {
 'indexer_sel_saved': rows*TOP_K*4, 'candidate_block_score': rows*math.ceil(MAX_KEYS/CAND_BLOCK)*4,
 'residual': rows*HC_DIM*4, 'residual_next': rows*HC_DIM*4, 'after_attn_hc': rows*HC_DIM*4,
 'engram_rows': rows*ENGRAM_IN*4, 'engram_xq': engram_chunk*ENGRAM_IN, 'engram_xscale': engram_chunk*ENGRAM_IN//32*4,
 'engram_kv': engram_chunk*ENGRAM_OUT*4, 'mtp_src': MTP_CAP_ROWS*MTP_SRC_LAYERS*N_EMBD*4,
 'ffn_input_norm': rows*N_EMBD*4, 'ffn_shared': rows*N_EMBD*4, 'ffn_moe_recv': rows*N_EMBD*4,
 'remote_ffn_moe': rows*N_EMBD*4, 'remote_xq_lane': rows*BQ8K_GATE*Q8K_BYTES,
 'small(d_sel/ew/alts/prior/pos/split/carry)': rows*(NU*4*4 + ROUTER_MAX_ALT*8 + 4 + 4 + HC_MIX*8) + N_EXPERT*4 + MTP_CAP_ROWS*4*4,
}
mb = rows  # max_boundaries = rows.div_ceil(min_ratio=1)
r1 = max(align256(x) for x in [rows*HC_DIM*4, rows*Q_FLAT*4, rows*MAX_KEYS*4, rows*Q_FLAT*4])
r3 = max(align256(rows*IDX_HD*4), align256(rows*sum(topk_levels(MAX_KEYS, TOP_K))*4))
legacy_keys = rows*N_HEAD*ATTN_SCORES_STRIDE
shared = {
 'r1_arena(indexer_scores)': r1, 'r3_arena(topk scratch)': r3, 'q_normed(R2)': rows*Q_FLAT*4,
 'mhc_*': 16*HC_DIM*4 + rows*8 + 64, 'mix': rows*HC_MIX*4,
 'attn_cur': rows*N_EMBD*4, 'attn_input_norm': rows*N_EMBD*4, 'ffn_cur': rows*N_EMBD*4,
 'kq_attn_q8k': rows*BQ8K_GATE*Q8K_BYTES, 'kq_ffn_q8k': rows*BQ8K_GATE*Q8K_BYTES, 'kq_mid_q8k': rows*BQ8K_DOWN*Q8K_BYTES,
 'xq_n_embd': rows*N_EMBD, 'x16_n_embd': rows*N_EMBD*2, 'qr16': rows*N_LORA_Q*2, 'heads16': rows*Q_FLAT*2,
 'low16': rows*OUT_LOW*2, 'mid_sh16': rows*N_FF_SHARED*2, 'xscale_n_embd': rows*(N_EMBD//32)*4,
 'qr': rows*N_LORA_Q*4, 'qr_normed': rows*N_LORA_Q*4, 'qr_xq': rows*N_LORA_Q, 'qr_xscale': rows*(N_LORA_Q//32)*4,
 'kv_raw': rows*HD*4, 'kv_normed': rows*HD*4, 'kv_ring_scratch': max(rows, SWA)*HD*2,
 'kv_cur': rows*2*HD*4, 'sc_cur': rows*2*HD*4, 'per_b i32 x5': rows*4*5,
 'verify_scores(DEAD)': N_HEAD*MAX_KEYS*4, 'verify_partials': N_HEAD*16*HD*4,
 'attn_scores': (legacy_keys // 2) * 4,
 'indexer_q16': rows*IDX_HD*2, 'indexer_head_weights': rows*32*4, 'indexer_selected': rows*TOP_K*4,
 'attn_active_comp_kv': rows*TOP_K*HD*2,
 'comp_state_kv_snapshots': max(mb*8*2*HD, 128*HD)*4, 'comp_state_score_snapshots': max(mb*8*2*HD, 128*HD)*4,
 'comp_pooled_batched': mb*HD*4, 'comp_rows_batched': mb*HD*4,
 'index_k_rows_batched': mb*128*4, 'index_k_normed_batched': mb*128*4,
 'router_logits x2': 2*rows*N_EXPERT*4, 'look_*': rows*NU*4*4,
 'gate_sh': rows*N_FF_SHARED*4, 'up_sh': rows*N_FF_SHARED*4, 'mid_sh': rows*N_FF_SHARED*4,
 'mid_sh_xq': rows*N_FF_SHARED, 'mid_sh_xscale': rows*(N_FF_SHARED//32)*4,
}
lane_tot = sum(lane.values()); sh_tot = sum(shared.values())
print(f"lane  @1024: {lane_tot/MB:8.1f} MB  (x2 = {2*lane_tot/MB:.1f})")
print(f"shared@1024: {sh_tot/MB:8.1f} MB   r1={r1/MB:.1f} r3={r3/MB:.1f} attn_scores={shared['attn_scores']/MB:.1f} active={shared['attn_active_comp_kv']/MB:.1f}")
print(f"set total  : {(2*lane_tot+sh_tot)/MB:8.1f} MB")

items = [(f"sd:{k}", v) for k, v in shared.items() if 'DEAD' not in k]
for L in ('a', 'b'):
    items += [(f"bd_{L}:{k}", v) for k, v in lane.items()]
items.append(('lm_store+lm_idx(4096 rows)', 4096*(HC_DIM+HC_MIX)*4 + 4096*TOP_K*4 + 4096*4))
# bins: pool first (r1 + lm_store forced), then typical free tails after compaction
pool = 2048*MB
caps = {'comp_kv1': 1331.2, 'comp_kv2a': 665.6, 'comp_kv2b': 665.6, 'comp_kv2c': 665.6,
        'index_k1': 104.0, 'index_k2a': 52.0, 'index_k2b': 52.0, 'index_k2c': 52.0}
tot = sum(caps.values()); live = 800.0
tails = {k: (v - live*v/tot)*MB for k, v in caps.items()}
bins = [('pool', pool)] + list(tails.items())
free = {n: c for n, c in bins}; place = {n: [] for n, _ in bins}
forced = ['sd:r1_arena(indexer_scores)', 'lm_store+lm_idx(4096 rows)']
rest = []
for name, sz in items:
    if name in forced:
        free['pool'] -= sz; place['pool'].append((name, sz))
    else:
        rest.append((name, sz))
rest.sort(key=lambda t: -t[1])
unplaced = []
for name, sz in rest:
    for bn, _ in bins:
        if free[bn] >= sz:
            free[bn] -= sz; place[bn].append((name, sz)); break
    else:
        unplaced.append((name, sz))
for bn, cap in bins:
    print(f"{bn:10s} cap {cap/MB:7.1f} MB  left {free[bn]/MB:7.1f} MB  items: " + ', '.join(f"{n.split(':')[-1]}={s/MB:.0f}" for n, s in place[bn]))
print("UNPLACED:", [(n, round(s/MB, 1)) for n, s in unplaced])
print(f"tails total {sum(tails.values())/MB:.0f} MB; needed outside pool {sum(s for n,s in rest)/MB:.0f} MB; slack {(sum(tails.values()) - sum(s for n,s in rest) - (free['pool'] if free['pool']<0 else 0))/MB:.0f} MB")
