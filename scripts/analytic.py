#!/usr/bin/env python3
"""Analytic bytes / FLOPs for the V4.1-Flash prefill and decode paths as the
engine runs them today (docs/v41/KERNEL_ROOFLINE.md §1). No measurements.

Model: 40 layers, dim 5120, mHC x4 (residual 20480), 384+1 experts top-6,
inter 2304, MXFP4 experts (17 B / 32), Q8_0 dense projections (34 B / 32),
f16 compressor + router weights, MLA 64 heads x 512 (rope 64, q_lora 1280),
1 KV head, compress ratio 2 on layers 2-19 (stores at 2/8/14), ratio 1 on
20-39 (store at 20), SWA 128, Engram at 1 and 14, vocab 129280.
CED prefill: layers 0-19 over the prompt + layer 20 KV-source-only, then the
last 128 tokens replayed through layers 20-39. Decode: all 40 layers, B=1,
dense attention over the whole compressed store (no indexer on V4.1).
"""
import sys

N_EMBD, N_HC, HC_DIM, HC_MIX = 5120, 4, 20480, 24
N_HEAD, HD, N_ROT, Q_LORA, Q_FLAT = 64, 512, 64, 1280, 32768
N_GROUPS, GROUP_DIM, RANK, OUT_LOW = 8, 4096, 1024, 8192
N_FF, N_EXP, TOPK, VOCAB = 2304, 384, 6, 129280
ENGRAM_IN, ENGRAM_OUT = 6144, 25600
Q8 = 34 / 32
MX = 17 / 32
F16 = 2

def q8(rows, k): return rows * k * Q8
def mx(rows, k): return rows * k * MX

# ---- per-layer dGPU weight bytes (decode reads each once per token) ----
W = {
    "hc_fn x2 (f16)": 2 * HC_MIX * HC_DIM * F16,
    "wq_a": q8(Q_LORA, N_EMBD),
    "wq_b": q8(Q_FLAT, Q_LORA),
    "wkv": q8(HD, N_EMBD),
    "wo_a": q8(N_GROUPS * RANK, GROUP_DIM),
    "wo_b": q8(N_EMBD, OUT_LOW),
    "router gate (f16)": N_EXP * N_EMBD * F16,
    "shared w1,w3,w2": 3 * q8(N_FF, N_EMBD),
}
per_layer_w = sum(W.values())
comp2 = 2 * HD * N_EMBD * F16      # wkv + wgate f16, layers 2/8/14
comp1 = HD * N_EMBD * F16          # wkv f16, layer 20
engram_w = q8(ENGRAM_OUT, ENGRAM_IN)  # layers 1, 14
head_w = q8(VOCAB, N_EMBD)
expert_w = 3 * mx(N_FF, N_EMBD)    # one routed expert (gate, up, down)

def decode(ctx):
    d = {}
    d["dgpu weights/token"] = 40 * per_layer_w + 3 * comp2 + comp1 + 2 * engram_w + head_w
    # attention: score + wsum each read the compressed store once (f16 rows),
    # plus scores f32 [64 x n_total] written once and read twice.
    kv2 = (ctx // 2) * HD * 2
    kv1 = ctx * HD * 2
    att = 0
    for n_comp, layers in ((ctx // 2, 18), (ctx, 20)):
        sb = N_HEAD * (128 + n_comp) * 4
        att += layers * (2 * n_comp * HD * 2 + 3 * sb + 2 * 128 * HD * 2)
    d["dgpu attention bytes/token"] = att
    d["dgpu attention FLOP/token"] = sum(layers * 2 * (2 * N_HEAD * (128 + n_comp) * HD) for n_comp, layers in ((ctx // 2, 18), (ctx, 20)))
    d["dgpu matvec FLOP/token"] = 40 * 2 * (Q_LORA * N_EMBD + Q_FLAT * Q_LORA + HD * N_EMBD + N_GROUPS * RANK * GROUP_DIM + N_EMBD * OUT_LOW + N_EXP * N_EMBD + 3 * N_FF * N_EMBD) + 2 * 2 * VOCAB * N_EMBD / 2 + 2 * 2 * ENGRAM_OUT * ENGRAM_IN
    d["igpu expert weights/token"] = 40 * TOPK * expert_w
    d["igpu expert FLOP/token"] = 40 * TOPK * 2 * 3 * N_FF * N_EMBD
    return d

def prefill(T, chunk_rows=512):
    """One lane-chunk of `chunk_rows` rows at depth T (encoder layers 0-19 +
    layer-20 KV-source pass), plus the 128-row replay through 20-39."""
    B = chunk_rows
    d = {}
    # dGPU GEMM FLOPs per encoder layer per lane-chunk
    gemm = 2 * B * (Q_LORA * N_EMBD + Q_FLAT * Q_LORA + HD * N_EMBD + N_GROUPS * RANK * GROUP_DIM + N_EMBD * OUT_LOW + N_EXP * N_EMBD + 3 * N_FF * N_EMBD + 2 * HC_MIX * HC_DIM)
    d["dgpu GEMM FLOP/lane-chunk (20 enc layers)"] = 20 * gemm
    d["dgpu weight bytes/lane-chunk (20 enc layers)"] = 20 * per_layer_w
    # attention: layers 0,1 SWA; 2-19 dense over n_comp = depth/2
    n_comp = T // 2
    att_fl = 2 * (2 * B * N_HEAD * (128 + n_comp) * HD)  # score + wsum
    d["dgpu attention FLOP/lane-chunk (18 layers)"] = 18 * att_fl + 2 * (2 * B * N_HEAD * 128 * HD * 2)
    d["dgpu attention bytes/lane-chunk (18 layers)"] = 18 * (2 * n_comp * HD * 2 + 2 * B * N_HEAD * (128 + n_comp) * 2 + 2 * B * Q_FLAT * 4)
    # activations that round-trip DRAM per layer (q f32 write+copy+read, heads, casts) ~ B x Q_FLAT x 4 x ~8
    d["dgpu activation bytes/lane-chunk (rough, 20 layers)"] = 20 * B * (Q_FLAT * 4 * 8 + HC_DIM * 4 * 10)
    # iGPU: every expert is touched by a 512-row lane-chunk -> full weight stream
    d["igpu expert bytes/lane-chunk (20 layers, all 384 experts)"] = 20 * N_EXP * expert_w
    d["igpu expert FLOP/lane-chunk (20 layers)"] = 20 * B * TOPK * 2 * 3 * N_FF * N_EMBD
    # replay: 128 rows through layers 20-39 with n_comp = T (ratio 1)
    Br = 128
    d["replay dgpu GEMM FLOP (20 dec layers, 128 rows)"] = 20 * gemm * Br / B
    d["replay dgpu attention FLOP (20 layers, n_comp=T)"] = 20 * 2 * (2 * Br * N_HEAD * (128 + T) * HD)
    d["replay dgpu weight bytes (20 dec layers)"] = 20 * per_layer_w
    d["replay igpu expert bytes (20 layers, ~min(384, 768 picks) experts)"] = 20 * min(N_EXP, Br * TOPK) * expert_w * (1 - (1 - 1 / N_EXP) ** (Br * TOPK)) / min(1, Br * TOPK / N_EXP)
    return d

def fmt(v, key):
    if "FLOP" in key:
        return f"{v/1e9:10.1f} GFLOP"
    return f"{v/1e9:10.3f} GB"

if __name__ == "__main__":
    bw_d, bw_i = float(sys.argv[1]) if len(sys.argv) > 1 else 560.0, float(sys.argv[2]) if len(sys.argv) > 2 else 214.0
    pk_d_wmma, pk_d_dp4a, pk_i_dp4a = 100.0, 100.0, 45.0
    print("per-layer dGPU dense weights (decode reads once per token):")
    for k, v in W.items():
        print(f"  {k:<22} {v/1e6:8.2f} MB")
    print(f"  {'total/layer':<22} {per_layer_w/1e6:8.2f} MB ; compressor r2 {comp2/1e6:.2f} MB x3, r1 {comp1/1e6:.2f} MB, engram {engram_w/1e6:.1f} MB x2, head {head_w/1e6:.1f} MB")
    print(f"one routed expert (gate+up+down MXFP4): {expert_w/1e6:.2f} MB; 6 experts {6*expert_w/1e6:.1f} MB/layer; all 384: {N_EXP*expert_w/1e9:.2f} GB/layer")
    for ctx in (8192, 32768, 102400):
        d = decode(ctx)
        print(f"\n== decode @ ctx {ctx} ==")
        for k, v in d.items():
            print(f"  {k:<36} {fmt(v, k)}")
        t_d = (d["dgpu weights/token"] + d["dgpu attention bytes/token"]) / (bw_d * 1e9)
        t_i = d["igpu expert weights/token"] / (bw_i * 1e9)
        print(f"  BW floor: dGPU {t_d*1e3:.1f} ms + iGPU {t_i*1e3:.1f} ms (serial per layer) = {1e3*(t_d+t_i):.1f} ms -> {1/(t_d+t_i):.1f} tok/s; if MoE overlapped shared+attn perfectly: max = {1e3*max(t_d,t_i):.1f} ms")
    for T in (4096, 32768, 102400):
        d = prefill(T)
        print(f"\n== prefill lane-chunk of 512 rows at depth {T} ==")
        for k, v in d.items():
            print(f"  {k:<64} {fmt(v, k)}")
        t_ig = d["igpu expert bytes/lane-chunk (20 layers, all 384 experts)"] / (bw_i * 1e9)
        t_dg_w = d["dgpu weight bytes/lane-chunk (20 enc layers)"] / (bw_d * 1e9)
        t_dg_att = d["dgpu attention FLOP/lane-chunk (18 layers)"] / (pk_d_wmma * 1e12)
        t_dg_gemm = d["dgpu GEMM FLOP/lane-chunk (20 enc layers)"] / (pk_d_wmma * 1e12)
        t_dg_act = d["dgpu activation bytes/lane-chunk (rough, 20 layers)"] / (bw_d * 1e9)
        print(f"  floors @ {bw_d:.0f}/{bw_i:.0f} GB/s, {pk_d_wmma:.0f} TF wmma: iGPU expert stream {t_ig*1e3:.0f} ms; dGPU weights {t_dg_w*1e3:.0f} + GEMM {t_dg_gemm*1e3:.0f} + attention {t_dg_att*1e3:.0f} + activations {t_dg_act*1e3:.0f} ms")
        lane = max(t_ig, t_dg_w + t_dg_gemm + t_dg_att + t_dg_act)
        print(f"  -> 2-lane pipelined ceiling ~ {512/lane:.0f} tok/s (iGPU-bound: {t_ig >= t_dg_w + t_dg_gemm + t_dg_att + t_dg_act})")
