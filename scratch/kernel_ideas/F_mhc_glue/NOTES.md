# F_mhc_glue — dGPU mHC / norms / router top-k / glue (gfx1201)

Working notes, written as I go. Commit base 361d4f9 (= production hub). All GPU runs via
`_infra/gpu_run.sh --dev dgpu`. Free dGPU VRAM at start: ~438 MB, so the practical cap is
~165 MB per run (gpu_run wants free - mb - 270 >= 0).

## 0. Production launch geometry (from the Rust wrappers, read 2026-09-26)

Constants (config.rs, v41): N_EMBD 5120, N_HC 4, HC_DIM 20480, HC_MIX_DIM 24, N_EXPERT 384,
N_EXPERT_USED 6, RMS_EPS 1e-20, SINKHORN_EPS 1e-6, SINKHORN_ITERS 20, EXPERT_WEIGHT_SCALE 1.5,
ROUTER_WEIGHT_EPS 6.103515625e-5 (forward_prefill.rs:50). ROUTER_MAX_EXPERTS 512 (build flag).

Production env (read from the live server's /proc/<pid>/environ): V41_SUB=3 (cache-prior),
V41_SUB_PROTECT=2, V41_SUB_LAMBDA=0.25, V41_PICK_TRACE set, V41_ROUTER_ALTS unset (= 0),
V41_MS_PIPELINE_MIN_ROWS=4, V41_MS_MHC_SPLIT=0, V41_MHC_* all default (fast/fused/late on).
=> decode `router_topk_par` runs WITH prior (n_protect 2, orig_sel, range_out non-null, n_alt 0)
on every layer box 2 owns; prefill rows run the plain form (no prior, no alts).

Decode (per lane-layer, b = 1..5, graph-captured, all warm):
- `mhc_fast_batched` pre-attn: grid (24+1+1, 1, b) x 256; mode 0, n_mix 24, do_collapse 1, write_carry 1.
- `mhc_fast_batched` pre-ffn collapse: grid (1, 1, b) x 256; n_mix 0, do_collapse 1, write_carry 0, mode 0.
- `mhc_fast_batched` late mix: grid (24, 1, b) x 256; mode 1, n_mix 24, do_collapse 0, write_carry 1.
- router: `f16_matvec_batched` grid (48, 1, b) x 256, W 384x5120 f16; then `router_topk_par` grid (b) x 512,
  n_expert 384, n_used 6, bias non-null.
- `hc_post_from_split_batched` grid (20, 4, b) x 256, n_w 4 (x2 per layer).
- `vec_add_inplace` grid (ceil(5120 b / 256)) (x2 per layer).
- `rms_norm_weighted_batched` grid (b) x 256, n 5120 (q_a / kv / comp norms).
- head prep per row: `hc_weighted_sum` (20) x 256 + `rms_norm_weighted` (1) x 256 (+ q8 quantize).

Prefill (b = 512 per lane, x2 chains per layer (pre-attn, pre-ffn); stage_cap only captures arena
rows, so these are direct launches):
- `rms_norm_no_weight_batched` grid (512, 1) x 256, n 20480  (residual -> flat)
- `f16_gemm_wmma_lds_tiled` grid (ceil(24/64)=1, 512/64=8) x 128, K 20480, M 24, N 512 (flat -> mix)
- `hc_sinkhorn_par_batched` grid (512) x 16
- `hc_weighted_sum_batched` grid (20, 1, 512) x 256, w = carry (stride 24)
- carry := split D2D copy (512*24*4 B)
- `rms_norm_weighted_batched` grid (512) x 256, n 5120
- router at 512: `f16_gemm_wmma_lds_tiled` grid (6, 8) x 128, K 5120, M 384; `router_topk_par` (512) x 512.

Cache regime for the prefill chain: the residual (42 MB at B=512) was written by the previous
layer's hc_post, with the OTHER lane's stages in between => partly cold. flat -> GEMM is
back-to-back => warm (42 MB fits the 64 MB MALL). I measure both warm and with a 40 MB flush.

## 1. Baseline measurements
(filled in below as they land)

## 2. Ideas
(see section further down)

## 3. Log
