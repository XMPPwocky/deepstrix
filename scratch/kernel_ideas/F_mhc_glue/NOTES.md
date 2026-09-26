# F_mhc_glue — dGPU mHC / norms / router top-k / glue (gfx1201)

Working notes, written as I go. Commit base 361d4f9 (= production hub). All GPU runs via the
scheduler (`_infra/gpu_submit.sh --dev dgpu` / `gpu_wait.sh`); job scripts under `jobs/`.
First attempt (2026-09-26 03:50-04:01) built the harness and took baselines with the hub LIVE
(polluted graph-mode numbers, kept as results/*_r1.txt); resumed 2026-09-26 evening with the hub
DOWN — all numbers below are from the resumed run unless marked.

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
- `rms_norm_weighted_batched` grid (b) x 256: decode's SEPARATE calls are q_a (n = N_LORA_Q = 1280,
  FP:3848), kv (n = N_HEAD_DIM = 512, FP:3996), comp (n = 512 per boundary, FP:4632). The n = 5120
  attn/ffn input norms are fused into mhc_fast at decode; they run standalone at prefill (B=512)
  and in head prep (`rms_norm_weighted`, 1 row).
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

## 1. Baseline measurements (hub DOWN, idle dGPU; results/base_*_r2.txt; job jobs/base_all.sh)

kb::ab, 40 rounds x 20 calls, warm; graph = captured graph of 20 calls (production decode path).

| kernel (decode, per call) | b=1 graph | b=1 direct | b=4 graph | b=4 direct | roofline | note |
|---|---|---|---|---|---|---|
| mhc_fast pre_attn (26,1,b) | 12.94 | 11.87 | 13.85 | 14.18 | 1.1 MB -> 1.7 us BW; launch floor ~3 us | latency-bound, known; not a target (see task) |
| mhc_fast pre_ffn collapse (1,1,b) | 4.70 | 4.63 | 4.29 | 4.66 | ~3 us floor | at floor |
| mhc_fast late mix (24,1,b) mode1 | 12.74 | 11.70 | 18.68 | 19.03 | 1.7 us BW | grows with b (RMS recomputed per WG) |
| router f16_matvec_batched (48,1,b) | **22.33** | 20.83 | 20.93 | 21.33 | 3.9 MB / 640 GB/s = **6.1 us** | **27% of roof**: one wave per row, 160 scalar u16 loads per lane |
| router_topk_par plain (b)x512 | 8.20 | 7.58 | 7.19 | 7.51 | ~3 us floor | 6 argmax x 10 barriers |
| router_topk_par prior (production) | **12.81** | 11.60 | 11.10 | 11.50 | ~3 us floor | 10 argmax + 2 range reductions = ~120 barriers |
| hc_post_from_split (20,4,b) | 3.61 | 3.79 | 4.33 | 4.70 | 0.1 MB -> <1 us; floor ~3 us | at floor |
| vec_add_inplace 5120b | 3.00 | 3.12 | 2.81 | 3.18 | floor | at floor |
| rms_norm_weighted_batched n=5120 | **11.35** | 10.92 | 10.32 | 10.65 | 40 KB -> <1 us; floor ~3 us | 20 serial load->DFMA iterations + re-read (prefill/head shape) |
| head hc_weighted_sum 1 row (20) | 3.41 | 3.56 | — | — | floor | at floor |
| head rms_norm_weighted 1 row (1) | **11.37** | 10.92 | — | — | floor ~3 us | same kernel body as above |

Decode rms at production n (1280 / 512): measured in the rms candidate run (section 3).

| kernel (prefill B=512, direct, per call) | warm | flush 40 MB | roofline (640 GB/s) | % roof |
|---|---|---|---|---|
| rms_norm_no_weight_batched (512,1) n=20480 | 123.8 | 147.2 | 84 MB -> 131 us | ~90-100% (warm reads from MALL) — closed |
| **f16_gemm_wmma_lds_tiled M=24 (1,8)x128** | **445.3** | **475.6** | 43 MB -> 67 us (0.5 GFLOP: nil) | **14-15%** — 8 WGs on 64 CUs, no load/compute overlap |
| hc_sinkhorn_par_batched (512)x16 | 8.8 | 9.0 | floor | at floor |
| hc_weighted_sum_batched (20,1,512) | 58.7 | 74.3 | 52 MB -> 82 us | >100% warm (MALL) — closed |
| rms_norm_weighted_batched (512) n=5120 | 34.6 | 36.0 | 21 MB -> 33 us | ~95% — closed |
| CHAIN (all of the above + carry copy) | 720 | 746 | ~200 MB -> 310 us | GEMM is 62% of the chain |
| mhc_fast collapse-only (1,1,512) [= hcw+rmsw fused, prod kernel, bit-exact vs the pair] | 37.7 | 71.8 | 62 MB -> 97 us | already exists; not wired for prefill (fast path is b<=8) |

Production's own figure for the GEMM is 478 us/call (FP:303-309) — matches the flushed number.

### Profile (first attempt, results/att_gemm, att_mhc; ATT on the dGPU)
- GEMM (dispatch 14, B=512): 66% of cycles stalled; **85% of the stall is `s_wait_loadcnt`**
  (48.5% loadcnt 0x3, 36.4% loadcnt_dscnt 0x0): pure memory wait, one 32-wide K step at a time,
  ~1430 cycles per step, 4 waves x 8 WGs on 64 CUs. Compute (WMMA) is invisible. Half of every
  WG's waves (rows 32..63 of the padded M) compute zeros.
- mhc_fast pre_attn: 72% stalled, the stalls are the up-front global loads (u16 W and f32 x) and
  the `s_wait_loadcnt 0x18` — i.e. it is the memory round trip of a 1-WG-per-row kernel, as
  documented in the kernel header. Not a target here (task: fuse neighbours, not re-tune).
- ISA (isa_all.sh, first attempt): no scratch/spills in any baseline kernel.

## 2. Ideas (ranked by expected gain x confidence / effort)

1. **[gemm_narrow] prefill pre-mix GEMM re-tiled for M=24** (structural; bit-exact): 256-thread WGs,
   ONE 16- or 32-column n-tile per WG (32 / 16 WGs at B=512 instead of 8), all 8 waves stream the
   X/W K-chunk with b128 loads, PF chunks in flight, double-buffered LDS, 2-4 waves own the WMMA
   chains. Same 1280-long wmma chain per output in the same K order with the same f16 fragments
   => bit-identical. Expected 5-10x (roof 67 us cold). cand_gemm.hip (6 variants: NT 16/32, BK 64/128, PF 2-6).
2. **[topk_wave] router_topk_par with wave-level argmax** (bit-exact): 12 registers per lane, 5-step
   xor butterfly per pass, winner masked in-register; phase 1 stays parallel. Removes ~120
   barriers. Expected 12.8 -> ~5 us at decode (x80 lane-layers = ~0.6 ms/step). cand_topk.hip.
3. **[rms_fast] rms_norm_weighted(_batched) with loads issued up front** (bit-exact): NPT template
   for n in {512, 1280, 5120}, same double accumulation order, same tree, output from registers.
   Expected 11 -> ~4 us at n=5120 (head prep x rows, prefill x2 per layer); smaller at 1280/512.
   cand_rms.hip.
4. **[router_mv]** router f16_matvec_batched is at 27% of its 6.1 us roof (22 us): scalar u16 loads,
   one wave per row. A bit-exact restructure = same per-lane element order (i = lane + 32 j), loads
   hoisted into register arrays in chunks of 32-40 (like mhc_fast) — expected 22 -> ~10 us
   (x80 = ~1 ms/step). Not bit-exact alternatives (b128 loads with contiguous per-lane chunks)
   would reach ~7 us but re-associate the dot.
5. **[mv_topk_fused]** fuse router_topk into the router matvec: the LAST of the 48 WGs of a row
   (atomic counter, mhc_fast pattern) runs the wave-level top-k on the logits it can see after a
   fence. Removes one graph node (~2.5 us + the top-k's fill/drain) per lane-layer; needs idea 2.
6. **[collapse_prefill]** wire mhc_fast collapse-only (1,1,B) for the prefill hcw + carry + rmsw
   trio: measured 37.7 us vs 58.7 + 34.6 (+ copy) = 93 us warm, bit-exact (CMP above). ~55 us x 2 x 20 x
   2 lanes = 4.4 ms per 1024-row chunk. Rust-side change only (the kernel exists).
7. **[hcpost_vecadd]** hc_post_from_split + vec_add: both at the ~3 us launch floor; a fusion saves
   one node (~3 us) x 2 per lane-layer — only if the two are adjacent in the stream (they are not:
   vec_add is the FFN combine, hc_post is the residual update after it). Not pursued.
8. **[mhc_late_b]** mhc_fast late mix grows 12.7 -> 18.7 us from b=1 to b=4 (every WG recomputes
   the row RMS in fp64): a separate RMS WG as in mode 0 would need the normed-mode dot order kept
   (dot over x*inv, not raw) — bit-exact if each mix WG still multiplies x*sc before the dot. ~5 us
   at b=4 x 80 = 0.4 ms/step. Not reached.

## 4. Summary (end of sweep, 2026-09-26 evening) — what to integrate, with the arithmetic

| idea | status | numbers (median of 3 process runs, graph mode unless noted) | est. leverage | integration |
|---|---|---|---|---|
| gemm_narrow_m24 (`f16_gemm_narrow_n16_bk128_pf2`, grid (1, ceil(B/16)) x 256) | WIN, bit-exact at B = 65/100/128/512/1024 | B=512 flushed 80.2 / 80.8 / 78.8 vs 452 / 457 / 446 us (5.6x); warm 74.0 / 73.7 / 71.4 vs 477 / 478 / 434 (6.4x); 84-90% of the 67 us roof | 2 calls x 20 layers x 2 lanes = 80 calls per 1024-row chunk x (478 - 80) us = 31.8 ms/chunk; 100K = 98 chunks = 3.1 s of 160 s = **~1.9% of prefill** (upper bound: the dGPU is not the pole; the `mhc_pre_attn` stage was 6.9 s) | small: new kernel symbol + `gemm_batched_wmma` dispatching narrow-M (out_dim <= 32, K % 128 == 0) to it in S/f16.rs; the M=384 router GEMM keeps the old kernel |
| rms_fast (`rms_norm_weighted_batched_fast`, same signature/grid) | WIN, bit-exact (n = 5120/1280/512/4864, b = 1/4/512, head 1-row) | n=5120 b=1 10.46 -> 4.07 us; head 1-row 10.46 -> 4.07; n=1280 4.91 -> 3.38; n=512 3.82 -> 3.25; B=512 n=5120 (prefill, direct) 33.4 -> 26.7 | decode per lane-layer: q_a -1.5 + kv -0.55 + comp -0.55 (ratio-2 layers) ~ -2.3 us x 80 = 0.18 ms/step; head prep -7.3 us x 8 rows = 0.06 ms/step => **~0.25 ms/step**; prefill 80 calls x 6.7 us = 0.5 ms/chunk (nil) | drop-in (replace the kernel body; `rms_norm_weighted` 1-row = grid 1) |
| router_mv_h20 (`f16_matvec_batched_h20`, same signature/grid) | WIN, bit-exact (k = 5120 and the 4992 generic tail, b = 1/4, 3 seeds) | COLD W (production regime): b=1 21.7 / 22.1 -> 10.4 / 10.3 us; b=4 21.9 / 21.8 -> 13.4 / 13.4. Warm: 20.3 -> 5.5 (b=1), 20.8 -> 8.7 (b=4) | (21.8 - 10.4..13.4) = 8.4-11.4 us x 80 lane-layers = **0.7-0.9 ms/step** | drop-in for the router; the same kernel serves other f16 matvec_batched callers (compressor/indexer f16 paths) — bit-exact there too, but re-measure |
| topk_wfred (`router_topk_wfred`, same signature/grid) | small WIN in production config, neutral plain, loss at b=512 | prior/n_protect 2/range: b=1 11.15 -> 9.20 us, b=4 11.15 -> 9.05; plain b=1 7.3 -> 7.3; plain b=512 33.0 -> 36.5 | -2.0 us x 80 = **0.16 ms/step**; keep `router_topk_par` for prefill (b > 64) | drop-in for decode (dispatch on b) |
| topk_wave (single-wave register argmax) | LOSS (bit-exact) | b=1 prior 15.9 vs 11.1; plain 14.2 vs 7.2; block-32 flavour 23 / 19; only b=512 with block 32 wins (20 vs 33) | — | dead end for decode |
| collapse_prefill (wire the existing `mhc_fast_batched` collapse-only (1,1,B) for the prefill hcw + carry-copy + rmsw) | measured, bit-exact (CMP in base_prefill), kernel exists | B=512 warm 37.7 vs 58.7 + 34.6 + copy (~93); flushed 71.8 vs 74.3 + 36.0 (~110) | 40-55 us x 80 calls = 3-4 ms/chunk = ~0.3 s / 100K (~0.2%) | small (Rust: call `launch_fast` collapse-only for b > 8 in the prefill chain) |
| mv_topk_fused | not attempted | — | one stage boundary, not a graph node (see log) | large |
| mhc_late_b (separate RMS WG for the normed late mix) | untested | late mix grows 12.7 -> 18.7 us from b=1 to b=4 | ~5 us x 40 = 0.2 ms/step at b=4 | medium |

Everything in the table came from the harness printouts in results/; repro.sh replays all of it.

## 3. Log
- 19:1x  hub down; baseline re-measured (results/base_*_r2.txt). GEMM 445/476 us (warm/flush) =
  14-15% of roof; router matvec 22 us = 27% of roof; topk prior 12.8 us; rms 5120 11.4 us.
- cand_gemm.hip (from the first attempt) compiles; harness GEMM_CANDS updated to its symbols
  (grid (1, ceil(B/NT)) x 256). cand_topk.hip and cand_rms.hip written; harness modes topk/rms.
- Submitted jobs/cands.sh r1 (gemm 512 warm+flush, tails 65/100/128/1024; topk b=1 prior/plain,
  b=4 prior, b=512 plain; rms b=1/b=4).
- r1 results (results/gemm_*_r1.txt, topk_*_r1.txt, rms_*_r1.txt):
  * GEMM: all 6 variants BIT-EXACT at B = 65, 100, 128, 512, 1024. Best `f16_gemm_narrow_n16_bk128_pf2`
    (grid (1, 32) x 256): 74.0 us warm / 80.2 us flushed vs 476.8 / 452.4 (6.4x / 5.6x); 535 GB/s
    flushed = 84% of the 640 GB/s roof. pf3 within 6%; bk64 variants 20-40% slower (more LDS
    stores per byte); n32 tiles slower (16 WGs instead of 32). At B=65..128 (replay-sized): 58-63 us
    vs 393-421 (~7x) -- there the per-WG K-loop (160 steps) is the floor, not bandwidth.
  * rms_fast: BIT-EXACT (n = 5120 / 1280 / 512 / 4864 tail, b = 1 and 4, 3 seeds, and the 1-row
    head kernel). Graph b=1: n=5120 10.46 -> 4.07 us; n=1280 4.91 -> 3.38; n=512 3.82 -> 3.25;
    head 1-row 10.46 -> 4.07. (49 VGPRs, no scratch.)
  * topk_wave (single wave, ds_bpermute butterflies + 16-slot serial scans): BIT-EXACT (sel, orig,
    weights, range; 4 seeds incl. a tie-heavy one; b = 1, 4, 512) but SLOWER: b=1 prior 15.9 vs
    11.1 us, plain 14.2 vs 7.2; block-32 flavour 23.0 / 19.4. Only at b=512 (prefill) does the
    32-thread flavour win (19.95 vs 32.8 us) because 512 tiny blocks then overlap. Dead end for
    decode: a lone wave's dependent chain (5 x 2 ds_bpermute + compares per pass, 12 passes) is
    longer than production's 9-level 512-thread LDS tree. => cand_topk2.hip: keep 512 threads,
    one `__ockl_wfred_max_u64` (DPP/permlane) per wave per pass on a (ord(score) << 32 | ~idx)
    sort key + a 16-slot LDS merge, one barrier per pass.
- r2 / r3 (separate processes, jobs/winners.sh): GEMM bk128 pf2 73.7 / 71.4 warm, 80.8 / 78.8
  flushed (baseline 477.5 / 434.2 warm, 457.3 / 445.8 flushed); rms_fast 4.07 / 4.06 (n=5120),
  3.38 / 3.37 (1280), 3.25 / 3.25 (512) vs 10.46 / 10.44, 4.91 / 4.90, 3.82 / 3.82. 36/36 CMPs
  bit-exact in every run. Consistent medians AND p10s across 3 runs.
- cand_router_mv.hip written (h20 88 VGPRs, h40 174, h80 spills 332 B scratch — expected loss);
  harness mode `mv`. Submitted jobs/mv_topk.sh r1 (mv b=1/b=4/direct; topk with wfred).
- mv / topk r1-r3 (results/mv_*_r{1,2,3}.txt, topk_*_r{1,2,3}.txt; 150/150 CMPs bit-exact per run,
  sel/orig/weights/range identical on 4 seeds incl. ties):
  * router matvec h20, graph b=1 WARM: 19.88 / 20.38 / 20.33 -> 5.43 / 5.50 / 5.50 us (3.7x);
    b=4 warm 20.5 / 20.8 / 20.8 -> 8.53 / 8.66 / 8.68. COLD W (32 rotating copies, r2/r3): b=1
    21.66 / 22.06 -> 10.36 / 10.31 (2.1x, 381 GB/s = 60% of the 640 GB/s roof); b=4 21.9 / 21.8
    -> 13.40 / 13.38. The baseline is the same warm and cold (pure latency chain: ATT shows 94%
    of its stall on ONE `s_wait_loadcnt 0x0` per loop iteration, 640 hits = 160 iterations x 4
    waves); h20 cold is DRAM-latency x occupancy bound (384 waves on 64 CUs, 40 loads in flight
    each). h40 = h20 within noise; h80 spills and loses (20 / 49 us).
  * router_topk_wfred, graph, PRODUCTION config (prior, n_protect 2, range): b=1 11.18 / 11.15 /
    11.13 -> 9.24 / 9.20 / 9.16 us (-18%); b=4 11.21 / 11.13 / 11.15 -> 9.10 / 9.04 / 9.05.
    Plain (prefill config): 7.35 -> 7.35 (neutral); b=512 plain 33.0 -> 36.5 (-10%, loss: the
    u64 wave reductions cost more than the LDS tree once 512 blocks overlap). 37 VGPRs. The
    remaining 9 us: ~3 us launch floor + 12 barriers + phase 1 transcendental chain.
  * ATT of the GEMM winner (results/att_gemmc): 66% stalled, 93% of it `s_wait_loadcnt` — it is
    now memory-bound (84-90% of roof), as intended.
- Adjacency check for idea 5 (mv + topk fusion): FP:6435-6466 the router matvec is the LAST node
  of a captured stage (`cap.end()` at 6466); the top-k launch (FP:6539) follows a host-side
  cache-prior fill + `d_prior.copy_from_host_async` and the look-ahead block. They are NOT in the
  same graph, so a kernel fusion would need the stage boundary moved (Rust restructuring, the
  prior H2D copy re-ordered before the matvec). Not attempted: integration cost large, the
  removable node is a stage boundary (event pair), not a plain graph node.
