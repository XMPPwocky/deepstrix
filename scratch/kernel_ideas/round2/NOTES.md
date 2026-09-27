# Round 2 — notes (one agent, items a-e, 2026-09-27)

Hub LIVE on this box during every measurement: all timing is interleaved `kb::ab` (both variants
see the same load), graph mode, >= 5 separate processes for anything wired; ratios are
candidate/baseline (< 1 = faster), "med / p10" = median over runs of the per-run median ratio /
of the per-run p10 ratio. Harnesses, scripts and raw outputs live under `round2/<item>/`.

## (a) Power-of-two / 2048-wave dispatch window scan — WIRED (`V41_GRID_PAD`)

### What the pathology is (synthetic probe, `a_gridpad/probe*.{hip,cpp}`, results/probe*.txt)
* gfx1201 only. gfx1151 shows nothing (probe: every block size x grid 128..65536 within 1%;
  real `q8_0_quantize_f32` at 1024..8192 blocks and `indexer_fp4` identical padded vs exact).
* It is a **~17 us floor on a short kernel**, not a per-WG cost: a guarded read/write kernel takes
  17 us instead of 3.7 us, and the same kernel with more work per thread stops paying it once its
  natural time passes ~20 us (spin probe: 2048x32 at 1024 FMAs/thread 28.5 vs 27.7 us; 1024x256
  at 256 FMAs 23.7 vs 23.6).
* The window is NOT "exact power of two": it is a total wave count at (or a few WGs below) a
  multiple of 2048 = the machine's wave slots (64 CUs x 32). Measured slow: 1-wave WGs at
  2040..2048 (2000 and 2049 fast); 2-wave at 1023/1024, 2047/2048, 3072, 4096; 8-wave at 255/256,
  511/512, 768, 1023/1024, 1280, 1536; 16-wave at 127/128, 255/256, 384, 512, 768. Not always:
  2D grids behave differently (gather (512,6)x64 = 6144 waves is fast) and the same +1 can HURT
  (f16_matvec_batched (4,1,64): +28%). So the pad is per launch site, measured, never global.

### Enumeration (production dGPU launches whose grid lands in / near the window)
Measured grid vs grid+1 WG in x (all kernels below already guard x: outputs byte-identical,
checked in every run). Regime = production: warm for just-written activations, cold (rotating
weight copies > MALL, or 64 MB flush) for streamed weights / the comp KV store.

| kernel (site) | shape(s) in window | regime | padded/exact med / p10 (runs) | verdict |
|---|---|---|---|---|
| fp8_act_quant_inplace (window KV, prefill) | b=128/256/512 x 512 | warm | 0.283/0.282, 0.337/0.336, 0.419/0.420 (5) | **WIRED** (17-22 -> 4.7-9.3 us) |
| indexer_fp4 (idx q, replay/prefill tails) | 32*b rows, b=16/32/64/128 | warm | 0.264, 0.302, 0.397/0.395, 0.689/0.626 (4-5) | **WIRED** |
| rope_tail_batched q (64,1,b) | b=32 | warm | 0.293/0.292 (5) | **WIRED** |
| rope_tail_batched idx-q (32,1,b) | b=64 (replay) | warm | 0.291/0.291 (5) | **WIRED** |
| f32_to_f16_cast_2d heads / low | 16x32768, 64x8192 (replay) | warm | 0.263/0.263 both (5) | **WIRED** |
| (same four kernels, every other b in 1..512) | — | warm | 0.97-1.02 (1-5) | neutral: pad costs nothing |
| gather_u4_r1 (512,b)x64 | b=2 (2048 waves), b=4 (4096) | WARM store | b=2 16.8 -> 4.56 us, b=4 18.2 -> 8.8 (3) | not production regime |
| gather_u4_r1 | b=2, b=4 | COLD store (prod: each layer's 178 MB weight stream evicts the MALL between gathers) | vs old kernel: b=4 exact 0.792 / +1 0.784; b=2 exact 1.258 / +1 1.284 (p10 0.876); pad ~neutral (3) | **no win in production regime — not wired** |
| indexer_gather_batched (512,b,2)x256 | b=1, 2 | warm + cold | 0.98-1.00 (3) | no |
| f16_matvec_batched idx-q (512,1,b) 4096x1280 | every b (4096*b waves) | cold W | b=1 0.823/0.798, b=2 0.789/0.770, b=3 0.830/0.836, b=4 0.839/0.985, b=5 0.919/0.861, b=8 0.910/0.900 (3) | real, but the kernel is replaced in item (c); see (c) |
| f16_matvec_batched ratio-1 comp (64,1,b) 512x5120 | b=4 (2048 w), 8 | cold W | b=4 0.909/0.877, others 0.99-1.04 (3) | unresolved (< 8% p10-consistent); see (c) |
| f16_matvec_batched idx proj (4,1,b) 32x5120 | b=64 (replay) | cold W | **1.279/1.249** (2) | pad HURTS — not wired |
| f16_matvec_batched_h20 router (48,1,b) | b=16/32/64 | cold W | 0.98-1.00 (2) | no |
| f16_matvec_narrow_batched mHC (24,1,b) | b=32/64 (replay) | cold W | 1.02-1.11 (2) | no (hurts) |
| vec_add_inplace (20b) | b=64 (10240 waves) | warm | 0.868/0.977 (2) | unresolved |
| hc_post_from_split_batched (20,4,b) | b=16, 64 | warm | 0.99-1.01 (2) | no |
| rms_norm_weighted_batched_fast (b)x256 (no x guard) | b=256/512, n=512/1280/5120 | warm, b vs b+1 REAL rows | n=512 b=512 p10 18.7 -> 8.7 us; n=1280 b=512 11.6 -> 8.1; n=5120 25.2 -> 20.5 (1) | win, needs a guarded twin kernel; prefill-only ~0.1% -> **next list** |
| kv_cache_append_batched (b)x512 (no guard) | b=128/256/512 | warm, b vs b+1 rows | 0.27-0.30 (1) | same: needs a twin -> **next list** |
| q8_0 GEMVs (qb 4096 WGs, wo_a 1024 x 8 waves) | — | cold | natural 77-90 us hides the floor (C1 measured qb/wo_a pad neutral) | no |
| head 16160, wo_b 640, q_a 160, kv 64, Engram 3200, hc_post, attention (4,b) | not in window at production b | — | — | not measured |

Value: all wired sites are prefill / replay (a 1024-row prefill chunk saves ~0.5 ms of dGPU time
on fp8 alone; replay ~1 ms per request over fp4 + rope + casts) — small, but free and bit-exact.
The decode-relevant candidate (gather) does not win once its comp-KV rows are cold, which they
are in production.

Test: `crates/v4flash-kernels/tests/grid_pad_bitexact.rs` (3 tests: fp8 13 row counts + indexer_fp4
10, rope 3 shapes x 8 b x fwd/inv, cast 8 shapes incl. a pitched destination; sentinel rows /
pitch padding checked untouched) passes with defaults and `V41_GRID_PAD=0`;
`q8_0_sweep_c2_bitexact` (cast_2d user) passes. Results: `a_gridpad/results/intree_tests.txt`.

## (b) Crossover scan of the deployed round-1 knobs — 4 gates changed

Family harnesses (copied where a limit had to move: `b_crossover/xattn.cpp` = D harness with 16-row
buffers, `xe.cpp` = E harness with a configurable scores stride, `xc2.cpp` = C2 harness + q_a /
shared replay shapes) and a new `xover_f.cpp` (F kernels, old vs new symbol from the SAME in-tree
code object). 5 separate scheduler jobs per family (r1..r5), interleaved kb::ab, production regime
per kernel (cold weights for GEMVs, cold comp-KV store for the gather, warm activations for glue).
Full table: `b_crossover/table.txt` (`bash b_crossover/table.sh`). Ratio = new/old median over runs
(p10 ratio in brackets where it matters); **bold = the new kernel loses > 5% or a gate moved**.

| knob (kernel) | gate before | b = 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | 16 | 24-32 | 48-64 | 128-192 | 512 | 1024 | action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V41_ATTN_DEC_FUSED (fused_vt_qreg_sp_d4) | b<=16 | .48 | .48 | .46 | .43 | .44 | .42 | .41 | .58 | .54 | | | | | | none |
| V41_ATTN_DEC_SCORE_BLK128 | pair only | .71 | .86 | .82 | .76 | .96 | .94 | .93 | .92 | .94 | | | | | | none |
| V41_GEMV_TB q_a (tB<b>) | b<=10 | .97 | .94 | .89 | .81 | .78 | .76 | .74 | (9: .72, 10: .70) | | | | | | | none (kv/gate/down same shape of curve, all < 1) |
| V41_GEMV_TB grouped wo_a | 2<=b<=8 | 1.02 (gated out) | .88 | .84 | .85 | .78 | .81 | .67 | | | | | | | | none (tB1 still +2%: stays gated) |
| V41_Q8_QUANT_WAVE (both padded) | all | .90 | | | .80 | | | .71 | | .64 | | .41-.42 | .35 | .40 | | none |
| V41_SHARED_FUSED (vs the tB chain) | b<=5 | .91 | .91 | .90 | .89 | .93 | .92 | **1.07** | | | | | | | | none (b=6/7 .92/.91 would win ~8%: not worth a rare lane size; b=8 loses, gated out) |
| V41_RMS_FAST n=5120/1280/512 | n in set | .42/.73/.88 | | | .43/.73/.88 | | | .44/.73/.88 | | .45/.74/.88 | | .53/.78/.90 | | .75/.57/.98 | .84/.51/.38 | none |
| V41_ROUTER_MV_H20 | b<=64 | .43 | .44 | .51 | .55 | .61 | .43 | .44 | .40 | .42 | .42 | .40 | | | | none |
| V41_TOPK_WFRED prior / plain | b<=16 | .81/.97 | .80 | .81 | .81/.98 | .81 | .81 | .81 | .82 | .82/.98 | .82/.99 | .89/**1.02** | 1.00/1.00 | **1.03/1.09** | | none (gate right; plain loses from 64) |
| V41_IDX_SCORE_QREG (n=32K; 12/16 at 12K) | all | .68 | .63 | .60 | .55 | .52 | .50 | .54 | .51 | .49 | | | | | | none |
| V41_IDX_TOPK_HYBRID (n=131K; 64/512 at 32K) | all | .66 | .66 | .66 | .65 | .65 | .65 | .65 | .66 | .66 | | .67 | | .63 | | none |
| **V41_CAND_THRESH_ILP** (n=131K / 32K) | **b<=8** | .57 | .61 | .61 | .60 | .62 | .60 | .60 | .60 | .61/.87 | .87-.88 | .94/.92 | **1.50** | | | **gate -> b<=32** |
| **V41_IDX_GATHER_B128** (cold store) | **b>=4** | .93 (+pad) | **1.35**, +pad .94 [.86] | **.80** | .76 | .75 | .75 | .73 | .66 | .60 | | | | | | **gate -> b>=3, launch padded** |
| V41_MOE_DOWN_DN2 (iGPU, chain, E=16 ppr=6) | rows<=8 | .94 | .88 | .94 | .92 | .86 | .93 | .97 | .94 | .93 | .89-.90 | .90 | | | | none (would win to 64: next list, regime not reproducible under the 600 MB cap) |
| **V41_GEMV_BPACK_Z16** kv (M=512) | 16<b<=64 | | | | | | | | | | (17: **1.72**) **1.04 / 1.11** | .88 / .83 | | | | **gate: n_rows>=2048 or b>=48** |
| V41_GEMV_BPACK_Z16 q_a (M=1280) | 16<b<=64 | | | | | | | | | | (17: **1.14**) .96 / .96 | .80 / .89 | | | | same gate |
| V41_GEMV_BPACK_Z16 q_b / wo_a / wo_b / shared | 16<b<=64 | | | | | | | | | | .38-.54 | .31-.43 | | | | none |
| V41_F16X_DB_BN64 kv / q_a | kv b>64; q_a b<=512 | | | | | | | | | | | (65) .57/.55 | .56/.54 | .55/.79 | .70/**1.07** | none (q_a at 1024 gated out; 768 1.00) |
| **V41_F16X_256** q_b / wo_a | **b>64** | | | | | | | | | | | (65) **1.27** / .98 | (128) **1.18** / 1.01; (192) .96 / .89 | | | **gate -> b>=192** |
| V41_ENGRAM_I8X (M=6400 slice) | b>8 | | | | | | | | | .54 | .55 | .54 | .32 | | | none |
| V41_MOE_WMMA_GATEUP / _DOWN | rows>=256 / >=128 | not re-measured: the regime needs all 384 experts (7.2 GB), over the live-hub iGPU cap; round-1 full-layer numbers stand | | | | | | | | | | | | | | none |

Gates changed (commit (b)): **F16X_256 >= 192 rows** (removes a 1.18-1.27x loss on every 65..191-row
prefill tail), **z16 only for M >= 2048 or b >= 48** (removes 1.11-1.72x on kv / q_a at 17..47
replay rows), **gather_b128 from b = 3 with the grid pad** (b=3 0.80; the b=2 "not root-caused"
loss IS the 2048-wave dispatch window of item a), **threshold ILP to b <= 32** (0.60-0.88).
Tests: `tests/round2_gates.rs` (selector asserts, knob-aware; padded gather at b = 2/3/4 and the ILP
threshold at b = 12/24/32 vs the production kernels) passes with defaults and all five knobs = 0;
`indexer_sweep_bitexact` and `q8_0_sweep_c2_bitexact` pass with defaults and knobs = 0
(`b_crossover/results/intree_tests.txt`).

## (c) z16 chunking of f16_matvec_batched — WIRED (`V41_F16_MV_Z16`)

Candidates `c_z16/cand_z16.hip`: grid.z = ceil(b/NB), one warp per weight row for NB batch rows,
NB accumulators, per-(row, b) order identical to the production kernel (so bit-exact by
construction and checked at every (shape, b)); "plain" (1 element / lane / step) vs "h8" / "h4" /
"h16" (8 / 4 / 16 weight elements hoisted into registers per step). h8 at the smallest NB >= b won
everywhere it wins; plain was worse than h8 at every b; h4 ~ h8; h16_n4 worse. In-tree:
`f16_matvec_batched_z16_n{1,2,4,8,16}` (f16_matvec.hip) + `F16Matvec::matvec_batched_z16`.
Cold weights (rotating copies > MALL, graph of `copies` calls), 6 runs (3 candidate file + 3 the
in-tree object, identical): ratio vs production `f16_matvec_batched` (med / p10):

| shape (site) | b=1 | 2 | 3 | 4 | 5 | 6 | 8 | 16 | 32 | 64 | wired |
|---|---|---|---|---|---|---|---|---|---|---|---|
| idx q 4096x1280, z16 + pad | .73/.74 | .61/.58 | .60/.50 | .49/.41 | .39/.37 | .47/.35 | .31/.32 | .31/.31 | .29/.28 | .26/.26 | all b |
| idx q, z16 unpadded | 1.00 | .68 | .61 | .63 | .48 | .55 | .37 | .32 | .29 | .27 | (pad kept) |
| idx q, production kernel +1 WG only | .82 | .78 | .84 | .89 | .89 | .91 | .88 | 1.02 | .99 | 1.00 | (not wired) |
| ratio-1 compressor 512x5120 | .51 | .85/.68 | **1.08** | .98 | .81/.77 | .88/.85 | .84 | .94/.80 | .49 | .30 | b not in 3..4 |
| idx proj 32x5120 | .28 | .33 | .64 | .80 | 1.00 | **1.10** | **1.44** | **2.46** | | | b <= 4 |
| router 384x5120 vs **_h20** (production) | | | | | | | (8) 1.9 | 2.2 | 1.43 | (48) .98, (64) .80 | b > 48 |

Absolute: idx q at decode b=4 72.8 -> 38.9 us (8 index layers per lane-step: ~0.27 ms/lane-step), at
replay b=64 639 -> 164 us (x8 layers x2 lanes: ~7.6 ms per replay); compressor b=64 325 -> 97 us;
replay router b=64 106 -> 85 us (x20 layers x2 lanes: ~0.85 ms per replay).
Test: `tests/f16_mv_z16_bitexact.rs` (every NB symbol x pad 0/1 + the z16 and router wrappers vs
`f16_matvec_batched` at 7 shapes incl. a row tail, a plain-loop k and b up to 64; site-gate asserts)
passes with defaults and `V41_F16_MV_Z16=0 V41_GRID_PAD=0`; `mhc_glue_bitexact` (router wrapper at
b = 1..64) passes with defaults and `V41_F16_MV_Z16=0 V41_ROUTER_MV_H20=0`.

## (d) Decode launch-count fusions — WIRED (`V41_DEC_FUSE`, `V41_DEC_SKIP_DEAD`)

Adjacency verified in `forward_prefill.rs` (the arena decode path; the q / kv / output-projection
stages are graph-captured, the FFN combine is not). Per lane-layer at decode the chain was
q chain: cast_input_f16 (DEAD) -> quantize(attn_input_norm) -> q_a GEMV -> rms_w(1280) ->
cast_qr_f16 (DEAD) -> quantize(qr_normed) -> q_b GEMV -> D2D copy q -> q_normed -> rope(q);
kv chain: quantize(attn_input_norm) (DUPLICATE of the q chain's) -> kv GEMV -> rms_w(512) ->
rope(1 head) -> fp8 window quant; output proj: rope_inverse -> cast_heads_f16 (DEAD) ->
quantize(heads) -> wo_a -> cast_low_f16 (DEAD) -> quantize(low) -> wo_b; combine: vec_add(local,
before the box-2 wait, off the critical path) -> [wait] -> vec_add(remote) -> hc_post.
Dead = its only reader is the f16x arm and the dp4a arm is taken at decode (b <= 8).

Candidates `d_fusion/cand_fuse.hip`, chain harness `d_fusion/chain_harness.cpp` (production
kernels from the in-tree code objects with their wrappers' grids incl. pads, graph inner=10, warm,
every output compared; plain AND YaRN rope params). In-tree kernels re-measured with CH_INTREE=1,
5 runs (combine: 2), new/old chain ratio, med [p10]:

| chain | b=1 | 2 | 3 | 4 | 5 | 8 | b=4 us | wired as |
|---|---|---|---|---|---|---|---|---|
| kv: rms + rope + fp8 -> 1 | .61 | .60 | .62 | .62 | .60 | .61 | 10.0 -> 6.0 | RopeTail::launch_kv_rms_rope_fp8 |
| q_a: rms + cast_qr + quant -> rms + quant | .74 | .74 | .74 | .75 | .75 | .74 | 8.9 -> 6.7 | dead cast skip |
| q_a: -> 1 (rms_quant) | .57 | .59 | .58 | .58 | .58 | .57 | 8.9 -> 5.2 | RmsNorm::launch_weighted_quant_q8_1280 |
| q: D2D copy + rope -> 1 | .67 | .69 | .71 | .74 | .74 | .78 | 6.7 -> 4.8 | RopeTail::launch_forward_batched_copy |
| oproj: rope_inv + cast_heads + quant -> 2 | .73 | .73 | .74 | .74 | .75 | .76 | 10.3 -> 7.4 | dead cast skip |
| oproj: -> 1 | .58 | .61 | .62 | .66 | .68 | .76 | 10.3 -> 6.6 | RopeTail::launch_inverse_quant_q8 |
| cast_input + quant + dup quant -> quant | .43 | .43 | .43 | .43 | .43 | .43 | 8.7 -> 3.7 | dead skip x2 |
| cast_low + quant -> quant | .62 | .61 | .61 | .62 | .62 | .62 | 6.2 -> 3.8 | dead cast skip |
| combine: vec_add + hc_post -> 1 | .64 | .64 | .65 | .68 | .69 | .72 | 7.2 -> 4.9 | HcPost::launch_from_split_batched_add |

~23 us of dGPU graph time per lane-layer at 4 rows -> ~0.9 ms per lane-step (40 layers), ~1.8 ms
per 8-row step (2 lanes). The candidate first ran the rope_copy / oproj kernels with the exact
(64, b) grids: at b = 8 they are 2048 waves and were 2.2x / 1.16x SLOWER than the chain -- the
item-(a) window again; the wired launches carry the `V41_GRID_PAD` column.
Gates: fusions only at b <= 16 (`DEC_FUSE_MAX_B`: decode / verify lanes; replay / prefill not
measured, a 512-row fused kv launch would sit in the 2048-wave window); the q_a / heads fusions only
on the dp4a arm (the f16x arm keeps its cast, no quantize); the combine fusion only when box 2's
partial is the last add (no resident-expert partial). Dead-launch skips use the consumers' own
predicates (any b). Not done: (4) router matvec + topk + readback_pack (the router matvec is a
48-WG GEMV: a single-WG fusion is the 24x-regression pattern; a last-WG-does-topk scheme needs an
atomic ticket -- next list) and (5) head prep (hc_weighted_sum + rms per row in forward_head.rs --
next list).
Test: `tests/decode_fusion_bitexact.rs` (each fused wrapper vs its chain of production wrappers,
b = 1..16, plain + YaRN rope, sentinel-filled outputs) passes with defaults and `V41_GRID_PAD=0`;
`grid_pad_bitexact` passes after the RopeTail arg refactor; server + expertd build.

## (e) int8-activation WMMA GEMM for the dense prefill sites — LOSS, NOT WIRED

Candidates `e_i8x/cand_i8x.hip`: the in-tree `f16x_core` (included verbatim) with int8 xq + per-32
xscale activations (Q8_0, dequantised at stage as i8x_db / lds_tiled do) and an explicit activation
pitch (grouped wo_a), in five geometries (i8x_db_ld = the Engram 128x128 DB PF2; 128x128 PF2;
256x128 PF1; 256x128 DB; 128x64 DB PF2). Harness `e_i8x/harness.cpp`: production = cast_2d + the
production f16x tile at that b (db_bn64 for q_a / kv, 256x128 for q_b / wo_a, base elsewhere),
candidate = q8_0_quantize_f32_wave (+pad) + i8x; cold weights (flush), direct launches, 3 runs;
b = 512 except wo_a at 256 (its f32 heads input alone is 67 MB at 512, over the 130 MB cap) and
q_a also at 1024. Best i8x geometry per site, ratio vs production (med / p10), GEMM alone and
with the activation prep:

| site | best i8x | GEMM alone | chain (prep + GEMM) |
|---|---|---|---|
| q_a 1280x5120 b=512 | bn64 | 1.08 / 1.07 | 1.08 / 1.05 |
| q_a b=1024 | db | 1.01 / 1.00 | 1.04 / 1.01 (128x128) |
| kv 512x5120 | bn64 | 1.05 / 1.06 | 0.99 / 0.97 |
| q_b 32768x1280 | 256x128 | 1.02 / 1.05 | 1.03 / 1.01 |
| wo_a 8x1024x4096 (b=256) | 256x128 | 1.03 / 1.03 | 1.09 / 1.05 |
| wo_b 5120x8192 | 256x128 | 1.09 / 1.08 | 1.02 / 1.15 (128x128) |
| shared gate/up 2304x5120 | db | 0.89 / 0.98 | 0.98 / 0.99 (128x128) |
| shared down 5120x2304 | 256x128 | 0.95 / 1.00 | 1.01 / 1.00 |

No site wins by >= 8% on both median and p10; most lose. Numerics (vs a double-accumulated
reference of dequantised weights x f32 activations, 32 rows): f16x rel_rmse 2.78-2.83e-4 at every
site; i8x 3.74-3.79e-3 (13x worse, the Q8_0 activation error the decode dp4a arm already carries);
i8x vs f16x 3.77e-3; all five i8x geometries give identical outputs. Why it loses: at b = 512 the
f16x GEMMs are not activation-bandwidth bound (the weight tile and the WMMA/LDS phases dominate), so
halving the activation bytes buys nothing, while the int8 -> f16 dequant moves into the staging
path; Engram's 1.87x came from replacing lds_tiled's core, not from the int8 activations.
`V41_DENSE_I8X` therefore not added (nothing to gate).

## Knobs on the branch after round 2 (value that restores the OLD path)

Round 1 (24): V41_ATTN_DEC_FUSED=0, V41_KV_F16_ROUNDTRIP=1 (inverted), V41_ATTN_DEC_SCORE_BLK128=0,
V41_GEMV_TB=0, V41_Q8_QUANT_WAVE=0, V41_Q8_QUANT_GRID_PAD=0, V41_SHARED_FUSED=0, V41_RMS_FAST=0,
V41_ROUTER_MV_H20=0, V41_TOPK_WFRED=0, V41_MHC_GEMM_NARROW=0, V41_IDX_SCORE_QREG=0,
V41_IDX_TOPK_HYBRID=0, V41_CAND_THRESH_ILP=0, V41_IDX_GATHER_B128=0, V41_MOE_DOWN_DN2=0,
V41_MOE_WMMA_GATEUP=0, V41_MOE_WMMA_DOWN=0, V41_ENGRAM_I8X=0, V41_ENGRAM_CHUNK128=0,
V41_GEMV_BPACK_Z16=0, V41_F16X_DB_BN64=0, V41_F16X_256=0, V41_REPLAY_F16X=0.
Round 2 (4 new): V41_GRID_PAD=0 (also un-pads the gather_b128, z16 idx-q and fused-rope launches),
V41_F16_MV_Z16=0, V41_DEC_SKIP_DEAD=0, V41_DEC_FUSE=0. Round-2 gate moves live under existing knobs
(V41_F16X_256 >= 192 rows, V41_GEMV_BPACK_Z16 M >= 2048 or b >= 48, V41_IDX_GATHER_B128 b >= 3,
V41_CAND_THRESH_ILP b <= 32): `=0` still restores the pre-round-1 kernel; there is no knob for the
round-1 gate values themselves. Non-bit-exact knobs unchanged: V41_MOE_WMMA_GATEUP/_DOWN,
V41_REPLAY_F16X (round 2 added none).

## Next (seen, not done), ranked by expected value / effort

1. **Gather dedupe at the S2 reuse layers.** Every reuse layer re-gathers the SAME top-512 rows from
   the SAME comp-KV store as its index source (30 of 38 gathers per lane-step), into the shared
   `sd.attn_active_comp_kv`. A per-lane (or per-index-group) destination would skip them: cold
   gather is 20-28 us at b = 1..5 -> ~0.6-0.8 ms per lane-step. Needs scratch per lane + a validity
   key (index group, lane).
2. **Root-cause / systematise the 2048-wave dispatch window** (item a): a wrapper-level helper that
   pads any short launch whose wave count lands at / just below a multiple of 2048 would catch the
   next occurrence (gather b=2/4, idx-q, rope b=32/64 were all found by accident); worth one probe
   of the HSA dispatch path (grid-size vs workgroup-count registers) and a note to AMD.
3. **Guarded twins for rms_norm_weighted_batched(_fast) and kv_cache_append_batched** so the pad can
   reach them: b+1-row runs 0.27-0.50x at prefill b >= 256 (n = 512 rms 18.7 -> 8.7 us, kv append
   18 -> 4.8 us); ~1 ms per 1024-row chunk, prefill-only.
4. **dn2 gate extension to rows < 128** (chain 0.89-0.94 at 12..64 rows in a 16-expert regime;
   the production 384-expert regime needs > the 600 MB iGPU cap: measure with the hub down).
5. **Router matvec + topk + readback_pack** in one launch via a last-WG-done ticket (the 48-WG h20
   matvec must stay multi-WG): ~2 graph nodes per lane-layer.
6. **Per-row head prep** (forward_head.rs hc_weighted_sum (20) + rms (1) per row): batch the rows
   bit-identically per row and hoist the loads (rms_fast already did the rms half).
7. **Shared-fused gate to b <= 7** (0.92 / 0.91 at 6 / 7 rows), **threshold ILP to 64**
   (0.92-0.94, unresolved), **vec_add pad at replay b = 64** (0.87 med / 0.98 p10, unresolved).
8. **Ratio-1 compressor at b = 3..4** still on grid.z = b (z16 1.05-1.08 there): a split-row
   variant with more WGs per row would need a re-associated sum (not bit-exact).
