# D_attention — dGPU attention + window/compressed KV (gfx1201) — sweep notes 2026-09-26

Base: 361d4f9 (production). All GPU runs via `../_infra/gpu_run.sh --dev dgpu`. Nothing here is merged.

## Production launch facts (read from S/attention_dec.rs, S/attention.rs, FP:5120-5910)

Decode (arena rows, b = 1..5 per lane, graph-captured stage):
- `attn_meta_fill` grid (1)x32: n_raw_per / n_raw_offset_per / n_comp_per as kernel args.
- `attention_dec_score_htiled_wmma_f16s` grid (ceil(n_total_max/256), n_head/16=4, b) x 512.
  args: scores(f16, stride `max_keys`), q f32 [b,64,512], raw_kv f16 (window cache, per-row
  offset n_raw_offset_per), comp_kv f16 (= `attn_active_comp_kv` [b, 512, 512] gathered top-K when
  the indexer fired -> comp_kv_batch_stride = INDEXER_TOP_K = 512, comp_base_per = null; else the
  dense per-row store with comp_base_per non-null and stride 0), mask = null,
  max_keys_words = ceil(82176/32) = 2568, n_head 64, max_keys = scores_stride = 3072
  (ATTN_SCORES_STRIDE whenever n_total_max <= 3072), kq_scale = 1/sqrt(512).
- `attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s` grid (4, b) x 512, same buffers +
  sinks[64], out f32 [b,64,512].
- n_raw = 128 (SWA_WINDOW) at any context past 128 tokens; n_comp = min(n_comp, 512) -> n_total <= 640
  at decode (no image rows at decode). Contexts 8K-235K all collapse to 640 keys.
- Per lane-layer: score ~13.6 us + smwsum ~16 us (incl. event pairs) x 80 lane-layers = ~2.4 ms/step.

Prefill (b = 512 per lane): `attention_mixed_score_batched_htiled_wmma_f16s` grid
(ceil(n_total/256), 1, B) x 512 (each warp = 16 keys, loops all 4 head tiles x 32 K-chunks),
smwsum grid (4, B) x 512. n_total = 128 + P/2 on ratio-2 encoder layers (up to ~50K keys at 100K),
scores streamed through DRAM (stride = n_total_max when > 3072).

Roofline, decode b = 1, 640 keys: K/V 640 KB (read twice: score + wsum, second pass L2-warm),
q 128 KB, scores 82 KB written + read + rewritten, out 128 KB. ~1.9 MB -> 3.0 us at 640 GB/s;
84 MFLOP WMMA -> ~2 us at 40 TF. Both kernels are far from any byte/flop roof: latency-bound.

## Log
(see below, appended as work proceeds)

### 2026-09-26 evening — RESUME after the 04:01 UTC box hang (hub DOWN, GPUs idle)

Attempt-1 state found: baseline harness + results (quiet run `results/baseline_decode_run1.txt`),
candidates `cand_attn.hip` (pf2 / regpf / fused_f16s / blk256) measured once under heavy noise
(`results/cand1_decode_run1.txt`: base pair med 94 us vs 42 us in the quiet run -> discard medians,
p10s only hint: fused 33.6 vs pair 41.7; blk256 6.9 vs score 9.7; pf2 pair 30.9).

Attempt-1 baseline (quiet, graph, per call, `results/baseline_decode_run1.txt`):
  b=1: pair 42.4 us = dec_score 9.9 + smwsum 33.7;  b=2: 42.5 = 10.0 + 33.7;  b=4: 48.9 = 16.1 + 34.0;
  b=5: 87.1 (=39.0+50.5, noisy p90 140); b=8: 53.8 = 21.7 + 33.5.
  NOTE: smwsum 33.7 us here vs "~16 us" in the inventory (commit 98bdd94 bench). Same grid (4,b)x512,
  same stride 3072, 640 keys = 40 serial V tiles (2 barriers + 1 L2 round trip each ~0.8 us). The
  inventory figure may be at a smaller n_comp; this harness is the number I can stand behind.

HARNESS BUG in attempt 1: `cmp_scores` compared candidate scores against the `scores` buffer AFTER the
production smwsum had overwritten it with softmax weights -> every "scores bitexact=no bit_diff=40960"
(= 64 heads x 640 keys) is an artefact; blk256 is bit-exact by construction. Fixed in attn_harness2.cpp
(pristine copy of the reference scores). The "weights" compare for pf2/regpf is void too (they keep the
weights in LDS by design; nothing downstream reads the weights: FP:5216/5889 are the only users of
sd.attn_scores, V41_VERIFY_DECODE_ATTN is off).

ATT of the baseline dec_score (b=1, `results/att_base_b1.summary`): 69% of wave latency is stall;
64.6% of stall = the FIRST `s_wait_kmcnt 0x0` (kernarg s_load_b512 -> dependent n_raw_per[b] /
n_comp_per[b] loads -> comp_base_per), 10% a second kmcnt, 6.7% s_wait_loadcnt (the K/q vector
loads), ~10% spread over 256 v_cvt_f16_f32 per lane (q f32->f16 recast by EVERY warp). The kernel
is latency-bound at the wave level (scalar-chain + one vector round trip) and CU-bandwidth-bound at
the WG level: each warp pulls 16 KB K + 32 KB f32 q through its CU's L0 (16 warps = 768 KB per WGP at
12 WGs / 64 CUs). Thin grid is the disease for both kernels.

f16_roundtrip (idea 4): after FP:4038 the only readers of sd.kv_normed are the two kv_cache_append
sites (FP:4187, 4204), which store `(_Float16)kv_new[d]` (RN). f16(f32(f16(x))) == f16(x) for RN,
so the kernel is a provable no-op on the cache contents regardless of the FP8 range argument
(the FP8 products CAN fall below f16's 2^-24 when a 32-block's amax < 448*2^-16 ~ 0.0068, so the
"exact in f16" argument alone is not enough; idempotence of the RN cast is). Verified on-GPU below.

#### Ideas (ranked by expected gain x confidence / effort)
1. Flash-decoding split-K, one fused launch + tiny combine: grid (4 head-tiles, S key-splits, b).
   Attacks both thin grids at once. NOT bit-exact (weights rounded vs the split max; f32 partials
   re-associated) -> fidelity gate. [cand2 flash_part1 -> spilled; cand3 flash2 = register-lean]
2. Bit-exact fused score+softmax+wsum, ONE launch, V fragments straight from global into WMMA B
   registers (no LDS V tile, no phase-B barriers), 8 tiles in flight. [cand2 fused_direct_g16]
3. dec score with fewer warps per WG (8/4/2 -> 2-8x more WGs). Bit-exact, drop-in (grid only).
   [cand2 blk256/blk128/blk64]
4. Drop f16_roundtrip (proved no-op). Drop-in.
5. (attempt 1) LDS-V smwsum with register prefetch 2 groups ahead + weights kept in LDS [pf2].
6. Fold the combine into part1 (last-arriving WG combines via an atomic ticket) to save a launch.
7. Widen the score phase inside the flash WG: all 16 warps split the 512-dim reduction of the
   NT key tiles and reduce partials through LDS (non-bit-exact scores, same precision class).
8. q pre-cast to f16 by the producer (rope_tail on q) -> halves q bytes, kills 256 cvt/lane in the
   score kernel. Needs a rope_tail change (family D, medium). Not measured.
9. attn_meta_fill folded into the score kernel (compute n_raw/n_comp from arena state in-kernel).
   Saves a launch (~2.5 us) per lane-layer; medium integration. Not measured.
10. Ratio-1 compressor at L20 (f16_matvec_batched grid.z=B): batch-tiled twin of the existing
    pair_tiled kernel (bit-exact by the same argument). Not reached.

#### Run 1 of the candidates (quiet box, graph mode, per call; `results/decode2_run1.txt`, `decode3_run1.txt`)
Correctness (`results/check2_run1.txt`, `check3_run1.txt`, 7 shapes incl. tails / dense store / empty
row / b=8): blk256/128/64 scores BIT-EXACT; fused_direct_g16 BIT-EXACT; fused_direct_g32 + g32d4
WRONG on partial tiles (spills 28 VGPRs; not debugged, dropped); flash/flash2 k32/64/128:
rel_rmse 2.1e-4 vs production, and vs an f64 CPU reference the flash kernels are at 1.99-2.07e-4
while production itself is at 2.13e-4 -> same precision class (f16 weight rounding), fidelity gate.
                        b=1     b=2     b=4     b=5     b=8      (us per call, medians)
base pair               42.5    42.3    47.8    47.6    53.4
  dec_score             9.9     10.1    15.7    15.8    21.5
  smwsum                34.0    33.6    33.2    33.0    33.0
score blk256            6.8     9.6     12.8    15.3    18.6
score blk128            7.0     8.6     11.8    15.1    20.1
score blk64             6.8     8.4     11.7    15.1    20.2
fused_direct_g16        43.2    42.5    42.4    42.5    42.9    (bit-exact, 1 launch; NO gain)
fused_f16s (att.1)      34.3    34.0    33.8    33.6    34.2    (bit-exact, 1 launch)
pair w/ smwsum_pf2      31.6    31.6    37.0    36.9    42.6    (bit-exact)
flash2 k32 (+combine)   18.5    29.5    46.7    51.7    80.2    (not bit-exact)
flash2 k64              16.8    31.0    43.2    44.1    70.1
flash2 k128             13.5    31.3    39.7    34.8    55.7
(cand2 flash k32/64/128 with spills: 23/26/27 at b=1, 66-79 at b=4 -> superseded by flash2)
Observations: flash2 wins big at b=1 (-68%) but scales badly with b: its WG latency (~7-9 us) x
rounds of concurrency (1-2 WGs per WGP at 149-256 VGPRs). fused_direct_g16: the compiler batches
the 128 u16 loads per group fine (s_clause x8, loadcnt up to 0x3e) but the kernel is no faster than
the LDS-V fused_f16s -> the direct u16 fragment loads (2 cache lines per wave-instruction, 128 per
group per wave, 16 waves) are TA/L0-throughput-bound, not latency-bound. ATT traces submitted.

#### ATT findings (dGPU, b=8 so CU 1 holds a WG; `results/att_*_b8`, `attsum.sh`)
* production smwsum: 80% of wave latency is stall; the two top `s_wait_loadcnt 0x0` (21% + 21%,
  128 hits = 32 tile iterations x 4 traced waves) are the per-tile V stage loads and the global
  weight-row loads: TWO dependent global round trips per tile, 40 tiles, ~750 cycles per iteration.
  Pure latency chain (grid (4,b)).
* fused_direct_g16: 66% stall, spread over 15 `s_wait_loadcnt 0x2` per group (~500 cycles each):
  the 128 u16 gathers per group per wave drain at ~3.7 cycles per wave-instruction through one
  CU's memory pipeline -> phase B ~15 us of INSTRUCTION throughput, not latency.
* flash2 k32 (b=4): only 30% stall; 17% of it the kernarg->meta scalar chain, then the q-stage
  wait and LDS waits; 42% "idle" -> per-WG fixed cost (q stage, 3 barriers, partial stores) and
  1-2 WGs per WGP; explains the bad scaling with b.
* fused_vt_qreg_d8: 13.7 us wave latency / 23 us life: 18% of stall = phase-S K round trips,
  10% = scalar prologue (2.2K cycles!), ~35% = per-tile `s_wait_dscnt 0x1` (store->load RAW in
  the private ring, ~150 cycles x 40); 40% of life "idle" (phase-S imbalance 3 vs 2 tiles per warp,
  4 waves/SIMD issue sharing).

#### Attempts 2c-2f (bit-exact fused, one launch, grid (4,b) x 512) — what moved the needle
* qlds (q staged in LDS, pf2 phase B): 37.5 us — SLOWER than fused_f16s (33.4): phase S is
  per-warp latency-bound, the LDS q adds a round trip + barrier.  [cand4]
* pf2r4 / pf1r8 (deeper V prefetch): = pf2r2 -> phase B not load-latency-bound.  [cand4]
* wpriv (warp-private LDS ring, NO phase-B barriers): 36-38 us, no gain -> not barrier-bound.  [cand5]
* vt (TRANSPOSED warp-private V tile [dim][key], 40 B row stride; weights tile-major so the A
  fragment is one ds_load_b128; B fragments 2 x ds_load_b64 instead of 8 x ds_load_u16): 27.3 us
  (-35%).  Phase B was LDS-INSTRUCTION-bound: 24 LDS ops per tile per lane -> 13.  [cand6]
* vt_qreg (+ the 32 q fragments cast ONCE per warp into 128 VGPRs; K loads only per tile):
  20.3-22.3 us at b=1..8 (x0.42-0.49 of the pair). Phase S was global-load-instruction-bound
  (64 q b128 loads + 32 K loads per tile per lane -> 32).  [cand6]  256 VGPRs, 1 spill (8 B).
* vt_qreg_sp_d4 (stage tile t+1 before the fragment loads of tile t): 19.9-21.8 us, -2% more,
  consistent over 3 runs x 6 shapes.  [cand7]  BEST BIT-EXACT.

#### FINAL numbers: 3 separate process runs each (`results/final_run{1,2,3}.txt`, `sp_run{1,2,3}.txt`,
`python3 agg.py ...`), quiet box, graph mode, per call, medians of 3 medians (ranges tight, <=2%):
  b:                          1       2       3       4       5       8
  base pair (prod)          41.9    41.1    43.8    46.9    47.4    53.3   us
    dec_score                9.8     9.7    12.3    15.4    15.9    21.4
    smwsum                  33.2    32.7    32.9    32.6    32.9    33.0
  fused_vt_qreg_sp_d4       19.9    19.9    20.0    20.4    20.6    21.8   x0.48 .. x0.41  BIT-EXACT
  fused_vt_qreg_d8          20.5    20.2    20.6    20.8    21.2    22.3   BIT-EXACT
  fused_vt_d8               27.4    26.9    27.2    27.4    27.8    28.5   BIT-EXACT
  pair w/ smwsum_pf2 (a.1)  30.9    30.6    33.2    36.3    36.8    42.6   BIT-EXACT
  fused_f16s (a.1)          33.5    32.9    33.1    33.2    33.5    34.1   BIT-EXACT
  score blk128 (vs score)   7.0     8.4    10.1    11.7    15.3    20.1   BIT-EXACT (blk256: 7.0/9.6/10.1/12.8/15.5/18.6)
  flash3 k128 (+combine)    12.0    28.2    18.7    35.4    29.5    49.4   NOT bit-exact
  flash3 k64                13.3    26.0    21.2    35.3    32.4    52.9   NOT bit-exact
  flash2 k128               13.4    30.8    21.3    39.3    34.8    55.7   NOT bit-exact
  kv chain (b=4): fp8_act_quant 4.2 + f16_roundtrip 3.2 + kv_cache_append 3.2; chain 8.6 -> 6.4
  without f16_roundtrip (-2.2..-2.4 us per lane-layer at b=1/4/8; `results/f16rt_timing_run1.txt`).

Leverage: at the typical b=3-4 per lane the pair costs 44-47 us per lane-layer in this harness;
fused_vt_qreg_sp_d4 saves 24-26.5 us per lane-layer -> 40 layers = ~1.0 ms per lane critical path,
80 lane-layers = ~2.0 ms/step of dGPU busy (both lanes). blk128 alone: -2..-4 us -> ~0.2 ms/step.
f16_roundtrip drop: -2.3 us x 80 = ~0.18 ms/step.

Caveats: harness = synthetic data, warm operands (as production decode), graph of 10 calls per
timed block; production numbers include event pairs (4.6 us) which the harness does not; the
production smwsum measured 33 us here vs the ~16 us quoted in the inventory (unexplained; same
grid/args, 640 keys). The fused kernels require n_total <= 640, n_head % 16 == 0, head_dim 512,
no comp_allowed_bits mask (decode passes none), and they do NOT write the softmax weights back to
sd.attn_scores (nothing reads them). Integration = new kernel + wrapper replacing the two launches
at FP:5805/5889 when attn_dec_score_for(b); scores buffer untouched (can stay allocated).

Dead ends: fused_direct (u16 global gathers, TA-bound; g32 variant miscomputes on partial tiles
with spills - not debugged); fused_qlds (slower); pf2r4/pf1r8 (no gain); wpriv (no gain, LDS-op
bound); wpriv_s2 (spills, 51 us); flash (cand2) spilled; flash2/flash3 scale badly with b
(per-WG fixed cost + 1-2 WGs/WGP + partials traffic) and are non-bit-exact: only worth it at b=1.

Repro (all from this dir; every GPU run via the scheduler):
  bash build2.sh all                                  # baselines from the unmodified in-tree .hip + cand2..7 + harness
  bash ../_infra/gpu_run.sh --dev dgpu --mb 96 --label D_attention/check -- ./attn_harness2_gfx1201 gfx1201 . check
  bash final.sh final 3 <CANDS csv>                   # 3 separate decode runs; then bash w.sh TICKET results/x.txt
  python3 agg.py results/final_run1.txt results/final_run2.txt results/final_run3.txt
  bash ../_infra/gpu_run.sh --dev dgpu --mb 96 --label D_attention/f16rt -- ./attn_harness2_gfx1201 gfx1201 . f16rt
  ATT: gpu_submit ... -- bash ../_infra/prof.sh att dgpu <regex> results/att_x -- ./attn_harness2_gfx1201 gfx1201 . prof 8 <cand>; bash attsum.sh results/att_x
