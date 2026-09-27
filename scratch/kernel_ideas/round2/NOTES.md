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
