# C1_dense_decode — dGPU dense Q8_0 projections at decode (gfx1201, RX 9070 XT)

Owner: kernel-ideas sweep 2026-09-26. Baseline = production 361d4f9 code objects built from the
UNMODIFIED in-tree `.hip` with `$KFLAGS_V41 --genco --offload-arch=gfx1201` (build_base.sh).
Candidates: cand_bpack.hip (build.sh). Harness: harness.cpp (modes gemv / grouped / quant / swiglu /
lanes / shared). Runs: run_one.sh, run_sweep.sh, run_small.sh, prof_att.sh. Raw outputs: results/.

## Production launch facts (read from the wrappers/call sites, not re-derived)

| site | kernel | M x K | grid x block | bytes/call | floor @640 GB/s |
|---|---|---|---|---|---|
| q_a (FP:3832 via dispatch.rs:302) | `q8_0_gemv_bpack_warp8` | 1280 x 5120 | (160)x256 | 6.96 MB | 10.9 us |
| qb (FP:3892) | same | 32768 x 1280 | (4096)x256 | 44.56 MB | 69.6 us |
| kv (FP:3984) | same | 512 x 5120 | (64)x256 | 2.79 MB | 4.35 us |
| wo_a (FP:6132 -> q8_0.rs:821 -> :760) | `q8_0_grouped_gemv_bpack` | 8 x (1024 x 4096) | (1024)x256 | 35.65 MB | 55.7 us |
| wo_b (FP:6167) | bpack | 5120 x 8192 | (640)x256 | 44.56 MB | 69.6 us |
| shared gate, up (FP:1847/1856) | bpack | 2304 x 5120 each | (288)x256 | 12.53 MB each | 19.6 us each |
| shared down (FP:1901) | bpack | 5120 x 2304 | (640)x256 | 12.53 MB | 19.6 us |
| Engram wkv (FP:3524, L1+L14) | bpack | 25600 x 6144 | (3200)x256 | 167.1 MB | 261 us |
| head (forward_head.rs:94, per lane) | bpack | 129280 x 5120 | (16160)x256 | 703.3 MB | 1099 us |
| quantize (7 sites/lane-layer) | `q8_0_quantize_f32` | K/32*b blocks | (K/32*b)x32 | tiny | launch-bound |
| swiglu (FP:1868) | `swiglu` | 2304*b | (ceil(2304b/256))x256 | tiny | launch-bound |

Per lane-layer dense weight bytes: 6.96+44.56+2.79+35.65+44.56+2*12.53+12.53 = **172.1 MB -> 269 us floor**.
Args: bpack = (out, w, xq, xscale, in_dim=K, out_dim=M, blocks=K/32, batch); grouped = (out, w, xq,
xscale, group_dim, rank, blocks_per_group, n_groups, batch). Block 256 = 8 waves x 32 lanes, one wave
per output row, lane-strided over K/32 blocks (lane `bl = lane, lane+32, ...`).

Regime: weights COLD (178 MB/lane-layer >> 64 MB MALL), activations warm, graph-captured stages.
Harness: `kb::ab` graph mode; a READ flush (72 MB streamed as clean lines, `c1_read_flush`) before
every timed block; rotation over N weight copies (sum <= 36 MB) as graph nodes where the weight is
small, inner=1 for the 35-45 MB weights. A `null 1-WG kernel` variant in every A/B measures the
launch floor of that graph shape.

Device budget: VRAM free 408-438 MB; gpu_run margin 270 MB -> <= 115 MB per run. Engram (167 MB)
and head (703 MB) cannot be allocated at full size: measured as shape proxies with the SAME K and
grid geometry but fewer rows (6400 x 6144 = 41.8 MB; 6464 x 5120 = 35.2 MB) — one wave per row,
no cross-row interaction, so per-byte behaviour transfers.

## ISA (isa.sh, production code objects)

| kernel | vgpr | sgpr | scratch | spills |
|---|---|---|---|---|
| q8_0_gemv_bpack_warp8 | 51 | 30 | 0 | 0 |
| q8_0_gemv_warp8 | 30 | 16 | 0 | 0 |
| q8_0_gemv_batched_warp8 | 31 | 22 | 0 | 0 |
| q8_0_grouped_gemv_bpack | 51 | 33 | 0 | 0 |
| q8_0_quantize_f32 | 21 | 54 | 0 | 0 |
| swiglu | 10 | 10 | 0 | 0 |

The bpack inner loop (base_q8_gfx1201.s): the runtime `for (b < batch)` is NOT unrolled. Per
(block, b): `v_movrels_b32 / v_movreld_b32` (M0-relative indexing of `acc[16]`), 2x
`global_load_b128` + 1x `global_load_b32` for that row's activation, then `s_wait_loadcnt 0x0`
INSIDE the b-loop. So after each weight block's DRAM loads a wave does `batch` serialised L2
round trips before issuing the next block's DRAM loads.

## What the baseline waits on (ATT, results/att_base_qa_b8/, q_a b=8, cold, dispatches 14-15)

stall/latency = 95.2-95.6%. Top stalls: **66-69% `s_wait_loadcnt 0x2` in the b-loop body** (400
hits = per (block, b) — the per-row activation L2 loads), **22-25% `s_wait_loadcnt 0x2` per block**
(50 hits — the weight DRAM loads), 4% `s_wait_loadcnt 0x0`, everything else < 1%. Two thirds of
the kernel's time at b=8 is the serialised per-row activation wait, not DRAM. That is the "same
bytes, 2x the time" mechanism the task asked about: not register pressure (51 VGPRs, no spills),
not re-reads (the weight is loaded once), but lost memory-level parallelism.

## Measurement log

### 04:00 — memset flush artefact
The first pass used `kb::Flusher` (memset): it leaves 80 MB of DIRTY lines in the MALL whose
write-back is charged to the kernel under test (q_a b=1 read 37 us). Production's "flush" is other
weights' clean reads, so the harness now uses a read flush: q_a b=1 16.4 us (rotation, 8 copies).

### Production-load hazard
The hub was at 56-100% dGPU busy during every run. The `null 1-WG kernel` reads ~4-7 us in a
rotation run (inner=N) and ~13-15 us in an inner=1 run when the hub is quiet, and 20-130 us when
it contends; in those windows all medians are 2-4x inflated. Criterion: a run is QUIET when the
null median < 2.5x its p10 (and < 20 us). Contaminated points are re-run.
Kernel-time estimate = med - null floor (the fixed per-node dispatch cost production also pays).

### Sweep r2 (results/gemv_*_r2.txt), quiet points only (med us; GB/s net of the null floor)
| shape | b | base med | tB med | ratio | base GB/s | tB GB/s |
|---|---|---|---|---|---|---|
| q_a 7.0 MB | 1 | 16.4 (rot8) | 15.5 | 0.95 | 600 | 650 |
| q_a | 2 | 18.7 | 16.6 | 0.89 | 499 | 588 |
| q_a | 4 | 21.6 | 17.1 | 0.79 | 417 | 575 |
| qb 44.6 MB | 1 | 89.4 | 88.4 | 0.99 | 583 | 591 |
| qb | 4 | 96.1 | 91.2 | 0.95 | 547 | 581 |
| qb | 8 | 117.1 | 102.6 | 0.88 | 436 | 508 |
| kv 2.8 MB | 1 | 9.3 | 8.9 | 0.96 | 511 | 552 |
| kv | 4 | 14.5 | 10.7 | 0.74 | 270 | 425 |
| wo_b 44.6 MB | 2 | 90.1 | 88.2 | 0.98 | 585 | 600 |
| wo_b | 4 | 105.5 | 88.0 | 0.83 | 487 | 603 |
| wo_b | 5 | 113.4 | 89.2 | 0.79 | 447 | 590 |
| gate 12.5 MB | 1 | 27.4 | 27.0 | 0.99 | 598 | 608 |
| gate | 4 | 37.0 | 28.1 | 0.76 | 411 | 580 |
| gate | 5 | 37.3 | 28.7 | 0.77 | 407 | 566 |
| down 12.5 MB | 4 | 32.1 | 28.2 | 0.88 | 489 | 576 |
| down | 5 | 42.1 | 32.8 | 0.78 | 358 | 488 |
| down | 8 | 49.7 | 35.5 | 0.71 | 294 | 440 |
| engram_proxy 41.8 MB | 1 | 83.3 | 83.3 | 1.00 | 605 | 605 |
| engram_proxy | 5 | 105.1 | 83.8 | 0.80 | 462 | 605 |
| engram_proxy | 8 | 126.8 | 85.8 | 0.68 | 371 | 584 |
| head_proxy 35.2 MB | 1 | 72.2 | 71.4 | 0.99 | 607 | 616 |
| head_proxy | 4 | 87.1 | 71.4 | 0.82 | 483 | 614 |
| head_proxy | 5 | 91.8 | 73.4 | 0.80 | 455 | 593 |
| head_proxy | 8 | 129.0 | 90.1 | 0.70 | 306 | 463 |

Reading: at b=1 the production kernel is at 92-95% of 640 GB/s on every shape >= 7 MB (kv, 2.8 MB
= 512 waves, is latency-bound at ~80%). From b=4 up the SAME bytes take 1.3-1.8x longer. The
compile-time-B twin (tB) holds 570-615 GB/s to b=5 on every shape and 440-580 at b=8, bit-exact
at every (shape, b) tested. K-unroll (ku2/ku4) on top of tB never wins (more VGPRs, no extra MLP
once the b-loop is unrolled): dead end. Warm-cache control (results/gemv_qa_8_warm.txt): base
18.4 us vs tB8 10.8 us (0.59) — the b-loop cost is even larger when DRAM is not the limiter.

## Ideas (ranked by expected gain x confidence / effort)

1. **tB — compile-time batch instantiations of `q8_0_gemv_bpack_warp8` (and the grouped twin)**.
   Bit-exact. Attacks the measured bottleneck (serialised b-loop). Drop-in: the wrapper switches on
   `batch` (1..8) to a symbol; b>8 keeps the runtime kernel. MEASURED WIN (above).
2. **Shared expert fused chain**: gate+up in ONE kernel (each wave computes gate row r and up row r,
   weight streams interleaved), swiglu in the epilogue, and — with a 32-wave (1024-thread) WG owning
   32 consecutive rows — the Q8_0 block quantize of `mid` in the epilogue via LDS: removes the `up`,
   `swiglu` and `quantize_mid` launches (3 graph nodes) and the f32 `mid` round trip. Bit-exact
   possible (same per-row GEMV math, elementwise swiglu, fmaxf amax is order-free). Medium effort.
3. **Two-lane merged dense pass**: at >= 6 rows both lanes stream every dense weight (FP:2753); one
   tB(b0+b1) launch reads it once. Kernel prototype = tB at b=8..10 with two (xq, out) bases;
   priced by `harness lanes`. Integration LARGE (lanes are separate graphs; needs a join per site).
4. **Activation quantize**: (a) drop the duplicate `attn_input_norm` quantize (FP:3979 is an
   identical no-op of FP:3811 — host-only, -1 node); (b) `q8_0_quantize_f32_wave` (one wave per
   block, 32x less redundant work; bit-exact) for the remaining launches; (c) quantize in the GEMV
   prologue (each WG re-quantizes its K-slice from L2) — costs B x K x 4 B of L2 reads per WG, 26 MB
   (q_a) .. 168 MB (qb) of L2 traffic per launch; only sane for the small-grid sites.
5. **Dead casts**: 4 `f32_to_f16_cast_2d` per lane-layer feed only the f16x arms (FP:3799, 3860,
   6093, 6147) — host-only; ~2.5 us node each = ~10 us/lane-layer = 0.8 ms/step at 80 lane-layers.
6. **K-unroll for latency-bound small-M shapes (kv 512 rows)**: measured — no gain at b=1 (the 5
   blocks per lane are already pipelined by the hardware's outstanding-load queue). Dead end.
7. **Q8_0 layout**: already split (scales | quants, M18 repack) with aligned b128 loads — closed.
8. **WG geometry (4/16 waves per WG)**: not tried; wave count is fixed by M so no MLP change expected.

## RESUME 2026-09-26 evening (hub DOWN, dGPU idle, 16.2 GB VRAM free, scheduler live)

State at resume: all r2 numbers above were taken under a 56-100% busy hub; the grouped/lanes/quant
files show null floors of 24-126 us in several points (contaminated). Plan:
1. Add a `sweep` harness mode (all shapes x b in ONE scheduler job), free device buffers per shape,
   and measure head (703 MB) and Engram (167 MB) at FULL size now that VRAM allows it (<= 4 GB/job).
2. Clean re-measure of base vs tB (3 separate runs), grouped tB, quant_wave, lanes merge.
3. Structural candidate: shared-expert fused chain `cand_shared.hip` — gate+up in one wave
   (both weight rows streamed per block, 2B accumulators), swiglu in the epilogue, and a variant
   with the Q8_0 quantize of `mid` fused via LDS (WG owns 32 consecutive rows = one Q8_0 block
   per b; rows-per-wave template RPW in {1,2,4} -> WG of 32/RPW waves). Expected bit-exact.
4. Cheap knob: WG geometry (4/16 waves per WG) on tB.
5. If time: LDS-staged activations for b>=8 (the merged-lane regime), where tB8 falls to 440-580 GB/s
   with 95 VGPRs (occupancy 16/16 -> not occupancy; suspect 8x L2 activation traffic per weight byte).

### r3 (hub DOWN, clean; results/gemv_sweep_r3.txt, shared_r3.txt, small_r3.txt) — ONE job each
Harness: `./harness <mode> ... : <mode> ...` (':'-separated modes in one process), 60 rounds, graph,
72 MB clean read flush between timed blocks; head and Engram now at FULL size (703 / 167 MB).
Null floor (1-WG kernel in the same graph shape) = 4-7 us with rotation (inner=N copies), 13-15 us at
inner=1 (one graph launch per timed block). Net GB/s below = bytes / (med - null).

gemv, base `q8_0_gemv_bpack_warp8` vs `tB<b>` (bit-exact at every (shape,b); w4/w16 = same within noise):
| shape (MB) | b | base med us | tB med us | ratio | base net GB/s | tB net GB/s |
|---|---|---|---|---|---|---|
| q_a 6.96 | 1 | 16.87 | 16.43 | 0.97 | 580 | 602 |
| q_a | 2 | 18.33 | 16.78 | 0.92 | 523 | 593 |
| q_a | 4 | 21.35 | 17.17 | 0.80 | 428 | 577 |
| q_a | 5 | 22.86 | 17.26 | 0.76 | 391 | 570 |
| q_a | 8 | 27.50 | 18.86 | 0.69 | 309 | 502 |
| qb 44.56 | 1 | 90.48 | 89.40 | 0.99 | 585 | 593 |
| qb | 4 | 95.12 | 92.32 | 0.97 | 551 | 571 |
| qb | 5 | 96.64 | 93.24 | 0.97 | 539 | 563 |
| qb | 8 | 104.44 | 96.56 | 0.93 | 496 | 543 |
| kv 2.79 | 1 | 9.11 | 8.88 | 0.98 | 536 | 561 |
| kv | 4 | 12.60 | 9.35 | 0.74 | 322 | 516 |
| kv | 5 | 13.66 | 9.56 | 0.70 | 286 | 494 |
| wo_b 44.56 | 1 | 88.32 | 88.52 | 1.00 | 601 | 600 |
| wo_b | 4 | 107.28 | 89.92 | 0.84 | 478 | 588 |
| wo_b | 5 | 114.84 | 90.52 | 0.79 | 443 | 585 |
| wo_b | 8 | 135.32 | 92.44 | 0.68 | 369 | 572 |
| gate 12.53 | 1 | 27.52 | 27.33 | 0.99 | 614 | 620 |
| gate | 4 | 33.95 | 27.55 | 0.81 | 457 | 597 |
| gate | 5 | 36.88 | 28.41 | 0.77 | 418 | 583 |
| down 12.53 | 1 | 27.75 | 27.88 | 1.00 | 608 | 605 |
| down | 4 | 32.36 | 28.55 | 0.88 | 495 | 583 |
| down | 5 | 33.84 | 29.15 | 0.86 | 466 | 564 |
| down | 8 | 40.61 | 31.75 | 0.78 | 373 | 506 |
| Engram 167.1 (FULL) | 1 | 280.44 | 280.36 | 1.00 | 628 | 628 |
| Engram | 4 | 328.80 | 284.00 | 0.86 | 531 | 620 |
| Engram | 5 | 349.84 | 285.12 | 0.82 | 498 | 617 |
| head 703.3 (FULL) | 1 | 1255.4 | 1255.2 | 1.00 | 567 | 567 |
| head | 4 | 1448.2 | 1269.5 | 0.88 | 490 | 560 |
| head | 5 | 1539.9 | 1273.2 | 0.83 | 461 | 559 |
| head | 8 | 1829.9 | 1289.7 | 0.70 | 387 | 552 |
(The full head at b=1 is 567 GB/s net with a cold flush — lower than the 630 GB/s in the inventory,
which was measured in the server without a flush; the MALL keeps ~9% of a 703 MB stream.)

grouped wo_a (35.65 MB, 1024 WGs), base vs `q8_0_grouped_gemv_bpack_tB<b>` (bit-exact):
b=1 78.76 / 80.32 (1.02, noise-level loss); b=2 79.08 / 71.76 (0.91); b=4 98.32 / 73.52 (0.75);
b=5 98.24 / 74.16 (0.75); b=8 122.04 / 82.92 (0.68). Net: base b=1 561 GB/s, b=4 426; tB4 605.

lanes (two lanes b0+b1 as ONE tB launch; bit-exact vs the two per-lane outputs), cold, inner=1:
| shape | 2 x base b=4 (sum) | 2 x tB4 (from sweep) | merged tB8 | merged / 2xtB4 |
|---|---|---|---|---|
| q_a 4+4 | 61.6 | 34.3 | 27.9 | 0.81 (0.45 vs base) |
| q_a 5+5 | 65.1 | 34.5 | 29.2 (tB10) | 0.85 |
| wo_b 4+4 | 216.4 | 179.8 | 90.7 | 0.50 |
| gate 3+3 | 79.8 | ~55 | 35.4 (tB6) | 0.64 |
| down 4+4 | 80.3 | 57.1 | 38.6 | 0.68 |
| qb 4+4 | 189.3 | 184.6 | 95.4 | 0.52 |
(each launch pays ~13 us of graph-launch floor in this inner=1 measurement, so the small shapes
show less than the 2x byte saving; the big shapes show it fully.)

quantize (warm, 10-node graph, per node): base 4.0-5.4 us vs wave 3.6-3.9 us at K in {1280,2304,
5120,8192} (null 3.3 us => launch-bound; the wave kernel removes 0.3-1.5 us per launch).
**K=32768 (the wo_a input, FP:6126) is 19.4 us at b=4 for BOTH kernels** — not launch-bound;
b=8 in r2 was 13 us base / 5.8 wave. To be understood (r4 runs b=1,2,4,5,8).
swiglu: 3.45-3.51 us per node vs null 3.3 => pure launch cost.

shared-expert chain (gate 2304x5120, up, swiglu, quant, down 5120x2304; 37.6 MB cold; 59 us byte floor):
| b | base 5 nodes | A (fused g+u+swiglu; quant; down) 3 nodes | A2 (+quant_wave) | B r1 (fused g+u+swiglu+quant, 1024-thr WG; down) 2 nodes | B r2 | B r4 |
|---|---|---|---|---|---|---|
| 1 | 88.00 | 81.72 (0.93) | 80.88 (0.92) | **76.56 (0.87)** | 77.36 | 77.52 |
| 2 | 92.32 | 84.08 (0.91) | 82.72 (0.90) | **78.92 (0.86)** | 79.44 | 81.56 |
| 4 | 107.48 | 88.48 (0.82) | 87.20 (0.81) | **83.12 (0.77)** | 85.64 | 85.44 |
| 5 | 114.84 | 90.68 (0.79) | 89.24 (0.78) | **85.44 (0.74)** | 88.60 | 91.64 |
| 8 | 137.96 | 108.56 (0.79) | **107.68 (0.78)** | 111.68 (0.81) | 117.24 | — |
All candidates bit-exact on mid, mid_xq, mid_xs and the final out at every b. Null floor 13.3 us.
The fused gate+up+swiglu alone: 53.2 us (1 node) vs 60.4 us (3 production nodes) at b=1 = 25 MB at
628 GB/s net; at b=5 54.2 vs 79.2. B r1 (1024-thread WG, 72 WGs) is best up to b=5; at b=8 its
100 VGPRs leave one 32-wave WG per WGP (32 resident of 72 WGs -> tail), so A2 wins there.
Production b per lane is 1-5, so B r1 is the pick; the down GEMV inside it is still the runtime
kernel (r4 adds "B r1 + tB down").

### r4, r5 = separate process runs (results/*_r4.txt, *_r5.txt): everything reproduces within 1-2%
gemv sweep r3/r4/r5 medians (base -> tB): q_a b=4 21.35/20.88/20.89 -> 17.17/16.63/16.62;
wo_b b=4 107.3/106.2/105.9 -> 89.9/89.0/89.0; head b=4 1448/1448/1445 -> 1269/1269/1269;
grouped b=4 98.3/98.0/98.3 -> 73.5/71.8/73.1. Shared chain b=4 base 107.5/106.6/106.7 -> B r1 83.1/82.5/82.5.

Shared chain attribution (r4/r5, b=4): tB-everything 5-node chain 83.6/84.0 (0.785); B r1 + tB down
79.16/79.16 (0.742); A2 + tB down 81.9/82.0. So at b=4-5 the compile-time-B kernels give ~23 of the
27 us; the fusion (3 fewer nodes, no f32 gate/up/mid round trip) adds 4.4 us at b=4, 5.8 at b=5,
7.8 at b=1 (where tB itself is worth nothing). Best chain = **B r1 + tB<b> down** for b<=5:
b=1 77.1-77.6, b=2 77.9, b=4 79.2, b=5 80.2-80.3 us vs base 86.7-88.0 / 91.4-92.3 / 106.6-107.5 / 113.8-114.8.

ATT of tB8 (q_a b=8, results/att_tB8_qa_b8): stall/latency 87%; the waits are `s_wait_loadcnt`
at partial counts (0x12, 0x4, 0xc, 0x1 ... = 26 loads issued per block, consumed in order); no
movrel, no VALU stall. It is memory latency with 1 KB of DRAM in flight per wave; only more bytes
in flight per wave (K-unroll: measured loss, VGPRs) or fewer L2 loads (LDS-staged activations, not
built) could move it. tB8 is already 502-572 GB/s net on >= 7 MB shapes; b=8 per lane does not occur
in production (b<=5 per lane), so this was not pursued.
ATT of shared B r1 (b=4, results/att_sharedB1_b4): stall/latency 79%, `s_wait_loadcnt` + global_load
issue back-pressure; 611 GB/s net for the 25 MB gate+up stream = 95% of roof. Done.

### QUANTIZE GRID PATHOLOGY (results/quant32k_q1.txt, quant_alias_a1.txt, quant_grid{0,1,3}_g1.txt, quant_nograph_g1.txt)
`q8_0_quantize_f32` (and the wave twin) take 16.7-18.6 us instead of 4-6 us when the block count is
2046-2048 or 4096 (65536 / 131072 elements): K=32768 b=2/4 (= the wo_a-input quantize FP:6126 at
2 and 4 rows per lane), 65536x1, 16384x4/8, 8192x8 (= wo_b input at b=8). NOT address aliasing
(shifting x / xq / xscale by 256 B .. 128 KB changes nothing), NOT graph-specific (plain launches
identical), NOT element-count (32767x2 = 2046 blocks slow, 32800x2 = 2050 fast). **Launching ONE
extra idle WG (grid = blocks + 1; the kernel's `b >= blocks` guard makes it a no-op) removes it
completely**: K=32768 b=2 16.8 -> 5.85 us (base) / 3.9 (wave); b=4 18.6 -> 8.2 / 4.5; 65536x1
16.7 -> 5.8; 16384x4 16.7 -> 5.8. Fast shapes are unchanged by the pad (+3 WGs: same). Production
impact: one launch per lane-layer at b=2 and b=4 per lane is ~11 us too slow -> 0.44 ms/step per lane
at 40 layers (0.9 ms of dGPU time at 2 x 4 rows). Fix = one line in the wrapper (grid + 1) or the
wave kernel at a non-power-of-two WG count. The GEMV grids that are powers of two (qb 4096 WGs,
wo_a 1024, kv 64) are NOT affected (results/gemv_gpad0.txt vs gemv_gpad1.txt: same within noise).

## FINAL SUMMARY (2026-09-26 ~20:00 UTC) — what to integrate, with the arithmetic

Per-lane-layer decode dense sites: q_a, qb, kv, wo_a (grouped), wo_b, shared gate/up/down (8 GEMVs),
7 quantize launches, 1 swiglu; Engram at L1/L14; head once per lane. 40 layers; 1 lane below 6 rows,
2 lanes (b=1-5 each) at >= 6 rows. All savings below are dGPU device time; the dense stages are on
the decode critical path (INVENTORY §2), so they should convert ~1:1 to step time for one lane and
mostly for two (the two lanes' dense kernels serialise on the one dGPU).

1. **tB — compile-time-B `q8_0_gemv_bpack_tB<b>` + `q8_0_grouped_gemv_bpack_tB<b>`** (cand_bpack.hip).
   Bit-exact. Drop-in: `matvec_bpack` / grouped wrapper select the symbol by `batch` (1..8), keep the
   runtime kernel for b>8. Savings per lane-layer at b=4 (r3-r5 medians): q_a 4.2 + qb 2.9 + kv 3.3 +
   wo_a 25.0 + wo_b 17.3 + gate 6.3 + up 6.3 + down 3.8 = **69 us -> 2.77 ms/step** + Engram 2x44.7 us
   + head 178 us = **3.03 ms/step at 4 rows (1 lane)**; at b=5 83 us/lane-layer -> 3.73 ms/lane;
   **2 lanes x 4 rows: 6.06 ms of dGPU time/step**. b=1: 0-3% (nothing to gain; kernel at 91-98% of roof).
2. **Shared-expert fused chain `shared_gateup_swiglu_q8_tB<b>_r1` (1024-thread WG) + tB down**
   (cand_shared.hip). Bit-exact (mid, mid_xq, mid_xs, out). Replaces 5 nodes by 2; drops the f32
   gate/up/mid buffers' round trips. Chain b=4: 106.9 -> 79.2 us (**-27.7 us/lane-layer = 1.11 ms/step
   at 4 rows, 2.2 ms dGPU at 2x4**), of which 4.4 us is the fusion itself on top of tB (0.18 / 0.35 ms).
   b=1: 87.4 -> 77.1 (-10.3 = 0.41 ms/step). Integration small: new kernel + wrapper; the call site
   FP:1847-1901 replaces gate/up/swiglu/quantize_mid launches by one launch when
   `small_b_dense_dp4a(b)` and b<=5 (use `_tB8` A-variant + quant for 6..8, or keep the tB chain).
3. **Quantize grid pad (grid = blocks + 1)** for `q8_0_quantize_f32`: bit-exact (same kernel), one-line
   wrapper change. Removes ~10-11 us per lane-layer at b=2/b=4 per lane (the K=32768 wo_a input):
   **0.41-0.44 ms/step per lane; 0.82 ms dGPU at 2x4**. Also protects the wo_b-input quantize at b=8.
4. **`q8_0_quantize_f32_wave`** (cand_bpack.hip): bit-exact; -0.4..-1.2 us per launch on the warm
   launch-bound sites (K 1280..8192), -3.7 us on K=32768 (with the pad). ~6.6 us/lane-layer over the
   5 remaining launches after (2) -> 0.26 ms/step (1 lane), 0.5 ms (2 lanes). Small.
5. **Two-lane merged dense pass** (prototype = tB at b0+b1 with both lanes' xq/out contiguous;
   harness `lanes`): bit-exact vs the two per-lane outputs. At 4+4 rows one launch costs 0.45-0.52 of
   the two production launches on the >= 7 MB shapes (q_a 27.8 vs 61.3; wo_b 90.7 vs 211; qb 91.9 vs
   182; down 38.5 vs 79.9) and 0.81-0.85 of two tB4 launches on the small ones. Summed over the 8
   sites vs 2 x tB4: ~315 us/layer -> **~12.6 ms dGPU/step at 8 rows (+1.8 ms Engram/head)**, i.e.
   the dense byte stream halves. Integration LARGE: lanes are separate graphs on separate streams,
   offset in time; each merged site needs a cross-lane join (event pair ~5-10 us x ~10 sites/layer
   = up to 4 ms/step of the gain back) and it removes the pipelining the lanes exist for. Price it
   in the pipeline, not here.
6. **WG geometry (4 / 16 waves per WG)**: neutral (+-1%) at every shape and b. Closed.
7. **K-unroll (ku2/ku4) on top of tB**: loss (VGPRs 95-144, no extra MLP). Closed (first attempt).
8. **Dead `f32_to_f16_cast_2d` x4 and the duplicate `attn_input_norm` quantize (FP:3979)**: host-only
   removals, not measured here; a warm graph node costs 3.2-3.4 us (null floor) -> ~13.6 + 4.5 us per
   lane-layer = 0.7 ms/step (1 lane), 1.4 ms (2 lanes). Untested.
9. **LDS-staged activations for b >= 8** (the merged-lane regime): not built; tB8 is at 502-572 GB/s
   net on the >= 7 MB shapes (ATT: pure load latency, 26 loads/block), so the ceiling is ~+10% at b=8
   only. Untested.

Reproduce: `bash build_base.sh; bash build.sh; bash submit_r3.sh rN` (3 tickets: shared / gemv sweep /
small), collect with `bash wait_to.sh TICKET file.txt`; quant pathology: `bash submit_quant.sh`,
`bash submit_quant_alias.sh`, `bash submit_quant_grid.sh`; ATT: `bash prof_att_submit.sh <regex> <outdir> <harness args>`
then `bash att_summary.sh <outdir>`.
