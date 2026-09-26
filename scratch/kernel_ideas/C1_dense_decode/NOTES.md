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
