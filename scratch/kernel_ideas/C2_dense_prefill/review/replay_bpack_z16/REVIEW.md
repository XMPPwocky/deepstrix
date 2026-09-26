# Review: C2_dense_prefill/replay_bpack_z16 — verdict CONFIRMED (kernel level)

Reviewer: skeptical re-run, 2026-09-26 ~20:50 UTC, dGPU via the scheduler (`--label review/C2_dense_prefill`).
Nothing of the engineer's was modified; everything here was rebuilt from read-only copies in this directory.

## 1. Baseline fidelity (matches production)

* In-tree sources `q8_0_matvec.hip`, `q8_0_grouped_matvec.hip`, `q8_0.rs`, `forward_prefill.rs` are unmodified
  vs 361d4f9 (`git diff --stat 361d4f9` empty). Baseline hsacos rebuilt here with the exact
  `$KFLAGS_V41 --genco --offload-arch=gfx1201` (sources.sha256).
* Symbols: `q8_0_gemv_batched_warp8` (q8_0.rs:261) and `q8_0_grouped_gemv_batched` (q8_0.rs:869) — the same the
  harness launches (`base_kernel()`).
* Grid/block/args: harness `launch_base` = grid (ceil(M/8),1,b) x 256, args (out,w,xq,xscale,K,M,blocks) and
  grid (ceil(G*rank/8),1,b) x 256, args (out,w,xq,xscale,group_dim,rank,blocks,n_groups): identical to the
  Rust `LaunchConfig`/`launch_kernel!` lists.
* Production regime for this path: replay runs ONCE per prefill job (`prefill_job_finish`, FP:687-750) with
  `b_seg = min(t, SWA_WINDOW=128)`; `lane_split` with no image spans cuts at `b.div_ceil(2)` = 64/64; per lane
  `prefill_f32_matvec(64)` is true -> "dp4a" arm -> `matvec_batched(.., 64)` -> `bpack_ok(64)` false (cap 16)
  -> the grid.z=64 kernel. Direct launches on `de.compute` (prefill is not graph-captured). Weights are read
  once per lane-layer -> cold. Harness: cold (96 MB memset flush > 64 MB MALL + 8 MB L2), one call per timed
  block, interleaved with rotated order, event pair per call. Inputs random (int8 activations, f32 scales in
  [0.002,0.02], f16 weight scales in [0.002,0.02]) — not degenerate.
* Candidate = production `q8_0_gemv_bpack_warp8` body with a `blockIdx.z*16` row offset; 51 VGPRs, 0 scratch,
  no LDS (the 20480 B in the isa.sh table belongs to the next row — off-by-one; the disassembly has only the 5
  `ds_bpermute` of `warp_sum_f32`, same as production). One kernel, no extra reduce/repack, writes all b*M outputs.

## 2. Re-run (three separate processes, results/replay_rv1..rv3; rv1 with --corr)

| site (b=64) | base med us (rv1/rv2/rv3) | z16 med us | ratio | engineer |
|---|---|---|---|---|
| qb64 32768x1280 | 1507.3 / 1508.1 / 1508.4 | 577.5 / 575.7 / 576.3 | 0.382 (2.62x) | 1519 -> 582-591 (2.58x) |
| wob64 5120x8192 | 1497.7 / 1498.4 / 1499.1 | 535.9 / 537.0 / 536.6 | 0.358 (2.79x) | 1492-1494 -> 536-540 |
| woa64 8x(4096->1024) | 1210.4 / 1209.7 / 1213.4 | 536.5 / 533.5 / 535.3 | 0.442 (2.26x) | 1203-1204 -> 533-542 |
| kv64 512x5120 | 52.3 / 51.5 / 51.9 | 47.5 / 47.4 / 47.4 | 0.914 (1.09x) | 52-53 -> 47-50 |

Per lane-layer: 4268 us -> 1695 us (-2.57 ms); x40 lane-layers (2 lanes x 20 decoder layers, once per
request) = -0.103 s of dGPU time per request. Matches the claim's 4.27 -> 1.71 ms and -0.10 s.

## 3. Correctness

* rv1 --corr: 24/24 CMP rows bit-exact (b=64,1,5,17,33,63 on 4 sites).
* Extra shapes the engineer did not run (results/extra): b=1, 7, 49 (3 full slices + 1-row tail), 128 (whole
  replay window) plus the harness tails -> 60/60 CMP rows bit-exact on all four sites.
* Numerics are identical by construction: same `dp4a_sudot` chain order (av0.x..av1.w), same
  `ws * xscale * dot` expression, same `bl = lane; bl += 32` striding, same `warp_sum_f32` shuffle tree.

Extra-shape timings (cold, 20 rounds): qb 0.555 @b=7, 0.380 @49, 0.385 @128; wob 0.677 / 0.386 / 0.330;
woa 0.708 / 0.461 / 0.417; kv 1.40 @7, 1.01 @49, 0.85 @128; b=1 all sites 1.00-1.02 (same work).
kv @b=7 is not a production regression: b<=16 already goes to `q8_0_gemv_bpack_warp8`, and z16 at b<=16 IS that
kernel (grid.z=1), so kv b<=16 timing is unchanged by construction.

## 4. Traps checked

* Warm-cache flattery: no (96 MB flush, weights 2.8-44.6 MB; candidate re-reads 4x, baseline 64x, both after a flush).
* Baseline flags: identical ($KFLAGS_V41); harness -O2 is host-side only.
* Skipped work: none (all outputs written; bounds `row >= out_dim`, `b0 >= batch`, `nb` clamp; no new alignment
  assumption — same 16 B `__builtin_memcpy` loads as production).
* Timing scope: single kernel each side; nothing excluded.
* Degenerate inputs: no (random).
* Call-count arithmetic: verified from code (once per job, 64 rows/lane, 20 layers) — but see caveats.

## 5. Caveats (do not change the verdict)

1. Device time, not wall time: the replay is box-2 paging-bound (memory: replay ~12.4 s, dense dGPU part
   ~0.19 s), and the two lanes' dGPU kernels may overlap each other, so the -0.10 s is an upper bound on
   the request-time gain; the claim already says 0.064%.
2. Measured from the engineer's 40-kernel cand module; the engineer documented a 1.6x module-composition
   effect on another kernel (i8x). Re-measure the exact kernel in the shipped `q8_0_matvec.hip` module.
3. kv64 gain (9%) is within the box's noise band for a single run but consistent across 6 processes (3 mine,
   3 engineer's); it is not what the claim rests on.

## 6. Integration notes

* Add the z-chunked bpack kernels to `q8_0_matvec.hip` / `q8_0_grouped_matvec.hip` (or add the `blockIdx.z*16`
  offset to the existing bpack kernels: at grid.z=1 the code path is unchanged for decode).
* `q8_0.rs`: `matvec_bpack` / `matvec_grouped_bpack` launch grid (grid_x, 1, batch.div_ceil(16)); drop or raise
  the `batch > GEMV_BPACK_MAX` error; `bpack_ok` keeps the `V41_GEMV_BPACK=0` rollback but stops capping at 16.
* `het/scratch.rs:38` sizes a head scratch on GEMV_BPACK_MAX (16 x N_VOCAB x 4 B) and `forward_head.rs:94`
  calls `matvec_bpack`: keep the head's own cap so lifting the kernel cap cannot hand the head b>16 with a
  16-row scratch.
* Tests: bit-exact vs `q8_0_gemv_batched_warp8` at b=17..64 incl. tails 49/63 and b=128; decode b<=16 unchanged
  (same kernel, grid.z=1); `prefill_job_replay` `elapsed_s` before/after on a >128-token prompt (expect <=0.1 s
  and possibly ~0 wall-clock, see caveat 1).
