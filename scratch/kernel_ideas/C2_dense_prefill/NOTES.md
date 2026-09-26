# C2_dense_prefill — dGPU dense projections at prefill/replay batch (gfx1201)

Working notes, 2026-09-26. Everything here is measured with standalone harnesses on synthetic
data at production shapes; baseline code objects are the UNMODIFIED in-tree .hip built with
`$KFLAGS_V41 --genco --offload-arch=gfx1201` (`build_base.sh`).

## Production facts (read from code, commit 361d4f9)

Constants (S/config.rs): N_EMBD 5120, N_LORA_Q 1280, Q_FLAT 32768, N_HEAD_DIM 512, GROUP_DIM 4096,
RANK 1024, N_GROUPS 8, OUT_LOW 8192, N_FF_SHARED 2304, ENGRAM_IN 6144, ENGRAM_OUT 25600,
ENGRAM_CHUNK 64. `f16_pitch(d) = d + 64` (batch_scratch.rs:171).

Dispatch at prefill b=512 (per lane, per encoder layer L0-19, 20 layers; plus the shared expert on
every layer): all Q8_0 dense projections go to `q8_0_gemm_wmma_f16x`, grid
(ceil(b/128), M/128, groups) x 256, preceded by `f32_to_f16_cast_2d` of the activation:

| site | M | K | groups | ldx | grid @512 | WGs |
|---|---|---|---|---|---|---|
| q_a (dispatch.rs:305) | 1280 | 5120 | 1 | 5184 | (4,10,1) | 40 |
| qb (FP:3887) | 32768 | 1280 | 1 | 1344 | (4,256,1) | 1024 |
| kv (FP:3990) | 512 | 5120 | 1 | 5184 | (4,4,1) | 16 |
| wo_a (FP:6129) | 1024 | 4096 | 8 | 32832 | (4,8,8) | 256 |
| wo_b (FP:6164) | 5120 | 8192 | 1 | 8256 | (4,40,1) | 160 |
| shared gate, up (FP:1847/1856) | 2304 | 5120 | 1 | 5184 | (4,18,1) | 72 |
| shared down (FP:1901) | 5120 | 2304 | 1 | 2368 | (4,40,1) | 160 |
| Engram wkv (FP:3526) `q8_0_gemm_wmma_lds_tiled` | 25600 | 6144 | 1 | (xq i8) | (400,1,1)x128 per 64-row chunk, 8 chunks/lane, layers 1 and 14 | 400 |

Replay (b~64 per lane, decoder layers 20-39): `prefill_f32_matvec(64)` is true -> qb, kv, wo_a, wo_b
take the "dp4a" arm = `matvec_batched` / `matvec_grouped_batched`; bpack_ok(64) is false (cap 16), so
they land on `q8_0_gemv_batched_warp8` grid (M/8, 1, 64) x 256 and `q8_0_grouped_gemv_batched`
grid (8192/8, 1, 64) x 256 — each z-slice re-reads the whole weight. q_a and the shared expert at
b=64 still take f16x (small_b_dense_dp4a(64) is false, cap 8).

## Rooflines (dGPU 640 GB/s DRAM, 194.6 TF dense f16 WMMA peak), b=512 unless noted

bytes = W (M*K/32*34) + x16 read (b*K*2) + out write (b*M*4); flops = 2*b*M*K.

| site | W MB | bytes MB | BW-time us | GF | matrix-time us | roof us (max) | bound |
|---|---|---|---|---|---|---|---|
| q_a | 6.96 | 14.8 | 23 | 6.71 | 34.5 | 34.5 | compute (but 40 WGs on 64 CUs) |
| qb | 44.56 | 113.0 | 176 | 42.95 | 221 | 221 | compute ~ BW |
| kv | 2.79 | 9.1 | 14 | 2.68 | 13.8 | 14 | 16 WGs on 64 CUs: occupancy |
| wo_a (8 groups) | 35.65 | 86.1 | 134 | 34.36 | 177 | 177 | compute |
| wo_b | 44.56 | 63.4 | 99 | 42.95 | 221 | 221 | compute |
| shared gate (=up) | 12.53 | 22.5 | 35 | 12.08 | 62 | 62 | compute (72 WGs) |
| shared down | 12.53 | 25.4 | 40 | 12.08 | 62 | 62 | compute |
| Engram chunk (b=64) | 167.1 | 174 | 272 | 20.1 | 103 | 272 | BW per chunk; x8 chunks = 2.2 ms/lane-layer; one 512-row pass would be max(272, 829) = 829 us |

Sum of f16x roofs per lane-layer: 34.5 + 221 + 14 + 177 + 221 + 2x62 + 62 = 853 us.
Lane-layers at 100K: 97 chunks x 2 lanes x 20 layers = 3880 -> 3.3 s at roof.

Replay regime, per lane-layer at b=64 (grid.z re-read, upper bound at DRAM rate; W < 64 MB MALL so
some z-slices may hit the MALL): qb 64 x 44.56 = 2.85 GB, wo_b 2.85 GB, wo_a 2.28 GB, kv 0.18 GB
-> 8.2 GB -> 12.8 ms per lane-layer if DRAM-bound; 40 lane-layers per request -> ~0.5 s of the
12.4 s replay (the rest is box-2 paging of 384 experts per decoder layer).

## Memory budget

dGPU cap for this sweep: free VRAM ~438 MB, gpu_run needs free - MB - 270 >= 0 -> at most ~165 MB
per harness. Engram's 167 MB weight does not fit: measure lds_tiled at M=6400 and M=12800 and check
the scaling before extrapolating. qb at b=1024 (128 MB output) does not fit either; production lanes
are 512 anyway.

Cold-weight regime: kb::Flusher memset between timed blocks, sized to what the budget leaves
(>= 32 MB + the output write itself evicts MALL lines). Warm numbers are reported as bounds only.

## Log

### Baseline r1 (results/baseline_r1/, cold = flush memset between calls, one call per timed block)

| shape | b | med us cold | med us warm | roof us | % of roof | GF/s cold | GB/s cold |
|---|---|---|---|---|---|---|---|
| qa f16x (40 WGs) | 512 | 127.7 | 99.0 | 34.5 | 27% | 52.5k | 116 |
| qb f16x (1024 WGs) | 512 | 447.8 | 489.6 | 221 | 49% | 95.9k | 252 |
| kv f16x (16 WGs) | 512 | 105.4 | 85.7 | 14 | 13% | 25.5k | 86 |
| kv f16x | 256 / 1024 | 105.2 / 111.8 | | | | | flat in b: per-WG latency bound |
| woa f16x (256 WGs, 8 groups) | 512 | 364.6 | 339.4 | 177 | 49% | 94.2k | 236 |
| wob f16x (160 WGs) | 512 | 483.8 | 408.9 | 221 | 46% | 88.8k | 131 |
| shg f16x (72 WGs) | 512 | 165.3 | 153.0 | 62 | 38% | 73.1k | 136 |
| shd f16x (160 WGs) | 512 | 133.5 | 125.8 | 62 | 46% | 90.5k | 190 |
| Engram lds_tiled M=12800 (half) | 64 | 499.2 | 537.8 | 136 (BW) | 27% | 20.2k | 175 |
| Engram lds_tiled M=6400 | 64 | 306.0 | | 68 | 22% | | 143 |
| cast_2d 512x5120 / 8192 / 32768 | 512 | 89 / 118 / 211 | | 25 / 39 / 157 | | | 176 / 214 / 477 |
| qb64 gemv_batched grid.z=64 | 64 | 1703 | 1536 | 70 (W once) | 4% | 3.2k | 31 (one-read basis) |
| kv64 gemv_batched | 64 | 53.1 | 38.5 | 4.4 | 8% | | |
| wob64 gemv_batched | 64 | 1691 | 1576 | 70 | 4% | | |
| woa64 grouped_gemv_batched | 64 | 1368 | 1210 | 56 | 4% | | |

Per lane-layer f16x sum at b=512 (cold): 127.7 + 447.8 + 105.4 + 364.6 + 483.8 + 2x165.3 + 133.5 = 1993 us
vs 853 us roof (43%). x 3880 lane-layers at 100K = 7.7 s of dGPU time (not the pole).
Engram: ~2 x 499 = ~1.0 ms per 64-row chunk at full M (linear extrapolation; M=6400 -> 12800 scaled
1.63x, so the tail effect is small) -> 8 chunks x 2 layers x 194 lanes = 3.1 s at 100K.
Replay dense per lane-layer at b=64: 1703 + 53 + 1691 + 1368 = 4.8 ms -> 40 lane-layers = 0.19 s per
request (0.12% of 160 s).

### ATT on f16x (results/att_qb, results/att_kv): the prefetch never overlaps

qb b=512: stall/latency 72%. Top stalls: `s_wait_loadcnt_dscnt 0x0` 27.1%, first WMMA after the
barrier 13.6% (LDS fragment latency), `s_wait_dscnt 0x0` before barrier-1 8.5%, ds_load_b128 5.7%,
`s_wait_loadcnt 0x0/0x1` 7.3% (dequant waiting for the A bytes / scale).
kv b=512 (1 WG per CU): `s_wait_loadcnt_dscnt 0x0` = 61% of all stall.

ISA (f16x_base.s): the compiled loop is rotated as
  [stage: s_wait_loadcnt 1 -> dequant -> s_wait_loadcnt 0 -> 4x ds_store -> s_wait_dscnt 0 -> barrier]
  -> issue global loads for k+1 -> 12x ds_load (fragments) -> **s_wait_loadcnt_dscnt 0x0** -> barrier
  -> global_inv -> 16x WMMA -> loop.
The `s_wait_loadcnt_dscnt 0x0` comes from `__syncthreads()`: on gfx12 (WGP mode) a workgroup-scope
release fence waits for ALL outstanding global loads, so the register prefetch issued a few
instructions earlier is waited on before the WMMAs it was meant to overlap. The prefetch has never
overlapped anything; every k-outer pays a full DRAM/L2 latency. (The `global_inv` after each barrier
is the matching acquire.)
Fix (fence_probe.hip): `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local")` +
`__builtin_amdgcn_s_barrier()` + acquire fence "local" -> ISA `s_wait_dscnt 0x0; s_barrier_signal;
s_barrier_wait` with the global load still in flight. Same arithmetic -> bit-exact by construction.

## Ideas (ranked by expected gain x confidence / effort)

1. **lb — LDS-only barrier** (f16x, all shapes): replace __syncthreads with the local fence pair so the
   existing prefetch overlaps the WMMAs. Bit-exact. Drop-in (kernel-only change). Expect the 27-61%
   loadcnt stall to mostly vanish: +20-50%.
2. **pf2 — prefetch 2 k-outers ahead** (+13 VGPRs), on top of lb: covers latency at 1 WG/CU (kv, qa).
3. **bn64 / 64x64 tiles** for small M (kv 16 WGs, qa 40, shg 72, shd/wob 160 with a 2.5-WG tail):
   BM=128xBN=64 (2x WGs, 15 KB LDS -> 4 WGs/CU) and 64x64 (4x WGs). Bit-exact (same k order per
   accumulator). Structural: fixes the occupancy floor of the small projections.
4. **Engram i8x** — f16x tile structure with int8+xscale activations dequantised at stage: drop-in for
   `gemm_lds_tiled` (same signature/inputs) at 64-row chunks; then optionally 128-row chunks (halves
   the weight re-reads; needs 13 MB more kv scratch).
5. **Replay b=64 routing**: f16x at b=64 on qb/kv/wo_a/wo_b instead of grid.z=64 GEMV (not bit-exact
   vs dp4a: f16 activations; needs the fidelity gate), and a bit-exact `bpack64` dp4a twin (weights
   read once, 64 accumulators).
6. Fuse the f32->f16 cast into the B-tile stage (f16x reading f32 activations): removes the cast
   kernels (89-211 us each, 6 per lane-layer ~ 0.7 ms) at the cost of 2x B-tile global bytes. Not
   attempted here (integration touches every call site's buffers); noted with the cast timings.
7. Larger BK (64) with int8 A in LDS: halves barriers per K; LDS 27 KB -> 2 WGs/CU. Not attempted.
8. Direct-to-LDS loads (global_load_lds_b128 on gfx12) for the B tile: bypass VGPR staging. Not attempted.

## Resumed 2026-09-26 19:10 UTC (after the 04:01 box hang; hub DOWN, both GPUs idle, per-device scheduler)

- cand.hip from the first attempt had never been compiled. Built now: all f16x twins compile without
  scratch (v2/lb 143 VGPRs, pf2 158, bn64 88, 64x64 95, i8x 148, i8x_bn64 100; base f16x 140).
- bpack64 twins first compiled with 272 B scratch (the predicated `break` blocked the x64 unroll and
  acc[64] went to memory). Fixed: `if (b < batch) {...}` predicate, no break -> 102 VGPRs, 0 scratch.
- Added `q8_0_gemv_bpack_z16` / `q8_0_grouped_gemv_bpack_z16`: the PRODUCTION bpack16 kernel body with
  a blockIdx.z chunk of 16 rows (grid.z = ceil(b/16)); weight re-read 4x at b=64 instead of 64x, and
  bit-identical per (row, b) by construction (same loop, same expression). 51 VGPRs. Cheapest possible
  integration (one kernel + the grid.z in `matvec_bpack`, drop the cap check).
- Budget: hub down -> flush stays 96 MB, jobs use --mb 400-700; Engram measured at FULL M=25600.
- Correctness r1 (results/corr_f16x_r1.txt): 315/315 CMP rows bit-exact (7 shapes x b in
  {512,1,3,17,100,129,255,256,500} x {v2, lb, pf2, bn64, 64x64}).

### A/B r1 (results/ab_f16x_r1/, cold, 60 rounds, ratio = med(cand)/med(base); all bit-exact)

| shape b | base us | v2 (control) | lb | pf2 (lb+2-deep) | bn64 (128x64) | 64x64 |
|---|---|---|---|---|---|---|
| qa 512 | 134.8 | 0.972 | **1.207** | 0.896 | 1.055 | 0.905 |
| qa 256 / 1024 | 108.4 / 163.5 | 1.03 / 1.01 | 1.02 / 1.07 | 0.859 / 0.964 | 0.965 / 1.076 | 0.886 / 1.133 |
| qb 512 | 451.9 | 0.991 | 1.022 | 0.968 | 1.248 | 1.423 |
| kv 512 | 104.4 | 1.027 | 1.046 | 0.857 | 0.833 | 0.834 |
| kv 256 / 1024 | 103.4 / 109.2 | 1.02 / 1.02 | 1.02 / 1.07 | 0.854 / 0.842 | 0.829 / 0.939 | 0.775 / 0.900 |
| woa 512 | 361.4 | 1.005 | 1.000 | 0.967 | 1.147 | 1.220 |
| wob 512 | 422.5 | 0.999 | 1.025 | 1.030 | 1.324 | 1.324 |
| shg 512 | 164.4 | 1.009 | 1.072 | 0.958 | 1.021 | 1.014 |
| shg 256 / 1024 | 128.1 / 256.8 | 0.97 / 1.02 | 1.14 / 1.03 | 0.866 / 0.955 | 0.999 / 1.231 | 0.876 / 1.172 |
| shd 512 | 137.8 | 1.009 | 1.010 | 0.972 | 1.201 | 1.199 |

Lessons: (a) the control v2 (generic core, production barrier) tracks the base within +-3% -> the
template core is a faithful copy. (b) **lb alone is a LOSS** (up to +21% on qa): the release-fence
theory from the ATT was wrong as a fix — the next stage needs those loads anyway and one k-outer
of WMMAs (16 x ~14 cycles) cannot cover an L2/DRAM round trip; letting the loads float only
changed the scheduling. (c) pf2 (+1 k-outer of loads in flight) is a modest win, 3-14% except wob
(+3% loss). (d) small tiles pay only on the occupancy-starved shapes (kv: 16 WGs -> 32/64 WGs,
-17%; qa/shg at b=256) and lose 15-42% on the big shapes (the per-WG arithmetic intensity halves).
(e) ISA of pf2: the compiler emitted `s_wait_loadcnt 0x0` before EVERY stage because the B loads
sat in `s_cbranch_execz` blocks (`if (b_valid)`), so the 2-deep prefetch only worked half the
time. Fixed in patch2.py (branch-free loads: clamped row + v_cndmask select; compile-time A
predicate): pf3 now shows `s_wait_loadcnt 0x2/0x3`, zero execz branches. pf2 140 / pf3 153 /
pf4 166 VGPRs — all free w.r.t. occupancy (LDS caps at 3 WGs/WGP = 6 waves/SIMD -> 256 VGPRs each).

Arithmetic that reframes the roofline: qb measured 452 us ~= BW-time 176 + matrix-time 221 (+55);
woa 361 ~= 134 + 177 (+50); wob 422 ~= 99 + 221 (+102). Memory and matrix work are NOT overlapping;
also per k-outer per WG the tile traffic is 12 KB (A 4 KB + B 8 KB) for 1 MF = 85 flop/B, and
1024 WGs x 40 k-outers x 12 KB = 492 MB of L2->CU traffic in 452 us = 1.1 TB/s. The 194 TF matrix
peak needs 2.3 TB/s at this intensity. So the binding ceiling is likely L2->LDS tile traffic (and/or
LDS bandwidth: 16 KB written + 48 KB read per k-outer per WG), not DRAM and not the matrix pipe.
Ablation kernels (abl_noglobal / abl_nolds / abl_nowmma) price each part in r2.

### Engram r1 (results/engram_r1/, FULL M=25600, cold; i8x = f16x tile structure with int8+xscale B)

| b | lds_tiled (prod) | i8x | i8x_bn64 |
|---|---|---|---|
| 64 (production chunk) | 904.7 us | **625.2 (0.691)** | 1245 (1.376) |
| 128 | 2244 (p10 1644) | 678.5 (0.302) | 634.0 (0.283) |

Both bit-exact vs lds_tiled at b in {64, 128, 1, 17, 63} (same `(f16)((float)q * scale)` dequant and
f32 WMMA accumulate in the same k order). Per chunk roof = 167 MB / 640 GB/s = 261 us; i8x at 42%.
128-row chunks: 678 us per 128 rows vs 2 x 625 = 1250 -> another 0.54x (needs 13 MB more scratch
and the Engram chunk loop at FP:3526 to step 128).

### r2: branch-free loads, PF2/3/4, and the ABLATIONS (results/ab_f16x_r2/; all pf* bit-exact, 189/189 rows)

| shape b | base us | pf2 | pf3 | pf4 | abl_noglobal | abl_nolds | abl_nowmma |
|---|---|---|---|---|---|---|---|
| qa 512 | 125.1 | 0.926 | 0.925 | 0.917 | 0.836 | 0.728 | 0.698 |
| qb 512 | 458.1 | 0.959 | 0.976 | 0.977 | 0.722 | 0.824 | 0.824 |
| kv 512 | 107.1 | 0.904 | 0.903 | 0.899 | 0.683 | 0.649 | 0.741 |
| woa 512 | 367.5 | 0.972 | 0.970 | 0.967 | 0.733 | 0.827 | 0.820 |
| wob 512 | 426.0 | 1.021 | 1.038 | 1.034 | 0.802 | 0.873 | 0.776 |
| shg 512 | 167.2 | 0.918 | 0.919 | 0.912 | 0.848 | 0.748 | 0.713 |
| shd 512 | 133.8 | 1.029 | 1.000 | 1.007 | 0.835 | 0.906 | 0.779 |

PF depth beyond 2 buys nothing (pf2 ~ pf3 ~ pf4). The ablations (NOT correct kernels; each removes one
resource): no single part dominates. Resource algebra at qb, if the parts overlapped perfectly the
time would be max(G, L, W): no-G -> max(L,W) = 330; no-L -> max(G,W) = 377; no-W -> max(G,L) = 377
=> G (global/L2 tile fetch + output write) = 377 us is the largest single resource, the LDS/WMMA
chain alone is 330 us, and the real kernel (458) still carries ~80 us of non-overlap. Every part is
within 1.4x of the total: the kernel is a serial chain per WG with too little cross-WG overlap.

ATT of pf3 at qb (results/att_pf3_qb, stall/latency 62.6% vs 72.5% for the base): the top stalls are
`s_wait_loadcnt 0x3` (~35% of stall summed over its 6 sites) = waiting for the OLDEST of 3 in-flight
k-outers. Three k-outers of work (~750+ cycles of WMMA + 3 barriers) do not cover the fetch: the
fetch path is throughput-bound (queued), not latency-bound. Deeper prefetch cannot fix it; only
fewer tile bytes per flop (bigger tiles) or fewer bytes (int8 in LDS) can.

### r3: double-buffered LDS `db` (one barrier per k-outer; frag ds_loads issued before the next
tile's VALU dequant + ds_store) — results/ab_f16x_r3/, all bit-exact (251/251 rows incl. tails)

| shape b | base | pf3 | db (40 KB, 1 WG/WGP) | db_pf1 | db_pf3 | db_bn64 (30 KB, 2 WGs/WGP) |
|---|---|---|---|---|---|---|
| kv 512 | 104.5 | 0.916 | 0.871 | 0.917 | 0.906 | **0.637** |
| kv 256 / 1024 | 103.5 / 110.0 | 0.916 / 0.914 | 0.873 / 0.867 | | | **0.621 / 0.763** |
| qa 512 | 124.5 | 0.915 | 0.965 | 1.003 | 1.002 | **0.817** |
| qa 256 / 1024 | 109.8 / 164.4 | 0.877 / 0.928 | 0.846 / 0.941 | | | 0.692 / 1.082 |
| qb 512 | 458.0 | 0.979 | 1.037 | 1.073 | 1.087 | 1.278 |
| woa 512 | 368.7 | 0.984 | 1.064 | 1.106 | 1.107 | 1.118 |
| wob 512 | 427.3 | 1.041 | 1.059 | 1.100 | 1.104 | 1.194 |
| shg 512 | 165.3 | 0.918 | 0.940 | 0.983 | 0.984 | 0.999 |
| shg 256 / 1024 | 128.0 / 268.8 | 0.894 / 0.956 | 0.945 / 0.944 | | | 0.800 / 1.058 |
| shd 512 | 137.4 | 1.000 | 0.975 | 1.004 | 1.013 | 1.117 |

db helps only where occupancy was the problem (kv 16 WGs, qa/shg at 256 rows); on the big shapes
the 1 WG/WGP occupancy (2 waves/SIMD) costs more than the pipelining gains. db_bn64 (2 WGs/WGP,
half the per-WG intensity) is the best kernel for kv (-36%) and qa at b<=512 (-18%) but loses 12-28%
on qb/woa/wob. Per-shape kernel selection is therefore the practical form of this win.

### Engram r2/r3 (results/engram_r2, engram_r3; all i8x* bit-exact vs lds_tiled incl. b=1,17,63)

| b | lds_tiled | i8x (brf) | i8x_grd (r1-style loads) | i8x_pf2 | **i8x_db** | i8x_db_grd |
|---|---|---|---|---|---|---|
| 64 | 911 / 917 | 1101 / 1105 | 1084 | 1014 / 1018 | **509.8 (0.560)** | 517 |
| 128 | 1676 / 1698 | 1130 / 1134 | 1153 | 1081 / 1078 | **542.8 (0.324)** | 585 |

The r1 i8x number (625 us at b=64) does NOT reproduce with the current source (1100 us; the load
style is not the cause: grd = brf). r4 re-measures the r1 binary itself (recovered from commit
aab5a67 as cand_r1.hip -> pseudo-candidate `i8x_r1`) side by side. i8x_db supersedes it anyway:
510 us per 64-row chunk = 51% of the 261 us DRAM roof... (167 MB weight per chunk at 640 GB/s);
at 128-row chunks 543 us = 2 x 261 -> 96% of the per-chunk roof, i.e. the weight stream is the
limit there and the chunk size is the lever (13 MB more scratch, FP:3526 loop step 128).

### r4: big tiles (results/ab_f16x_r4/, all bit-exact 315/315 incl. tails)

256x128 (256 thr, wave 64x64, 30 KB LDS = 2 WGs/WGP): qb 0.930, woa 0.960, shd 0.950, shg-1024 0.945;
loses on qa/kv/shg-512 (1.05-1.2: fewer WGs). 256x256 (512 thr, 40 KB): loses everywhere (1.05-1.8).
256x128_db: 60 KB LDS + 31 VGPR spills: loses everywhere. (All measured BEFORE patch6, see below.)

### THE LOAD-SELECT TRAP (patch6) — the biggest single lesson of this family

Patch2's "branch-free" B loads did `vb = b_on ? loaded : 0` right after the load. The compiler
materialises that select immediately, which needs the data, so it emitted `s_wait_loadcnt` right
after the global_load — the prefetch became a synchronous load for the B tile in EVERY variant
measured in r2-r4 (pf*, db*, 256x*, i8x*). The r1 code (guarded `if (b_valid)` loads, as in
production) waited at the stage instead. Fix (patch6): load the raw registers unconditionally
(clamped row, always mapped) and apply the select in stage(), after the barrier the scheduler
cannot cross. ISA after: `global_load x4 | 12 ds_load | s_wait_dscnt | barrier | 16 wmma | ... |
s_wait_loadcnt 0x3/0x2 at the stage`. Rule: never touch a prefetched register before its use.

### r5/r6/r7 (post-patch6; three separate processes; results/ab_f16x_r5..r7/, all bit-exact
378/378 CMP rows in results/corr_final_f16x.txt) — ratio = med(cand)/med(base), r5 / r6 / r7

| shape b | base us (r5/r6/r7) | lb | pf2 | pf3 (r5) | db (40 KB, 1 WG/WGP) | db_bn64 (30 KB, 2/WGP) | 256x128 |
|---|---|---|---|---|---|---|---|
| kv 512 | 106.1 / 106.0 / 107.1 | 0.791 | 0.776 / 0.770 / 0.761 | 0.768 | 0.706 / 0.711 / 0.701 | **0.557 / 0.560 / 0.553** | 1.099 |
| kv 256 | 104.2 / 104.4 / 104.4 | 0.794 | 0.772 / 0.777 / 0.777 | 0.762 | 0.703 / 0.707 / 0.703 | **0.546 / 0.542 / 0.543** | 1.057 |
| kv 1024 | 110.4 / 110.9 / 110.4 | 0.794 | 0.773 / 0.769 / 0.775 | 0.771 | 0.732 / 0.734 / 0.736 | **0.717 / 0.719 / 0.720** | 1.125 |
| qa 512 | 125.7 / 136.2 / 125.1 | 0.981 | 0.855 / 0.875 / 0.859 | 0.869 | 0.884 / 0.919 / 0.888 | **0.774 / 0.801 / 0.780** | 0.982 |
| qa 256 | 110.3 / 109.8 / 109.6 | 0.779 | 0.793 / 0.790 / 0.794 | 0.799 | 0.697 / 0.703 / 0.706 | **0.649 / 0.655 / 0.662** | 1.076 |
| qa 1024 | 165.6 / 165.8 / 166.3 | 0.965 | 0.910 / 0.908 / 0.904 | 0.913 | **0.894 / 0.892 / 0.889** | 1.044 | 1.031 |
| qb 512 | 463.1 / 454.2 / 457.7 | 0.960 | 0.944 / 0.955 / 0.954 | 0.968 | 0.995 / 0.989 / 0.987 | 1.250 | **0.901 / 0.900 / 0.898** |
| woa 512 | 368.9 / 366.6 / 367.0 | 0.956 | 0.928 / 0.920 / 0.932 | 0.938 | 0.958 / 0.963 / 0.961 | 1.087 | **0.886 / 0.883 / 0.884** |
| wob 512 | 429.4 / 426.8 / 423.7 | 0.998 | **0.956 / 0.973 / 0.978** | 0.985 | 0.961 / 0.960 / 0.966 | 1.239 | 0.992 / 0.985 / 0.997 |
| shg 512 | 165.8 / 165.7 / 165.6 | 0.969 | **0.890 / 0.894 / 0.893** | 0.894 | 0.890 / 0.893 / 0.892 | 1.039 | 1.039 |
| shg 256 | 128.6 / 128.6 / 129.6 | 0.969 | 0.845 / 0.844 / 0.841 | 0.853 | 0.870 / 0.867 / 0.866 | **0.759 / 0.756 / 0.756** | 1.007 |
| shg 1024 | 261.8 / 265.5 / 269.4 | 1.003 | 1.014 / 1.001 / 0.999 | 0.916 | **0.906 / 0.889 / 0.876** | 1.069 | 0.969 |
| shd 512 | 135.2 / 137.6 / 136.2 | 1.002 | 1.026 / 1.014 / 1.025 | 0.992 | **0.939 / 0.933 / 0.939** | 1.141 | 0.982 |

Best-per-shape (b=512 lane): kv db_bn64 59 us (from 106), qa db_bn64 97 (from 125), qb 256x128 410
(from 458), woa 256x128 324 (from 367), wob pf2/db 410 (from 427), shg pf2/db 148 (from 166), shd db
127 (from 136). Lane-layer sum at b=512: 1993 -> 59 + 97 + 410 + 324 + 410 + 2x148 + 127 = 1723 us
(-13.5%; the big three qb/woa/wob are 66% of the sum and improve only 4-12%). Roof: 853 us -> 50%.

### Engram, three processes (r5/r6/r7; results/engram_r5..r7/; bit-exact 16/16 in corr_final_engram.txt)

| b | lds_tiled | i8x_db | i8x_r1 (recovered r1 binary) |
|---|---|---|---|
| 64 | 913 / 899 / 892 | **489 / 484 / 482 (0.536-0.541)** | 631 / 623 / 623 (0.69) |
| 128 | 1687 / 2254 / 2252 (bimodal: p10 1640) | **514 / 510 / 512 (0.23-0.30)** | 682 / 675 / 676 |

i8x_db at 128-row chunks: 512 us vs 2 x 484 = 968 for two 64-row chunks -> the chunk size is the
second lever (needs +13 MB kv scratch and the FP:3526 chunk loop stepping 128).

### Replay b=64, three processes (results/replay_r1..r3 + corr_final_replay: z16/bpack64 bit-exact
48/48 incl. b=1,5,17,33,63; f16x_at_b NOT bit-exact, rel_rmse 2.9e-4 = f16 activation numerics)

| shape (b=64) | grid.z=64 base | **bpack_z16** (bit-exact) | bpack64 (bit-exact) | f16x (production kernel, f16 acts) | f16x_bn64 |
|---|---|---|---|---|---|
| qb64 | 1519 / 1521 / 1519 | **589 / 582 / 582 (0.383-0.388)** | 1023 / 1007 / 1009 | 269 (0.177) | 376 |
| wob64 | 1494 / 1494 / 1492 | **540 / 536 / 537 (0.36)** | 985 / 980 / 980 | 231 (0.155) | 207-231 |
| woa64 | 1204 / 1203 / 1204 | **542 / 533 / 536 (0.44-0.45)** | 674 / 672 / 672 | 168 (0.14) | 148 (0.12) |
| kv64 | 53 / 52 / 52 | **50 / 47 / 47 (0.91-0.95)** | 189 (3.6x) | 98 (1.9x) | 64-74 |

bpack64 (weight read once, 64 accumulators) LOSES to z16 (4 re-reads): the per-lane 64-way activation
loop has too little memory parallelism (2 loads per row per block, in order); 4x the WGs wins.
Per lane-layer: 4270 us -> 1723 us (z16) or ~620 us (f16x, needs the fidelity gate).

### The r1 i8x gap is a MODULE-COMPOSITION effect (results/engram_bisect/, results/modtest_r1/)

Bisect on the r1 source (patch8: the r1 f16x_core copied verbatim into cand.hip as namespace r1c):

| b=64 | i8x_r1 (r1 BINARY, 8-kernel module) | r1c_i8x (same source, 40-kernel module) | r1c + UNC | r1c + AONC | current i8x | i8x_db |
|---|---|---|---|---|---|---|
| us | 623 | 983 | 993 | 1072 | 1078 | 484 |

The identical source is 1.6x slower when compiled into the big module. Source-level changes
(unconditional loads UNC: 0; compile-time a_on AONC: +9%) do not explain it. Suspect: the LDS
lowering of `__shared__` arrays declared inside a template function and referenced from lambdas
(every instantiation + every lambda closure = a distinct LDS variable and a "module LDS" struct
whose layout/addressing depends on the whole module). Checked and ruled out: LDS size per kernel
(descriptor reads 20480 for both; the isa.sh/kd_meta table is off by one row — the kernel's .name
follows its sizes in the metadata — all sizes are consistent once shifted), VGPR count (148 vs
139, both well under the 256 that 3 WGs/WGP allows), instruction count (r1 has MORE VALU).
The modtest job (small 8-kernel module with the shortlisted kernels vs the 40-kernel module, same
process) settles whether the winners get faster in a production-sized module. Integration note:
production q8_0_matvec_wmma.hip has 4 kernels; a candidate must be re-measured IN that module.

## Ideas (final ranking; all measured unless marked)

1. **Engram i8x_db** (measured_win, bit-exact): f16x tile structure + int8/xscale activations,
   double-buffered LDS, one barrier per k-outer. 0.54x per 64-row chunk, 0.30x at 128-row chunks.
2. **Replay bpack_z16** (measured_win, bit-exact): production bpack16 with grid.z = ceil(b/16).
   0.36-0.45x on qb/wo_a/wo_b at b=64, neutral on kv. Cheapest integration in this family.
3. **f16x per-shape selection** (measured_win, bit-exact): db_bn64 for kv/qa (0.56 / 0.78),
   256x128 for qb/wo_a (0.90 / 0.88), pf2 (or db) for wo_b/shared (0.96 / 0.89-0.94).
4. **Replay f16x at b=64** (measured_win, NOT bit-exact): route qb/kv/wo_a/wo_b to the production
   f16x below the 64-row threshold (0.14-0.18x); f16 activation numerics = the prefill path's; needs
   the fidelity gate (KLD vs golden CPU reference with pinned routing) before merging.
5. Engram 128-row chunks (measured at the kernel level: 512 vs 2x484 us; integration = +13 MB scratch
   and the FP:3526 chunk loop) — untested end to end.
6. Fuse the f32->f16 cast into the B-tile stage (untested; would remove 6 cast launches = ~0.7 ms
   per lane-layer at 512 rows, 89-211 us each, at the cost of 2x B-tile global bytes).
7. A tile as int8 in LDS (untested): 128x40 B + scales = 5.4 KB + B f16 10 KB = 15.6 KB -> 4 WGs/WGP
   with the production tile; dequant at fragment load (~48 VALU per k-outer per wave).
8. Split-K for wo_b (K=8192, 160 WGs = 1.67 rounds of 96 slots): untested; needs an f32 reduction
   (atomics are not deterministic; a second pass costs 2 x 10 MB).

## Dead ends (measured)

- lb alone with guarded loads (r1): loss up to +21% — letting loads float only reordered the wait.
- PF depth > 2: pf3/pf4 = pf2 within noise everywhere; ATT of pf3 still waits on the OLDEST stage
  (`s_wait_loadcnt 0x3` = 35% of stall): the fetch path is throughput-bound, not latency-bound.
- db (1 WG/WGP) on the big shapes (qb/wo_a/wo_b): 0.96-1.00 — the occupancy loss cancels the pipelining.
- bn64 / 64x64 / db_bn64 on qb/wo_a/wo_b: 1.09-1.42 (per-WG intensity halves).
- 256x256 (512 thr): 1.05-1.8 everywhere; 256x128_db: 61 KB LDS + spills, 1.2-1.9.
- bpack64 (weight read once, 64 accumulators): 0.66 vs z16's 0.36-0.45; 3.6x slower on kv.
- i8x_bn64 for Engram at b=64: 1.34-1.38 (grid.x = 1 either way; half the WMMAs per barrier).
- Branch-free loads with the select at load time (patch2): silently serialised the B prefetch in
  every variant (r2-r4 numbers are all pessimistic by 5-25%).

### Module-size and W0 tests (results/modtest_r1, modtest2, w0_r1) — both hypotheses for the r1 gap REFUTED

- modtest: every shortlisted kernel runs identically from the 8-kernel and the 40-kernel module
  (i8x_db 485 vs 486, pf2 80.1 vs 80.0, db_bn64 58.8 vs 58.7, 256x128 418 vs 413, ...); the r1
  SOURCE (cand_r1.hip) + 22 dummy instantiations still gives 629 us for its i8x. Module size is out.
- W0 (explicit `s_wait_loadcnt 0x0` before the WMMA block, as the fast r1 binary happens to do):
  i8x_w0 = i8x (1084 vs 1085), i8x_db_w0 507 vs 486, pf2_w0 kv 0.963 vs 0.772, qb 1.003 vs 0.955,
  db_w0 kv 0.875 vs 0.713. Loads floating across the WMMAs are GOOD; the r1 wait placement is not it.
- Status: `i8x_r1` (cand_r1.hip, commit aab5a67, 20480 B LDS, 148 VGPRs, bit-exact, 623-631 us in 6
  runs) is 1.6x faster than the same algorithm re-expressed in the current template (r1c_i8x 983-993,
  i8x 1078-1100); the histogram shows r1 unpacking the int8 activations with ~100 more shift/and/or
  ops and higher ILP. Unexplained at the source level; NOT needed for the deliverable because i8x_db
  (482-489 us) beats it. Flag for integration: re-measure the exact file that ships.

## Summary for the ledger (all numbers: median of 60 interleaved rounds, cold weights, >= 3 processes)

| idea | kernel | shape | base -> cand us | speedup | bit-exact | runs |
|---|---|---|---|---|---|---|
| engram_i8x_db | q8_0_gemm_wmma_lds_tiled -> q8_0_gemm_wmma_i8x_db | 25600x6144, b=64 | 899-913 -> 482-489 | 1.86 | yes | 6 |
| engram_i8x_db (128-row chunk) | same | b=128 | 1676-2254 -> 510-514 | 3.3-4.4 | yes | 5 |
| replay_bpack_z16 | q8_0_gemv_batched_warp8 -> q8_0_gemv_bpack_z16 | qb 32768x1280 b=64 | 1519 -> 582-591 | 2.58 | yes | 4 |
| replay_bpack_z16 | same | wob 5120x8192 b=64 | 1492-1494 -> 536-540 | 2.77 | yes | 4 |
| replay_bpack_z16 | q8_0_grouped_gemv_batched -> q8_0_grouped_gemv_bpack_z16 | woa 8x(4096->1024) b=64 | 1203-1204 -> 533-542 | 2.24 | yes | 4 |
| f16x_db_bn64 | q8_0_gemm_wmma_f16x -> _db_bn64 | kv 512x5120 b=512 | 106-107 -> 58.7-59.4 | 1.79 | yes | 3 |
| f16x_db_bn64 | same | qa 1280x5120 b=512 | 125-136 -> 97-109 | 1.28 | yes | 3 |
| f16x_256x128 | q8_0_gemm_wmma_f16x -> _256x128 | qb 32768x1280 b=512 | 454-463 -> 409-417 | 1.11 | yes | 3 |
| f16x_256x128 | same | wo_a b=512 | 367-369 -> 324-327 | 1.13 | yes | 3 |
| f16x_pf2 | q8_0_gemm_wmma_f16x -> _pf2 | wo_b b=512 | 424-429 -> 410-415 | 1.03 | yes | 3 |
| f16x_pf2 / db | same | shared gate/up 2304x5120 b=512 | 166 -> 148 | 1.12 | yes | 3 |
| f16x_db | same | shared down 5120x2304 b=512 | 135-138 -> 127-128 | 1.07 | yes | 3 |
| replay_f16x_b64 | grid.z GEMV -> production f16x | qb/wob/woa b=64 | 1519/1494/1204 -> 269/232/168 | 5.6/6.4/7.1 | NO (f16 acts) | 4 |

## Repro (from scratch/kernel_ideas/C2_dense_prefill/)

    bash build_base.sh; bash build.sh; bash build_r1.sh; bash build_small.sh
    bash ../_infra/gpu_submit.sh --dev dgpu --mb 400 --label C2_dense_prefill/corr -- bash run_corr_f16x.sh <cands>
    bash ../_infra/gpu_submit.sh --dev dgpu --mb 500 --label C2_dense_prefill/ab -- bash run_ab_f16x.sh rN <cands>
    bash ../_infra/gpu_submit.sh --dev dgpu --mb 700 --label C2_dense_prefill/engram -- bash run_engram.sh rN [--corr]
    bash ../_infra/gpu_submit.sh --dev dgpu --mb 500 --label C2_dense_prefill/replay -- bash run_replay.sh rN [--corr]
    ATT: gpu_submit ... -- bash ../_infra/prof.sh att dgpu <kernel> results/att_x -- ./harness_gfx1201 gfx1201 . qb --short --cand <kernel>
    cand.hip is the accumulated candidate file (patch2..patch11 applied in order to the aab5a67 version);
    cand_r1.hip = the aab5a67 version (git show), cand_small.hip / cand_r1plus.hip = module tests.
