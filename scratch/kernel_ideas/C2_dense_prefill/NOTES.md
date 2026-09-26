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
