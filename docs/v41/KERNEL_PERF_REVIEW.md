# V4.1-Flash kernel-path performance review (2026-09-13, read-only)

Scope: the `--features v41` kernel path — `kernels/mxfp4_*`, router, mHC, compressor, f16 pair,
Engram, SWA/compressed attention — their launchers, and the call sites in `het/forward_layer.rs`
(decode), `het/forward_prefill.rs` (CED prefill: encoder 0..19 over the prompt, layer 20 as
KV-source-only, fixed ≤128-row decoder replay over 20..39) and `het/expert_pager.rs`.
No GPU was touched. Kernel resource numbers below come from the V4.1 hsacos in `target-v41`
(`clang-offload-bundler` + `llvm-readelf --notes` + `llvm-objdump`); every "measured" figure is
either from a V4-Flash trace of the *identical* kernel at the same shape or from a V4.1 log, and
is labelled as such. **There is no per-kernel device timing of the V4.1 path anywhere in the repo**
(every `het.token.summary` in `logs/` has `dgpu_busy_us=0 igpu_busy_us=0`; ENGINE_PORT M1: "kwide
perf unmeasured"), so several rows end with the measurement that would settle them.

## 0. Ceilings and shapes used for every roofline

| device / resource | ceiling used |
|---|---|
| dGPU gfx1201 (RX 9070 XT) | 640 GB/s DRAM; 8 MB L2 + 64 MB MALL; f16 WMMA ≈ 194 TF peak, **35–50 TF realised** (attention 20–25 % of peak per `reference_rdna4_flash_attention_ceiling`); f32 VALU ≈ 49 TF |
| iGPU gfx1151 (Strix Halo) | **214 GB/s** measured expert streaming (256 theoretical); 40 CUs; 64 KB LDS/CU; dp4a ≡ f16-WMMA ≈ 55 TOPS |
| NVMe (experts, Engram) | 4.31 GB/s measured ceiling, 3.83 achieved with 4 reader threads |

| V4.1 object | bytes |
|---|---|
| one routed expert (MXFP4, 17 B / 32) | gate+up 2 × 2304 × 20 sb × 136 B = **12.53 MB**; down 5120 × 9 × 136 = **6.27 MB**; total 18.8 MB |
| decode MoE per layer (6 picks) | 112.8 MB → **0.527 ms** at 214 GB/s → 21.1 ms/token over 40 layers |
| Engram `wkv` (Q8_0, 25600 × 6144) | **167 MB** per Engram layer (layers 1, 14) |
| non-routed per layer, Q8_0 | ≈ 178 MB → 0.28 ms/layer at 640 GB/s → 11 ms/token byte floor for the dGPU chain |
| Q8_K activation row | 5120 → 5840 B; 2304 → 2628 B |
| attention K = V latent row | 512 × f16 = 1 KB; window 128 rows; compressed store = pos/ratio rows (ratio 2 on layers 2–19, **1** on 20–39) |
| `hc_fn` per chain (f16) | 24 × 20480 × 2 = 983 KB |

Budget the findings are priced against: the goal in memory is ≥ 1000 tok/s prefill at 100K and
≥ 30 tok/s decode. That is **512 ms per 512-row lane-chunk** (1024 ms per 1024-row chunk) and
**33 ms per token**; DECODE_M8_PLAN's no-miss single-box budget is 47.7 ms/token.

## 1. Ranked findings

| # | kernel / site | what is wrong | measured vs roofline | estimated win | conf. | confirming measurement |
|---|---|---|---|---|---|---|
| 1 | **Dense compressed attention — sparse indexer never fires on V4.1.** `forward_layer.rs:1049` (`use_sparse = ratio == 4 && …`), `:1225-1240` dense fallback; `forward_prefill.rs:2430-2432` (`dense_needed = !(ratio == 4 …)`), `:2491`; caps `attention.rs:53` (`ATTN_MIXED_MAX_KEYS`), `:87` (`ATTN_SCORES_STRIDE = 3072`), `:679`; `engine_worker.rs:764` (`--ctx` check divides by 4) | Every layer ≥ 2 attends over its whole compressed store (V4.1 keys/indexer/candidate pool are the ENGINE_PORT M5 leftover). Decode: 29·P rows/token (18 layers × P/2 + 20 × P). Prefill: keys/query = 128 + P/2 on encoder layers, = N on the replay. **All three caps assume ratio ≥ 4**: CED replay errors at N > 2944 tokens and the encoder at pos ≥ 5.9K (`n_total_max > ATTN_SCORES_STRIDE`, `attention.rs:679`); decode errors past 82K tokens (`ATTN_MIXED_MAX_KEYS`) while the server accepts `--ctx 328K`. The 100K goal cannot run today. | Decode (M58 B=1 rates, 19.4 ns/row): **4.6 ms @8K, 18.4 @32K, 58 @100K, 173 @300K per token** vs ~0.5 ms sparse (38 × 640 rows) + ≤0.5 ms indexer. Prefill (V4-Flash measured 1.15 ms per 640 keys per 512 rows, ≤1.8 µs/key): 70 ms per 512 rows at P=4K (14 % of budget), 100 ms at the 5.9K cap; hypothetically 535 ms at 32K (> whole budget), 1.6 s at 100K (≤ 320 tok/s ceiling). Sparse ≈ 21 ms + ~20 ms indexer per 512 rows (8 %). | Decode: **−4.6/−18/−57 ms per token** at 8K/32K/100K (the difference between ~10 and 30 tok/s at 100K). Prefill: the difference between "cannot run" and ~8 % of budget. | HIGH (code + arithmetic; M58 rates are measured) | `V41_ATTN_COMP_CAP=512` (existing measurement knob, `forward_layer.rs:1231-1239`) vs uncapped, ms/token at ctx 8K and 32K; a 3000-token CED prompt (expect the stride error) |
| 2 | **Engram `wkv` in prefill runs the `grid.z = batch` GEMV.** `forward_prefill.rs:1344` → `q8_0.rs:190-239` (`q8_0_gemv_batched_warp8`, grid `(M/8, 1, batch)`) → `q8_0_matvec.hip:196-228`; `config.rs:31` (`ENGRAM_CHUNK = 64`) | Each z-slice re-streams the 167 MB weight (≫ 64 MB MALL; x-fastest dispatch puts 3200 WGs between reuses of a row tile). Per 64-row pass: 64 × 167 MB = **10.7 GB DRAM = 16.7 ms**; 8 passes per 512-row lane; two Engram layers in the encoder. | **267 ms per 512 rows = 52 % of the 1000 tok/s budget** vs roofline 0.27 ms BW (167 MB once) + 161 GFLOP ≈ 4 ms WMMA → ~65× over. Invisible today only because the chunk is NVMe-bound (~37 s). | **−255 ms per 512 rows** once experts are resident. Fix = `q8_wmma.gemm_lds_tiled` (exists, M=25600 K=6144 N=64..512; the kernel that took qb 8.8 → 1.4 ms) | HIGH (structure), MED (exact DRAM factor) | perfetto `dgpu.engram` on a pinned-window layer at B=512 (expect ~135 ms per pass-set), or rocprofv3 on `q8_0_gemv_batched_warp8` |
| 3 | **iGPU prefill MoE: MXFP4 has no WMMA arm; the resident-iGPU expert-streaming roofline is below target at B=512/lane.** `dispatch.rs:278-293` (`igpu_moe_wmma_selected` requires IQ2_XS/IQ2_S gate + IQ3_XXS down), `forward_prefill.rs:3513`; fallback `mxfp4_pair_matvec.hip:334` (kwide) + `mxfp4_matvec.hip:204` (`kwide2`, per-member global Q8_K reads at `:311-312`) | (a) V4-Flash's WMMA arm was +62 % (448 → 727 tok/s) and now runs the iGPU near its weight-BW floor; V4.1 is on the older path. `kwide2` re-reads activations from L2/MALL per (member, lane, superblock): ≈ 320 row-pair tiles × 384 work items × 8 warps × 8 members × 2 iters × 2 KB ≈ **31 GB per layer per 512 rows vs 2.4 GB of down weights (13×)**; the identical structure on V4-Flash (17 GB) measured 12–20 ms vs a 3.8 ms DRAM roofline (3–5×). (b) Each lane dispatches its own MoE, so **expert bytes are streamed once per lane**: tok/s ≤ lane_rows × 214e9 / (20 layers × distinct × 18.8e6) = **787 tok/s at 512 rows/lane** (distinct ≈ 370 of 384; routing measured flat) even at 100 % kernel efficiency; ≈ 450 at kwide's V4-Flash efficiency. | per lane per layer: roofline 32.5 ms (gate/up 21.7 + down 10.9) vs est. 55–70 ms today | **~+60 % prefill once resident** (WMMA arm); then the *structure* decides: ≥ 1000 tok/s needs lane_rows ≥ ~700 (B_MAX 2048), or one MoE dispatch over both lanes (super-chunk layer-major, which memory says "still stands" for V4.1), or box 2 owning ≥ 30 % of picks | MED-HIGH | rocprofv3 on `mxfp4_pair_matvec_fused_swiglu_kwide` + `mxfp4_matvec_par_by_expert_kwide2` on a pinned-window layer at B=512 vs 21.7 + 10.9 ms |
| 4 | **`attention_swa_batched` ~80× over roofline.** `attention_swa.hip:120-187` (per key row: 2 FMAs/thread + 8-level LDS tree with 9 `__syncthreads`), launch `forward_prefill.rs:2397`, `attention.rs:201-232` (grid `(64, B)`, block 256) | One 256-thread WG per (head, token); ≈ 1150 barriers per WG, 32768 WGs per call; VGPR 15 / LDS 2 KB so occupancy is not the issue — it is barrier/latency bound by construction. **Measured on V4-Flash, same kernel, same shape: 16.2 ms per call at B=512** (`PREFILL_BALANCE_2026-09-12`: 2.07 s / 128 calls). | Roofline 0.21 ms (q 67 MB + out 67 MB + kv 0.6 MB) / 0.18 ms f32-VALU (8.6 GFLOP) → **~78×**. V4.1: layers 0–1 → 32 ms per 512 rows = 6.3 % of budget | **−31 ms per 512 rows.** Route `ratio == 0` through the existing batched WMMA `score_…_f16s` + `smwsum_…_ldsv_f16s` with `comp_kv = None`, `n_comp = 0` (≈ 0.3–0.5 ms at 128 keys). Decode twin: 2 × ~40 µs/token, ignore | HIGH (measured) | `k.attn.swa` stage in a V4.1 prefill perfetto |
| 5 | **mHC pre-mix matvec re-reads each activation row 24× and each weight row B×.** `forward_prefill.rs:1358` and `:3030` → `f16.rs:511-541` (`f16_matvec_narrow_batched`, grid `(24, 1, B)`) → `f16_matvec_narrow.hip:309-344` | 12,288 WGs each read an 80 KB activation row + a 40 KB weight row: **1.47 GB of L2/MALL traffic per launch for 42 MB of unique bytes**. V4-Flash measured 559/544 µs per call at B=512 (16384-wide, 1.18 GB → 2.1 TB/s = MALL-bound), 7.7 % of the binding device. | 10.4× over the 54 µs roofline on V4-Flash; V4.1 (20480-wide): ~0.7 ms × 2 chains × 20 layers = **28 ms per 512 rows (5.5 %)** vs ~3 ms | **−25 ms per 512 rows.** Use `f16.gemm_batched_wmma` (`f16_gemm_wmma_lds_tiled`: f16 W × f32 X cast at LDS load, already the router's kernel at `:3095`) with M=24 padded to BM=64 — BW-bound on X, ~0.1 ms; fold `rms_nw`'s per-row `inv_rms` into the epilogue and delete the 84 MB `flat` round trip (`:1352-1355`; decode already does "inv only"). Gate with the 2× parity floor (mHC is fp32 in the reference; the weights are already f16) | MED-HIGH | `k.mhc_pre_attn.f16_matvec` stage at B=512 on V4.1 (expect ~0.7 ms), then A/B |
| 6 | **Decode MXFP4 MoE kernels: unmeasured, three V4-Flash-shape mis-fits.** `mxfp4_pair_matvec.hip:136` (16 block-lanes over 20 superblocks), `:109-117` (`MXFP4_PAIR_STAGE`), `:94-98` (LDS LUT); `mxfp4_matvec.hip:190` (8 block-lanes over 9 superblocks), `:143` (hetsplit) | (a) Lane geometry inherited from 4096/2048: gate/up lanes 0..7 run 2 iterations while 24 lanes idle (62.5 % issue efficiency, 4× less memory-level parallelism in the tail); down 9-over-8 → 56 %. Exact tilings exist at 17-B block granularity: 160 blocks per 5120-row = 32 lanes × 5; 72 per 2304-row = 8 lanes × 9 (4 rows/warp). (b) nibble→int8 through a 16-B LDS LUT: ISA shows **256 `ds_load_u8` per 64 `v_dot4`** (4 LDS ops per dot4) in the batch/hetsplit kernels, 64 per 16 in the down kernel; two `v_perm_b32` + a select (llama.cpp's HIP `get_int_from_table_16`) does 4 values in ~6 VALU with zero LDS. (c) ISA confirms the xq staging loop is `global_load_u8` + `ds_store_b8`, 23 serialised iterations per thread before the barrier, 1728 WGs/layer. Weight loads themselves are fine (`global_load_b128` ×3 + u8 per 4 blocks). VGPR 66/103, LDS 9.4 KB → 14–16 waves/SIMD, 6 WGs/CU: occupancy is not the limiter. | Roofline 0.527 ms/layer = 21 ms/token; expectation 1.2–1.5× → **4–10 ms/token** recoverable; DECODE_M8_PLAN budgets exactly the roofline (0.58 ms "local MoE"), which nobody has measured | 4–10 ms/token (10–20 % of the 47.7 ms budget); fix order: dword/b128 staging (trivial), register LUT, block-granular lane map | MED-LOW — measure first | rocprofv3 (iGPU) on `mxfp4_pair_matvec_fused_swiglu_batch_hetsplit` and `mxfp4_matvec_par_batched_hetsplit` per layer vs 351 + 176 µs; iGPU PMC `SQ_INSTS_LDS`, `MemUnitBusy` (PMC works on the iGPU) |
| 7 | **NVMe miss/stream path does the HF→ggml nibble repack on the CPU reader threads.** `hf_v41.rs:585-635` (`read_expert_raw`: 2 Vec allocs + 2 preads + `out × nb × 8` = 2.95 M bounds-checked iterations per role), `expert_pager.rs:108` (4 threads), `:339` (`ensure_layer_dense`), `:506` (`ensure`) | Per expert ≈ 9 M scalar iterations ≈ 10 ms of CPU vs 4.4 ms of NVMe (18.8 MB / 4.31 GB/s). With 4 threads the pipeline is CPU-co-limited: 4 × 18.8 MB / (4.4 + ~10 ms) ≈ 5.2 GB/s theoretical → observed **3.83 vs 4.31 GB/s (−11 %)**; the single-threaded decode miss (10.7 ms) is ≈ 4.4 read + 4–6 repack + 2.6 copy. | streaming at 89 % of the NVMe ceiling; decode miss 2.4× the NVMe floor | +12 % on today's NVMe-bound prefill and replay (14.75 → ~16.5 tok/s); decode miss 10.7 → ~5–7 ms (M8 R1.8's corrected target). Fix: pread raw HF bytes into the slot and repack on the iGPU (18.8 MB @ 214 GB/s = 0.1 ms), or read the HF layout directly in the two MXFP4 kernels (a lane's 16 elements are 8 *contiguous* HF bytes) | MED | `PagerCounters.decode_repack_ns` vs `decode_pread_ns`, and `hf_v41::expert_read_profile()` after a prefill — the counters already exist |
| 8 | **CED replay pages all 384 experts per decoder layer for ≤ 128 rows.** `forward_prefill.rs:3307` → `expert_pager.rs:339` | 20 × 384 × 18.8 MB = 144 GB per request = the fixed ~31 s replay (ENGINE_PORT M7 (c)). The replay's union is 768 picks over 384 experts → E[distinct] = 384(1 − e⁻²) = 332 under flat routing → 13.5 % of the bytes are never read by a kernel. A per-layer `d_selected` readback (3 KB, ~50 µs) is free against 1.5 s of paging. | 13.5 % of 31 s | **−4 s per request** (of 76 s at 1.8K tokens) until box 2 holds the decoder set | MED-LOW | log \|∪ d_selected\| per replay layer |
| 9 | **V4.1 decode chain launch structure** (DECODE_M8_PLAN step 3 owns this; one item it misses). `forward_layer.rs:497` (qkv graph off under v41), `:571-635` (6 un-graphed kv launches), `:617` (`f16rt` after `fp8_act_quant_inplace`), `:506/:547` (`q_normed` D2D copy), `:342-343` (Engram forces standalone graphs) | `f16rt` is numerically a no-op after the V4.1 window quant (E4M3 × 2ᵉ products are exact in f16) — delete it; the 128 KB `q_normed` copy exists only to keep the buffer flow (rope can read `q`). | ~1–3 ms/token of un-graphed launches (×40) | 1–3 ms/token, mostly covered by M8 step 3 | HIGH (structure) | M8's M-b (harvest device events on the paged path) |
| 10 | Minor bundle: `router_topk_par.hip:161-197` at block 512 = 6 argmax passes × 10 barriers (~10–15 µs × 40 = 0.4–0.6 ms/token; a wave-shuffle top-6 over 16 candidates/lane is ~2 µs); ratio-1 compressor at layer 20 uses `f16.matvec_batched` grid.z = B (`forward_prefill.rs:1761`, `f16.rs:499`: 512 × 5.2 MB = 2.7 GB MALL traffic per 512 rows ≈ 2 ms, 0.4 %); prefill `q_normed` copy (`:1516`: 64 MB D2D per layer per lane, 0.2 ms × 20 = 4 ms per 512 rows, 0.8 %) | | | ≤ 1 % each | HIGH | — |

## 2. Detail and arithmetic

### 2.1 Dense attention (finding 1)

Decode rows attended per token at context P: layers 2–19 at ratio 2 contribute 18 × P/2, layers
20–39 at ratio 1 contribute 20 × P → **29·P rows**. The only measured B=1 rate on this hardware is
M58's "score 58 µs + smwsum 260 µs at 16384 rows" = 19.4 ns/row (which is itself 12.5× the 1 KB/row
byte floor, but that matters only while dense exists). Sparse: 38 layers × (128 + 512) rows.

| P | dense ms/token | sparse ms/token | bytes floor (dense) |
|---|---|---|---|
| 8K | 4.6 | 0.5 | 0.37 ms |
| 32K | 18.4 | 0.5 | 1.5 ms |
| 100K | 58 | 0.5 + indexer ≤ 0.5 | 4.6 ms |
| 300K | 173 | — | 14 ms |

Prefill per 512-row lane per compressed encoder layer: keys/query = 128 + P/2. Anchor: V4-Flash's
32K trace, sparse (640 keys), `k.attn.score` 534 µs + `k.attn.smwsum` 616 µs = 1.15 ms → ≤ 1.8 µs/key
(fixed costs included, so the dense per-key rate is if anything lower — call it 1.2–1.8).
18 encoder layers × 1.8 µs × (128 + P/2): P=4K → 70 ms; P=5.9K → 100 ms and the stride error;
32K → 535 ms; 100K → 1.6 s per 512 rows. The scores scratch is 64 heads × stride × 2 B per row, so
dense at 100K (50K keys) would need 3.3 GB per lane — sparse is the only design, not an
optimisation. The three caps (`ATTN_SCORES_STRIDE`, `ATTN_MIXED_MAX_KEYS`, the `--ctx` check) were
all derived for ratio ≥ 4 and need a V4.1 derivation with ratio 1 (n_comp = N).

### 2.2 Engram prefill GEMV (finding 2)

`q8_0_gemv_batched_warp8` is one warp per (row, b) with `b = blockIdx.z`; the weight row tile is
52 KB and 3200 WGs (one z-slice) = 167 MB separate two visits to the same tile, so every z-slice is
a DRAM pass. Per 64-row pass: 64 × 167 MB = 10.7 GB → 16.7 ms at 640 GB/s. `ENGRAM_CHUNK = 64` →
8 passes per 512-row lane → 134 ms per Engram layer; layers 1 and 14 → **267 ms per 512 rows**.
Roofline: weights once (0.26 ms) + 512 × 25600 × 6144 × 2 = 161 GFLOP (≈ 3–4 ms at 40–50 TF).
`q8_0_gemm_wmma_lds_tiled` measured 6× the dp4a batched GEMV on exactly this class of shape
(qb: M=32768 K=1024 B=512 → 1.38 ms). Keeping the 8 passes costs 8 × 0.26 ms of weight re-reads,
still < 8 ms total. Decode uses `q8_0_gemv_warp8` (one pass, 0.26 ms/layer at roofline) — fine.

### 2.3 iGPU MoE and the per-lane streaming roofline (finding 3)

Weights per layer per lane at B=512: distinct experts × 18.8 MB. With 3072 picks over 384 flat
experts the union is ≈ 370–384 → 7.0 GB → 32.5 ms at 214 GB/s. Twenty encoder layers, two lanes
per 1024-row chunk: 1.3 s → **787 tok/s** at 100 % kernel efficiency. At B=1024 rows/lane: 1533.
The two-lane pipeline exists to overlap the two devices; it doubles iGPU expert traffic relative
to one dispatch over both lanes' rows. V4-Flash's post-WMMA iGPU sits near its own BW floor
(0.8 s/chunk floor vs 716 tok/s measured, 62 % iGPU busy), so the WMMA arm is the known way to get
the *kernel* to the floor; the *structure* (rows per dispatch, or expert ownership split with box 2)
sets the floor itself. The MXFP4 LDS-fill dequant for the WMMA arm is a 16-entry table — the
cheapest of the four formats the arm already handles.

Why kwide2's down is cache-bound (the V4-Flash q2k evidence transfers): per warp it streams
`members × 9 sb × 32 lanes × 64 B` of Q8_K from global (L2/MALL) against 2 rows × 1.2 KB of
weights; at 8 members/expert (B=512 / 384 experts × 6) the per-member cost is amortised over
nothing, and the 8 MB `xq` working set exceeds the iGPU L2.

### 2.4 SWA batched (finding 4)

`attention_swa_batched`: per key row a 512-wide dot spread over 256 threads (2 FMAs each) followed
by an 8-level LDS tree with a barrier per level plus two more — ~1150 barrier round trips per
(head, token) WG; 32768 WGs per 512-row call. Bytes: q 512 × 128 KB = 67 MB, out 67 MB, kv 0.6 MB
(shared) → 0.21 ms at 640 GB/s; FLOPs 512 × 64 × 128 × 512 × 4 = 8.6 GFLOP → 0.18 ms at f32 VALU
peak. Measured 16.2 ms on V4-Flash for the identical kernel at the identical shape. The batched
mixed path (`launch_score_batched_htiled_wmma_f16s` + `…_ldsv_f16s`) already accepts
`comp_kv = None` and per-row `n_comp = 0`; the ratio-0 branch at `forward_prefill.rs:2384-2397`
is the only reason this kernel is still on the path.

### 2.5 mHC pre-mix (finding 5)

`f16_matvec_narrow_batched`: grid (24, 1, B), 256 threads, one (row, b) per WG, VGPR 8. Per WG it
reads its full 80 KB activation row and 40 KB weight row; nothing is shared across the 24 rows or
the B rows. 12,288 WGs × 120 KB = 1.47 GB of cache traffic per launch for 42 MB of unique bytes.
The V4-Flash measurement (559 µs for 1.18 GB) is 2.1 TB/s — the MALL/L2 wall, not DRAM. The GEMM
form ([B × 20480] × [20480 × 24]) is 503 MFLOP — trivial; it is a pure re-read problem.

### 2.6 Decode MXFP4 kernels (finding 6) — static evidence

From the gfx1151 ELF: `mxfp4_pair_matvec_fused_swiglu_batch{,_hetsplit}` VGPR 66, LDS 9360 B,
no spills; loop body 256 `ds_load_u8` + 16 `ds_load_2addr` + 8 `global_load_b128`/`u8` + 64
`v_dot4_i32_iu8` + 208 `s_waitcnt`; staging loop = `global_load_u8` → `ds_store_b8` (byte
granular, confirmed). `mxfp4_matvec_par_batched_hetsplit` VGPR 103 (the plain twin is 45),
64 `ds_load_u8` per 16 `v_dot4`. `kwide` VGPR 119 / LDS 16.8 KB → 3 WGs/CU (same class as the
V4-Flash iq3_s kwide); `kwide2` VGPR 171 → 8 waves/SIMD; `private_segment 0` everywhere.
LDS issue is not a throughput wall at decode (1728 WGs × 8 waves × 512 LDS ops ≈ 7 M wave-ops ≈
60 µs on 40 CUs), so the three items are latency/MLP costs on a kernel that should be purely
BW-bound — hence "measure first": if the kernel already sits at ~0.55 ms/layer, only (a) is worth
doing; if it sits at ≥ 0.8 ms, all three are.

## 3. Structural pass (additive-discipline check, per binding device)

1. **NVMe (binding today: 474 GB per warm 1.8K-token request).** The structure is "stream every
   expert of every layer, per lane, per chunk". It should not exist; the plan (CED + two-box
   residency) removes it. Two pieces survive residency and are findings above: per-lane expert
   streaming on the iGPU (#3b) and dense per-layer paging for a 128-row replay (#8). The CPU repack
   (#7) is an 11 % tax on the structure while it exists.
2. **dGPU (binding once experts are resident — and the current prefill balance on V4-Flash).**
   Dense compressed attention should not exist at any context (#1); it is not tunable into the
   budget and hard-fails before 100K. After that, the three re-read kernels (#2, #4, #5) are
   together ~325 ms per 512 rows ≈ 63 % of the 1000 tok/s budget, all replaceable by kernels that
   already exist in the tree (`q8_0_gemm_wmma_lds_tiled`, the batched WMMA attention pair,
   `f16_gemm_wmma_lds_tiled`). None of these requires a new kernel.
3. **iGPU.** The by-expert `kwide2` down (activations re-read 13× the weight bytes) and the kwide
   gate/up are the structure V4-Flash replaced with the WMMA arm; at V4.1's 384 experts × 6 used
   the per-expert member count halves, which is the regime where per-member costs dominate. Beyond
   the kernel, `tok/s ≤ lane_rows × 214e9 / (20 × distinct × 18.8e6)` says 512 rows/lane cannot
   reach 1000 tok/s on one box regardless of kernel quality; the lever is rows per dispatch or
   expert ownership across boxes, and that decision should be made before tuning the kernels.
4. **Decode.** DECODE_M8_PLAN owns the chain (its 0.55 ms/layer is 2× the 0.28 ms byte floor; the
   plan defers that as "design A"). What this review adds: the plan's MoE phase (0.58 ms) is an
   assumed roofline, and at any context past ~8K the dense-attention term (#1) is larger than
   every other item in its budget table.

Lessons applied (nothing above contradicts them): no integer WMMA proposed (WMMA-IU8 ≡ dp4a); the
WMMA MoE arm is the *f16 LDS-tiled* structure that already won on this hardware, not a FLOP
argument; no LDS-adding attention variants (finding 4 *removes* a kernel in favour of the
`_ldsv_f16s` pair that survived the exhausted axis); no multi-WG-into-single-WG fusions;
launch-count items are listed as fuse/delete, not graph.

## 4. Checked and found adequate (do not re-review)

- **Compressor** (V4-Flash's 28 % item): V4.1 has three ratio-2 source layers (2, 8, 14) on the
  tiled pair matvec (`f16_matvec_pair_batched_tiled`, TILE_B = 8, VGPR 117 → 4 waves/SIMD) at
  ≈ 1.3 GB of cache traffic per layer per 512 rows (~1 ms) plus one ratio-1 layer (finding 10).
  ≈ 3–4 ms per 512 rows (< 1 %). Not a V4.1 lever; the V4-Flash finding does not transfer.
- **Router gate in prefill** already uses `f16_gemm_wmma_lds_tiled` (`forward_prefill.rs:3095`);
  decode router matvec is 3.9 MB (~10 µs). `router_topk_par` is finding 10 only.
- **Decode Engram** `q8_0_gemv_warp8` over 167 MB: 0.26 ms/layer at roofline; the dense Q8 GEMVs
  measured 85–99 % of BW (M29 audit). `engram_gate_add` (grid (4, B), three warp-reduced sums):
  320 KB/token, noise. `fp4_kv_quant_inplace` / `fp8_act_quant_inplace`: one thread per element,
  ≤ 512 rows per chunk, noise (their *launch count* in decode is finding 9).
- **`hc_post`/`hc_post_batched`** elementwise BW-bound; **`hc_sinkhorn_par`** is a single wave
  (its barriers are intra-wave). **MXFP4 weight streams are vectorised** (`global_load_b128`), so
  the 17-B block layout is not a load-granularity problem. **No VGPR spills** in any V4.1 kernel.
- **`mhc_pre_fused`** is not on the path (`MHC_FUSED` default off) — and must stay off: the loop
  bounds at `mhc_pre_fused.hip:294` (32 = 16384/512) and `:437` (8 = 4096/512) are hard-coded for
  V4-Flash, so at V4.1 dims it would normalise over 16384 of 20480 elements and cover 4096 of 5120
  outputs. Trap, not a perf item.
- **Q8_0 projections from fp8**: 1.0625 vs 1.0 B/elem; fp8-native kernels are the known quality
  item, not a bytes lever (−6 %). **Head**: 703 MB Q8_0 → 1.1 ms/token at roofline, unchanged.
- **Sampler**, **compressor state/snapshot/pool** at ratio 2, **`moe_group_builder`** at 384
  experts: nothing shape-dependent found.

## 5. Measurements needed, in order

1. Harvest device events on the paged decode path (M8's M-b) — settles #6 and #9 and gives the
   first real V4.1 per-stage numbers.
2. `V41_ATTN_COMP_CAP=512` vs uncapped at ctx 8K and 32K — prices #1 on this box.
3. One V4.1 prefill perfetto on pinned-window (warm) layers at B=512: `dgpu.engram`, `k.attn.swa`,
   `k.mhc_pre_*.f16_matvec`, `igpu.pair_kwide`, the down stage — #2, #3, #4, #5.
4. `PagerCounters.decode_repack_ns` / `decode_pread_ns` and `expert_read_profile()` after a
   prefill — #7.
5. |∪ `d_selected`| per replay layer — #8.
