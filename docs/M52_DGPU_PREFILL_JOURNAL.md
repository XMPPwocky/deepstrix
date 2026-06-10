# M52 — dGPU prefill (long-context) journal

Context: after M51 (kwide MoE, 320 tok/s @4K), the depth curve sagged at long
ctx (290.8 @64K, 261.1 @96K) — the dGPU attention/indexer side stopped being
hidden behind the now-faster iGPU. All profiling per user guidance at LONG
contexts via FAKE_PREFILL_POS (dGPU is invisible at short ctx).

## 2026-06-09 — Phase 0: kernel-trace + pftrace at 96K

rocprofv3 kernel-trace (FAKE_PREFILL_POS=98304, PIPELINE_LANES=2, B=1024/iter,
per 1024-tok chunk): iGPU busy 3231 ms; dGPU busy 1812 ms; wall 3922.
Top dGPU: q8_0_gemm_wmma_lds_tiled 415, **indexer_score_wmma_batched 402
(28.7 ms/launch)**, f16_matvec_pair 238, wmma_grouped 141.

pftrace gaps (per 512-tok lane-chunk): iGPU busy 1568 + 388 idle (9 ms/layer
after q2k_down waiting on dGPU); dGPU busy 1411 + 527 idle (12.4 ms/layer
waiting on iGPU MoE). Perfect packing floor = iGPU busy = 326 tok/s; ~835 ms
of unfilled per-layer ping-pong stalls. Steady-state check (4096 tok/iter):
261.6 — identical to 2-chunk number → stalls are per-layer structural, NOT
pipeline edges.

indexer_score roofline: B=1024 × n_comp 24.5K × 64 heads × 128 dim ≈ 412
GFLOP ≈ 3.4 ms at WMMA peak; measured 28.7 — 8× over. Cause (read from
kernel): 1-wave WGs, each re-staging the token's FULL Q (8192 f32→f16, 256
serial iters/lane) per 128 cols → ~6.3 GB redundant Q reads/launch + 4×
staging:WMMA instruction ratio.

## 2026-06-09 — indexer_score_wmma_batched_mw SHIPPED as default

Design: 8 waves/WG, Q+hw staged cooperatively ONCE per 1024 cols; NO K LDS
(B-fragments straight from global — K is MALL-resident); single barrier, so
per-wave early-outs can't deadlock. Grid (ceil(cols/1024), B), block 256.

- Oracle: **bit-exact** on [0, n_comp); -INF tail contract on
  [n_comp, stride) (the 1-wave kernel leaves cols past its n_idx_max-sized
  grid UNWRITTEN — fine, topk only reads [0, n_comp)).
- Isolated (B=512, n_idx=24576): **32.6 → 7.45 ms (4.4×)**.
- E2e cold (back-to-back):

| depth | sw (1-wave) | mw | Δ |
|---|---|---|---|
| 4096  | 319.7 | 318.6 | noise |
| 32768 | 312.0 | 311.6 | noise |
| 65536 | 291.8 | **314.6** | +7.8% |
| 98304 | 262.3 | **318.0** | +21.2% |

Wall saved at 96K (684 ms) > kernel time saved (~310 ms): the shorter dGPU
per-layer critical path also dissolved most of the per-layer packing stalls.
**Depth penalty eliminated — 96K runs at 4K speed.** Rollback:
INDEXER_SCORE_VARIANT=sw.

## Remaining (next levers at long ctx)

- Depth curve now flat 4K-96K at ~312-320 → the wall is back to the iGPU MoE
  side everywhere; M51's "remaining headroom" list applies at all depths
  (B_MAX=2048, q2k traffic halving, FAST_FULL hotlist → ~400).
- mw kernel could still drop K MALL traffic ~2× (2 tokens/WG sharing
  B-fragments) if indexer ever re-surfaces in a trace.
- q8_0_gemm_wmma_lds_tiled (415 ms/chunk dGPU, depth-independent) is the next
  dGPU item if packing ever exposes it; WMMA-i8 idea from [[q8-lds-tiled-wmma]].

# M53 — iGPU round 2 (no hot-expert cache)

## 2026-06-09 — kwide staging ISA audit

The xq staging loop compiles terribly: per dword — 64-bit addr chain, exec-mask
branching for the ii==0 yd case, and `s_waitcnt vmcnt(0)` before EVERY
ds_store (zero MLP); members copied sequentially with only 130/256 threads.
Plan: branch-free run-per-thread staging (all members concurrent, 16
pipelined loads/thread), then q2k row-pair reuse, then chunk=64 sequential
halves.

## 2026-06-09 — M53.1 staging rewrite SHIPPED: iq2 50.6 → 30.0 ms (−41%)

Branch-free run-per-thread staging (all members concurrent, 16 pipelined
loads/thread, yd by separate thread subset). Oracle unchanged (1.1e-4).
The serialized staging stalled all 8 waves at the barrier — its true cost was
~2× its PMC instruction share.

E2e: **4K 419.9 / 32K 409.8 / 96K 384.0 tok/s** (was 318/312/318).
iq2 now ≈ at the (conservative-estimate) dp4a compute roofline.
96K dips again as the iGPU shrinks → dGPU partially re-exposed (expected).
Next: q2k row-pair (q2k now ~36% of iGPU busy).

## 2026-06-09 — M53.2 q2k kwide2 + dead-fill removal SHIPPED as defaults

- `q2_k_matvec_par_by_expert_kwide2` (Q2K_VARIANT=kwide2, now default): each
  warp dots one loaded q8/bsums set against TWO rows' weights (16 rows/WG) —
  halves the cross-WG activation traffic the kernel is bound on. Members in
  halves of 16 (register budget; 125 VGPR / 10 waves — below the usual 12-wave
  gate, but the kernel is BW-bound and the cold e2e decides). **Bit-exact** vs
  by_expert. Isolated 19.7 → 16.65 ms (−15.5%).
- q2k partials zero-fill REMOVED: router topk gives 8 distinct experts/token →
  group_count[e] ≤ B = max_per_expert → builder overflow guard can't fire →
  every (b, slot) written. Was 128 MB/layer ≈ 44 ms/chunk of fillBuffer.

E2e (defaults, cold, B=1024/iter):

| depth | M52 end | M53 end |
|---|---|---|
| 4096  | 318.6 | **459.7** |
| 32768 | 311.6 | **443.4** |
| 98304 | 318.0 | **402.7** |

(M53.1 staging rewrite contributed 318→420; kwide2 420→450; fill removal
450→460 at 4K.)

Server gate: real cold prefill **420.6 tok/s** (4606-tok prompt); completion
coherent + content-accurate (correctly summarized our own M51 journal).

Day total: **230 → 460 tok/s @4K (2.0×), 224 → 443 @32K, 223 → 403 @96K.**

## Future (logged, not started)
- chunk=64 for q2k only (dual work-item lists): iq2 can't use it (accumulator
  VGPR cap), q2k weight-DRAM halves → ~+2%. Plumbing: second work_items build.
- 96K gap vs 4K (403 vs 460) = dGPU re-exposure round 2: next dGPU item is
  q8_0_gemm_wmma_lds_tiled (~415 ms/chunk) + attention smwsum growth.
- iq2 now ~30 ms/launch ≈ 1.6× over theoretical dp4a peak; remaining levers
  are tail-amortization geometry changes (SBG=4 needs chunk=16 → dequant 2×,
  pencils ~neutral) — likely format-bound without the f16 cache route.

# M61 — prefill het-split MoE (dGPU computes hot experts during its slack)

**Hypothesis.** At PIPELINE_LANES=2 the iGPU MoE chain is the prefill wall
(~2200 ms/chunk iGPU vs ~625 ms dGPU at 4K). The decode het-split (M56)
already keeps the K hottest experts/layer packed dense in dGPU VRAM. If the
dGPU also computes those experts' (b, slot) members during prefill — in its
~1575 ms/chunk of slack, fully async with the iGPU misses — every hit member
comes off the iGPU wall. With K=8 hot experts covering share `s` of
selections, projected iGPU MoE wall shrinks ~s; at s≈0.2 → ~460→
~530-560 tok/s @4K.

**Design** (no changes to the matvec kernels):
- `moe_group_builder_hetsplit` (new): same inversion as `moe_group_builder`
  but residency-aware. Resident slots of a token are ranked in slot order;
  rank < cap (`DGPU_HOT_CAP_PREFILL`, default 255 = all) goes to the dGPU.
  mode=0 (iGPU) keeps the complement in ORIGINAL id space; mode=1 (dGPU)
  keeps hits in DENSE (remap) id space.
- dGPU side avoids the per-layer host readback of n_work_items entirely:
  a STATIC e-major work-items list (e<<16 | c*32, c<B_MAX/32) is uploaded
  once; grid.y = n_hot × B_MAX/32 and the existing kwide/kwide2 guard
  (`member_end <= member_start → return`) early-exits empty chunks. The
  whole hot chain (group build → q8k → iq2 kwide → q8k → q2k kwide2 →
  reduce) queues on de.compute with zero host syncs.
- `q2_k_reduce_partials_hetsplit` (new): each side's reduce sums ONLY the
  slots it computed (recomputes the same resident-rank), preserving the
  M53 no-zero-fill invariant on both devices.
- Combine: extra batched vec_add of `ffn_moe_dgpu` at ffn_combine
  (mirrors decode M56).
- Numerics: dGPU q8k input is the same f32 ffn_input_norm both devices
  hold → bit-identical quantization; per-member sums bit-identical (same
  kernels); only the final slot-sum association changes (hot partial added
  after own-slot sum) → tiny f32 drift, same class as decode M56.
- Rollback: DGPU_HOT_PREFILL=0 (default 1 when hot experts loaded).
  Forbidden combos: Q2K_VARIANT=bxn and IQ2_VARIANT=hybrid error out when
  hot prefill is active.
- VRAM: +~280 MB dGPU batch scratch at B_MAX=1024 (xq 8.4 + mid 50 +
  midq 15 + partials 176 + out 29 + group arrays ~1), gated on
  DGPU_HOT_EXPERTS>0.

**Gates.** forward_prompt_batch_matches_sequential (hot on), then cold
back-to-back A/B DGPU_HOT_PREFILL=0/1 at 4K/32K/96K, PIPELINE_LANES=2,
DGPU_HOT_EXPERTS=8 + decode_hot_experts.txt placement.

## M61 results (2026-06-10, HEAD 801a5bb)

Oracles: all 4 green. pipelined-vs-single-lane **BIT-EXACT** (0.0 diff —
confirms per-member math is batch-split-independent). last_only 4.99e-2
scaled (bound 5e-2, argmax match) — thin margin; the M61 marginal drift was
fixed by defaulting the prefill cap to decode's DGPU_HOT_CAP=4 so the
slot→device partition (and f32 sum association) matches the sequential
reference exactly. Watch this one for flapping.

Two implementation incidents:
1. Uncapped prefill (cap=∞ vs decode 4) pushed last_only to 5.44e-2 — over
   the bound. Matched caps fixed it; offload cost ~nil (P(>4 resident
   slots) ≈ 0 at K≈8/256).
2. +280 MB/lane hot scratch OOM'd the 3-scratch pipelined oracle. Fixed by
   carving the five big buffers as views into attn_active_comp_kv (537 MB,
   idle during the MoE phase; single de.compute stream ⇒ serial-safe).
   Marginal VRAM now ~5 MB/lane.

Cold A/B, PIPELINE_LANES=2, K=8 (decode placement file), median of 3:

| depth | hot=0 | hot=1 | Δ |
|---|---|---|---|
| 4096  | 459.7 | **500.5** | +8.9% |
| 32768 | 444.8 | **470.6** | +5.8% |
| 98176 | 402.4 | **417.0** | +3.6% |

Ships default-on (DGPU_HOT_PREFILL=1 when DGPU_HOT_EXPERTS set).

Saved iGPU wall at 4K ≈ 190 ms/chunk ≈ 9% of the MoE leg — consistent with
a ~10-12% hot member share under prefill's flatter routing (placement file
is from DECODE token stats). Future levers:
- prefill-specific placement (collect prefill expert stats per layer;
  re-rank); could also bump per-layer K if the 128K-KV margin allows.
- raise DGPU_HOT_CAP_PREFILL above 4 only if last_only oracle margin is
  re-examined (association then diverges from decode again).

# M62 — expert-selection stats: device collection → sidecar aggregate → load-time placement

**Hypothesis.** M61's hot set is ranked by decode-token stats
(reference/decode_hot_experts.txt, collected with sync-heavy diagnostics).
Prefill routes flatter; placement ranked on real prefill+decode volume should
raise the hot member share (~10-12% today) and with it the M61 win.

**Design** (user-approved plan, 2026-06-10):
- `expert_sel_count` kernel: counts[layer*256+sel] atomicAdd, launched on
  de.compute after router top-k in BOTH paths (dGPU holds d_selected
  natively). Two persistent u32[N_LAYER×256] banks (prefill/decode) on dGPU,
  88 KB each; no hot-path readbacks. DEEPSTRIX_SEL_STATS=0 opts out.
- Persistence: NOT in KV snapshots (global workload aggregate vs per-prefix
  LRU-evicted state). Sidecar expert_stats.json in the deepstrix cache root;
  flushed at the existing snapshot-save points (turn end / shutdown);
  halving decay past ~10M tokens/bank; model-fingerprint guarded.
- Placement: DGPU_HOT_EXPERTS_FILE may be the JSON; score =
  (1-α)·prefill_freq + α·decode_freq (DGPU_HOT_ALPHA, default 0.5) into the
  EXISTING global-greedy budget. Raw counts stored, placement computed at
  load → adaptive refresh stays open.

**Gates.** (1) device histogram == PrefillStats pick_counts exactly;
(2) prefill A/B stats on/off ≤0.5%; (3) server flush e2e + restart with JSON
placement: 4 oracles green + placement A/B bench (old txt vs JSON).

## M62 results (2026-06-10, HEAD 84994ac)

All three gates green:
1. **Counts oracle**: device histogram == PrefillStats pick_counts, 0 diffs
   over 43×256; token counter exact; banks zeroed after harvest
   (tests/expert_sel_stats_oracle.rs).
2. **Cost**: 4K prefill A/B stats off/on = 504.8 vs 505.6 tok/s median —
   pure noise (the 6144-atomic kernel hides in dGPU slack).
3. **Server e2e**: two requests across sessions → expert_stats.json banks
   exact (prefill 16 tok ⇒ 4128 picks; decode 40 ⇒ 10320) + 43-line
   hot_experts.txt placement; second request's stats flush at next save
   point as designed (kill -9 loses only last-turn stats).

Restarts now self-improve: loader picks up <cache>/hot_experts.txt unless
DGPU_HOT_EXPERTS_FILE is pinned. Placement A/B (accumulated vs legacy
decode txt) deferred until real workload volume accrues — meaningful only
after letta runs feed the prefill bank.
