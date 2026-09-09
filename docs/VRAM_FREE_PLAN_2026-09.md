# Freeing dGPU VRAM (and therefore host RAM) — plan, 2026-09-09 (rev 3)

Rev 3 = rev 2 + the second-pass corrections (naive IndexerScore converted, not deleted;
lever-4 rationale corrected; index_comp_kv v3 recovery + per-blob format flags;
exhaustive expansion oracles; 60-min window; back-to-back baseline).
Rev 2 folded in the first architect review: lever 1 recomputed from device sizes (0.30 GiB, not
0.55), lever 3 changed to E2M1 + per-32-block exponents (the indexer's own QAT format),
the decode producer for lever 2 corrected, the prefill dense path added, snapshot
compatibility made real, the coldest-slot policy defined over ids absent from the
placement file, lever 5 dropped (not lossless), K chosen in-window from sysfs.

## Why VRAM

Host RAM is the binding constraint (93.4 GiB total, IQ3_XXS experts hold 85.2 GiB of
GTT, ~3.5 GiB available). Hot experts are deduplicated: an expert resident on the dGPU
has no host copy. So every GiB of dGPU VRAM freed becomes a GiB of host RAM via a
larger hot-expert budget K (0.348 GiB per K at 8.69 MB/expert x 43 layers). The dGPU
sits at 15.11 / 15.92 GiB at K=15, 192K ctx.

Placement analysis (2026-09-09, real routing stats): decode is insensitive to which
experts are resident within any feasible K (current global-greedy is within 0.2% of the
leg-model optimum), so extra slots are a pure memory lever and should go to the coolest
experts available (see lever 4 for what the placement file can and cannot express).

## Levers (all lossless — bit-identical device state)

| # | lever | dGPU freed @192K | host via K | effort |
|---|---|---|---|---|
| 1 | per-layer weight arenas (kill 2 MiB allocation rounding) | 0.30 GiB (+0.09 hot stacks) | same | 0.5 day |
| 2 | FP8 storage for compressed KV (ratio-4 layers) | 0.42 GiB | same | 1.5 days |
| 3 | E2M1 storage for indexer keys (per-32-block exponents) | 0.17 GiB | same | 1 day |
| 4 | spend today's free VRAM + the above on K; memory slots → coolest experts | — | 0.348 GiB per K | 0.2 day |

Sum freed ≈ 0.30 + 0.09 + 0.42 + 0.17 + 0.80 (free today) = 1.78 GiB. K is chosen
IN-WINDOW from measured free VRAM after the oracle load (see execution), not from this
table; the arithmetic supports +3 K (K=18, 1.04 GiB host, ~0.7 GiB margin) or +4 K
(K=19, 1.39 GiB, ~0.4 GiB margin) once the margin has been measured, not asserted.

Direct host levers (separate track, not VRAM): lazy-loaded vision tower 0.87 GiB;
on-demand iGPU prefill scratch ~0.3 GiB; embedding table as page cache 0.4 GiB; stream
`HotExpertWeights::load` per expert instead of reading whole 256-expert tensors into
host RAM (removes a ~0.8 GiB host spike at the tightest moment of load,
het/weights.rs:703 vs the streaming pattern at :636-647).

Follow-on (touches attention kernels, deliberately out of scope here): FP8 rows in the
gathered `active_comp_kv` / `attn_active_comp_kv` buffers (256 MiB per prefill lane x2)
with expansion at LDS staging: ~0.2 GiB plus fewer V bytes in the BW-bound smwsum.

## Lever 1 — per-layer weight arenas

Facts: the driver charges allocations > 1 MiB in 2 MiB granules on both GPUs
(alloc_granularity_probe.rs). `weights::load_to_device` makes one hipMalloc per tensor.
Waste must be computed from DEVICE sizes, not the GGUF header: the ToF16 roles
(weight_contract.rs:92-103; `attn_compressor_{kv,gate}` x43, `indexer_compressor_*` x21,
`indexer.attn_q_b` x21) are Q8_0 on disk but f16 on device (4096x1024x2 = 8.00 MiB, zero
granule waste), F32 roles halve. Recomputed over shards 2-4: 0.296 GiB on the dGPU.
Hot expert stacks are 3 buffers/layer (one per matrix): ~0.09 GiB. Cold stacks on the
iGPU: ~0.13 GiB of GTT (direct host saving, same change).

Design: an arena entry point beside `load_to_device` (its signature stays; the vision
tower, Laguna and tests keep using it). Per layer per device: sum the POST-CONTRACT
sizes (`n_elements*2` for ToF16, `byte_size` otherwise; the Q8_0 repack is
size-preserving, weights.rs:106-121), allocate one `DeviceBuffer<u8>`, carve 256-B
aligned `slice_view`s (non-owning, buffer.rs:85-132), load each tensor into its view.
The layer weight struct owns the arena. Non-layer tensors → one global arena. Hot stacks:
one buffer per layer holding gate|up|down (they load after the layer loop,
het/weights.rs:1017-1049, so this is a separate per-layer allocation).

Safety: views are raw pointers with no lifetime; safe because nothing frees or replaces
a tensor after load (`hot_experts` assigned once at het/weights.rs:1035, no reload path).
Graph-captured decode kernels bake weight pointers at capture; arena views are load-time
stable, so unaffected.

Verify: one-layer arena-vs-per-tensor byte-equality test (runs beside the live server);
VRAM used (sysfs) before/after at the same K; all oracles bit-identical (bytes
unchanged); bench numbers unchanged.

## Lever 2 — FP8 compressed KV

Facts: the compressor rounds each stored row's 448 non-RoPE dims through E4M3 with a
block-of-64 power-of-two scale: `c_e4m3fn_values[best] * ldexpf(1, e)` with
`e = ceil(log2f(amax/448))` (fp8_e4m3fn.hip:96,137,184), then RN-to-half
(`f16rt`, comp_kv_append.hip:19,35). Expanding `half_rn(table[code] * 2^e)` in f32 is
bit-identical to what the cache holds today, subnormal flushes included. Cross-checked
on a 92K-token snapshot: 7056/7056 sampled blocks exactly E4M3 x 2^e, e in [-8, -5].

Row format (ratio-4 layers only; ratio-128 layers stay f16 — their caches are ~1.5 MB
each and are read directly by the dense path; `HetCompressorState` (het/state.rs:25-37)
therefore carries two formats, selected per layer): 448 x u8 codes (sign + 7-bit index)
| 8 x u8 block exponents (7 used) | 64 x f16 RoPE dims = 584 B, padded to 592 B (16-B
aligned) → −42% vs 1024 B; 21 layers x 49152 rows x 432 B = 0.42 GiB at 192K.

Producers:
- decode: the direct-launch chain `de.fp8.launch → f16rt → comp_kv_append` at
  forward_layer.rs:638-670 (NOT `kv_post_fused`, which writes the raw SWA cache inside
  the captured qkv_chain graph, forward_layer.rs:390-410, and must not be touched).
- prefill: `fp8_e4m3fn_quantize_batched` + `comp_kv_append_*` at forward_prefill.rs:1482-1500.
- refactor `dsv4_e4m3fn_dequant()` to also return the index; keep the exponent expression
  `e = ceil(log2f(amax/448))` VERBATIM (device log2f is not correctly rounded; a different
  expression can flip e at exact powers of two — see the indexer_qat.hip:90-92 comment).
None of these launches are graph-captured (no baked-pointer issue).

Consumers:
- sparse path (decode and prefill, n_index_comp > 512): `indexer_gather` (+ batched)
  expands FP8 → f16 into `active_comp_kv` / `attn_active_comp_kv`; the attention kernels
  are untouched.
- decode dense path (forward_layer.rs:995, ctx ≤ 2K): identity gather into `active_comp_kv`.
  `DECODE_INDEXER=off` (forward_layer.rs:867-869) forces dense at any depth with no f16
  scratch for it: make it a hard error.
- prefill dense path: `need_mask` is per-chunk `any(n > 512)` (forward_prefill.rs:1819-1822);
  below it, score/smwsum read `cs.comp_kv` directly (:1766, :2027-2034). Expand the
  `n_comp` rows ONCE per chunk into a shared f16 scratch (batch_stride = 0) — not a
  per-token gather, which would turn a shared 0.5 MiB read into B x 512 KiB.
- ds4 oracle dump `attn_comp_kv` (het/engine.rs:517-522) reads f16: expand there too.
- vision/image rows use the same compressor chain and snapshot code: no special case.

Snapshots: FORMAT_VERSION 3 → 4 with FP8 rows on disk (new snapshots shrink 42%), AND
explicit v3 acceptance in BOTH places that filter on the version — the index loader at
startup (snapshot.rs:307-315) and `restore_vl` (:1051) — with f16→FP8 conversion on
restore: per block recover e' = ceil(log2(amax_stored/448)) ∈ {e0−1, e0}, verify every
value is exactly `table[code] * 2^e'` after RN-to-half, refuse the row otherwise (the
amax-floor case e = −22 is f16-flushed and unrecoverable; refuse = fall back to prefill).
Byte variants of `stream_u16`/`load_u16` (snapshot.rs:901,935,1153,1203). Keeps the 7 GB
of on-disk snapshots usable across the restart.
v4 `meta.json` carries per-blob format flags (`comp_kv_format`, `index_comp_kv_format`,
row strides) so lever 2 and lever 3 can ship independently under one version number.
`index_comp_kv.bin` is f16 in v3 too (snapshot.rs:935,1203): recover per 32-block
e' = ceil(log2(amax_stored/6)) ∈ {e0−1, e0} (the stored block max is one of {3,4,6} x 2^e0;
the e0−1 case doubles codes ≤ 3, all still E2M1), verify every value equals
half_rn(e2m1[code] x 2^e'), refuse the row otherwise.

Verify: EXHAUSTIVE expansion oracle — the code space is tiny (254 E4M3 codes x the ~31
exponents e ∈ [−22, 8] the kernel can emit ≈ 7.9K values): enumerate every (code, e)
through the old path (fp8_e4m3fn.hip:96-137 → f16rt) versus the new expand, bit-identical.
Pin the expand arithmetic: f32 product then `__float2half_rn` (the same intrinsic and
compile flags as `f16rt`), never a half-typed multiply, so denormal-mode differences
cannot creep in. Plus random-row oracle beside the live server; ALL existing decode +
prefill oracles unchanged (bit-exact expectation); VRAM delta at 192K = 0.42 GiB.

## Lever 3 — E2M1 indexer keys

Facts: indexer K rows are produced by `indexer_qat` (kernels/indexer_qat.hip:75-101):
Hadamard128, then per-32-element block `scale = 2^ceil(log2(amax/6))`, E2M1
nearest-even. The structural invariant is E2M1 x per-block 2^e (4 exponents per
128-dim row). (Per-row E4M3, the rev-1 format, is only exact while a row's four block
exponents spread ≤ ~7 binades — true on the sample, not by construction. Rejected.)

Row format: 64 B of E2M1 codes (4 bits each) + 4 e8m0 exponent bytes = 68 B → 72 B
(8-B aligned) vs 256 B → −72%; 21 layers x 49152 rows x 184 B = 0.17 GiB at 192K.
Exact by construction: expansion = f32 `e2m1[code] * 2^e` then `__float2half_rn`
(same intrinsic as the producer's f16 write). Verify EXHAUSTIVELY: 16 codes x 254
exponents = 4,064 values through indexer_qat.hip:98-101 → f16 versus the new expand,
bit-identical.

Producer: `indexer_qat` already has the scale (indexer_qat.hip:97); fuse with the
indexer-compressor append so codes + exponents are written directly.
Consumers (ALL SIX, no runtime format toggle exists so an unconverted variant reads
garbage): decode `launch_mw`, `launch` (sw), naive `indexer_score.launch`
(forward_layer.rs:930-960); prefill `launch_batched_gemm`, `_mw`, `launch_batched`
(forward_prefill.rs:1925-1960). Convert at LDS staging in the three kernels that stay
in production (mw B=1, batched mw, gemm) AND convert the naive `IndexerScore` kernel:
it is the only implementation on gfx1151 and the ds4-dump-backed oracles deliberately
run there (tests/indexer_pipeline.rs:73-81,461; tests/indexer_score.rs:18-26,126;
`IndexerScoreWmma` exists only for gfx12, het/engine.rs:234), so it stays as the
reference. DELETE only the WMMA `sw` entry points (`launch`, `launch_batched`) and
hard-error the env values that select them (`INDEXER_DECODE=sw`,
`INDEXER_SCORE_VARIANT=sw`; `mw` stays valid — it maps to `launch_batched_mw`,
forward_prefill.rs:1938-1948). Test harnesses upload f16 rows directly
(tests/indexer_pipeline.rs:193-221, tests/indexer_score.rs, bench_indexer_score_isolated.rs):
add host-side packing. Rows are 72 B: legal for unaligned dwordx4 loads on gfx12, but the
LDS staging code must not assume 16-B row alignment (80 B buys it for 0.008 GiB).

Perf gate (not just exactness): the gemm score kernel runs at 68% of matrix peak and
the staging-time expansion adds VALU work on the decode critical path at depth. Gate on
decode @192K (bench_decode FAKE_POS) and prefill @192K within noise of today; if the
expansion costs > 1%, keep a 16-row LDS expansion buffer per tile rather than per-load
conversion.

## Lever 4 — spend it

After 1-3: K chosen in-window (below). Placement policy: the sidecar `hot_experts.txt`
carries only the top 64 nonzero experts per layer (expert_stats.rs:170,173), so "rest by
ascending frequency" over the file would pick the WARM ranks 17-64 and add real dGPU
work. Define memory slots over ids ABSENT from the file, deterministic ascending-id
order. Honest rationale: absent means rank 65-256, not never-picked — in the served
stats no expert has a zero count in both banks, and ranks 65-256 carry 33-57% of a
layer's picks (~0.2% each). Cost of an absent-id slot: ~0.2% x 6 picks x 12.3 µs per
layer, i.e. a few extra dGPU expert evals per token (~20 µs) that come off the iGPU long
pole — negligible either way. Exact alternative if wanted: make `write_placement` emit
all 256 entries (140 KB file) and rank memory slots by true ascending score. The first
645 slots stay exactly today's global-greedy placement. Implement in
`parse_hot_expert_file` (het/weights.rs:897-953) behind `DGPU_MEMORY_SLOTS=<n>`.
Verify: decode and prefill tok/s within noise; host MemAvailable +0.348 GiB per K.

## Execution

Branch `vram-free`. Order 1 → 2 → 3 → 4. Each lever independently shippable; kernels
and unit oracles run beside the live server (small buffers). Before the window:
snapshots must be readable (v3 acceptance) so the restart does not cold-start every
session; all three levers complete and unit-verified.

Single server-down window (~60 min: a 192K prefill alone is 5-6 min and there are two
model loads, the oracle process and then the server):
1. full-model oracles, all in-process (decode, prefill, pipelined, indexer, batch-vs-seq),
   one weight load.
2. measure PEAK sysfs VRAM in the SERVER binary at K=15 (HIP graphs, vision tower and
   snapshot staging included) over: a 192K prefill, an image request, a decode run.
   The run script's own note says K=17 at ~0.1 GiB free was unsafe; the margin is
   measured here, not assumed, and 0.35 GiB is kept on top of the measured peak.
3. choose K = floor((free_at_peak − 0.35 GiB) / 0.348), set it and `DGPU_MEMORY_SLOTS` in
   the run script, restart, smoke. Perf gate: prefill and decode tok/s BACK-TO-BACK on the
   same binary and quant (bench A/B rule) — the old binary's numbers from the log
   (736 tok/s prefill on the IQ2_S WMMA path, 29.0 decode at short ctx, both on
   UD-IQ3_XXS) are the reference only if taken in the same window.
Rollback: previous K in the run script + the branch's parent commit. FP8/E2M1 formats
have no runtime toggle; a lever that fails an oracle does not ship.

## Not in this plan (measured dead, stale, or not lossless)

Native FP8 for attention/shared projections (Q8_0 keeps 7 magnitude bits, E4M3 keeps
3; only lossless from the safetensors' 128x128 block scales, which the GGUF lacks);
f16 scratch unions (already arenas; ~30 MiB/lane left); entropy-coding expert
codebooks (3%); cold-expert NVMe paging (prefill touches every expert); placement
re-packing for speed (within 0.2% of optimal).
