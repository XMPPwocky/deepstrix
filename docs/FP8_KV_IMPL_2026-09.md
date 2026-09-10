# FP8 compressed-KV storage — implementation (2026-09-10, rev 2)

Implements lever 2 of `VRAM_FREE_PLAN_2026-09.md` (rev 3). Read that section first;
this document is the build order, the design choices where the two differ, and the
gates. Priority order for every choice below: quality > RAM > decode perf > prefill
perf > implementation complexity.

Rev 2 = rev 1 after the architect review and the build. Status: steps 1-4 are
committed and verified beside the live server (1c23813 kernels + proof, b3d6d08
engine + snapshots, da03067 one-load A/B); step 5 (full-model oracles, benches, VRAM
delta, restart) needs the server-down window. Review corrections folded in: the
sign-of-zero code (below), the restore error path (cache miss, not a failed request),
the exponent field (plain i8, producer clamps), the startup guard for `--ctx` vs
`ATTN_MIXED_MAX_KEYS`, the context table (shared scratch counted once), and the D2/D5
rationales.

## What the proof found (worth knowing before touching the format)

**Sign of zero is part of the format.** The quantiser computes `sign * table[idx]`
with `sign = x < 0 ? -1 : 1`, so a negative value that snaps to code 0 is stored as
`-0.0` (f16 `0x8000`); `+0.0` and `-0.0` inputs both store `+0.0`. The packed code
keeps that as the sign bit (`0x80`). The expand must apply the sign to the f16 BITS,
not to the f32 product: the compiler contracts `cvt_f16(a * b)` into `v_fma_mix`
with a `+0.0` addend, and `(-0.0 * 2^e) + 0.0 = +0.0` — the old chain never
contracted because the product crossed a kernel boundary through memory. A
single-kernel "replica" of the old chain is therefore NOT a faithful oracle; the
proof pushes every reachable (code, e) through the real old kernels instead.

## What ships

Ratio-4 layers (21 of 43) store each compressed-KV row as FP8 codes + block exponents
+ f16 RoPE tail instead of 512 x f16. Reconstruction is bit-identical to today's f16
cache (the compressor already E4M3-rounds every stored row with a power-of-two block
scale; see the plan's "Facts"). Nothing numerical changes anywhere; every existing
oracle must pass unchanged.

Row format (unchanged from the plan): `448 x u8 code (sign<<7 | 7-bit table index)
| 8 x u8 block exponent (7 used; biased i8) | 64 x f16 RoPE dims` = 584 B, padded to
592 B (16-B aligned). 21 layers x 49152 rows x (1024 - 592) B = 0.42 GiB at 192K.

Ratio-128 layers and the indexer compressor stay f16 (lever 3 handles indexer keys).

## Design choices that differ from the plan

### D1. Dense path reads an f16 "head shadow" — no identity gather

The plan routes the dense case (n_index_comp <= 512, decode at <= 2K ctx and prefill
chunks entirely below 512 comp rows) through the gather kernel with an identity
selection. That adds one launch per ratio-4 layer on every short-context decode
token: ~21 x 5 us = ~0.1 ms/token, ~0.3% at 32 ms/token. Decode perf outranks
complexity, and short-context decode is the common case, so instead:

- `HetCompressorState` (ratio-4) keeps a `comp_kv_head: DeviceBuffer<u16>` of
  `INDEXER_TOP_K` (512) x 512 f16 = 512 KiB/layer, 10.5 MiB total.
- The new append kernel writes the FP8 row always, and ALSO the f16 row into
  `comp_kv_head` when `row < 512`. Same values the f16 cache holds today.
- The dense path (decode `forward_layer.rs:993-995`, prefill
  `forward_prefill.rs:1766` and `:2032-2034` with batch_stride 0) reads
  `comp_kv_head` exactly where it read `comp_kv`. Zero new launches, zero attention
  kernel changes, and bit-exactness is trivial (the shadow IS the old cache prefix).
- Guard: the dense path hard-errors if `n_comp > 512` (this is the
  `DECODE_INDEXER=off` case from the plan; also the FAKE_POS bench overrun). The
  sparse/dense decision is made on `n_index_comp` (`forward_layer.rs:863`); assert
  `n_comp_full <= 512` whenever the dense path is taken, on both decode and prefill.

Cost: 10.5 MiB VRAM. Benefit: dense-path decode and prefill are launch-for-launch
identical to today.

### D2. Arithmetic E4M3 decode in the expand, not the `__constant__` LUT

Decode the code arithmetically (sign, 4-bit exponent, 3-bit mantissa; subnormal
when exp==0) — pure VALU, no memory traffic. (On AMDGPU `__constant__` is ordinary
global memory and divergent indices are plain vector loads, so this is a small win,
not a serialisation fix.) `fp8_kv_table_check` asserts `decode(i) ==
c_e4m3fn_values[i]` for all 127 reachable indices (index 127 is E4M3FN NaN and the
search never emits it).

### D3. Fewer launches on the producer side (free decode win)

Today the decode boundary chain is `fp8.launch -> f16rt -> comp_kv_append` (3
launches, `forward_layer.rs:638-670`); prefill is `fp8_e4m3fn_quantize_batched ->
f16rt -> comp_kv_append_batched` (`forward_prefill.rs:1482-1500`). The new
`comp_kv_append_fp8{,_batched}` takes the post-RoPE f32 row and does quantise + pack +
head-shadow write in one kernel. 3 -> 1 launches per boundary (boundaries fire every
4 tokens per ratio-4 layer: ~0.04 ms/token average, small but positive). Check first
that nothing reads the row buffer after the append (the compressor-state
snapshot/shuffle kernels read `state_kv`, not the row; confirm in both call sites).

The exponent expression `e = ceil(log2f(amax/448))` and the E4M3 nearest-code search
are moved VERBATIM into a shared include (`kernels/fp8_e4m3fn_common.inc`, returning
the index, not just the value) and reused by the old kernels, so the old f16 path and
the new packed path share one implementation (pure move: the ds4-dump fp8_quantize
oracle stays at max_abs 0). The exponent is stored as a plain i8; the producer clamps
into the field (unreachable for finite rows: `[-22, 120]`). Non-finite rows are out
of domain (the old chain stores garbage for such a block too). Device `log2f` is not correctly
rounded; a re-derived expression could flip `e` at exact powers of two
(`indexer_qat.hip:90-92`).

### D4. Expansion arithmetic pinned to the producer's cast

Expand = f32 `decode(code) * ldexpf(1.0f, e)` then the SAME f32->f16 conversion the
append uses today (`(_Float16)x`, `comp_kv_append.hip:19,35`), in the same
translation unit and compile flags. Never a half-typed multiply. The product is exact
in f32 (4 significant bits x power of two), so the only rounding is the final cast,
which is what the old path did to the same f32 value.

### D5. v3 snapshot conversion: host recovers, host verifies against a device-proven expand

The restore path recovers codes on the host (per block `e' = ceil(log2(amax/448))`
from the STORED values, then `e'+1`, `e'-1`; per value the code whose expansion is
exactly the stored f16, sign of zero included) and accepts a row only if every value
round-trips through the host expand. The host expand is proven equal to the device
expand for EVERY (code, e) over the whole i8 field (`fp8_kv_format`), so a host-verified
row is a device-verified row; no per-restore device scratch or launches are needed.
(The rev-1 worry about host/device f16 denormal disagreement was unfounded: the
denormal flag is f32-only and `v_cvt_f16_f32` keeps f16 denormals; the exhaustive
comparison settles it either way.) The exponent field can legitimately come back one
lower than the producer wrote (codes doubled) — the expansion is what must match, and
does. A row that does not round-trip refuses the snapshot, which the server treats as a
cache miss (reset, evict the entry, full prefill), not a failed request.

`COMP_KV_FP8=0` keeps the f16 store (rollback and the one-load A/B knob); restore
converts in both directions, so v3/v4 files and f16/FP8 stores are all mutually
readable.

## Build order

Each step is independently testable beside the live server (small buffers). Steps 1-2
are kernel-only and are where the bit-exactness proof lives; nothing in the engine
changes until they are green.

### Step 1 — expansion oracle first (test-driven) — DONE (1c23813)

`kernels/fp8_e4m3fn.hip`:
- `__device__ int e4m3fn_nearest_index(float ax)` — the existing search, returning
  the index; `dsv4_e4m3fn_dequant` becomes `sign * table[idx]`.
- `__device__ float e4m3fn_decode(uint8_t code)` — arithmetic decode (D2).
- `__device__ int comp_row_block_exp(float amax)` — the VERBATIM exponent expression.
- Test kernels: `fp8_e4m3fn_table_check` (128 codes), `fp8_expand_exhaustive`
  (every (code, e), e in [-24, 10], both the old path `quantize -> (_Float16)` and
  the new `expand -> (_Float16)`, writes both u16s).

`tests/fp8_kv_format.rs` (runs on the dGPU beside the server, KiB of memory):
- table check = 128/128; exhaustive = all pairs bit-identical, INCLUDING the
  `e = -22` floor (whatever the cast does to subnormals, both paths do it).
- host pack/unpack reference (`v4flash_kernels::fp8_kv::{pack_row, unpack_row}`)
  cross-checked against the device kernels on random rows.

### Step 2 — producer and consumer kernels — DONE (1c23813)

- `kernels/comp_kv_append_fp8.hip`: `comp_kv_append_fp8` (block 512, one row) and
  `_batched` (grid (1, n_boundaries)). Input: post-RoPE f32 row. Per row: 7 block
  amax reductions (warp shuffles) -> e -> code per dim, pack bytes, write the 64
  RoPE dims as f16, write the head shadow when `row < 512`. Output row stride 592.
- `kernels/indexer_gather.hip`: `indexer_gather_fp8{,_batched}`, same grid shape as
  the f16 kernels (256 threads x 2 dims). Thread t < 224 reads two code bytes and
  its block exponent (the 8 exponent bytes are one u64 broadcast load); t >= 224
  copies two f16 RoPE values. Writes f16 into `active_comp_kv` exactly as today.
- Rust wrappers in `src/comp_kv_append.rs` / `src/indexer.rs`, `for_arch` like the
  existing ones (both GPUs: gfx1151 tests run the same kernels).
- Test `tests/fp8_kv_format.rs` extended: random f32 rows -> old chain
  (`fp8_e4m3fn_quantize -> f16rt -> comp_kv_append`) vs new chain
  (`comp_kv_append_fp8 -> indexer_gather_fp8` with identity selection): u16-exact,
  and the head shadow equals the old rows for row < 512. Include rows with a
  zero block (amax = 0) and rows at the exponent floor.

### Step 3 — state and engine plumbing — DONE (b3d6d08)

- `het/state.rs`: `HetCompressorState` gains a format:
  `enum CompKvStore { F16(DeviceBuffer<u16>), Fp8 { rows: DeviceBuffer<u8>, head: DeviceBuffer<u16> } }`
  (ratio-4 main compressor -> Fp8; ratio-128 and the indexer compressor -> F16).
  Allocation: `max_n_comp x 592` bytes + 512 KiB head. `reset_in_place` unchanged
  (counters only).
- `het/forward_layer.rs`: decode boundary chain (`:638-670`) -> one
  `comp_kv_append_fp8` launch; gather (`:976-985`) -> `indexer_gather_fp8`; dense
  select (`:993-995`) -> `comp_kv_head` with the `n_comp <= 512` assert.
  The `kv_post_fused` graph (`:390-410`) writes the RAW SWA cache and is untouched.
- `het/forward_prefill.rs`: producer (`:1482-1500`) -> `comp_kv_append_fp8_batched`;
  gather (`:1995`) -> `_fp8_batched`; dense reads (`:1766`, `:2032-2034`) ->
  `comp_kv_head` with the same assert. The indexer compressor chain (`:1687`) stays.
- `het/engine.rs:517-522` oracle dump: identity-gather (`indexer_gather_fp8`) the
  live rows into a temporary f16 buffer, dump that. Same kernel as production, so
  the dumped values are what attention sees.
- 19 `.comp_kv` references in Rust (5 forward_layer, 7 forward_prefill, 1 engine,
  6 snapshot); the compiler finds them all once the field becomes an enum.

### Step 4 — snapshots — DONE (b3d6d08; round trip + both conversions verified)

Build on the streamed snapshot code just committed (93256c6: BlobWriter/BlobReader,
stream_u16/load_u16).
- FORMAT_VERSION 3 -> 4. `meta.json` gains per-blob `comp_kv_format`
  (`"f16"` | `"fp8_e4m3_b64"`), `comp_kv_row_bytes`, and the same two fields for
  `index_comp_kv` (lever 3 reuses them; f16 for now).
- Save: ratio-4 layers stream the raw 592-B device rows (`stream_u8`); ratio-128
  layers stream f16 as today. Head shadow is not saved.
- Restore: `load_u8` into the FP8 rows, then rebuild the head shadow with one
  identity-gather launch per ratio-4 layer over `min(n_comp, 512)` rows.
- v3 acceptance in BOTH version checks (`snapshot.rs:307` index loader, `:1051`
  `restore_vl`): convert per D5. Refusal = log + treat as cache miss. A converted
  session is re-saved as v4 by the next normal save; v3 files age out under the
  existing disk-cap LRU.
- `n_kv_max` compatibility (`:1073-1082`) unchanged: row counts, not bytes.

### Step 5 — in-window verification and ship — PENDING (needs the server down)

Each weight load is ~2 min; keep it to two (one oracle process, then the server).
1. `fp8_kv_store_ab_one_load` (forward_prompt_batch_matches_sequential.rs): packed vs
   f16 store, BIT-IDENTICAL logits on prefill + decode at T=3000 (sparse path) and
   T=200 (dense/head-shadow path), with a same-store determinism control. This is
   the format proof at model scale. Then `forward_prefill_all_oracles_one_load`
   (ds4-dump reference, dense path; needs `DEEPSTRIX_GGUF` + the dump for the served
   quant) in the same session if a second load is affordable, else skip: the A/B
   plus the unchanged f16 path's existing validation covers it.
2. Perf, same process where possible (`COMP_KV_FP8=0` flips the store per
   allocation): `bench_decode` at short ctx (dense path: launch-for-launch identical,
   equal within noise) and at 192K `FAKE_POS` (gather expand); `bench_prefill_chunked`
   `PIPELINE_LANES=2` at 4K and 192K. Back-to-back, same binary (bench A/B rule).
3. sysfs VRAM at 192K/K=15: expect −0.42 GiB + 10.5 MiB versus the previous binary.
4. Restart with `~/run_deepstrix.sh --bg`; the on-disk v3 snapshots convert on first
   use (log line `snapshot.restore: converted compressed-KV encoding`); smoke a
   restored session and a fresh one. Spend the freed VRAM per the section below.
Rollback: `COMP_KV_FP8=0` in the run script (no rebuild), or the parent commit.

No runtime toggle (as the plan says): a format that fails an oracle does not ship.
Rollback is the parent commit.

## Effort

Steps 1-4 took one session (2026-09-10, beside the live server). Step 5 is the
60-min window.

## What to spend it on: host RAM (K) or context

KV lives entirely on the dGPU, so the 0.42 GiB is a dGPU lever that can go either to
hot-expert K (host RAM via dedup, 0.348 GiB per K) or to context. The two compete.

Per-token dGPU bytes (n = context tokens), ratio-4 layers x 21, ratio-128 x 20.
Scratch that scales with `ATTN_MIXED_MAX_KEYS` is allocated ONCE (`BatchDgpuShared`,
`DgpuScratch`), not per lane:

| component | today (f16) | after FP8 KV | after FP8 KV + lever 3 |
|---|---|---|---|
| comp_kv ratio-4 (rows/4) | 5376 | 3108 | 3108 |
| comp_kv ratio-128 (rows/128) | 160 | 160 | 160 |
| index comp_kv ratio-4 | 1344 | 1344 | 378 |
| prefill indexer_scores scratch (shared, 512 rows x 4 B / 4) | 512 | 512 | 512 |
| decode attn_scores (N_HEAD x MAX_KEYS f32 / 4) + top-k scratch | ~130 | ~130 | ~130 |
| **total B/token** | **~7460** | **~5190** | **~4220** |

(Check: 6880 B/token of KV x 196608 = 1.26 GiB, the audited KV@192K figure.)

dGPU today at K=15/192K: 15.13 of 15.92 GiB used at idle, 0.79 GiB free. Keeping the
plan's 0.35 GiB margin over the (to-be-measured) peak leaves ~0.45 GiB spendable now:

| configuration | spendable | extra tokens | max ctx (text) |
|---|---|---|---|
| today, no changes | 0.45 GiB | ~65K | ~255K |
| + FP8 KV | 0.87 GiB | ~180K | ~370K |
| + FP8 KV + E2M1 indexer keys | 1.04 GiB | ~265K | ~455K |

The model's RoPE is YaRN factor 16 over a 64K base (`ROPE_ORIG_CTX`,
`DS4_ROPE_SCALE_FACTOR`), i.e. a 1M design ceiling; we are memory-bound, not
model-bound. Raising the context is a separate small change with four knobs:
- `ATTN_MIXED_MAX_KEYS` (49408, `attention.rs:50` AND the `#define` in
  `attention_mixed.hip:27`) = n/4 + 256. This also grows the per-lane
  `indexer_scores` scratch (the 1024 B/token row above).
- With the vision tower loaded, `ATTN_SCORES_STRIDE` (2048) must satisfy
  n/128 + 512 <= stride (`check_vision_ctx_fits`); at 370K that is 3402 -> 3456,
  costing +90 MiB once (shared scratch; ~17K tokens of budget). Text-only servers
  are unaffected.
- `--ctx` in the run script. Snapshots from a smaller `n_kv_max` restore into a
  larger state already. The server now refuses `--ctx > 4 x ATTN_MIXED_MAX_KEYS` at
  startup (the indexer clamp would otherwise silently drop the newest rows).
Cost of using it: sessions that actually reach 370K pay the depth tax (decode ~38
ms/token beyond 128K today, indexer-chain bound; prefill tok/s keeps sliding with n).
Sessions that don't are unaffected.

Recommendation given RAM > decode perf: the plan's K route (+1 K = 0.348 GiB host)
and the context route are both available; decide per window from measured peak VRAM.
