# E2M1 indexer keys — implementation (2026-09-10, rev 2)

Lever 3 of `VRAM_FREE_PLAN_2026-09.md`, built on the packed-FP8 compressed-KV work
(`FP8_KV_IMPL_2026-09.md`, shipped 2026-09-10). Priority order: quality > RAM >
decode perf > prefill perf > implementation complexity.

Rev 2 = rev 1 after the architect review and the build (steps 1-4 committed and
verified beside the live server; step 5 needs the window). Review corrections:
the expansion is a byte-permute fast path plus a branch-free general path (D2
rewritten); the decode twins load all nine words before expanding (the range
branch inside the expansion otherwise serialised one memory round trip per
k-tile); realistic perf expectations (below); stale-format snapshot dirs are
deleted at index load, not skipped.

## What ships

The 21 ratio-4 indexer compressors store each 128-dim key row as 64 B of E2M1 nibbles
+ 4 block exponents instead of 256 B of f16. Bit-identical: `indexer_qat`
(kernels/indexer_qat.hip:75-101) already snaps every value to `±m x 2^e` with
`m ∈ {0, 0.5, 1, 1.5, 2, 3, 4, 6}` and one power-of-two `e` per 32-element block, and
the f16 cache holds `half_rn(±m x 2^e)`; storing `(sign, m-index, e)` and recomputing
that expression on read reproduces the same bits.

Row format (`E2M1_KEY_ROW_BYTES` = 80, 16-B aligned so every staging load is an
aligned b128): `[0,64)` nibbles, element `i` in byte `i/2`, low nibble for even `i`,
nibble = `sign<<3 | idx` (the standard FP4 E2M1 bit layout, idx order = the
`e2m1fn_value` table); `[64,68)` i8 exponents for blocks 0..3; `[68,80)` zero pad.
21 layers x 76800 rows x (256 - 80) B = 0.28 GiB at 300K (0.17 at 192K); per-token
dGPU cost of context drops 1344 -> 420 B.

## Design choices

### D1. Producer re-derives (code, e) from the QAT'd row; `indexer_qat` is untouched

The plan fused the pack into `indexer_qat`. Instead a new `index_kv_append_e2m1`
(+ `_batched`) takes the QAT'd f32 row (exactly `±m x 2^e0`) and replaces
`f16_roundtrip -> comp_kv_append` (3 launches -> 2 per boundary). Per 32-block it
recomputes the exponent with the SAME expression as indexer_qat.hip:89-97 (frexpf
form, the 6 x 2^-126 amax floor) from the QAT'd values: the block max is
`{3, 4, 6} x 2^e0`, so the recovered `e'` is `e0` (max 4 or 6) or `e0 - 1` (max 3,
codes doubled, all still `<= 6`), and `v / 2^e'` is exactly a table value: the code
is an exact match, never a rounding. Expansion `half_rn(m x 2^e')` equals the old
`half_rn(v)` because both are the same f32. This keeps the ds4-faithful QAT kernel
byte-for-byte and needs no new numerics on the producer side.

**Sign of zero (again).** The QAT'd row holds `-0.0` for a negative value that
snapped to code 0 (`sign * e2m1fn_value(0)`, kernel line 46), and the old chain stores
it as `0x8000` (memory round trip, no contraction). The packer takes the sign from
the f32 SIGN BIT (`__float_as_uint(v) >> 31`), not from `v < 0`, and the expand
applies it to the f16 bits, not the product — the FP8 lesson.

### D2. Consumers expand at their existing load points; no attention-kernel change

- `indexer_score_wmma_mw` (decode, B=1) and `indexer_score_wmma_batched_mw`: each
  lane loads its comp row's 8 keys for a k-tile as one b128 from global. Packed: one
  b32 of 8 nibbles (byte offset `row*80 + (kbase + k_off)/2`, 4-B aligned) plus the
  block's exponent byte (`row*80 + 64 + kbase/32`), expanded to `half8` in registers.
  Bytes per row per lane drop 256 -> 68.
- `indexer_score_wmma_gemm` (prefill): the staging role (row `tid>>2`, quarter
  `tid&3` = 32 halves) becomes one aligned b128 of 32 nibbles + one exponent byte,
  expanded to 4 x b128 at publish into the same LDS tile (stride 136 halves); the
  register prefetch shrinks from 4 x uint4 to 1 x uint4 + 1 u32. WMMA inputs are
  the identical f16, so scores are bit-identical to the f16 kernel (the existing
  oracle applies unchanged).
- `indexer_score` (naive, the gfx1151 ds4-dump reference): reads
  `expand(nibble, e)` instead of `kv[i]`.
- `indexer_score_wmma` / `_batched` (the 1-wave `sw` entry points): hard-error when
  the store is packed (`INDEXER_DECODE=sw`, `INDEXER_SCORE_VARIANT=sw`).
- Expansion arithmetic, FAST PATH (`e ∈ [-13, 13]`, i.e. every result is a normal
  f16 — every real block): the f16 of `m x 2^e` has a zero low byte and a high byte
  `hb[m] + 4e` (`hb = {0, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46}`), so eight
  nibbles expand with two `v_perm_b32` table picks from a two-register table
  pre-biased by the block exponent, two ORs for the signs and four interleaving
  perms: ~12 VALU ops, no float math. GENERAL PATH otherwise: `(float)(2m) x 0.5f
  x 2^e` (the `2m` byte also via `v_perm`, so no branch tree) then `__float2half_rn`,
  sign on the bits. Both proven against the host expand over the whole i8 field and
  against each other for 1024 words x 256 exponents (`e2m1_key_expand8_check`).
  The first (float, per-nibble) version made the decode kernel 2x SLOWER.
- Decode twins load all eight words + the four exponent bytes of a lane's row
  before any expansion (`e2m1_key_load_row_words`), matching the f16 kernel's
  "eight loads, one wait" shape; the data-dependent range branch would otherwise
  split the basic block and serialise the loads (review finding, ISA-verified).
- GEMM twin: expansion at publish (~48 VALU per thread per 64-row tile). Expanding
  in the shadow of the previous tile's WMMAs was tried and is worse (+12% vs +7.4%):
  it forces an early wait on the just-issued prefetch.

### D3. Store enum, not a toggle in the kernels

`CompKvStore` gains `E2m1(DeviceBuffer<u8>)`; the indexer compressor allocates it
(`INDEXER_KEYS_E2M1=0` keeps f16 — rollback / one-load A/B). Every consumer
dispatches on the variant, so no unconverted variant can read the packed bytes as
f16.

### D4. Snapshots: v5, no cross-format conversion (owner decision 2026-09-10)

`FORMAT_VERSION` 4 -> 5 with `index_comp_kv_format` / `index_comp_kv_row_bytes`
alongside the comp_kv fields; a snapshot whose index encoding differs from the live
store is refused (evicted, full prefill). Old cache entries are cache entries: the
v3->v4 f16->FP8 conversion of the compressed KV is DELETED in the same commit and
`MIN_FORMAT_VERSION` = 5, so the restore path has exactly one branch per blob
(same encoding, byte copy). Cost: the existing ~62 GB of v3/v4 snapshots cold-prefill
once after the restart.

## Build order

1. `kernels/e2m1_key_common.inc` (table, nibble decode, exponent expression VERBATIM,
   expand) + `kernels/index_kv_e2m1.hip`: `index_kv_append_e2m1{,_batched}`,
   `index_kv_e2m1_expand` (rows -> f16, for the oracle dump and tests), test kernels.
   `tests/e2m1_keys_format.rs` (both GPUs, no model): table check; host expand ==
   device expand over the whole i8 exponent field; every (code, e) for e in
   [-30, 20] through the REAL old chain (`indexer_qat -> f16_roundtrip ->
   comp_kv_append`) vs the new chain — rows built by the inverse Hadamard so the
   QAT lands on the intended code with a `6 x 2^e` anchor per block; random-row
   chain oracle (tiny negatives, zero blocks, floor blocks, large values).
2. Packed score kernels in `indexer_score_wmma.hip` / `indexer_score.hip`
   (`*_e2m1` entry points sharing the epilogues). `tests/indexer_score_e2m1.rs`:
   random Q/K, packed vs f16 for mw, batched_mw, gemm, naive — bit-identical scores
   including the -inf tails; `bench_indexer_score_isolated` gains the packed variants.
   Perf gate at kernel level: gemm at B=512, n=76800 within noise of f16; mw B=1
   should be FASTER (bytes).
3. Plumbing: `CompKvStore::E2m1`, indexer compressor alloc, decode chain
   (forward_layer.rs:790-806 + score dispatch :924-960), prefill chain (:1687 +
   :1925-1960), oracle dump. Startup hard-error for the `sw` env values with a
   packed store.
4. Snapshot v5 (D4); `snapshot_fp8_roundtrip.rs` becomes the v5 round trip for both
   blobs; delete the conversion code.
5. Window: `fp8_kv_store_ab_one_load` extended with the key-store toggle
   (bit-identical logits across all four store combinations); `bench_decode`
   `FAKE_POS=300000`; `bench_prefill` at 4K/300K, back-to-back key-store A/B in
   process; VRAM; restart with the snapshot dir wiped.

## Measured (kernel level, n=76800 rows = 300K, beside the live server; bit-identical
scores on 39.8M values per batched kernel, every run)

| kernel | f16 | packed | change |
|---|---|---|---|
| decode score, B=1 (`_mw`) | ~74 µs | ~74 µs | 0% (latency-bound: 75 WGs on 64 CUs; bytes are not the limiter) |
| batched multi-wave, B=512 | 22.7 ms | 22.3 ms | -2% |
| GEMM prefill, B=512 | 5.01 ms | 5.38 ms | +7.4% (~0.5% of prefill at 300K, less at shorter contexts) |

Expectations for the window: decode ms/token unchanged (a decode win from this
kernel needs a bigger grid, e.g. NT_PER_WG 8 -> 2, or n-tile software pipelining —
a follow-on that also helps the f16 kernel); prefill @300K within ~0.5% of today;
dGPU -0.28 GiB at 300K (index key cache 0.39 -> 0.12 GiB). RAM is the lever.

## Step 5 — window (pending)

1. `fp8_kv_store_ab_one_load` now covers all four (main, keys) store combinations
   with a same-config control: bit-identical logits required, T=3000 and T=200.
2. `bench_decode COMP_KV_FP8_SWEEP=packed,keys16 FAKE_POS=300000`;
   `bench_prefill COMP_KV_FP8_AB=packed,keys16,packed FAKE_PREFILL_POS=4096,300000`.
3. sysfs VRAM at 300K/K=15: expect -0.28 GiB.
4. Restart. The index loader deletes every pre-v5 snapshot dir (cache entries;
   ~62 GB); sessions cold-prefill once. Rollback: `INDEXER_KEYS_E2M1=0`
   (invalidates v5 files on first touch — they are refused and evicted).
