# V4.1 sparse indexer port — plan

> **STATUS 2026-09-14: DO NOT START THIS AS A PREFILL PROJECT.**
>
> The premise below ("this gates the 1000 tok/s prefill goal") was WRONG. It was
> derived from a *decode* ablation slope and a row-count table, never from a
> prefill trace. **Measured** (CTX=40960, 32,545-token prompt, TOKEN_PROFILE,
> HTTP 200, 0 errors):
>
>     prefill @32K = 586 tok/s   (the 6.0k probe reads 255 — fixed costs dominate there)
>     attention    = attn_smwsum 13,827 us + attn_score 3,811 us = 17.6 ms
>                  = ~7.0% of ~252 ms of stage time
>     igpu.routed_moe = 140,811 us = 56%  <-- the actual prefill bottleneck
>
> Removing attention ENTIRELY would take prefill 586 -> ~630. Prefill's path to
> 1000 runs through the MoE leg (expert paging + kwide kernels), not the indexer.
>
> The indexer is still needed, but for DECODE at long context (dense scoring grows
> linearly; sparse is flat from ~1K) and for VRAM (the `indexer_gathers` flip alone
> reclaims the multi-GB dense scores scratch, and is testable by allocation delta
> without running the indexer). Re-scope it as that, much smaller, project.
>
> **Also: the bit-exactness oracle proposed below is a TAUTOLOGY** — the gate is
> `n_index_comp > INDEXER_TOP_K`, so at <=512 rows the sparse code does not execute
> and you would compare the dense path to itself. It would pass with garbage
> weights. Use the SELECTION-SET oracle instead: D2H the 512 indices and compare as
> a set against a CPU recompute (`tests/indexer_score.rs:88` has
> `cpu_indexer_score`). Exact at ANY context.
>
> Further corrections from review (all verified against the code):
> * `indexer_qat.hip` applies a **Hadamard128** rotation before FP4; V4.1's indexer
>   does **not** rotate (`model.py:535-537`). Using it = plausible output, wrong
>   selection, silent. Needs a Hadamard-free twin.
> * V4.1 has **no indexer compressor**: index K is `k_norm(wk(latent))` on the
>   PRE-RoPE latent. Allocating `HetCompressorState` is the wrong structure — put
>   the K cache inside the kv-source's state so `with_kv_source` carries it to
>   reuse layers for free.
> * `hf_v41.rs` also maps `indexer.k_norm.weight` (:381-385), omitted below.
> * `sd.indexer_selected` is in lane-**shared** scratch (`batch_scratch.rs:348-362`)
>   — a reuse layer would read the other lane's selection.
> * E2M1 index-K row stride is **80 B**, not 68 (`index_kv_e2m1.rs:30`).
> * Decode hard-codes `attn_n_comp = INDEXER_TOP_K` (`forward_layer.rs:1367`), so at
>   n < 512 it would score stale rows — a real bug to fix before any test.
> * §1.5 candidate pool is **inert below 16K** (`min(2048, num_blocks)`), so the
>   existing 3,072-token CPU oracle cannot observe it at all.
> * The 838 MB/token figure is 3.5x low: the engine's V4.1 store is F16 (1,024
>   B/row) => ~2.9 GB/token at 100K.


## Why

`forward_layer.rs:1193` gates the sparse path on `ratio == 4`. V4.1's compress
ratios are 1 and 2, so the gate NEVER fires and every V4.1 layer >= 2 scores
**densely over the entire compressed store**.

Cost (docs/v41/INDEXER_DENSE_VS_SPARSE.md, arithmetic from inference/config.json):

| context | compressed rows read/token, sparse | dense | ratio | KV bytes/token |
|---|---|---|---|---|
| 4,096 | 19,456 | 118,784 | 6.1x | 8.6 -> 37.2 MB |
| 32,768 | 19,456 | 950,272 | 48.8x | 8.6 -> 276.6 MB |
| 100,000 | 19,456 | 2,900,000 | **149x** | 8.6 -> **838 MB** |

The sparse path is FLAT in context from ~1K onwards (512 rows/layer + a 128-row
window, forever) — that flatness is the entire point of CSA2. Dense throws it away
and goes linear. At 100K a decode step would stream 838 MB/token = ~1.3 s at the
dGPU's 640 GB/s, i.e. ~0.8 tok/s. It also costs decode +18 ms/token at 32K.

**This gates the 1000 tok/s prefill goal**, which is a 100K-context statement and
cannot even be measured until dense scoring is gone.

## SELECTION-SET ORACLE: BUILT AND PASSING (2026-09-14)

`crates/v4flash-kernels/tests/v41_indexer_selection_oracle.rs` (`--ignored`). Drives the
REAL packed-E2M1 key chain and the REAL score + top-512 kernels at V4.1 shapes
(n_head=32, dim=128, n_comp=4096, top_k=512) against a CPU recompute:

    score: max_abs_diff=1.526e-5 over magnitude 6.536e1   (rel ~2.3e-7)
    selection: |cpu\gpu| = 0 of 512, hard misses outside the error band = 0
    PASS

Design notes that matter:
  * the CPU reference scores the EXPANDED keys (`index_kv_e2m1_expand`), i.e. exactly
    the quantized values the GPU reads, so E2M1 error cannot masquerade as a kernel bug;
  * selection is compared as a SET with a tolerance band around the cut, because the
    WMMA score path's ~4.2e-4 absolute error legitimately swaps rows at the boundary.
    Rows OUTSIDE the band must match exactly — that is the assertion. (Here nothing
    swapped at all.)

**This replaces the "bit-identical at 512 tokens" proof, which is a TAUTOLOGY** — the
gate cannot fire at <=512 rows, so it compares the dense path to itself.

**The gate for S1 is therefore GREEN.** What remains is the four-source rewire below.

## S1b/S1c WIRING MAP (the scoring path already exists; do not rewrite it)

`forward_layer.rs:1274+` already implements the whole chain for V4-Flash:
`matvec(attn_q_b) -> RoPE -> matvec(proj) -> scale -> IndexerScore -> IndexerTopk ->
IndexerGather`, producing a dense `active_comp_kv` of <=512 rows that the attention
kernels consume instead of the full `cs.comp_kv`. `dgpu_scratch.indexer_q` is allocated.

To reach it under v41, REWIRE these four sources — the block is otherwise shape-generic:

  1. `n_index_comp`: currently `ls.indexer_compressor.n_comp`. V4.1 has NO indexer
     compressor — use `ls.compressor.n_index_comp`, which S1a now maintains.
  2. Index K: currently the indexer compressor's `comp_kv`. V4.1 -> `cs.index_k`,
     PACKED E2M1 + E8M0/32, 80 B/row. The `*_e2m1` kernel variants
     (`indexer.rs:146/293/358/442`) already read that format.
  3. **SKIP `de.indexer_qat.launch` (`:1201`)** — it applies a Hadamard128 rotation
     that V4.1's indexer does NOT (`model.py:535-537`). Using it gives plausible output
     with a WRONG selection, silently. This is the single highest-risk line in the port.
  4. Gate: `ratio == 4 && n_index_comp > INDEXER_TOP_K` -> for v41, `is_index_source(layer)`
     (layers 2,8,14,20,24,28,32,36) with the same `> INDEXER_TOP_K` condition, and reuse
     layers consuming the shared selection (that is S2).

Shapes are already right under v41: `N_INDEXER_HEAD`=32, `N_INDEXER_HEAD_DIM`=128,
`INDEXER_TOP_K`=512.

Order: build the SELECTION-SET oracle first (see the WMMA-error constraint below), then
flip. Do not flip and eyeball the text — a wrong selection reads as fluent.

## Existing kernels RE-VALIDATED at V4.1 shapes (2026-09-14, measured)

Both ran clean on today's tree, so S1 is mostly WIRING, not new kernels:

* **`indexer_topk_select_oracle`** (`--ignored`): 0 mismatched tokens, 0 fallbacks, at
  n_idx_max 24,576 and 49,152 with `top_k=512` — i.e. V4.1 scale. Fast too: 0.598 ms at
  96K and 0.743 ms at 192K for B=512 (4.9x / 8.3x over the chain). **The top-k half of
  S1 is de-risked.**
* **`indexer_score`** (`--ignored`) vs `cpu_indexer_score`: scalar path max_abs_diff
  ~3e-7. WMMA path max_abs_diff **4.22e-4** at n_comp=16,384 (magnitude ~1.9).

**Design constraint found by that second number**: top-512 out of ~103,845 is a RANKING
decision, and 4.2e-4 of absolute score error will flip rows across the selection cut.
So the S1 selection-set oracle must either (a) compare against the SCALAR score path,
or (b) allow boundary disagreement and assert only on rows whose score gap exceeds the
kernel's error bound. A strict set-equality assert against the WMMA path WILL flake.

Also note `cpu_indexer_score` (`tests/indexer_score.rs:88`) already implements V4.1
semantics exactly — per-head dot, ReLU, then head-weighted sum — and is generic in
(n_head, head_dim), so it works at 32x128 unchanged. The oracle's core already exists.

## What already exists (verified, not assumed)

* **Kernels** — `indexer_score.hip`, `indexer_score_wmma.hip`, `indexer_topk.hip`,
  `indexer_topk_bitonic.hip`, `indexer_gather.hip`, `indexer_bitpack.hip`,
  `indexer_qat.hip`, `index_kv_e2m1.hip`, plus `src/indexer.rs`,
  `src/index_kv_e2m1.rs`. Built for V4-Flash; unreachable under v41.
* **HF tensors** — `hf_v41.rs:370-379` already maps
  `attn.indexer.{wq_b,weights_proj,wk}.weight` to
  `indexer.{attn_q_b,proj,attn_k}.weight`.
* **Constants** — v41 arm is correct: `N_INDEXER_HEAD=32` (V4-Flash 64),
  `N_INDEXER_HEAD_DIM=128`, `INDEXER_TOP_K=512`, `N_LORA_Q=1280`.
* **A CPU oracle for quality** — `scripts/v41_oracle/dense_index.py` already A/Bs
  dense vs sparse against DeepSeek's unmodified `model.py`.

## What is missing

1. `het::weights` loads `indexer` / `indexer_compressor` **only at ratio 4**, so
   `dlw.indexer` is None under v41.
2. `indexer_compressor` **state** is not allocated under v41 (`state.rs`);
   `n_index_comp` is 0, which is the second reason the gate cannot fire.
3. **Shared selection** — V4.1 has 8 index-source layers (2, 8, 14, 20, 24, 28,
   32, 36); every other layer reuses `shared.topk_idxs` from the most recent
   source. No such plumbing exists.
4. **§1.5 hierarchical candidate pool** — built at layer 20, consumed by 24/28/
   32/36. Block score = max over 8 consecutive positions, block holding the
   query's newest position pinned (+inf), top-2048 blocks, mask = union of
   selected blocks (16,384 positions). Entirely new.
5. Index K lives only on the 4 **kv_source** layers (2, 8, 14 at ratio 2; 20 at
   ratio 1), stored E2M1 + E8M0 per 32 = 68 B/entry.

## The free correctness oracle

`topk = min(index_topk, store_rows)` and unreachable rows score `-inf`, so when a
query's reachable set is <= 512 rows the top-512 selects **all of them** and
**dense === sparse, bit for bit**. That holds for context <= 512 (ratio-1 layers)
and <= 1024 (ratio-2 layers).

So the port has a built-in exactness test: flip the gate, run at <= 512 tokens,
and the logits must be BIT-IDENTICAL to today's dense path. Only past that
threshold do the paths legitimately diverge. Every V4.1 parity run to date was at
6 and 129 tokens — inside this regime — which is exactly why the missing indexer
survived validation.

## Staging

> **S0 WEIGHTS LANDED + VALIDATED 2026-09-14.** `het/weights.rs`: `IndexerWeights`
> gained `attn_k` / `k_norm` (both `None` on V4-Flash), plus `is_index_source()` and
> `owns_index_k()`. Proof: server starts, decode output BIT-IDENTICAL to pre-S0
> (sha b7f58f53b529), weights load in 10.0 s.
>
> **Plan correction found by implementing it:** all 8 index-source layers SCORE, but
> index K exists only on the 4 KV-SOURCE layers (2, 8, 14, 20) — loading it on all 8
> fails loudly with `blk.24.indexer.attn_k.weight not found in GGUF`. The other four
> reuse the nearest source's keys, mirroring the main compressed store's reuse. The
> plan said this in "What is missing #5"; the staging section did not.
>
> STILL TO DO for S0: allocate the index-K CACHE on the 4 kv-source layers (it belongs
> in the kv-source's state so `with_kv_source` carries it to reuse layers for free —
> do NOT allocate a `HetCompressorState`, V4.1 has no indexer compressor).

**S0 — weights + state, no behaviour change.** Load indexer tensors for the 8
index-source layers and allocate the index-K cache on the 4 kv-source layers.
Gate still false; nothing calls them. Proof: server starts, VRAM/GTT delta matches
68 B/entry x rows, all existing outputs bit-identical.

> **S1a LANDED (decode path) + VALIDATED 2026-09-14.** `V41_INDEX_K=1`, default OFF,
> inert (nothing reads `index_k` until the sparse gate flips).
> `forward_layer.rs`: compute between the compressor's `rms_w` and its `rope`
> (`wk(latent)` via `f16.matvec` -> `k_norm` via `rms_w.launch_weighted` -> RoPE on the
> tail N_ROT at `comp_pos = pos+1-ratio`), STORE beside the `comp_kv` append via
> `q8k.launch_cast_f16` into `cs.index_k` at row `n_index_comp`.
> Proof: decode sha b7f58f53b529 (BIT-IDENTICAL), and both stages report **calls=4** —
> exactly the 4 KV-source layers, not all 8 index-source layers. Cost 104 us/token
> (72 compute + 32 store).
>
> **PREFILL twin LANDED + VALIDATED too** (`forward_prefill.rs`, batched):
> `f16.gemm_batched_wmma` (128x512, batch=n_boundaries) -> `rms_w.launch_weighted_batched`
> -> `rope.launch_forward_batched` at the same `comp_pos_per_boundary` -> reuse
> `comp_kv_append.launch_batched` (f32 rows -> f16 store) at 128 wide into `index_k`.
> Proof: 32K prefill sha b1854a8ad2f1 BIT-IDENTICAL both with and without
> `DEEPSTRIX_PREFILL_PROFILE=1`; `k.comp_b.index_k` = 12.2 ms over **calls=248**
> (4 KV-source layers x ~62 chunks). Cost is negligible: 12 ms across a 32K prefill.
>
> **FP4 DONE too**: index K is stored PACKED E2M1 + one E8M0 per 32 via the existing
> `index_kv_e2m1.launch_append{,_batched}` (`E2M1_KEY_ROW_BYTES` = 80, `E2M1_KEY_DIM` =
> 128) — exactly the reference's `fp4_act_quant(k, 32, True)`, and NOT the compressed-KV
> 16-block E4M3 path. Cache is ~26 MB at --ctx 130688 (80 B/row vs 256 for f16).
> Re-validated: 32K sha b1854a8ad2f1 BIT-IDENTICAL, `k.comp_b.index_k` 12.3 ms calls=248.
> **`HetCompressorState::index_k` is `DeviceBuffer<u8>`, packed — not f16.**
>
> STILL TO DO for S1: (b) the indexer Q path (`wq_b` -> RoPE -> fp4) + score + top-512 + the causal
> block mask; (c) the SELECTION-SET oracle; (d) then flip the gate.
>
> **S1 SPEC, transcribed from the reference** `/persist/lumi/models/dsv4.1f-full/inference/model.py`
> (`Indexer.forward`, :528-565). Implement against THIS, not from memory — three of these
> would be silent wrong-selection bugs:
>
> 1. **Index K (owners only: layers 2, 8, 14, 20)**
>    `k = k_norm(wk(latent))` — `wk: head_dim 512 -> index_head_dim 128`, RMSNorm eps = norm_eps.
>    `latent` is this layer's RoPE-FREE compressed latent, and this must run BEFORE Attention
>    overwrites that storage with the RoPE'd quantized values (:749).
> 2. **RoPE IS applied to the index key** — `apply_rotary_emb(k[..., -64:], freqs)` over the
>    last `rope_head_dim`=64 dims only. "Pre-RoPE" refers to the INPUT latent, not the key.
>    Group j takes position `j * ratio`: at start_pos==0 the freqs are
>    `freqs_cis[: seqlen - seqlen % ratio : ratio]`; on a decode step, `freqs_cis[start_pos+1-ratio]`.
> 3. **FP4 quantization, NO Hadamard** — `fp4_act_quant(k, fp4_block_size, True)` and the same
>    on q. `indexer_qat.hip` applies a Hadamard128 rotation and is therefore the WRONG kernel:
>    it yields plausible output with a wrong selection, silently.
> 4. **Query** — `q = wq_b(qr)` unflattened to `[n_heads=32, 128]`, RoPE on its last 64 dims over
>    `freqs_cis[start_pos:end_pos]`, then fp4 quant.
> 5. **Score** — `weights = weights_proj(x) * (128**-0.5 * 32**-0.5)`;
>    `score = einsum("bshd,btd->bsht", q, index_k)` (K is shared across the 32 heads, MLA-style);
>    then `score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)`. **ReLU BEFORE the
>    head-weighted sum** — not after, and not softmax.
> 6. **Causal mask** — a compressed block is visible only once the query has passed its LAST
>    token: mask `arange(seqlen//ratio) >= (arange(1, seqlen+1) // ratio)` to `-inf`.
>
> **Exact hook point in OUR code** (`model.py:_compress_kv`, :740-758 vs
> `forward_layer.rs`): the reference order is
> `latent = compressor(x)` -> INDEXER READS LATENT -> `apply_rotary_emb(latent[...,-64:])`
> -> `fp4_act_quant(latent, 16, e4m3)` -> write compress_kv_cache.
> Our decode compressor is `pool` (:918) -> `rms_w` (:929) -> `rope` (:941) ->
> `fp4kv` (:956) -> append, and `pool+rms_w` is what corresponds to the reference's
> `compressor(x)` output. **So index K hooks in between `rms_w` and `rope`** — after the
> compressor norm, before the main path rotates and quantizes that same storage.
>
> The two quantizations are NOT the same and must not share a kernel: compressed KV uses
> groups of **16 with E4M3** scales; the indexer uses groups of **32 with E8M0**.
>
> **Kernel APIs for S1a, all already on the dGPU engine** (decode fires one latent row
> per group, so this is a MATVEC chain, not a GEMM):
>   1. `de.f16.matvec(&de.compute, out, wk_buf, &dgpu_scratch.comp_row, 128, 512)`
>      (`f16.rs:412`) — `comp_row` IS the latent after `rms_w`.
>   2. `de.rms_w.launch_weighted(.., out, x, k_norm_buf, 128, RMS_EPS)` (`rms_norm.rs:52`).
>   3. `de.rope.launch_forward(.., x, n_head=1, head_dim=128, n_rot=N_ROT, comp_pos, &dlw.rope_params)`
>      (`rope.rs:77`). Use the SAME `comp_pos = pos + 1 - ratio` the compressor's own rope
>      uses at `forward_layer.rs:941` — the reference shares `freqs_cis` with Attention
>      (`model.py:733-734`), so `dlw.rope_params` is correct.
>   4. FP4 pack, 32-block E8M0 — NOT `de.fp4kv` (that is 16-block E4M3 for compressed KV).
>   5. Append into `compressor.index_k` at row `n_index_comp`.
> NEW scratch needed: two 128-wide f32 rows in `DgpuScratch` (wk output, normed output).
>
> S1a (index-K write) is then a 5-step chain on KV-source layers only:
>   1. GEMM  latent[rows,512] x wk -> k[rows,128]
>   2. RMSNorm(k) with `k_norm`, eps = norm_eps
>   3. RoPE on k[..., -64:] at GROUP positions (group j -> position j*ratio)
>   4. fp4 quant, 32-block E8M0 (NOT `indexer_qat.hip` — that one Hadamards)
>   5. write at row `start_pos / ratio` into `compressor.index_k`, bump `n_index_comp`
>
> Proof for S1 is the SELECTION-SET oracle (D2H the 512 indices, compare as a set against a CPU
> recompute), NOT "bit-identical at 512 tokens" — see the tautology note in the header.

**S1 — flip the gate, single source, no sharing, no candidates.** Make
`use_sparse` fire for v41 on an index-source layer, with reuse layers still dense.
Proof: **bit-identical at 512 tokens**; divergence only above it.

**S2 — shared selection.** Reuse layers consume the most recent source's
`topk_idxs`. Proof: still bit-identical at 512 tokens; measure the 32K decode
delta (expect the +18 ms/token to go away).

**S3 — §1.5 candidate pool.** Layer 20 builds it; 24/28/32/36 mask against it.
Proof: CPU oracle only (no bit-exact test exists above 512 tokens) — use
`dense_index.py`'s teacher-forced NLL and attention-mass probes, with the
duplicate-row control, since two equally-correct runs of this model diverge.

**S4 — measure the goal.** Prefill at >= 32K, then 100K. Only here does the
1000 tok/s target become measurable at all.

## Risks

* S3 has no bit-exact oracle. The control row in `dense_index.py` is mandatory:
  torch CPU GEMM is not row-position deterministic (~2e-6 relative), bf16 turns
  that into 1-ULP flips, and by layer 3 it flips router top-k decisions. "Top-1
  agreement" without the control is meaningless.
* Index K on kv-source layers only means the cache is shared across layers that
  do not own it — a lifetime/ownership question the current state model does not
  express.
* `ATTN_MIXED_MAX_KEYS` currently clamps `n_index_comp`; check it does not
  silently truncate the store at long context.
* At 100K the dense **scores scratch** is ~3.2 GB/lane; if S1-S3 leave any dense
  path reachable at long context it will OOM rather than merely be slow.

## BLOCKER FOUND 2026-09-14: the WMMA score kernels are 64-head (V4-Flash) only

`N_INDEXER_HEAD` is **64** on V4-Flash and **32** on V4.1. The indexer score variants split:

| variant | takes `n_head`? | V4.1-usable |
|---|---|---|
| `IndexerScore::launch_e2m1` (scalar, `indexer.rs:146`) | YES `(n_head, head_dim)` | **yes** |
| `IndexerScoreWmma::launch_mw_e2m1` | no | no — hard-coded 64 |
| `launch_batched_mw_e2m1` / `launch_batched_gemm_e2m1` | no | no — hard-coded 64 |

The size assert is literal: `q16.len() < batch * 64 * 128` (`indexer.rs:437`).

Consequences:
* DECODE preferred WMMA whenever `indexer_score_wmma` existed, so it fed 32-head Q to a
  kernel reading 64 heads' worth. Now forced to `launch_e2m1` (correctness first; a
  32-head WMMA twin is the perf follow-up). This is a strong candidate for the
  unexplained 0.433 logit residual.
* PREFILL *requires* WMMA (`forward_prefill.rs`: "batched prefill indexer requires gfx12
  IndexerScoreWmma") and fails outright: `indexer gemm e2m1: q16 too small for batch=512`.
  **Prefill needs either a batched SCALAR e2m1 path or a 32-head WMMA twin before the
  prefill indexer can run at all.** This is the remaining S1 work for long-context prefill.
* The passing synthetic oracle (`tests/v41_indexer_selection_oracle.rs`) used the SCALAR
  kernel at 32x128 — which is why it was green while the wired paths were not.

Also fixed en route: the three per-token indexer scratch buffers were gated on the static
`indexer_ever_fires()` (false for v41), so the gather failed with `active_comp_kv_b has 1
f16, need 4718592`. New `attention::indexer_scratch_needed()` ORs in `V41_INDEX_K=1`.
NOT applied to `attn_max_scored_keys` — only 8 of 40 layers gather under V4.1, so the
scores-scratch and `--ctx` caps must stay dense-sized until S2.

## S1 FUNCTIONAL ON PREFILL 2026-09-14 — measured +7.4% at 32K

After the 32-head fixes, the prefill sparse indexer runs end to end:

    32K prefill, dense  : 271 tok/s  "...the IQ2_XXS/Q2_K MoE expert GEMV/"
    32K prefill, sparse : 291 tok/s  "...the IQ2_XXS/Q2_K routed-expert"   (+7.4%)

Both coherent and semantically equivalent; different shas are EXPECTED (sparse attends
512 selected rows, dense attends the whole store).

+7.4% is consistent with what is wired: only the **8 index-source layers** gather; the
32 reuse layers still score densely. Attention is ~30% of prefill at 32K, so 8/40 of it
is ~6%. **S2 (shared selection — reuse layers consume the most recent source's
`topk_idxs`) is what turns this into the real win**, and it matters more at 100K where
attention is ~59% of prefill.

Enable with `V41_INDEX_K=1`. Still default OFF.

SEPARATE REGRESSION TO CHASE: the 32K prefill baseline moved 393 -> 271 tok/s between the
window sweep and this run. Both arms above share a config so the +7.4% is sound, but
something cost prefill ~30% — prime suspect is box 2's decoder regions being
frequency-re-ranked (the CED replay runs decoder layers through box 2).

## S2 (shared selection) — DECODE path landed 2026-09-14

`HeterogeneousEngine::last_idx_gather_{src,rows}` (atomics, reset per token). An index
source publishes its gathered rows and the store group it belongs to
(`kv_source_of(layer).unwrap_or(layer)`); a non-source V4.1 layer whose store group
matches reuses `active_comp_kv` as-is — no rescore, no regather. Guarding on the store
group is what stops a selection leaking across a kv-source boundary (layer 8 opens a new
store, so layer 2's selection must not carry into it).

Validated:
* 37-token prompt (n_comp ~293 < INDEXER_TOP_K): output BIT-IDENTICAL to dense and
  `dgpu.attn_compute` 323 vs 324 us — correctly INERT below the gate.
* 8K prompt (n_comp ~7373 > 512): selection differs (sha 86a69797a478 -> 59bd49b75e8d),
  both outputs coherent. `dgpu.attn_compute` 325 vs 326 us — NO WIN, and that is expected:
  at 8K the compressed store is only ~7.5 MB/layer, so 7373 rows vs 512 is 7.5 MB vs
  0.5 MB and neither is expensive. **The indexer pays at 100K**, where the store is
  ~102 MB/layer and 40 layers move ~4 GB/token.

**STILL TO DO: S2 on the PREFILL path.** `forward_prefill.rs` has S1 only (the 8 index
source layers gather; the other 32 still score densely), which is why prefill measured
+10.8% and not more. Prefill's `sd.attn_active_comp_kv` is SHARED between the two
pipelined lanes, so the decode trick (engine-level atomics) is NOT directly safe there —
the cached selection must be keyed per lane, or held in the per-lane scratch.

### S2-on-prefill: concrete design (not yet implemented)

The decode approach (engine-level atomics + reuse `active_comp_kv` as-is) is NOT safe here.
`forward_prompt_batch_v2_pipelined_range` splits ONE chunk into lane A (rows `[0, b_a)`)
and lane B (the rest) and runs both at the SAME layer, lane A's pre-MoE before lane B's
(the KV ordering note at `forward_prefill.rs:425-429`). `sd` is one shared instance, so
`sd.attn_active_comp_kv` holds lane A's gather and is immediately overwritten by lane B's.
The two lanes are different TOKENS, so their selections legitimately differ.

Do NOT fix this by duplicating `attn_active_comp_kv` (268 MB at lane_rows=512, x2, and it
already competes with the 4.3 GB attention scratch at 130K). Instead save the INDICES,
which are tiny, and re-run only the gather:

1. Add `sd.indexer_sel_saved: DeviceBuffer<i32>` sized `2 * b * INDEXER_TOP_K`
   (2 x 512 x 512 x 4 = 2 MB) plus, per lane, `saved_store: i32` and
   `saved_n_sparse: DeviceBuffer<i32>` (the per-token `min(n_comp, TOP_K)` counts, b i32).
2. Thread a `lane: usize` (0/1) into the batched entry point — the callers already know it
   (`bd_a`/`bd_b` at `:476`); it is a parameter add, not new state.
3. At an INDEX-SOURCE layer, after `indexer_topk_bitonic.launch_batched`, device-to-device
   copy `sd.indexer_selected -> sd.indexer_sel_saved[lane]`, and record
   `saved_store[lane] = kv_source_of(layer).unwrap_or(layer)` and the sparse n_comp counts.
4. At a NON-source layer whose `kv_source_of(...).unwrap_or(layer) == saved_store[lane]`:
   SKIP the matvec_q / RoPE / fp4 / score / topk entirely, and run ONLY
   `indexer_gather.launch_batched` from `sd.indexer_sel_saved[lane]`, then take the same
   `eff_comp_kv_buf` / `eff_n_total_max` / `eff_comp_kv_batch_stride` branch S1 uses.
5. Invalidate `saved_store[lane] = -1` at the start of each chunk (a selection must not
   cross chunks), mirroring the per-token reset the decode path does in `engine.rs`.

Why gather-not-cache: scoring is the expensive part (the whole store per query); the
gather is ~512 rows per token. Re-gathering per reuse layer keeps the win and avoids
both the memory and the lane-aliasing hazard.

Validation: the same two-ended check S2-decode passed — BIT-IDENTICAL below the gate
(n_comp <= INDEXER_TOP_K, where top-512 selects everything), and coherent output with a
changed selection above it. Then measure at 100K, NOT 8K: at 8K the store is ~7.5 MB/layer
and the indexer cannot show a win (measured: `dgpu.attn_compute` 325 vs 326 us).

## S2-on-prefill IMPLEMENTED 2026-09-14 (engaging; win not yet measured)

Implemented as designed above, with one simplification that removed all the lane
plumbing: the top-k now writes DIRECTLY into the lane's own
`BatchDgpuScratch::indexer_sel_saved` instead of shared `sd.indexer_selected`, so the
selection persists per-lane with NO copy, and the gather reads it from there. The sparse
per-token counts are recomputed at reuse layers (`min(n_comp_after, INDEXER_TOP_K)`)
rather than saved. `indexer_saved_store` records the store group.

Verified engaging: `S2 shared selection ACTIVE ... layer=3 store=2` — layer 3 reuses
layer 2's selection and skips score+topk.
Verified safe below the gate: short-prompt output BIT-IDENTICAL to dense (sha
443e71cb8342).

NO cross-chunk reset is needed, and this is why: every store group's FIRST layer
(2, 8, 14, 20) is itself an index source, so a group's selection is always refreshed
within the chunk before any reuse layer can consume it. Layers 0-1 have no compressor.

**WIN NOT YET MEASURED.** 32K gave 474 tok/s (S1+S2) vs 488 (dense), but dense itself
swung 399 -> 488 tok/s between runs, so this is inside run-to-run variance and proves
nothing either way. A valid A/B needs: ONE server, arms alternated, >=3 passes each, and
at 100K (attention is ~30% of prefill at 32K but ~59% at 100K — see
`project_v41_prefill_aggregate_stages_2026-09-14`). Do that before drawing any conclusion
about S2's value.
