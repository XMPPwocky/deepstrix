# Embed phase: Qwen3-Embedding-4B in the hub process

**Status: rev 3 (2026-10-04), for review round 3.** Branch `worktree-embed-phase`, base `origin/main` c2db000. Rev 1 and rev 2 each got APPROVE WITH CHANGES; §12 maps round 1's findings and §13 round 2's. Code is on the branch (host tests pass). The CPU oracle has run on the real Q8_0 GGUF and reproduces the model card's scores (§10, `model_card_scores`). Nothing has run on a GPU yet.

## 0. The ask and the constraints

The owner asked (2026-10-04) for the hub to serve [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF) next to DeepSeek V4.1, from the **same process**, behind a standard OpenAI `/v1/embeddings` endpoint. Their constraints:

1. **Computing embeddings may block LLM inference.** Embedding is a third scheduler phase, next to prefill and decode. It runs alone.
2. **Nothing the embedding model holds may stay resident between phases.** Every resident byte costs hit rate: in the box-1 pool, the box-2 pool, or box 1's page cache. §4.5 lists what stays resident anyway; the owner must sign that list off.
3. **If it fits on the dGPU, run it there**, evicting other dGPU state *temporarily* during the phase, to box-1 RAM or NVMe.
4. **There are no dGPU hot-tier slots to borrow.** Laguna code is not a template: it was never really tested.

## 1. Where things stand (measured 2026-10-04 ~20:20 UTC, hub on the 128 GB box)

| Resource | State |
|---|---|
| dGPU (9070 XT, gfx1201) | 16,135 / 16,304 MiB used. About 9 GiB is V4.1 non-MoE weights (startup log). V4.1 KV is small: only the 4 KV-source layers hold KV, FP8 at 592 B/row for `n_kv_max/4` rows. The rest is arena, scratch, lane buffers and graph execs. |
| iGPU (gfx1151) | GTT 107 / 127 GB. Box-1 pool `V41_PAGER_POOL_GB=95`, drafter 7.93 GB, vision 0.9 GB. |
| Box-1 RAM | 124 GB total, ~8 GB available. Most of that is page cache serving expert and Engram reads. |
| dGPU link | OCuLink Gen4 x4. |
| NVMe reads | Latency-bound on dm-crypt. One pread runs at 2.4–2.55 GB/s; four readers at 5.52 GB/s (`mapped.rs:157-161`, `het/weights.rs` fast load). |

## 2. The model and its weight format

**Shape.** Qwen3 decoder (`general.architecture = qwen3`):

- 36 layers, hidden 2560, 32 Q heads and 8 KV heads (kv_group 4), head_dim 128.
- Per-head RMSNorm on Q and K before RoPE. RoPE is **NeoX**, θ = 1e6.
- SwiGLU FFN with intermediate 9728. RMSNorm eps 1e-6. Vocab 151,665.
- Tied embeddings, so there is no LM head.
- **Causal** attention.

**Output.** The **last** token's hidden state goes through `output_norm`, then L2 normalization. With MRL `dimensions` in 32..2560, the result is truncated to that length and normalized again.

**Prompts.** Queries are formatted by the client as `Instruct: {task}\nQuery:{query}`. The reference appends `<|endoftext|>` (151643); gate E0 confirms this from the reference token ids.

**Weights: Q8_0 only** (`Qwen3-Embedding-4B-Q8_0.gguf`, 4.28 GB).

- On disk a Q8_0 row is interleaved per block: `[f16 scale | 32 × i8]`.
- The production prefill GEMM `Q8_0MatvecWmma::gemm_f16x` (Q8_0 weights × f16 activations, 128×128 WMMA) reads the **M18 split layout** instead: per row, all scales, then all quants (`weights::repack_q8_0`; kernel `q8_0_matvec_wmma.hip:605-630`).
- Each streamed layer is therefore **repacked in place on the dGPU** (§5.2).
- `gemm_f16x` is tolerance-tested (`rel < 5e-3`, `rms < 1e-3`) at k ∈ {1024, 4096, 8192} (`tests/q8_0_gemm_f16x.rs`). It is not bit-exact. Gate E2 covers all four Qwen (m, k) shapes. Every one meets its contract: m ∈ {6144, 2560, 19456} are multiples of 128, k ∈ {2560, 4096, 9728} are multiples of 32.
- Other quants are refused at startup.

Byte sizes (Q8_0 = 34 B per 32 weights):

| Tensor (per layer) | Rows × K | Bytes |
|---|---|---|
| `attn_q` | 4096 × 2560 | 11,141,120 |
| `attn_k`, `attn_v` | 1024 × 2560 each | 2,785,280 each |
| `attn_output` | 2560 × 4096 | 11,141,120 |
| `ffn_gate`, `ffn_up` | 9728 × 2560 each | 26,460,160 each |
| `ffn_down` | 2560 × 9728 | 26,460,160 |
| 4 norms (f32) | — | 21,504 |
| **one layer** | | **107,254,784** |
| **36 layers** | | **3.86 GB** |
| `token_embd` (host-side row reads only) | 151,665 × 2560 | 412,528,800 |

## 3. Design summary

One **embed phase** runs **between scheduler ticks**, in `worker_loop_ms`. It never runs inside `Sched::tick`, so an embedding error cannot reach the step-failure path. During the phase, no other hub work is queued on any device. The Tier B thread may record events on its own stream, and box 2 may finish in-flight work; neither touches the dGPU donors.

1. **Batch.** Take queued inputs up to the token budget, round-robin across requests.
2. **Quiesce.** `hipDeviceSynchronize` the dGPU.
3. **Lend.** Load the phase's kernel modules and allocate two pinned host buffers. Lend about 575 MB of dGPU memory **in place** from immutable V4.1 weights (the donors).
4. **Embed tokens.** Read the token-embedding rows on the host (4 reader threads, then Q8_0 dequant) and upload the f32 residual.
5. **Run the layers.** All 36 layers, **layer-major over every token of the phase**. Weights **stream** from the GGUF: 4 reader threads per layer fill a pinned host buffer, one layer ahead of compute. Each layer is copied H2D into a two-buffer dGPU ring, **repacked in place**, then computed.
6. **Finish.** Gather each input's last row, then on the host: `output_norm`, L2, MRL. A non-finite result fails its request.
7. **Return the loan**, success or error, from the **loan image** (written once at startup), and verify it. Guard bands catch out-of-loan writes. If the return fails, or the thread unwinds with the loan out, the process aborts.
8. **Release.** Reply to finished requests; free the pinned buffers and kernel modules; trim the heap.

Fixed cost per phase, estimated from the in-tree read rates (not measured):

- layer stream: 3.86 GB at ~5.5 GB/s ≈ 0.70 s, with compute overlapped;
- loan return: ~0.58 GB single-reader at ~2.4 GB/s ≈ 0.25 s, hashed as it is read;
- verify (on by default): +~0.25 s;
- pinned alloc with zero-fill: ~0.05–0.1 s;
- token rows: < 0.1 s.

That totals **≈ 1.3–1.5 s** with verify, or **≈ 1.0–1.2 s** without, plus whatever compute does not overlap. The LLM stalls for that long. Gate E7 measures it.

## 4. The loan (dGPU memory borrowed in place)

### 4.1 Donors

At startup the server builds a candidate list, best first:

1. `global.output`, the V4.1 LM head: 129,280 × 5,440 B = 703,283,200 B in the M18 layout (`hf_v41.rs:393`, `weights.rs:170-190`). It is read only by the head and the drafter exit, both on the engine thread inside ticks (`het/engine.rs:908-920`, `forward_head.rs:114,226`).
2. Then the per-layer `attn_q_b`, `attn_output_a/b` and `shared.{gate,up,down}`, largest first.

All are immutable after load. `Loan::new` places the phase's buffers first-fit, in `EmbedSizing::buffer_sizes` order, adding donors until everything fits. `LoanAlloc::take` repeats that placement exactly, because the same sizes are requested in the same order. At the defaults, the head alone holds the whole loan.

**Never donors:** the KV arena, prefill scratch states, lane scratch, DSpark rings, or anything else whose contents live across ticks.

Dead-between-ticks scratch would also cost nothing to borrow, but it stays out. Its liveness audit is per buffer and easy to get wrong silently. The weights-only rule needs no audit.

### 4.2 The loan image

At startup, after the V4.1 weights load and before the scheduler starts:

- The donors' imaged ranges are copied D2H in chunks through a pinned buffer and written to `V41_EMBED_LOAN_IMAGE` (default `$HOME/.cache/deepstrix/embed-loan.img`), on the engine's dGPU transfer stream.
- One 64-bit hash is kept per chunk. Chunks are one layer buffer, 107 MB.
- The file is opened **without truncating** and taken under `flock(LOCK_EX|LOCK_NB)` *before* any byte changes. It is held for the life of the process, so a second hub (a private-port gate run) fails at startup instead of overwriting production's image.
- After the write: `fdatasync`, then `FADV_DONTNEED`.

Why an image, and not RAM:

- The donors' bytes are the M18 repack of a checkpoint conversion. A plain read cannot recreate them.
- A RAM copy would spend box-1 page-cache budget.
- Reading the image back drops its pages (DONTNEED), and the startup write costs ~0.3 s.

The hash is a 64-bit multiply-xorshift mix (`dgpu_loan::chunk_hash`), not blake3. It guards against accidental corruption, and adding blake3 to the kernels crate would need dependency sign-off. Only the server has blake3 today.

### 4.3 Return and verification

`Loan::give_back` runs at the end of every phase.

1. **Canary check first** (rev 3). `Loan::new` *plans* every lent buffer once (`LoanAlloc` then hands out exactly those ranges) and surrounds each with **canaries**, which are imaged but never lent:
   - a 64 KiB band before each donor's first buffer;
   - a 64 KiB gap after every buffer;
   - a 1 MiB guard band after each donor's last buffer.

   Every canary region is hashed at startup. At each return it is read D2H and compared before anything is overwritten. A mismatch means a kernel wrote outside its buffer, and possibly outside the loan into V4.1 state no image covers, so it **aborts** the process.
2. **Return the bytes.** The image is pread chunk by chunk into one of two pinned buffers. Each chunk is **checked against its startup hash (always)**, which catches disk or page corruption, then copied H2D. Per-buffer events let the next chunk's read overlap the previous copy.
3. **Read-back** (`V41_EMBED_VERIFY`, live, default on): each chunk is copied back D2H into the other buffer and hashed. This catches DMA corruption. A mismatch is copied once more.

**Any** error on this path aborts the process, so the supervisor restarts it. That covers a canary violation, a second read-back mismatch, a hash-on-read mismatch, an image read error, an H2D error and a sticky HIP error. A hub whose V4.1 weights differ from what it loaded must not serve.

`LoanOut`, a drop guard, also aborts if the engine thread unwinds with the loan out. It is armed only once the phase's pinned buffers exist: a failure to allocate them (host memory pressure) fails that phase's requests and touches nothing on the device.

The image is lent + canaries + guard, about **0.58 GB** at the defaults. A reader thread's panic becomes an error, not an unwind.

### 4.4 Invariants

- **L1.** Nothing reads a donor while it is lent. The phase runs between ticks and starts with `hipDeviceSynchronize` on the dGPU. Every dGPU kernel and copy is enqueued by the engine thread, and other threads touch no donor (review round 1 verified each thread).
  - The last tick's DSpark ring writes may still be in flight on `igpu.compute` (`ms_dspark.rs:672-678`). They are iGPU-side and read no donor.
- **L3.** Every donor byte is back, and the transfer stream synchronized, before the next tick. The return runs on every path; failure aborts.
- **L4.** No device allocation or free happens during the phase. The loan is views only. The pinned buffers are host memory. Kernel modules are code objects that load per phase (~ms) and unload after.
- **L5.** The image is written from donor bytes before the first loan, and donors never change, so the image always equals the live bytes.

Rev 1's L2, "not mid layer-major group", is **not** a loan invariant. It is a scheduling policy (§6.2).

### 4.5 What stays resident between phases (owner sign-off needed)

| Item | Where | Size | Why |
|---|---|---|---|
| Embedding tokenizer (`BpeVocab`: 151,665 tokens + ~151k merges, `Vec` + 2 `HashMap`s) | box-1 host heap | ~20 MB (estimate) | The HTTP handler tokenizes. Rebuilding it per request costs ~100+ ms. A lazy load with an idle drop is possible if the owner wants zero. |
| GGUF tensor directory + 36 × 11 `TensorLoc` + `output_norm` (10 KB) + open file handle | host heap | < 1 MB | Locating tensors. The parsed metadata (the tokenizer arrays) is **dropped** after load (`MappedGguf::drop_metadata`). |
| Loan bookkeeping (donor views, hashes) | host heap | < 1 KB | — |
| Loan image | disk only, page cache dropped | ~0.58 GB | §4.2 |
| Pages of the embed GGUF | page cache: **none** | 0 | The GGUF is opened with `POSIX_FADV_RANDOM` (no readahead around the phase's reads), and every phase ends with a **whole-file** `FADV_DONTNEED` (`MappedGguf::drop_page_cache`). Per-range DONTNEED keeps partial edge pages, and every 2.7 KB token row is one (rev 3). |

Allocated **per phase only:**

- kernel modules (`qwen3_embed`, `GqaAttention`);
- pinned host buffers (2 × 107 MB);
- the loan views.

Rev 1's `V41_EMBED_KEEP_PINNED` is removed. The GEMM module and the streams are the engine's own, already resident.

## 5. The forward

### 5.1 Layout of one phase

The phase holds `T` tokens: `n` inputs of `len_s` tokens each, every one ending in EOS, packed back to back. `seq_start[s]` is the prefix sum. Token `t` of input `s` is row `seq_start[s] + t`, at RoPE position `t`.

| Device buffer, in carve order (`EmbedSizing::buffer_sizes`) | Shape | Bytes |
|---|---|---|
| `resid` | f32 [T, 2560] | 10,240 · T |
| `ring0`, `ring1` | one layer each (`LayerLayout`) | 107.25 MB each |
| `gemm_out` | f32 [R, 19456] (gate‖up is widest; also the last-row gather) | 77,824 · R |
| `kc`, `vc` (current layer) | f16 [T, 8, 128] each | 2,048 · T each |
| `x16` (GEMM input) | f16 [R, 9728] | 19,456 · R |
| `attn_out` | f32 [R, 4096] | 16,384 · R |
| `q16` | f16 [R, 32, 128] | 8,192 · R |
| `pos`, `idx` | u32 [T] each | 4 · T each |

`R` = `V41_EMBED_SUB_ROWS` (default 1024); `T` ≤ `V41_EMBED_PHASE_TOKENS` (default 16384). At the defaults that is **575 MB** (unit test `default_loan_is_about_575_mb`).

There is no separate last-rows buffer. The last rows are gathered into `gemm_out` in chunks (up to 7,782 rows of 2560 per chunk) and read back from there. Empty inputs are refused, so every input has at least 2 tokens (content + EOS).

### 5.2 Per layer

Each ring buffer uses `LayerLayout`: `[q‖k‖v]` (6144 rows, one GEMM), `[attn_output]`, `[gate‖up]` (19456 rows, one GEMM), `[down]`, then the norms.

1. **Upload.** The copy stream waits for the event of layer `l−2` (the last reader of this ring buffer), copies H2D and records an event. The compute stream waits on it.
2. **Repack.** `qe_q8_0_repack_rows` runs on the compute stream, once per matrix group (qkv, o, gate‖up, down). It uses one workgroup per row and stages the row in LDS (≤ 480 blocks; the widest Qwen row is 304 blocks = 10,336 B). It writes the M18 layout in place.
   - Device traffic is ~2 × 107 MB per layer, under 1 ms.
   - It is never done on the host: the pipeline is read-bound, and a host repack would add straight to it.
   - GPU test `repack_matches_host` checks it against `weights::repack_q8_0` at all three Qwen row widths.
3. **Compute.** For each sub-batch of `R` rows, in token order:
   1. `qe_rmsnorm_f16`: `x16 ← f16(rmsnorm(resid) ⊙ attn_norm)`.
   2. GEMM qkv.
   3. `qe_qk_norm_rope`: per (row, head), q/k RMSNorm ⊙ `q_norm`/`k_norm`, then NeoX RoPE with HF numerics (`inv_freq[i] = 1/θ^(2i/128)`, f32). q goes to `q16`; k and v go to `kc`/`vc` at `row`.
   4. Attention: `prefill_flash_wmma_fa2`, one launch per input segment of the sub-batch.
      - Settings: `q_offset` = the segment's first position; K/V views start at `seq_start[s]`; `kv_capacity = len_s`; full causal (`swa_window 0`); scale `1/√128`; `kv_first = true`.
      - Tile loads past `n_kv_total` are guarded and zero-filled (`gqa_attention.hip:979-995`), so stale loan bytes cannot inject NaN.
      - This is the Laguna-era kernel, kernel-tested at kv_group 6/9 against a CPU reference; E2 covers kv_group 4.
   5. `qe_cast_f16` (pitch 4104), then GEMM o, then `qe_add`.
   6. `qe_rmsnorm_f16` (`ffn_norm`), then GEMM gate‖up, then `qe_swiglu_f16`, then GEMM down, then `qe_add`.
4. **Hand back.** Record the compute event for this ring buffer, then call `after_layer(l)`: the hub resets the watchdog there, and gate E6 injects its fault there.

Every f32→f16 cast **saturates** at ±65504 (Qwen-family activations can be large) and keeps NaN. NaN surfaces in the host-side finiteness check: a non-finite embedding fails its request and counts in `nonfinite`.

**Causality across sub-batches.** A row's K/V depend only on its own residual, and sub-batches run in token order. So every key an attention launch reads was written by an earlier sub-batch, or the same one, of the same layer.

**Launch count.** Attention launches once per input segment: about n × 36 per phase, and 73k at 8-token inputs. Gate E7 measures this. If it costs, a varlen variant (per-row segment start) is the fix.

### 5.3 Streaming

- **Reader threads** (scoped to the phase) pread layer `l+1` into pinned host buffer `(l+1) % 2` while the dGPU computes layer `l`. Each tensor is split into ≥ 4 MiB pieces over 4 threads (`read_layer_into_par`), using pread + `FADV_DONTNEED`.
- **Refill rule.** The engine thread hands a host buffer back to the readers only after that buffer's H2D event completes. A ring buffer is refilled only after the compute event of the layer that last read it.
- **Errors.** On any error the engine thread returns from the scope. Dropping the job channel ends the reader loop. A reader never waits for a buffer: it only receives jobs. `run` then synchronizes both streams, also on error.
- **Pinned buffers** (2 × 107 MB) are allocated per phase (`PinnedBuffer::new` zero-fills). The return reuses them.

### 5.4 Token-embedding rows

The phase's ids are deduplicated and sorted, and runs of adjacent ids are coalesced. Four threads pread the runs (Q8_0 rows are 2,720 B each) and dequantize. The f32 residual goes up in one H2D.

## 6. Scheduler

### 6.1 Requests reach the worker

- **`EmbedQueue`.** A mutex-guarded `VecDeque<EmbedRequest>`, bounded by the tokens of **every unfinished request**: queued, or taken by the worker and not yet replied to (`V41_EMBED_QUEUE_TOKENS`, default 1,048,576 = 64 full phases).
  - A request's tokens are released when it is replied to, failed or dropped, not when the worker takes it. That was rev 2's bug: the worker drained the queue every phase, so the cap never fired.
  - When full, the handler returns 503 with `Retry-After`. With nothing unfinished, one request is always accepted.
- **Engine death.** `submit_embed` refuses when the engine channel is closed. The handler waits for its reply in 5 s slices and fails if the worker is gone, since a request queued before the death keeps its reply sender alive in the queue.
- **`EmbedRequest`.** Holds the inputs (EOS appended), `dims`, per-input results, `next`, a `oneshot` reply and the queue time. A request whose client went away (`reply` closed) is dropped before it is computed.
- **Wake: edge-triggered.**
  - The handler sends `EngineRequest::EmbedWake` only when `EmbedQueue::claim_wake` flips `wake_pending` from false to true, so at most one wake is ever in the channel.
  - If that `try_send` fails because the channel is full, the claim is released: the worker is busy and checks the queue every iteration.
  - The worker clears the flag when it receives the wake. Before it blocks idle, it clears the flag and **re-checks** the queue; a push after the clear sends a new wake.
  - At most one channel slot goes to a wake, so chat keeps at least 7 of 8.
- **One `has_work` predicate** is used at both of `worker_loop_ms`'s idle checks: `Sched::has_llm_work() || embed.has_work()`. In rev 1 only the first was patched, so embed-only work spun forever.

### 6.2 Placement: between ticks, in `worker_loop_ms`

After intake and before `sched.tick`, the phase runs when all of these hold:

- the embed queue (or a part-done request) has work;
- **scheduling policy:** not in the middle of a layer-major group (`finish_group() && prefills.any(lm_mid_group)`). This protects the group's box-2 pages, as for decode;
- **duty cycle:** with LLM work pending, the next phase starts no earlier than `last_end + last_dur · (100/share − 1)`, where `share` = `V41_EMBED_MAX_SHARE` (live, 1..100, default 50). With no LLM work there is no gap.

Then the iteration runs the phase and `continue`s. The tick waits for the next iteration, so intake runs between a phase and the next tick.

`run_phase` cannot fail. Its errors go to its own requests, and only a failed return aborts.

Accounting:

- `self.phase` is unchanged, so bursts resume.
- `phase_since` is advanced by the phase's duration, so the interrupted burst is not charged for it.
- No box-2 pin hooks fire.
- `progress` is reset after every layer.
- Afterwards: `invalidate_device_cache` and `dgpu.set_current`.

**Watchdog.** The deadline is `--hang-deadline-ms` (default 60 s; the prod env has 120 s). It is reset after every layer, and a layer takes tens of ms.

### 6.3 Batching

- **Round-robin.** One input from each active request in turn, starting one request later each phase, until the next input would push `T` past `V41_EMBED_PHASE_TOKENS`. A request replies when its last input is done, so a big request spans phases without starving a small one behind it.
- **Limits** are enforced in the handler, as 400s:
  - each input ≤ `V41_EMBED_MAX_INPUT_TOKENS` (default 8192, clamped to `n_ctx_train`; startup checks it ≤ the phase tokens);
  - ≤ 2048 inputs per request;
  - ≤ `V41_EMBED_MAX_REQUEST_TOKENS` (default 262,144) tokens per request;
  - no empty strings or empty token arrays.
- **Bigger inputs** (up to the model's 32K) need `V41_EMBED_PHASE_TOKENS ≥ 32768`. That loan is ~1.0 GB: past the head, so the startup placement adds per-layer donors or fails with a clear error.

### 6.4 Serial loop

`serial_next`: with embed work waiting, a pending channel message goes first; otherwise exactly **one** phase runs and the loop asks again, so an embed backlog yields to every arriving request. Before blocking, it uses the same clear-then-recheck wake protocol.

### 6.5 Shutdown

`Shutdown` fails every queued and active embedding request ("server shutting down") before the engine exits. `abort_all` (a failed LLM step) leaves embedding requests alone: they are independent of the streams.

## 7. HTTP: `POST /v1/embeddings`

Request (OpenAI):

- `input`: a string, an array of strings, an array of token ids, or an array of arrays of token ids;
- `model`: accepted, not routed on;
- `encoding_format`: `float` (default) or `base64` (little-endian f32, hand-rolled RFC 4648 encoder, no new crate);
- `dimensions`: 32..`n_embd`;
- `user`: ignored.

Token ids must be `< vocab`.

**Body limit.** 64 MiB for this route. axum's default is 2 MiB, and 262,144 ids as JSON is ~1.8 MB, while 2048 texts easily exceed 2 MiB.

**Tokenization** runs in `spawn_blocking`, with the **embedding model's** vocab (`encode_qwen2`, then `<|endoftext|>` appended).

**Response:** `{object:"list", data:[{object:"embedding", index, embedding}], model: <the embedding model's name>, usage:{prompt_tokens, total_tokens}}`. `prompt_tokens` counts the EOS.

Without `--embed-gguf`, the route answers 404 (`embeddings_disabled`). `/v1/models` lists the chat model **first**, then the embedding model (`type: embeddings`, `embedding_dimensions`).

## 8. Tokenizer

`pre = qwen2` is `laguna_qwen2_split` (a port of llama.cpp's `unicode_regex_split_custom_qwen2`) run over the whole text, without Laguna's newline pre-split. New: `qwen2_pre_tokenize`, `BpeVocab::encode_qwen2` (no BOS), and an `encode_auto` arm.

Known divergences from HF:

- **No NFC.** It needs a Unicode-tables crate, which is a sign-off question.
- **Special-token text in the input is encoded as text.**
- **`\p{L}` / `\p{N}` are approximated** by `char::is_alphabetic` / `is_numeric`. `is_alphabetic` includes Other_Alphabetic combining marks, which `\p{L}` excludes: Indic and Thai vowel signs are the likely divergence.

The E0 corpus covers each of these.

## 9. Code layout

| Where | What |
|---|---|
| `v4flash-core/src/tokenizer.rs` | `qwen2_pre_tokenize`, `encode_qwen2`, the `encode_auto` arm |
| `v4flash-core/src/qwen3_embed.rs` | `Qwen3EmbedModel::from_gguf` (validation, tensor locations); `LayerLayout`; `read_layer_into_par`; threaded `token_rows`; `finish_embedding`; `cpu_forward` (the f32 oracle); `testing::write_synthetic` (a tiny random Qwen3 GGUF) |
| `v4flash-core/src/{gguf,mapped}.rs` | `drop_metadata` |
| `v4flash-kernels/kernels/qwen3_embed.hip` | `qe_q8_0_repack_rows`, `qe_rmsnorm_f16`, `qe_qk_norm_rope`, `qe_cast_f16`, `qe_swiglu_f16`, `qe_add`, `qe_gather_rows` |
| `v4flash-kernels/src/qwen3_embed.rs` | `Qwen3EmbedKernels` (per phase), `EmbedSizing`, `EmbedBuffers`, `run` |
| `v4flash-kernels/src/dgpu_loan.rs` | `Loan` (placement, image + `flock`, guard bands, `give_back`), `LoanAlloc`, `chunk_hash`. Generic: it knows device ranges, not what they hold. |
| `v4flash-kernels/tests/qwen3_embed_gpu.rs` | GPU gates: `tiny_gpu_matches_cpu`, `real_gpu_matches_cpu` (E2), `loan_round_trip`, `repack_matches_host` |
| `deepstrix-server/src/embed_phase.rs` | `EmbedQueue`, `EmbedRequest`, `EmbedInfo`, `EmbedCtx` (`load`, `due`, `run_phase`, `fail_all`), `LoanOut` |
| `deepstrix-server/src/openai/embeddings.rs` | Types, handler, `base64` |
| `multistream.rs`, `engine_worker.rs`, `main.rs`, `handler.rs` | The between-ticks hook and the idle predicate; `EmbedWake` and `submit_embed`; `serial_next`; `--embed-gguf` / `--embed-model-name`; the route and body limit; `/v1/models` |
| `knobs.rs` | `V41_EMBED_*` |
| `het/evtrace_kinds.rs` | `HUB_EMBED`, id **14** (hub band 10..19). `scripts/evtrace.py` reads kinds from the trace header, so it needs no change. |

**Knobs.** `MAX_SHARE` and `VERIFY` are live; the rest are static.

| Knob | Default | Meaning |
|---|---|---|
| `V41_EMBED_PHASE_TOKENS` | 16384 | max tokens per phase (sizes the loan) |
| `V41_EMBED_SUB_ROWS` | 1024 | rows per sub-batch (sizes the loan) |
| `V41_EMBED_MAX_INPUT_TOKENS` | 8192 | per input, at most the phase tokens |
| `V41_EMBED_MAX_REQUEST_TOKENS` | 262144 | per request |
| `V41_EMBED_QUEUE_TOKENS` | 1048576 | queued tokens before 503 |
| `V41_EMBED_MAX_SHARE` | 50 | % of wall time while the LLM has work (1..100) |
| `V41_EMBED_VERIFY` | 1 | hash-on-read + read-back on return |
| `V41_EMBED_LOAN_IMAGE` | `$HOME/.cache/deepstrix/embed-loan.img` | image path |
| `V41_EMBED_FAULT_LAYER` | off | gates only: fail every forward after this layer |

**Telemetry.** One `ms.embed` line and one `hub_embed` record per phase. Fields: `requests`, `inputs`, `tokens`, `lent_bytes`, `wait_ms`, `rows_ms`, `read_ms`, `wait_read_ms`, `fwd_ms`, `return_ms`, `verify_ms`, `pinned_ms`, `total_ms`, `live`, `prefills`, `ok`, `nonfinite`, `guard_violations`.

## 10. Gates

All thresholds are pre-registered. Every cosine gate reports the **minimum** over inputs, not the mean.

| Gate | What | Pass |
|---|---|---|
| **E0** tokenizer | `encode_qwen2` + EOS vs HF `AutoTokenizer` ids on the fixture corpus: queries with instruction; documents; code; CJK, **Indic and Thai** (combining marks); emoji; **uppercase contractions** (`IT'S`, `WE'LL`); **digit runs**; **CRLF runs**; **trailing and mixed whitespace**; long (8K) | identical ids on every input, or each divergence explained by §8 and signed off |
| **E1** CPU oracle | `cpu_forward` on the real Q8_0 GGUF vs HF bf16 reference embeddings (`scripts/qwen3_embed/ref_embed.py`) | min cos ≥ 0.998 (Q8_0 budget; revisit against the measured distribution) |
| **E2** GPU forward | `real_gpu_matches_cpu`: `run` on the real GGUF vs the CPU oracle (an unset `QWEN3_EMBED_GGUF` is an error). It covers all four Qwen GEMM shapes and kv_group 4, on: the model card's texts, code, CJK, a ~4.5K-token input, and adversarial inputs (a ~2K-token repetition, a punctuation run). It runs twice: at the **production sizing** (T 16384, R 1024, so one input spans sub-batches with `q_offset` ≥ 1024) and with 64-row sub-batches. Also `repack_matches_host` and `tiny_gpu_matches_cpu`. | min cos ≥ 0.9999 in both runs; repack byte-identical; all outputs finite |
| **E3** loan integrity | (a) **A/A control first**: the decode oracle twice with no phases, under pinned settings (`V41_SUB_DRY=1` or λ = 0, one stream, a fixed lane rule), must be identical, or E3 cannot be judged. (b) The same with embed phases interleaved. (c) A test build hashes **all** immutable dGPU weights (~9 GiB) before and after 100 phases. (d) `loan_round_trip`. | (a) identical; (b) identical; (c) all hashes equal; (d) passes; zero verify retries and zero `guard_violations` |
| **E4** e2e | `/v1/embeddings` on a private port (own `V41_EMBED_LOAN_IMAGE`): every input shape, `dimensions`, `base64`, 503 on a full queue, a request spanning several phases, concurrent chat | correct responses; chat output unchanged |
| **E5** residency (constraint 2) | After a phase: `fincore` on the embed GGUF and the image shows 0 pages; RSS (after trim), pinned memory and dGPU free are back at the pre-phase baseline ± noise. With `--embed-gguf` and no embedding traffic over a 2 h window: decode tok/s and box-1 hit rate equal to the same window without it (`feedback_box2_warming_dominates_ab`: warm first). | all at baseline |
| **E6** failure injection | `V41_EMBED_FAULT_LAYER=k` (k = 0, 17, 35) with chat streams live; a client cancelling mid-phase; shutdown with requests queued | the faulted requests get errors; the loan returns (verify clean); chat streams continue unharmed; cancelled and shut-down requests get errors or are dropped; no abort |
| **E7** performance | Phase fixed cost (one 16-token input; it includes the per-phase `malloc_trim` and pinned zero-fill) and per-token cost (slope over 1K..16K tokens; `feedback_per_token_cost_is_a_slope`); attention launch overhead at 8-token inputs; chat ITL p50/p99 with embedding traffic at share 50 vs none | recorded; fixed cost within 2× of §3's estimate, or the estimate is revised |
| **E1a** model card *(PASSED 2026-10-04)* | `model_card_scores`: the CPU oracle on the real Q8_0 GGUF, the card's 2 instructed queries × 2 documents | every score within 0.01 of `[[0.7534, 0.1147], [0.0320, 0.6258]]`. Got `[[0.7515, 0.1155], [0.0328, 0.6257]]`, max \|Δ\| 0.0019 (vLLM's own is 0.002). The GGUF also confirms `add_eos_token = true` with eos 151643 and `pooling_type` 3 (last). |

E2–E7 need a GPU window (the hub down) and the weights. E0–E1 need only the weights. The host tests run anywhere: core 11, kernels 4, server 161.

## 11. Risks and open questions

1. **Phase cost is unmeasured** (E7). Possible levers: parallel image reads in `give_back` (today a single reader), verify off after a soak, Q4_K_M (half the stream, but needs the K ≤ 4096 cap of the quantized GEMM lifted, and costs quality).
2. **§4.5 residency needs owner sign-off**, mainly the ~20 MB tokenizer.
3. **The duty-cycle default** (50%) is a guess.
4. **NFC** (§8), if E0 shows divergence on real inputs.
5. **Instructions** are the client's job, as with vLLM and llama-server.
6. **Attention launch count** for many short inputs (E7). The fix is a varlen variant.
7. **E3(a) may show V4.1 is not deterministic** run to run even under pinned settings. Then E3 rests on (c) and (d).

## 12. Rev 1 review → rev 2

| # | Finding | Resolution |
|---|---|---|
| 1 | BLOCKER: `gemm_f16x` reads the M18 split layout | `qe_q8_0_repack_rows` in place after each H2D (§5.2); GPU test vs `weights::repack_q8_0`; §2 corrected (tolerance-tested, k coverage) |
| 2 | Embed-only work spins | One `has_llm_work() \|\| embed.has_work()` predicate at both idle checks (§6.1) |
| 3 | An embedding failure kills chat streams | Phase moved out of `tick` into `worker_loop_ms`; `run_phase` is infallible; any return-path failure or unwind aborts (§4.3, §6.2) |
| 4 | `EmbedWake` can fill the channel | Edge-triggered `wake_pending`; full channel releases the claim; clear-then-recheck before blocking (§6.1) |
| 5 | "Zero residency" was not true | §4.5 lists what stays, for sign-off; metadata dropped; kernels per phase; engine streams and `q8_wmma` reused; `KEEP_PINNED` removed; heap trimmed after each phase |
| 6 | Gates insufficient | E3 A/A control + whole-weights hash; E5 residency; E6 fault injection (`V41_EMBED_FAULT_LAYER`, through the now-fallible `after_layer`); E7 performance |
| 7 | Cost estimate ignored measured read rates | 4-reader layer reads, threaded token rows; §3 restated from 2.4 / 5.5 GB/s |
| 8 | Attention is Laguna code | Relabelled (§5.2); launch count in E7 |
| 9 | `last rows` unaccounted | No such buffer: gathered through `gemm_out`; empty inputs refused (§5.1) |
| 10 | L2's reason was wrong | Now a scheduling policy, not an invariant (§4.4, §6.2) |
| 11 | Verification checked the wrong things | Hash-on-read + read-back + guard bands (§4.3); hash choice documented (§4.2) |
| 12 | Tokenizer corner cases | E0 corpus extended (§10); combining marks named (§8) |
| 13 | f16 overflow | Saturating casts; host finiteness check fails the request; `nonfinite` telemetry; adversarial inputs in E2 |
| 14 | Image write left page cache; no lock | fdatasync + DONTNEED; `flock` before any write, no truncate (§4.2) |
| 15 | HTTP and queue gaps | 64 MiB body limit; `spawn_blocking`; Shutdown fails requests; token-bounded queue; round-robin (§6, §7) |
| 16 | Serial loop backlog | One phase per iteration, channel first (§6.4) |
| 17 | Reader thread on errors | The job channel drops on scope exit; readers never wait on buffers (§5.3) |
| 18 | Watchdog default | 60 s default, 120 s in prod (§6.2) |
| 19 | Evtrace id band | id 14 (hub band) |
| 20 | Glossary | `SlotLayout` → `LayerLayout`, ring/host "slots" → buffers, "job" → `EmbedRequest`; GLOSSARY.md gains embed phase, loan, donor, loan image, guard band |
| 21 | "Nothing else runs" overstated | Qualified (§3) |
| 22 | Tile and `kv_first` | `Base` tile for v1 (variants are bit-identical; T256x128 fits 6144 / 19456, a later perf lever); `kv_first = true` explicit |
| 23 | Share = 0; unreachable max | Knob range 1..100; §6.3 states what >8K inputs need |
| 24 | `/v1/models` order; echoed name | Chat model first; responses carry the embedding model's name |

## 13. Rev 2 review → rev 3

| # | Finding | Resolution |
|---|---|---|
| 1 | MAJOR: the queue cap never bounded anything | Tokens count until reply, fail or drop (`EmbedQueue::release`), not until the worker takes them; `settle_requests` is the one exit (§6.1) |
| 2 | MAJOR: page-cache residue (partial edge pages, readahead) | `FADV_RANDOM` at open; whole-file `drop_page_cache` after every phase (§4.5) |
| 3 | MAJOR: E2 as coded ≠ E2 as registered | 4.5K + adversarial inputs, production sizing + 64-row run, unset GGUF is an error (§10) |
| 4 | Pinned-alloc failure aborted the hub | Pinned buffers allocated before `LoanOut` is armed; failure fails the batch (§4.3) |
| 5 | Serial loop skipped the device-cache reset | `invalidate_device_cache` + `set_current` moved into `run_phase` |
| 6 | Guard bands covered little; a violation did not stop anything | Planned placement with canaries before, between and after every buffer; violation aborts (§4.3) |
| 7 | Hash-on-read only under verify | Always on; `V41_EMBED_VERIFY` controls only the read-back |
| 8 | Requests hang if the engine dies | `submit_embed` checks the channel; the handler watches worker liveness (§6.1) |
| 9 | One bad id fails a phase | Startup: vocab ≤ `token_embd` rows; the handler range-checks text-derived ids too |
| 10 | Events dropped before the error-path sync | The events live in `run`, past both streams' synchronize |
| 11 | `SUB_ROWS` could exceed `grid.y` | Knob max 65535 |
| 12 | Doc precision | DSpark iGPU writes noted (L1); image 0.58 GB; chat keeps 7 of 8 slots; `wait_ms` is the max over active requests |
| 13 | Trim cost | Counted in E7 |
| 14 | CPU oracle attention single-threaded | Threaded over rows |
| 15 | Tiny gate's unaligned scale sections | Tiny `n_ff` 384 → 512 (every K a multiple of 256) |
| 16 | Reader panic re-panicked with the loan out | `catch_unwind` in the reader; the panic becomes an error |
