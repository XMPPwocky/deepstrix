# Embed phase: Qwen3-Embedding-4B in the hub process

**Status: DRAFT rev 1 (2026-10-04), for review. No code merged.** Branch `worktree-embed-phase`, base `origin/main` c2db000.

## 0. The ask and the constraints

The owner asked (2026-10-04) for the hub to serve [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF) next to DeepSeek V4.1, from the **same process**, behind a standard OpenAI `/v1/embeddings` endpoint. Their constraints:

1. **Computing embeddings may block LLM inference.** Embedding work is a third scheduler phase next to prefill and decode. It runs alone.
2. **Nothing the embedding model holds may stay resident between phases.** Every byte held outside the phase costs hit rate: in the box-1 pool, in the box-2 pool, or in the box-1 page cache. That is why embedding is a separate phase in the first place.
3. **If it fits on the dGPU, run it there.** Other dGPU state may be evicted *temporarily* during the phase, to box-1 RAM or NVMe.
4. **There are no dGPU hot-tier slots to borrow.** The dGPU hot tier is inert under paged experts (glossary). Laguna code is not a template: it was never really tested.

## 1. Where things stand (measured 2026-10-04 ~20:20 UTC, hub on the 128 GB box)

| Resource | State |
|---|---|
| dGPU (9070 XT, gfx1201) | 16,135 / 16,304 MiB used. V4.1 non-MoE weights are ~9 GiB of it (startup log). V4.1 KV is small (4 KV-source layers; FP8 592 B/row at `n_kv_max/4`). The rest is arena / scratch / lane buffers / graph execs. |
| iGPU (gfx1151) | GTT 107 / 127 GB. Box-1 pool `V41_PAGER_POOL_GB=95`, drafter 7.93 GB, vision 0.9 GB. |
| Box-1 RAM | 124 GB total, ~8 GB available, mostly page cache that serves expert / Engram reads. |
| dGPU link | OCuLink Gen4 x4 (16 GT/s). |

There is no free memory anywhere to park 4 GB permanently. The one large, cheap-to-borrow thing on the dGPU is **immutable V4.1 weights**: they can be overwritten in place and put back byte-exact from a copy, with no write-out at borrow time.

## 2. The model

From `config.json` (HF) and the GGUF repo:

- Qwen3 decoder (`general.architecture = qwen3`): 36 layers, hidden 2560, 32 Q heads / 8 KV heads, head_dim 128.
- Per-head RMSNorm on Q and K before RoPE (`attn_q_norm`, `attn_k_norm`). RoPE is **NeoX** (pairs `i`, `i+64`), θ = 1e6.
- SwiGLU FFN with intermediate 9728. RMSNorm eps 1e-6. Vocab 151,665. Tied embeddings, so there is no LM head to load.
- **Causal** attention. Output = **last-token** hidden state → `output_norm` → L2 normalize.
- MRL: any `dimensions` in 32..2560 means truncate, then L2 normalize.
- Queries are instruction-formatted by the client: `Instruct: {task}\nQuery:{query}`. Documents are not.
- The reference appends `<|endoftext|>` (151643). **To verify** against the reference tokenizer's ids (gate E0).

**Weight format: Q8_0 only in v1** (`Qwen3-Embedding-4B-Q8_0.gguf`, 4.28 GB). Every Q8_0 projection runs on the production prefill GEMM `Q8_0MatvecWmma::gemm_f16x` (Q8_0 weights × f16 activations, 128×128 WMMA, bit-exact-tested). Every Qwen3 shape fits its contract (`m % 128 == 0`, `k % 32 == 0`). Other quants are refused at startup.

Byte sizes at Q8_0 (34 B per 32 weights):

| Tensor (per layer) | Rows × K | Bytes |
|---|---|---|
| `attn_q` | 4096 × 2560 | 11,141,120 |
| `attn_k`, `attn_v` | 1024 × 2560 each | 2,785,280 each |
| `attn_output` | 2560 × 4096 | 11,141,120 |
| `ffn_gate`, `ffn_up` | 9728 × 2560 each | 26,460,160 each |
| `ffn_down` | 2560 × 9728 | 26,460,160 |
| 4 norms (f32) | — | 21,504 |
| **one layer** | | **107,254,784 (102.3 MiB)** |
| **36 layers** | | **3.86 GB** |
| `token_embd` (host-side row gather only) | 151,665 × 2560 | 412,528,800 |

## 3. Design summary

One **embed phase** = one scheduler tick, run between ticks. Nothing else runs on any device while it is open.

1. Take queued embedding inputs up to a token budget.
2. **Loan**: in-place views over immutable V4.1 dGPU weight buffers (the donors). Pointers do not change. The loan holds about 0.6 GB: a two-layer weight ring plus activations and per-layer K/V.
3. Gather token-embedding rows on the host (pread + Q8_0 dequant) and upload the f32 residual.
4. Run the 36 layers **layer-major over every token of the phase**. Weights **stream** from the GGUF on NVMe through a pinned host ring into the dGPU ring, one layer ahead of compute.
5. Gather each input's last row, then on the host: `output_norm`, L2, MRL.
6. **Return** the loan: re-upload the donors' bytes from the **loan image** (an on-disk copy written once at startup), synchronize, and optionally verify.
7. Reply.

What it costs: zero residency outside the phase. About 0.6 GB of the dGPU is borrowed during it. A phase's fixed cost is about one NVMe stream of 3.86 GB plus a ~0.7 GB return read (estimated 0.9–1.8 s, not measured), with compute overlapped. The LLM stalls for that long.

## 4. The loan (dGPU memory borrowed in place)

### 4.1 Donors

At startup, with `--embed-gguf` given, the server builds a **donor list** from `state.weights`. Each donor is an immutable dGPU `DeviceWeight` buffer, taken largest first, starting with `global.output` (the V4.1 LM head: 129,280 × 5120, ≥ 0.7 GB). Donors are added until their total covers the **loan size** (§6.3).

A donor must be:

- **immutable after load**: no kernel writes it;
- **read only inside steps / prefill units**: never between ticks, never by a background thread;
- **not referenced by a pointer that changes when it is overwritten**: in-place views keep every captured HIP graph valid (graphs bake addresses, not contents).

`global.output` satisfies all three. It is read by the head kernel and by the drafter exit, both inside ticks. The per-layer weights are fallbacks (`shared` expert, `attn_q_b`, `attn_output_a/b`), with the same property.

Each embed buffer is a sub-allocation (256-B aligned, first-fit) inside one donor range. No embed buffer spans two donors. The ring slots (107 MB each) need the big donor.

**Never donors:** the KV arena, prefill scratch states, lane scratch, DSpark rings, and anything else whose contents live across ticks (dirty state). Scratch that happens to be dead at a tick boundary would also cost nothing to borrow. It is deliberately left out of v1: the "dead between ticks" audit is per buffer and easy to get wrong silently, and the weights-only rule needs no audit.

### 4.2 The loan image

At startup (after the V4.1 weights are loaded, before the scheduler starts), the donor ranges are copied D2H through a pinned chunk buffer and written to **`V41_EMBED_LOAN_IMAGE`** (default `$HOME/.cache/deepstrix/embed-loan.img`; one hub per box, overwritten at every start). A blake3 hash is kept per 64 MiB chunk.

- Startup cost: about 0.7 GB D2H + write, roughly 0.3 s, against an ~80 s load.
- A startup failure (disk full, read-only) fails the start, as a bad `--mmproj` does.

Why an image and not RAM:

- A RAM copy would sit in box-1 RAM, the page cache's budget, either for the whole process or for the whole phase.
- The image is read once per phase (pread + `POSIX_FADV_DONTNEED`, as `MappedGguf::read_range_into` does), so it leaves no page-cache residue.
- The donors are clean. Nothing is written out at borrow time.

### 4.3 Return and verification

At the end of every phase, success or error, the loan image is read chunk by chunk into the pinned ring and copied H2D into each donor range, then the device is synchronized.

With **`V41_EMBED_VERIFY`** (default 1 in v1), each chunk is then read back D2H and its blake3 compared with the startup hash.

- A mismatch is retried once.
- A second mismatch logs an error and **aborts the process**: the supervisor restarts it, as the watchdog does. A hub with a corrupt V4.1 head must not serve.

Verification costs one extra D2H of the loan (~0.12 s at Gen4) plus hashing. The default can drop to 0 once the gates (§10) and a soak have passed.

### 4.4 Invariants (the loan is correct iff all hold)

- **L1.** The phase starts only at a tick boundary, after `MsDspark::settle_writes`. It also requires `HeterogeneousEngine` idle: `state.dgpu.synchronize()` is the first thing the phase does. Nothing reads a donor while it is loaned.
- **L2.** The phase never runs while a layer-major group is half done (`finish_group() && prefills.any(lm_mid_group)`): its checkpoints are valid only between windows.
- **L3.** Every donor byte is back, and the device synchronized, before `tick` returns. No early return skips the return path; it runs in a guard.
- **L4.** No allocation or free happens on the dGPU during the phase. The loan uses views only, so `GraphCache::refresh_room` and the layer-major store see the same free memory before and after.
- **L5.** The image is written from donor bytes before any phase overwrites them, and donors never change afterwards, so the image always equals the live bytes.

## 5. The forward

### 5.1 Layout of one phase

The phase takes `T` tokens: `n` inputs of lengths `len_s`, each ending in EOS, packed back to back. `seq_start[s]` = prefix sum. Token `t` of input `s` is row `seq_start[s] + t` at RoPE position `t`.

| Device buffer (in the loan) | Shape | Bytes |
|---|---|---|
| residual | f32 [T, 2560] | 10,240 · T |
| K, V (current layer only) | f16 [T, 8, 128] each | 4,096 · T |
| `x16` (GEMM input; normed hidden, attention output, SwiGLU output) | f16 [R, 9728] | 19,456 · R |
| `gemm_out` | f32 [R, 19456] (gate‖up, largest) | 77,824 · R |
| `q16` | f16 [R, 32, 128] | 8,192 · R |
| `attn_out` | f32 [R, 4096] | 16,384 · R |
| weight ring | 2 × 107,254,784 | 214.5 MB |
| last rows | f32 [n, 2560] | 10,240 · n |

`R` = sub-batch rows (`V41_EMBED_SUB_ROWS`, default 1024). `T` ≤ `V41_EMBED_PHASE_TOKENS` (default 16384). At the defaults the loan is about 214.5 + 168 + 67 + 125 ≈ **575 MB**.

### 5.2 Per layer

In each ring slot the layer's tensors are laid out so one GEMM covers several of them, because Q8_0 rows are independent:

`[q ‖ k ‖ v]` (6144 rows) · `[attn_output]` · `[gate ‖ up]` (19456 rows) · `[down]` · `[norms]`

For each sub-batch of `R` rows, in token order:

1. `qe_rmsnorm_f16`: `x16 ← f16(rmsnorm(residual[rows]) ⊙ attn_norm)`.
2. GEMM qkv: `gemm_out[R, 6144] ← x16 · Wqkv` (k = 2560).
3. `qe_qk_norm_rope`, per (row, head):
   - q: per-head RMSNorm ⊙ `q_norm`, then NeoX RoPE at `pos[row]`, written to `q16`;
   - k: the same with `k_norm`, written to `K[row]`;
   - v: cast to f16 into `V[row]`.
4. Attention, once per input segment inside the sub-batch: `GqaAttention::prefill_flash_wmma_fa2` with `q_offset` = the segment's first position, K/V = views starting at `seq_start[s]`, `kv_capacity = len_s`, `swa_window = 0` (full causal), scale `1/√128`. Output goes to `attn_out`.
5. `qe_cast_f16`: `x16 ← f16(attn_out)` (pitch 4104, off a power of two per the GEMM note).
6. GEMM o: `gemm_out[R, 2560] ← x16 · Wo` (k = 4096). Then `qe_add`: `residual[rows] += gemm_out`.
7. `qe_rmsnorm_f16` with `ffn_norm`. GEMM gate‖up: `gemm_out[R, 19456]` (k = 2560).
8. `qe_swiglu_f16`: `x16 ← f16(silu(g) ⊙ u)`. GEMM down: `gemm_out[R, 2560]` (k = 9728). Then `qe_add`.

Causality holds across sub-batches. A row's K/V depend only on its own residual. Sub-batches run in token order, so every key an attention launch needs was written by an earlier or the same sub-batch of the same layer.

After layer 35, `qe_gather_rows` copies each input's last row into `last rows`, followed by one D2H. On the host: RMSNorm ⊙ `output_norm`, then L2 normalize; with `dimensions = d`, truncate to `d` and L2 normalize again.

The kernels are **new**, in `kernels/qwen3_embed.hip`: `qe_rmsnorm_f16`, `qe_qk_norm_rope`, `qe_cast_f16`, `qe_swiglu_f16`, `qe_add`, `qe_gather_rows`. They are small and each is tested against a CPU reference. The GEMM and attention are reused production / kernel-tested code.

RoPE follows the HF numerics: `inv_freq[i] = 1 / θ^(2i/128)` and `angle = pos · inv_freq[i]` in f32.

### 5.3 Streaming the weights

A reader thread (scoped to the phase) preads layer `l+1`'s tensors into pinned host buffer `(l+1) % 2` while the GPU computes layer `l`. It uses `MappedGguf::read_range_into`: pread + DONTNEED, so no page-cache residue. One pread per tensor, in ring-slot order.

The engine thread issues the H2D copy into dGPU ring slot `(l+1) % 2` on a **copy stream** and records an event, which the compute stream waits on before layer `l+1`.

A slot is refilled only after the compute stream's event for the layer that last used it. The pinned host buffer is reused only after its H2D event.

Pinned host buffers (2 × 107 MB) are allocated **per phase** and freed at its end (constraint 2). `V41_EMBED_KEEP_PINNED=1` keeps them for the process if their allocation cost shows up in the phase timing.

### 5.4 Token-embedding rows

The phase's token ids are deduplicated and sorted. Each `token_embd` row (2720 B at Q8_0) is read with one pread; runs of adjacent ids coalesce. Rows are dequantized on the host (`kquants::dequant_to_f32`) into the f32 residual, which goes up in one H2D.

## 6. Scheduler

### 6.1 Requests reach the worker

- **`EmbedQueue`**: `Arc<Mutex<VecDeque<EmbedJob>>>`, shared by `EngineHandle` and the worker, bounded by `V41_EMBED_QUEUE_INPUTS` (default 4096 queued inputs). Full → HTTP 503 (`SubmitError::Busy`).
- **`EmbedJob`**:
  - `inputs: Vec<Vec<u32>>`: token ids, EOS appended by the handler;
  - `dims: Option<u32>`;
  - `results: Vec<Option<Vec<f32>>>`;
  - `next: usize`;
  - `reply: oneshot::Sender<Result<EmbedOutput, String>>`;
  - `queued: Instant`.
  A job whose `reply` is closed (client gone) is dropped before it is computed.
- **Wake.** After pushing, the handler `try_send`s `EngineRequest::EmbedWake` on the existing channel. The variant carries no payload.
  - If the channel is full, the wake is dropped. That is harmless: a full channel means the worker is busy, and it checks `EmbedQueue` every loop iteration.
  - Chat keeps all eight of its slots: embed jobs never sit in the channel.
- **Idle check.** The multistream loop's idle test (`multistream.rs` `worker_loop_ms`) gains `&& embed_queue.is_empty()`. Without it, a phase that leaves inputs queued (token budget) would block in `blocking_recv`.

### 6.2 The phase in `Sched::tick`

The phase runs right after `settle_writes` and the perfetto export, before the cancel sweep. When it is due, the tick runs the embed phase **and nothing else**, then returns.

It is due when all of these hold:

- `EmbedQueue` is not empty;
- **L2** holds (not mid layer-major group);
- the **duty-cycle gap** has passed. When LLM work exists (streams, prefills, queued or parked requests), the next phase may start no earlier than `last_end + last_dur · (100/share − 1)`, with `share = V41_EMBED_MAX_SHARE` (default 50). With no LLM work there is no gap.

The gap is the only throttle. With a 1 s fixed cost, an indexing client sending one small request at a time would otherwise take the engine.

Accounting:

- `self.phase` is not changed. Prefill / decode bursts resume where they were.
- `phase_since` is advanced by the phase's duration, so the interrupted burst is not charged.
- Box-2 pin hooks do not fire: box 2 sees no requests during the phase.
- `state.progress.pet()` runs after every layer (the 120 s watchdog).
- After the phase: `state.engine.invalidate_device_cache()` and `state.dgpu.set_current()`, as after a vision encode.

### 6.3 Batching

The phase takes **inputs**, not whole jobs, FIFO across jobs, until the next input would push `T` past `V41_EMBED_PHASE_TOKENS`. A job finishes, and is replied to, when its last input is done, so a large request spans several phases.

Limits, enforced in the handler as HTTP 400 so the worker never sees an impossible input:

- each input ≤ `V41_EMBED_MAX_INPUT_TOKENS` (default 8192, Qwen's own example; max 32768). Startup checks it ≤ `V41_EMBED_PHASE_TOKENS`;
- inputs per request ≤ 2048;
- tokens per request ≤ `V41_EMBED_MAX_REQUEST_TOKENS` (default 262144).

The loan size is computed at startup from `V41_EMBED_PHASE_TOKENS` and `V41_EMBED_SUB_ROWS` (§5.1). Both are **static** knobs, because the donor list is fixed then.

### 6.4 The serial loop

The non-multistream `worker_loop` runs pending embed work after each request and on `EmbedWake`. Production runs multistream; this keeps tests and the fallback honest.

## 7. HTTP: `POST /v1/embeddings`

Request (OpenAI):

- `input`: one string, an array of strings, an array of token ids, or an array of arrays of token ids;
- `model` (accepted and echoed; not routed on);
- `encoding_format`: `float` (default) or `base64` (little-endian f32, hand-rolled encoder, no new crate);
- `dimensions` (32..2560);
- `user` (ignored).

Token-id inputs are checked `< vocab`. Empty strings are refused (400), as OpenAI does.

Response: `{object:"list", data:[{object:"embedding", index, embedding}], model, usage:{prompt_tokens, total_tokens}}`. `prompt_tokens` counts the EOS.

Tokenization happens in the handler with the **embedding model's own vocab** (`BpeVocab::from_gguf` of the embed GGUF): `encode_qwen2`, then `<|endoftext|>` appended.

Without `--embed-gguf`, the route answers 404 with a JSON error ("embeddings are not enabled on this server"). `/v1/models` lists the embedding model (`--embed-model-name`, default `qwen3-embedding-4b`) next to the chat model.

## 8. Tokenizer

`pre = qwen2` is Laguna's `laguna_qwen2_split` (a port of llama.cpp's `unicode_regex_split_custom_qwen2`) over the whole text, **without** Laguna's newline pre-split. New: `qwen2_pre_tokenize`, `BpeVocab::encode_qwen2`, and an `encode_auto` arm for `"qwen2"`. `"laguna"` and the joyai default are unchanged.

Known divergences from HF:

- **No NFC normalization.** It needs a Unicode-tables crate, which is a sign-off question.
- **Special-token text in the input is not split out.** `<|endoftext|>` typed by a client is byte-pair encoded as text.
- `\p{L}` / `\p{N}` are approximated by `char::is_alphabetic` / `is_numeric`, as Laguna does.

Gate E0 measures the damage on the fixture corpus.

## 9. Code layout

| Where | What |
|---|---|
| `v4flash-core/src/tokenizer.rs` | `qwen2_pre_tokenize`, `encode_qwen2`, `encode_auto` arm |
| `v4flash-core/src/qwen3_embed.rs` | `Qwen3EmbedConfig` + tensor layout from GGUF metadata and dims (validates arch, shapes, Q8_0), ring-slot layout, host pooling (`output_norm`, L2, MRL), and a layer-streamed **f32 CPU reference forward** (the oracle). Host tests on a synthetic tiny Qwen3 GGUF written with `gguf_write`. |
| `v4flash-kernels/kernels/qwen3_embed.hip` + `src/qwen3_embed.rs` | The six kernels, `Qwen3EmbedKernels`, `EmbedForward` (one phase's forward over a `Loan`) |
| `v4flash-kernels/src/dgpu_loan.rs` | `Loan`: donor ranges, sub-allocation, image write at startup, return + verify. Generic over `DeviceBuffer<u8>` ranges; it knows nothing about Qwen. |
| `deepstrix-server/src/embed_phase.rs` | `EmbedQueue`, `EmbedJob`, startup (`EmbedCtx`: GGUF, vocab, config, kernels, loan), `run_embed_phase` (the tick body), duty-cycle state. Not `embed.rs`: that module is the token-embedding lookup. |
| `deepstrix-server/src/openai/embeddings.rs` | Types + handler |
| `main.rs`, `engine_worker.rs`, `multistream.rs` | `--embed-gguf`, `--embed-model-name`; `EngineRequest::EmbedWake`; `EngineHandle.embed`; the tick hook; the idle check |
| `deepstrix-server/src/knobs.rs` | `V41_EMBED_*` (below) |
| `het/evtrace_kinds.rs` | `HUB_EMBED` (id 24) |

Knobs. All are static except `V41_EMBED_MAX_SHARE` and `V41_EMBED_VERIFY`, which are live: a change between two phases is safe.

| Knob | Default | Meaning |
|---|---|---|
| `V41_EMBED_PHASE_TOKENS` | 16384 | max tokens per phase (sizes the loan) |
| `V41_EMBED_SUB_ROWS` | 1024 | rows per sub-batch (sizes the loan) |
| `V41_EMBED_MAX_INPUT_TOKENS` | 8192 | per input (≤ phase tokens) |
| `V41_EMBED_MAX_REQUEST_TOKENS` | 262144 | per request |
| `V41_EMBED_QUEUE_INPUTS` | 4096 | queued inputs before 503 |
| `V41_EMBED_MAX_SHARE` | 50 | % of wall time embed phases may take while the LLM has work |
| `V41_EMBED_VERIFY` | 1 | verify the return against the image hashes |
| `V41_EMBED_KEEP_PINNED` | 0 | keep the pinned ring across phases |
| `V41_EMBED_LOAN_IMAGE` | `$HOME/.cache/deepstrix/embed-loan.img` | image path |

**Telemetry.** One `ms.embed` info line and one `hub_embed` evtrace record per phase. Fields: `t`, `jobs`, `inputs`, `tokens`, `loan_bytes`, `wait_ms` (oldest input's queue time), `rows_ms` (token-embedding rows), `stream_ms` (reader busy), `compute_ms`, `return_ms`, `verify_ms`, `total_ms`, `live`, `prefills`, `pinned_ms` (pinned alloc).

## 10. Gates

All thresholds are pre-registered here. Each reports the **minimum** cosine over inputs, not the mean.

| Gate | What | Pass |
|---|---|---|
| **E0** tokenizer | `encode_qwen2` + EOS vs HF `AutoTokenizer` ids on the fixture corpus (~64 inputs: queries with instruction, documents, code, CJK, emoji, mixed whitespace, long) | identical ids on every input, or each divergence explained by §8 |
| **E1** CPU oracle | Rust CPU reference on the real Q8_0 GGUF vs HF bf16 reference embeddings (`scripts/qwen3_embed/ref_embed.py`) | min cos ≥ 0.998 (Q8_0 quantization budget; revisit with the measured distribution) |
| **E2** GPU forward | `EmbedForward` on the real GGUF vs the CPU oracle, same inputs | min cos ≥ 0.9999. Plus each kernel vs CPU at our shapes (32/8 heads, hd 128, `q_offset > 0`, multi-segment sub-batches). |
| **E3** loan integrity | Hub with `V41_EMBED_VERIFY=1`: decode oracle tokens with embed phases interleaved vs without (temp 0, private port) | bit-identical tokens; zero verify mismatches |
| **E4** e2e | `/v1/embeddings` on a private port: shapes, `dimensions`, `base64`, 503 on full queue, concurrent chat; phase timing logged | correct responses; `ms.embed` timings recorded |

E2–E4 need a GPU window (the hub down: today the dGPU has ~170 MiB free) and the weights. E0–E1 need only the weights. The synthetic-GGUF host tests run anywhere.

## 11. Risks and open questions

1. **Phase cost is not measured.** NVMe read rate on box 1 and pinned-alloc cost decide it. Q4_K_M would halve the stream but needs the quantized GEMM's K ≤ 4096 cap lifted and costs quality: a later lever.
2. **The head donor's size.** If `global.output` is smaller than the loan, the per-layer fallbacks join, with one more image range each. Startup logs the donor list.
3. **Duty-cycle default.** 50% is a guess. The owner may want a lower share or a minimum gap.
4. **Verify cost** (~0.2–0.3 s per phase) while `V41_EMBED_VERIFY=1`.
5. **NFC.** If E0 shows divergence on real inputs, NFC needs a crate (sign-off).
6. **Instruction format** is the client's job, as with vLLM / llama-server. A server-side `instruction` extension is possible, but it is not OpenAI.
