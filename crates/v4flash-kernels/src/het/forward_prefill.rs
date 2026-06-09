//! Layer-major batched prefill.
//!
//! Two production entry points:
//!
//! * `forward_prompt_batch_v2` — single-lane batched prefill. Stateless
//!   matmuls + HC stages run in single B-wide launches against
//!   [`BatchDgpuScratch`] (B-extended per-token buffers); stateful kernels
//!   (rope, KV append, compressor, attention, iGPU MoE) stay in a serial
//!   inner loop using `DeviceBuffer::slice_view` per batch element.
//! * `forward_prefill_pipelined` — two-lane pipelined wrapper around v2;
//!   alternates lane A / lane B so iGPU MoE on chunk N overlaps with
//!   dGPU work on chunk N+1. Used by `deepstrix-chat`.
//!
//! ## Layer-major vs token-major
//!
//! Both produce identical state (per-layer KV cache + compressor are
//! commutative across batch elements only because layer N's per-position
//! state for token b at position pos0+b is only ever written by that
//! one call). Layer-major lets batched kernels amortize per-layer
//! weight reads across the batch.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::DeviceBuffer;

use crate::config::{
    BLOCKS_N_EMBD, BLOCKS_N_FF_SHARED, BLOCKS_N_LORA_Q, BLOCKS_OUT_LOW, EXPERT_WEIGHT_SCALE,
    GROUP_DIM, HC_DIM, HC_MIX_DIM, INDEXER_COMP_WIDTH, INDEXER_TOP_K, N_EMBD, N_EXPERT,
    N_EXPERT_USED, N_FF_SHARED, N_GROUPS, N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD,
    N_INDEXER_HEAD_DIM, N_LAYER, N_LORA_Q, N_ROT, N_VOCAB, OUT_LOW, Q_FLAT, RANK, RMS_EPS,
    SINKHORN_EPS, SINKHORN_ITERS, SWA_WINDOW,
};
use crate::attention::ATTN_MIXED_MAX_KEYS;
use crate::routing::hash_router_select;

use super::batch_scratch::{BatchDgpuScratch, BatchIgpuScratch, B_MAX};
use super::engine::HeterogeneousEngine;
use super::prefill_stats::PrefillStats;
use super::scratch::{DgpuScratch, IgpuScratch};
use super::state::{HetLayerState, HetModelState, KV_CACHE_ROWS};
use super::sync::{peer_push_f32, peer_push_i32};
use super::weights::{DgpuLayerWeights, HetModelWeights, IgpuLayerWeights};

const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;

impl HeterogeneousEngine {
    /// Layer-major batched prefill using batched kernels.
    ///
    /// Reads `input_hcs[i]` = layer-0 input HC for token `i`, broadcast of
    /// `embed(tokens[i])` to HC_DIM. `tokens[i]` is the token id at
    /// position `pos0 + i` (used by the hash router on bootstrap layers).
    ///
    /// Stateless big matmuls + HC stages run in single B-wide kernel
    /// launches against `batch_dgpu` (B-extended contiguous buffers).
    /// Stateful per-token kernels (rope, kv_append, compressor, attn,
    /// iGPU MoE) loop in a serial inner B loop using `slice_view`.
    ///
    /// After return, `batch_dgpu.residual` (or `residual_next` if the
    /// model layer count is odd — V4-Flash has 43, so `residual` after
    /// post-loop swap holds it) contains per-token post-last-layer HC.
    /// Does NOT compute logits / head — the caller picks which batch
    /// element(s) to feed `forward_head`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prompt_batch_v2(
        &self,
        batch_dgpu: &mut BatchDgpuScratch,
        batch_igpu: &mut BatchIgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        mut stats: Option<&mut PrefillStats>,
    ) -> eyre::Result<()> {
        let b = tokens.len();
        if b == 0 {
            return Ok(());
        }
        if input_hcs.len() != b {
            return Err(eyre!(
                "forward_prompt_batch_v2: input_hcs len {} != tokens len {b}",
                input_hcs.len()
            ));
        }
        for (i, hc) in input_hcs.iter().enumerate() {
            if hc.len() != HC_DIM as usize {
                return Err(eyre!(
                    "forward_prompt_batch_v2: input_hcs[{i}] len {} != HC_DIM {}",
                    hc.len(),
                    HC_DIM
                ));
            }
        }

        self.current_device
            .store(-1, std::sync::atomic::Ordering::Relaxed);
        self.set_current_cached(self.dgpu.device)?;

        // 1. Seed per-token residual buffers in `batch_dgpu.residual`.
        //    `residual` is laid out [B, HC_DIM] contiguous. Each token's
        //    input HC is copied into its slot.
        for i in 0..b {
            let mut slot = batch_dgpu
                .residual
                .slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(&input_hcs[i])?;
        }

        // 1b. Upload pos_per_b = [pos0, pos0+1, ..., pos0+B-1] once per chunk.
        //     Used by batched rope kernels in Stages 2/3/6. Constant across
        //     layers, so uploaded outside the layer loop.
        {
            let pos_host: Vec<i32> = (0..b).map(|i| (pos0 + i as u32) as i32).collect();
            let mut pos_v = batch_dgpu.pos_per_b.slice_view_mut(0, b);
            pos_v.copy_from_host_async(&pos_host, &self.dgpu.compute)?;
        }

        // 2. Layer loop: invoke forward_layer_batch_v2 once per layer.
        //    Each call swaps residual / residual_next internally (we do
        //    the swap here for clarity, mirroring forward_token's per-
        //    layer swap).
        for layer in 0..N_LAYER as usize {
            self.forward_layer_batch_v2(
                batch_dgpu,
                batch_igpu,
                &mut state.layers[layer],
                &weights.dgpu_layers[layer],
                &weights.igpu_layers[layer],
                pos0,
                tokens,
                stats.as_deref_mut(),
            )?;
            // Swap residual / residual_next for the next layer: the
            // layer wrote residual_next; next layer reads residual.
            std::mem::swap(&mut batch_dgpu.residual, &mut batch_dgpu.residual_next);
        }

        // Drain any pending async work.
        self.dgpu.compute.synchronize()?;
        Ok(())
    }

    /// Two-lane pipelined version of `forward_prompt_batch_v2`. Splits the
    /// chunk into lane A (first ceil(B/2) tokens) and lane B (the rest)
    /// and interleaves them on the layer loop so:
    ///   per-layer order on de.compute = pre_A stages, pre_B stages,
    ///                                   ffn_combine_A, ffn_combine_B
    ///   per-layer order on ie.compute = q8k+group+wis+iq2+q2k_A,
    ///                                   q8k+group+wis+iq2+q2k_B
    /// The cross-lane dependency that matters is KV writes: lane B's
    /// attention at layer L reads KV slots lane A wrote at the same
    /// layer L. Both lanes share de.compute (FIFO), and pre_A is queued
    /// before pre_B, so lane A's kv_chain/kv_append always sequences
    /// before lane B's attn — no event needed.
    ///
    /// Lane A uses `self.sync_events`, lane B uses `self.sync_events_t1`
    /// (both pre-allocated at engine construction for the decode pair
    /// path; we reuse them here).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prompt_batch_v2_pipelined(
        &self,
        bd_a: &mut BatchDgpuScratch,
        bi_a: &mut BatchIgpuScratch,
        bd_b: &mut BatchDgpuScratch,
        bi_b: &mut BatchIgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        stats: Option<&mut PrefillStats>,
    ) -> eyre::Result<()> {
        let b = tokens.len();
        if b == 0 {
            return Ok(());
        }
        if input_hcs.len() != b {
            return Err(eyre!(
                "forward_prompt_batch_v2_pipelined: input_hcs len {} != tokens len {b}",
                input_hcs.len()
            ));
        }
        // For chunks too small to bother pipelining, fall back to single-lane.
        if b < 2 {
            return self.forward_prompt_batch_v2(
                bd_a, bi_a, state, weights, input_hcs, tokens, pos0, stats,
            );
        }
        let b_a = b.div_ceil(2);
        let b_b = b - b_a;
        let tokens_a = &tokens[..b_a];
        let tokens_b = &tokens[b_a..];
        let input_a = &input_hcs[..b_a];
        let input_b = &input_hcs[b_a..];
        let pos0_a = pos0;
        let pos0_b = pos0 + b_a as u32;

        self.current_device
            .store(-1, std::sync::atomic::Ordering::Relaxed);
        self.set_current_cached(self.dgpu.device)?;

        for i in 0..b_a {
            let mut slot = bd_a
                .residual
                .slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(&input_a[i])?;
        }
        for i in 0..b_b {
            let mut slot = bd_b
                .residual
                .slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(&input_b[i])?;
        }
        {
            let pos_a: Vec<i32> = (0..b_a).map(|i| (pos0_a + i as u32) as i32).collect();
            let mut va = bd_a.pos_per_b.slice_view_mut(0, b_a);
            va.copy_from_host_async(&pos_a, &self.dgpu.compute)?;
            let pos_b: Vec<i32> = (0..b_b).map(|i| (pos0_b + i as u32) as i32).collect();
            let mut vb = bd_b.pos_per_b.slice_view_mut(0, b_b);
            vb.copy_from_host_async(&pos_b, &self.dgpu.compute)?;
        }

        // Stats: only lane A collects, to avoid a double mutable borrow of
        // PrefillStats inside the loop. Pipelined mode is for perf benches,
        // not for stats collection — if a caller wants per-batch picks,
        // they should use the single-lane forward_prompt_batch_v2.
        let mut stats_a = stats;

        // Deep pipeline: queue lane A's L+1 pre-MoE immediately after lane A's
        // L post-MoE (NOT after lane B's L post-MoE). On the dgpu.compute FIFO
        // this means lane A's next layer can start as soon as lane A's MoE is
        // back, regardless of how long lane B's MoE still has to run.
        //
        // Stream order in steady state:
        //   ... [wait moe_A(L)] post_A(L) pre_A(L+1) [wait moe_B(L)] post_B(L) pre_B(L+1) ...
        //
        // vs. the shallow version (which we replaced):
        //   ... [wait moe_A(L)] post_A(L) [wait moe_B(L)] post_B(L) pre_A(L+1) pre_B(L+1) ...
        // — the shallow version stalled pre_A(L+1) behind moe_arrived_B(L).

        // Warmup: queue layer 0 pre-MoE for both lanes.
        let layer0 = 0usize;
        self.forward_layer_pre_moe_v2(
            bd_a,
            bi_a,
            &mut state.layers[layer0],
            &weights.dgpu_layers[layer0],
            &weights.igpu_layers[layer0],
            pos0_a,
            tokens_a,
            stats_a.as_deref_mut(),
            &self.sync_events.layers[layer0],
        )?;
        self.forward_layer_pre_moe_v2(
            bd_b,
            bi_b,
            &mut state.layers[layer0],
            &weights.dgpu_layers[layer0],
            &weights.igpu_layers[layer0],
            pos0_b,
            tokens_b,
            None,
            &self.sync_events_t1.layers[layer0],
        )?;

        // Steady state: for each layer L in 0..N_LAYER-1, queue post_X(L)
        // followed by pre_X(L+1) for the SAME lane, before moving to lane B.
        for layer in 0..(N_LAYER as usize - 1) {
            let sev_a_cur = &self.sync_events.layers[layer];
            let sev_b_cur = &self.sync_events_t1.layers[layer];

            // Lane A: finish layer L, then start layer L+1.
            self.forward_layer_post_moe_v2(bd_a, b_a as u32, sev_a_cur)?;
            std::mem::swap(&mut bd_a.residual, &mut bd_a.residual_next);
            self.forward_layer_pre_moe_v2(
                bd_a,
                bi_a,
                &mut state.layers[layer + 1],
                &weights.dgpu_layers[layer + 1],
                &weights.igpu_layers[layer + 1],
                pos0_a,
                tokens_a,
                stats_a.as_deref_mut(),
                &self.sync_events.layers[layer + 1],
            )?;

            // Lane B: same.
            self.forward_layer_post_moe_v2(bd_b, b_b as u32, sev_b_cur)?;
            std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);
            self.forward_layer_pre_moe_v2(
                bd_b,
                bi_b,
                &mut state.layers[layer + 1],
                &weights.dgpu_layers[layer + 1],
                &weights.igpu_layers[layer + 1],
                pos0_b,
                tokens_b,
                None,
                &self.sync_events_t1.layers[layer + 1],
            )?;
        }

        // Cooldown: post-MoE for the final layer on both lanes.
        let last = N_LAYER as usize - 1;
        self.forward_layer_post_moe_v2(bd_a, b_a as u32, &self.sync_events.layers[last])?;
        std::mem::swap(&mut bd_a.residual, &mut bd_a.residual_next);
        self.forward_layer_post_moe_v2(bd_b, b_b as u32, &self.sync_events_t1.layers[last])?;
        std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);

        self.dgpu.compute.synchronize()?;
        Ok(())
    }

    /// Chunked prefill driver. Processes `tokens` (length T)
    /// through the v2 batched pipeline in chunks of CHUNK_SIZE=B_MAX. State
    /// carries across chunks via `state.layers[*].{kv_cache,n_raw,compressor}`
    /// (the per-layer fields just keep growing — no special handling).
    ///
    /// Returns logits:
    /// * `last_only=true`: `[N_VOCAB]` for the last token only — typical
    ///   generation start path. Each per-token head is ~16 ms; skipping
    ///   all but the last saves ~T × head_cost wall on the prefill.
    /// * `last_only=false`: `[T × N_VOCAB]` — full per-token logits, for
    ///   prompt-eval / log-prob scoring.
    ///
    /// `head_scratch` is a single-token `DgpuScratch` used for the head
    /// matvec — the head buffers (`head_flat`, `head_pre`, …, `logits`)
    /// aren't B-extended in `BatchDgpuScratch` yet. Per-token head is fast
    /// enough that this isn't on the critical path for `last_only=true`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        head_scratch: &mut DgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        last_only: bool,
        mut stats: Option<&mut PrefillStats>,
    ) -> eyre::Result<Vec<f32>> {
        let t = tokens.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        if input_hcs.len() != t {
            return Err(eyre!(
                "forward_prefill: input_hcs len {} != tokens len {t}",
                input_hcs.len()
            ));
        }
        let chunk_size = B_MAX;
        let cs_hc = HC_DIM as usize;
        let cs_vocab = N_VOCAB as usize;

        let mut out_logits: Vec<f32> = if last_only {
            Vec::with_capacity(cs_vocab)
        } else {
            Vec::with_capacity(t * cs_vocab)
        };

        let mut chunk_start = 0usize;
        while chunk_start < t {
            let chunk_end = (chunk_start + chunk_size).min(t);
            let chunk_b = chunk_end - chunk_start;
            let is_last_chunk = chunk_end == t;
            let chunk_input = &input_hcs[chunk_start..chunk_end];
            let chunk_tokens = &tokens[chunk_start..chunk_end];
            let chunk_pos0 = pos0 + chunk_start as u32;

            // Reset event pools at chunk start (mirrors decode's per-token cycle).
            self.dgpu.events.reset();
            self.igpu.events.reset();

            self.forward_prompt_batch_v2(
                bd,
                bi,
                state,
                weights,
                chunk_input,
                chunk_tokens,
                chunk_pos0,
                stats.as_deref_mut(),
            )?;

            // After this chunk: if perfetto is attached, emit slices + re-anchor.
            if let Some(exp_lock) = &self.perfetto {
                let mut exp = exp_lock.lock().unwrap();
                self.dgpu.events.for_each_pair(|name, s, e| {
                    let track = if name.contains(".xfer") || name.contains(".peer_push") {
                        &exp.dgpu_xfer
                    } else {
                        &exp.dgpu_compute
                    };
                    exp.emit_slice(track, name, s, e)
                })?;
                self.igpu.events.for_each_pair(|name, s, e| {
                    let track = if name.contains(".xfer") || name.contains(".peer_push") {
                        &exp.igpu_xfer
                    } else {
                        &exp.igpu_compute
                    };
                    exp.emit_slice(track, name, s, e)
                })?;
                exp.re_anchor(
                    self.dgpu.device,
                    &self.dgpu.compute,
                    &self.dgpu.xfer,
                    self.igpu.device,
                    &self.igpu.compute,
                    &self.igpu.xfer,
                )?;
                self.current_device.store(-1, std::sync::atomic::Ordering::Relaxed);
            }

            // residual post-loop holds layer-N output in bd.residual (43
            // layers + 43 swaps = even number of mutations to residual).
            if last_only {
                if is_last_chunk {
                    let last_b = chunk_b - 1;
                    head_scratch.residual.copy_from_buffer(
                        &bd.residual.slice_view(last_b * cs_hc, cs_hc),
                    )?;
                    self.forward_head(head_scratch, &weights.global)?;
                    let mut logits = vec![0f32; cs_vocab];
                    head_scratch.logits.copy_to_host(&mut logits)?;
                    out_logits = logits;
                }
            } else {
                for i in 0..chunk_b {
                    head_scratch
                        .residual
                        .copy_from_buffer(&bd.residual.slice_view(i * cs_hc, cs_hc))?;
                    self.forward_head(head_scratch, &weights.global)?;
                    let mut logits = vec![0f32; cs_vocab];
                    head_scratch.logits.copy_to_host(&mut logits)?;
                    out_logits.extend_from_slice(&logits);
                }
            }

            chunk_start = chunk_end;
        }
        Ok(out_logits)
    }

    /// Two-lane pipelined chunked prefill. Same contract as
    /// `forward_prefill` but takes two BatchDgpu/BatchIgpu scratch sets
    /// (one per lane) and calls `forward_prompt_batch_v2_pipelined`.
    /// For `last_only`, the last token of each chunk lives in lane B if
    /// `chunk_b > 1`, otherwise in lane A.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_pipelined(
        &self,
        bd_a: &mut BatchDgpuScratch,
        bi_a: &mut BatchIgpuScratch,
        bd_b: &mut BatchDgpuScratch,
        bi_b: &mut BatchIgpuScratch,
        head_scratch: &mut DgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        last_only: bool,
        mut stats: Option<&mut PrefillStats>,
        cancel: Option<&std::sync::atomic::AtomicBool>,
        // Called after each prefill chunk completes (after the head
        // pass for last_only=true, after the per-token logits copy for
        // last_only=false). Used by the server to pet its forward-
        // progress watchdog — the chunk-grain matters because long-ctx
        // chunks can take many seconds and per-call petting would
        // false-fire the watchdog mid-chunk.
        on_chunk_done: Option<&dyn Fn()>,
    ) -> eyre::Result<Vec<f32>> {
        let t = tokens.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        if input_hcs.len() != t {
            return Err(eyre!(
                "forward_prefill_pipelined: input_hcs len {} != tokens len {t}",
                input_hcs.len()
            ));
        }
        let chunk_size = B_MAX;
        let cs_hc = HC_DIM as usize;
        let cs_vocab = N_VOCAB as usize;

        let mut out_logits: Vec<f32> = if last_only {
            Vec::with_capacity(cs_vocab)
        } else {
            Vec::with_capacity(t * cs_vocab)
        };

        // Progress log: long cold prefills can take many minutes;
        // emit a heartbeat every ~16 chunks so the operator can see
        // forward progress instead of guessing whether the engine is
        // stuck. Wall-clock + chunk index lets you extrapolate ETA.
        let prefill_start = std::time::Instant::now();
        let total_chunks = t.div_ceil(chunk_size);
        let mut chunk_idx = 0usize;
        let mut chunk_start = 0usize;
        while chunk_start < t {
            // Caller-driven cancel (typically: HTTP client disconnect).
            // Checked at chunk boundary so latency is bounded by one
            // chunk's wall-clock — ~hundreds of ms at long ctx, cheap
            // at short. Returning an empty Vec is fine for the
            // server's `last_only=true` path; the caller knows to
            // discard the result when the cancel bool is set.
            if let Some(c) = cancel {
                if c.load(std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!(
                        chunk_idx,
                        tokens_done = chunk_start,
                        tokens_total = t,
                        "prefill cancelled by caller"
                    );
                    return Ok(Vec::new());
                }
            }
            let chunk_end = (chunk_start + chunk_size).min(t);
            let chunk_b = chunk_end - chunk_start;
            let is_last_chunk = chunk_end == t;
            let chunk_input = &input_hcs[chunk_start..chunk_end];
            let chunk_tokens = &tokens[chunk_start..chunk_end];
            let chunk_pos0 = pos0 + chunk_start as u32;

            if chunk_idx == 0 || chunk_idx % 16 == 0 || is_last_chunk {
                let elapsed_s = prefill_start.elapsed().as_secs_f32();
                let toks_per_s = if elapsed_s > 0.001 {
                    chunk_start as f32 / elapsed_s
                } else {
                    0.0
                };
                let eta_s = if toks_per_s > 1.0 {
                    (t - chunk_start) as f32 / toks_per_s
                } else {
                    -1.0
                };
                tracing::warn!(
                    chunk = chunk_idx,
                    total_chunks,
                    chunk_pos0,
                    tokens_done = chunk_start,
                    tokens_total = t,
                    elapsed_s = format!("{elapsed_s:.1}"),
                    tok_per_s = format!("{toks_per_s:.1}"),
                    eta_s = format!("{eta_s:.1}"),
                    "prefill_progress"
                );
            }

            self.dgpu.events.reset();
            self.igpu.events.reset();

            self.forward_prompt_batch_v2_pipelined(
                bd_a,
                bi_a,
                bd_b,
                bi_b,
                state,
                weights,
                chunk_input,
                chunk_tokens,
                chunk_pos0,
                stats.as_deref_mut(),
            )?;

            if let Some(exp_lock) = &self.perfetto {
                let mut exp = exp_lock.lock().unwrap();
                self.dgpu.events.for_each_pair(|name, s, e| {
                    let track = if name.contains(".xfer") || name.contains(".peer_push") {
                        &exp.dgpu_xfer
                    } else {
                        &exp.dgpu_compute
                    };
                    exp.emit_slice(track, name, s, e)
                })?;
                self.igpu.events.for_each_pair(|name, s, e| {
                    let track = if name.contains(".xfer") || name.contains(".peer_push") {
                        &exp.igpu_xfer
                    } else {
                        &exp.igpu_compute
                    };
                    exp.emit_slice(track, name, s, e)
                })?;
                exp.re_anchor(
                    self.dgpu.device,
                    &self.dgpu.compute,
                    &self.dgpu.xfer,
                    self.igpu.device,
                    &self.igpu.compute,
                    &self.igpu.xfer,
                )?;
                self.current_device.store(-1, std::sync::atomic::Ordering::Relaxed);
            }

            // Split point mirrors forward_prompt_batch_v2_pipelined:
            // first ceil(b/2) tokens → lane A, rest → lane B.
            let b_a = chunk_b.div_ceil(2);
            let b_b = chunk_b - b_a;

            if last_only {
                if is_last_chunk {
                    // Last token: lives in lane B if b_b > 0, else lane A.
                    let (src_bd, last_idx) = if b_b > 0 {
                        (&*bd_b, b_b - 1)
                    } else {
                        (&*bd_a, b_a - 1)
                    };
                    head_scratch
                        .residual
                        .copy_from_buffer(&src_bd.residual.slice_view(last_idx * cs_hc, cs_hc))?;
                    self.forward_head(head_scratch, &weights.global)?;
                    let mut logits = vec![0f32; cs_vocab];
                    head_scratch.logits.copy_to_host(&mut logits)?;
                    out_logits = logits;
                }
            } else {
                for i in 0..b_a {
                    head_scratch
                        .residual
                        .copy_from_buffer(&bd_a.residual.slice_view(i * cs_hc, cs_hc))?;
                    self.forward_head(head_scratch, &weights.global)?;
                    let mut logits = vec![0f32; cs_vocab];
                    head_scratch.logits.copy_to_host(&mut logits)?;
                    out_logits.extend_from_slice(&logits);
                }
                for i in 0..b_b {
                    head_scratch
                        .residual
                        .copy_from_buffer(&bd_b.residual.slice_view(i * cs_hc, cs_hc))?;
                    self.forward_head(head_scratch, &weights.global)?;
                    let mut logits = vec![0f32; cs_vocab];
                    head_scratch.logits.copy_to_host(&mut logits)?;
                    out_logits.extend_from_slice(&logits);
                }
            }

            chunk_start = chunk_end;
            chunk_idx += 1;
            if let Some(f) = on_chunk_done {
                f();
            }
        }
        Ok(out_logits)
    }
}

impl HeterogeneousEngine {
    /// One layer of batched prefill, all phases. Reads
    /// `batch_dgpu.residual` (per-token input HC), writes
    /// `batch_dgpu.residual_next` (per-token output HC). All other
    /// `batch_dgpu` fields are scratch. Thin wrapper over the split
    /// pre-MoE + post-MoE methods; single-lane callers use this.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer_batch_v2(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos0: u32,
        tokens: &[i32],
        stats: Option<&mut PrefillStats>,
    ) -> eyre::Result<()> {
        let layer = dlw.layer_idx as usize;
        let b = tokens.len() as u32;
        if b == 0 {
            return Ok(());
        }
        let sev = &self.sync_events.layers[layer];
        self.forward_layer_pre_moe_v2(bd, bi, ls, dlw, ilw, pos0, tokens, stats, sev)?;
        self.forward_layer_post_moe_v2(bd, b, sev)?;
        Ok(())
    }

    /// Pre-MoE phase of one prefill layer. Submits dGPU stages 1-10
    /// (attn + router + shared expert), the dGPU→iGPU peer push, the
    /// iGPU MoE chain (iq2 + q2k_down), and the iGPU→dGPU peer push of
    /// the MoE output. Records `sev.moe_arrived` once the MoE result has
    /// landed on dGPU. Does NOT queue `ffn_combine` — the caller drives
    /// that via `forward_layer_post_moe_v2`, which lets two lanes share
    /// the de.compute stream without ffn_combine serializing the second
    /// lane's pre-MoE work behind the first lane's ffn_combine.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer_pre_moe_v2(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos0: u32,
        tokens: &[i32],
        stats: Option<&mut PrefillStats>,
        sev: &super::engine::LayerSyncEvents,
    ) -> eyre::Result<()> {
        let layer = dlw.layer_idx;
        if ilw.layer_idx != layer {
            return Err(eyre!(
                "forward_layer_pre_moe_v2: dgpu L{} != igpu L{}",
                layer,
                ilw.layer_idx
            ));
        }
        let ratio = dlw.ratio;
        let b = tokens.len() as u32;
        if b == 0 {
            return Ok(());
        }

        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;
        let cs_n_embd = N_EMBD as usize;
        let cs_qflat = Q_FLAT as usize;
        let cs_kvhd = N_HEAD_DIM as usize;
        let cs_n_used = N_EXPERT_USED;

        // ========================================================
        // Stage 1: mhc_pre_attn (BATCHED)
        // rms_nw → f16_narrow → sinkhorn → hc_weighted → rms_w
        // ========================================================
        let _t_mhc_pre = de.events.stage("dgpu.mhc_pre_attn", &de.compute)?;
        {
            let _t = de.events.stage("k.mhc_pre_attn.rms_nw", &de.compute)?;
            de.rms_nw
                .launch_batched(&de.compute, &mut bd.flat, &bd.residual, 1, HC_DIM, RMS_EPS, b)?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.f16_matvec", &de.compute)?;
            de.f16.matvec_narrow_batched(
                &de.compute,
                &mut bd.mix,
                &dlw.hc_attn_fn.buffer,
                &bd.flat,
                HC_MIX_DIM,
                HC_DIM,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.sinkhorn", &de.compute)?;
            de.hc_sinkhorn.launch_batched(
                &de.compute,
                &mut bd.split,
                &bd.mix,
                &dlw.hc_attn_scale,
                &dlw.hc_attn_base,
                N_HC,
                SINKHORN_ITERS,
                SINKHORN_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.hc_weighted", &de.compute)?;
            de.hc_weighted.launch_batched(
                &de.compute,
                &mut bd.attn_cur,
                &bd.residual,
                &bd.split,
                N_EMBD,
                N_HC,
                HC_MIX_DIM, // w_stride: split is [B, HC_MIX_DIM]; pre-sigmoid w is first n_hc
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.rms_w", &de.compute)?;
            de.rms_w.launch_weighted_batched(
                &de.compute,
                &mut bd.attn_input_norm,
                &bd.attn_cur,
                &dlw.attn_norm,
                N_EMBD,
                RMS_EPS,
                b,
            )?;
        }

        drop(_t_mhc_pre);

        // ========================================================
        // Stage 2: Q chain (BATCHED quantize + matvec + rms + ...)
        // ========================================================
        let _t_q = de.events.stage("dgpu.q_chain", &de.compute)?;
        {
            let _t = de.events.stage("k.q_chain.quantize_input", &de.compute)?;
            de.q8.quantize_input_batched(
                &de.compute,
                &mut bd.xq_n_embd,
                &mut bd.xscale_n_embd,
                &bd.attn_input_norm,
                N_EMBD,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.qa_matvec", &de.compute)?;
            // LDS-tiled WMMA GEMM: matvec_batched re-reads weight per
            // batch (weight-BW-bound at B=512). GEMM shares weight across
            // BN=64 batch cols.
            de.q8_wmma.gemm_lds_tiled(
                &de.compute,
                &mut bd.qr,
                &dlw.attn_q_a.buffer,
                &bd.xq_n_embd,
                &bd.xscale_n_embd,
                N_LORA_Q,
                N_EMBD,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.rms_w", &de.compute)?;
            de.rms_w.launch_weighted_batched(
                &de.compute,
                &mut bd.qr_normed,
                &bd.qr,
                &dlw.q_a_norm,
                N_LORA_Q,
                RMS_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.quantize_qr", &de.compute)?;
            de.q8.quantize_input_batched(
                &de.compute,
                &mut bd.qr_xq,
                &mut bd.qr_xscale,
                &bd.qr_normed,
                N_LORA_Q,
                b,
            )?;
        }
        // qb up-projection (M=Q_FLAT=32768, K=N_LORA_Q=1024). Default
        // LDS-tiled WMMA: cooperative-load A+B into LDS per K-outer iter,
        // then WMMA from LDS — kills the s_wait_loadcnt latency throttle
        // that capped both dp4a and the older non-tiled WMMA. Isolated A/B
        // at B=512: dp4a 8.82ms / wmma_old 4.24ms / wmma_lds_tiled 1.38ms
        // → 6.4× over dp4a, 3.1× over the older WMMA. Q_FLAT % 64 == 0 ✓.
        // QB_WMMA=wmma forces the older non-tiled WMMA; QB_WMMA=0 forces dp4a.
        let qb_variant = std::env::var("QB_WMMA").unwrap_or_else(|_| "lds_tiled".into());
        match qb_variant.as_str() {
            "0" | "dp4a" => {
                let _t = de.events.stage("k.q_chain.qb_matvec", &de.compute)?;
                de.q8.matvec_batched(
                    &de.compute, &mut bd.q, &dlw.attn_q_b.buffer,
                    &bd.qr_xq, &bd.qr_xscale, Q_FLAT, N_LORA_Q, b,
                )?;
            }
            "wmma" => {
                let _t = de.events.stage("k.q_chain.qb_wmma", &de.compute)?;
                de.q8_wmma.gemm(
                    &de.compute, &mut bd.q, &dlw.attn_q_b.buffer,
                    &bd.qr_xq, &bd.qr_xscale, Q_FLAT, N_LORA_Q, b,
                )?;
            }
            _ => {
                let _t = de.events.stage("k.q_chain.qb_lds", &de.compute)?;
                de.q8_wmma.gemm_lds_tiled(
                    &de.compute, &mut bd.q, &dlw.attn_q_b.buffer,
                    &bd.qr_xq, &bd.qr_xscale, Q_FLAT, N_LORA_Q, b,
                )?;
            }
        }
        {
            let _t = de.events.stage("k.q_chain.rms_nw_heads", &de.compute)?;
            // rms_nw over batch: each batch has [N_HEAD, N_HEAD_DIM] rows.
            // batched API: grid (B, N_HEAD, 1), inner row of N_HEAD_DIM.
            de.rms_nw.launch_batched(
                &de.compute,
                &mut bd.q_normed,
                &bd.q,
                N_HEAD,
                N_HEAD_DIM,
                RMS_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.rope", &de.compute)?;
            let pos_v = bd.pos_per_b.slice_view(0, b as usize);
            de.rope.launch_forward_batched(
                &de.compute,
                &mut bd.q_normed,
                &pos_v,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                b,
                &dlw.rope_params,
            )?;
        }

        drop(_t_q);

        // ========================================================
        // Stage 3: KV chain (BATCHED matvec + rms; per-token rope/fp8/f16rt)
        // ========================================================
        let _t_kv = de.events.stage("dgpu.kv_chain", &de.compute)?;
        {
            let _t = de.events.stage("k.kv_chain.matvec", &de.compute)?;
            de.q8_wmma.gemm_lds_tiled(
                &de.compute,
                &mut bd.kv_raw,
                &dlw.attn_kv.buffer,
                &bd.xq_n_embd,
                &bd.xscale_n_embd,
                N_HEAD_DIM,
                N_EMBD,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.kv_chain.rms_w", &de.compute)?;
            de.rms_w.launch_weighted_batched(
                &de.compute,
                &mut bd.kv_normed,
                &bd.kv_raw,
                &dlw.kv_a_norm,
                N_HEAD_DIM,
                RMS_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.kv_chain.rope", &de.compute)?;
            let pos_v = bd.pos_per_b.slice_view(0, b as usize);
            de.rope.launch_forward_batched(
                &de.compute,
                &mut bd.kv_normed,
                &pos_v,
                1,
                N_HEAD_DIM,
                N_ROT,
                b,
                &dlw.rope_params,
            )?;
        }
        {
            let _t = de.events.stage("k.kv_chain.fp8", &de.compute)?;
            de.fp8.launch_batched(
                &de.compute,
                &mut bd.kv_normed,
                N_HEAD_DIM - N_ROT,
                N_HEAD_DIM,
                b,
            )?;
        }
        {
            // f16rt is pure elementwise — stretch n by B for a single launch.
            let _t = de.events.stage("k.kv_chain.f16rt", &de.compute)?;
            de.f16rt.launch(&de.compute, &mut bd.kv_normed, b * N_HEAD_DIM)?;
        }

        // ========================================================
        // Stage 4: KV cache append + compressor (SERIAL per batch)
        //
        // We capture per-batch n_raw_after / n_comp_after snapshots so
        // Stage 5 attention can use causal prefix lengths instead of the
        // final post-loop values (which would let token i attend to
        // future tokens i+1..B-1).
        // ========================================================
        drop(_t_kv);
        let _t_kv_append_comp = de.events.stage("dgpu.kv_append_compressor_serial", &de.compute)?;
        let mut n_raw_after: Vec<u32> = Vec::with_capacity(b as usize);
        let mut n_comp_after: Vec<u32> = Vec::with_capacity(b as usize);

        // Oversized cache + per-token offset. The cache (ls.kv_cache) is
        // sized SWA_WINDOW + B_MAX rows so a prefill chunk can write its full
        // batch into slots [n_raw_before .. n_raw_before + b) WITHOUT evicting
        // any prior content. Attention's `n_raw_offset_per[i]` tells the
        // kernel where each token's causally-valid window begins in the cache:
        //
        //   absolute_position p_i  = chunk_pos0 + i  (lives at cache slot
        //                            n_raw_before + i)
        //   causal window         = [max(0, p_i - W + 1) .. p_i]
        //   in cache slots         = [max(0, n_raw_before+i+1-W) .. n_raw_before+i+1)
        //
        // So n_raw_per[i] = min(n_raw_before+i+1, W) and
        //    n_raw_offset_per[i] = max(0, n_raw_before+i+1 - W).
        //
        // After the chunk's attention runs, an explicit eviction pass copies
        // the last SWA_WINDOW rows back to slots [0..W) and resets ls.n_raw
        // to W (or less, for short prompts) so the steady-state SWA invariant
        // holds for decode and for the next chunk.
        let n_raw_before = ls.n_raw;
        let mut n_raw_offset_after: Vec<u32> = Vec::with_capacity(b as usize);
        for i in 0..b as usize {
            let causal_end = n_raw_before + i as u32 + 1; // exclusive upper slot
            let n_per = causal_end.min(SWA_WINDOW);
            let offset = causal_end.saturating_sub(SWA_WINDOW);
            n_raw_after.push(n_per);
            n_raw_offset_after.push(offset);
        }
        // The cache is oversized to SWA_WINDOW + B_MAX rows, so we always
        // use the no-eviction batched append. The post-chunk eviction pass at
        // the end of this layer (after attention) copies the last SWA_WINDOW
        // rows down to slots [0..W) and updates ls.n_raw, restoring the
        // steady-state SWA invariant before the next chunk or decode.
        //
        // The cache bound is tight (no margin): n_raw_before ≤ SWA_WINDOW and
        // b ≤ B_MAX must hold, otherwise launch_batched will OOB-write into
        // the next layer's KV allocation. Guard in debug builds.
        debug_assert!(
            (n_raw_before + b) as usize <= KV_CACHE_ROWS,
            "kv_append OOB: n_raw_before={n_raw_before} + b={b} > KV_CACHE_ROWS={}",
            KV_CACHE_ROWS,
        );
        de.kv_append.launch_batched(
            &de.compute,
            &mut ls.kv_cache,
            &bd.kv_normed,
            n_raw_before,
            N_HEAD_DIM,
            b,
        )?;
        // Cache now holds n_raw_before + b rows; attention will index with
        // n_raw_offset_per. ls.n_raw is updated to its post-eviction value
        // at the END of this layer (see the eviction-down pass).
        let n_raw_during_chunk = n_raw_before + b;

        // Batched matvec_pair across all B for ratio>0 layers. Produces
        // bd.kv_cur[B, comp_width] + bd.sc_cur[B, comp_width] in one launch.
        // The per-token loop below just READS from those buffers.
        if ratio > 0 {
            let cw = dlw
                .compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing compressor weights"))?;
            let comp_width = cw.width;
            de.f16.matvec_pair_batched(
                &de.compute,
                &mut bd.kv_cur,
                &mut bd.sc_cur,
                &cw.wkv.buffer,
                &cw.wgate.buffer,
                &bd.attn_input_norm,
                comp_width,
                N_EMBD,
                b,
            )?;
        }

        // Per-segment batched state_write. Each segment is ≤ `ratio`
        // positions long; within a segment, state_writes go to distinct
        // rows (rows {pos_mod_start..pos_mod_start+seg_len}) so they're
        // safely batched. Segments are bounded by compressor boundaries
        // (where pool+shuffle fire serially) or the chunk end.
        if ratio > 0 {
            let cw = dlw
                .compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing compressor weights"))?;
            let comp_width = cw.width;

            // Precompute per-b (row, pos_mod) and upload once.
            let row_host: Vec<i32> = (0..b)
                .map(|i| {
                    let pos = pos0 + i;
                    let pm = pos % ratio;
                    let row = if ratio == 4 { 4 + pm } else { pm };
                    row as i32
                })
                .collect();
            let pos_mod_host: Vec<i32> =
                (0..b).map(|i| ((pos0 + i) % ratio) as i32).collect();
            {
                let mut row_v = bd.row_per_b.slice_view_mut(0, b as usize);
                row_v.copy_from_host_async(&row_host, &de.compute)?;
                let mut pm_v = bd.pos_mod_per_b.slice_view_mut(0, b as usize);
                pm_v.copy_from_host_async(&pos_mod_host, &de.compute)?;
            }

            let cs = ls
                .compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{layer}: missing compressor state"))?;
            // Per-boundary snapshot scratch slot size for THIS compressor.
            let coff_main: u32 = if ratio == 4 { 2 } else { 1 };
            let snap_elems = (coff_main * ratio * comp_width) as usize;
            let n_comp_start = cs.n_comp;
            let mut pos_per_boundary_host: Vec<i32> = Vec::new();

            let mut i: u32 = 0;
            while i < b {
                let pos_mod_now = (pos0 + i) % ratio;
                let seg_len = std::cmp::min(ratio - pos_mod_now, b - i);
                let seg_end = i + seg_len;

                // Batched state_write for this segment.
                let comp_stride = comp_width as usize;
                let kv_seg = bd.kv_cur.slice_view(
                    (i as usize) * comp_stride,
                    (seg_len as usize) * comp_stride,
                );
                let sc_seg = bd.sc_cur.slice_view(
                    (i as usize) * comp_stride,
                    (seg_len as usize) * comp_stride,
                );
                let row_seg = bd.row_per_b.slice_view(i as usize, seg_len as usize);
                let pm_seg = bd.pos_mod_per_b.slice_view(i as usize, seg_len as usize);
                de.compressor_state_write.launch_batched(
                    &de.compute,
                    &mut cs.state_kv,
                    &mut cs.state_score,
                    &kv_seg,
                    &sc_seg,
                    &cw.ape.buffer,
                    &row_seg,
                    &pm_seg,
                    comp_width,
                    seg_len,
                )?;

                // Boundary fire? Snapshot state for batched post-pass at
                // end-of-chunk. Shuffle still runs immediately so the
                // NEXT segment's state_write sees correct "old" rows.
                let comp_fires = (pos0 + seg_end) % ratio == 0;
                if comp_fires {
                    let k = pos_per_boundary_host.len();
                    let snap_off = k * snap_elems;
                    let mut snap_kv = bd
                        .comp_state_kv_snapshots
                        .slice_view_mut(snap_off, snap_elems);
                    let mut snap_sc = bd
                        .comp_state_score_snapshots
                        .slice_view_mut(snap_off, snap_elems);
                    de.compressor_state_snapshot.launch(
                        &de.compute,
                        &mut snap_kv,
                        &mut snap_sc,
                        &cs.state_kv,
                        &cs.state_score,
                        snap_elems as u32,
                    )?;
                    if ratio == 4 {
                        de.compressor_shuffle.launch(
                            &de.compute,
                            &mut cs.state_kv,
                            &mut cs.state_score,
                            comp_width,
                        )?;
                    }
                    pos_per_boundary_host.push((pos0 + seg_end - ratio) as i32);
                    cs.n_comp += 1;
                }

                // n_comp_after semantics: for token at pos = pos0+k, value
                // reflects cs.n_comp AFTER processing that position. If the
                // boundary fires at the end of this segment, only the LAST
                // position sees the post-fire n_comp; earlier positions see pre-fire.
                let post_fire = cs.n_comp;
                let pre_fire = if comp_fires { post_fire - 1 } else { post_fire };
                for k in i..seg_end {
                    let snap = if comp_fires && k == seg_end - 1 {
                        post_fire
                    } else {
                        pre_fire
                    };
                    n_comp_after.push(snap);
                }
                i = seg_end;
            }

            // Batched per-boundary stages: pool → rms_w → rope → fp8 →
            // f16rt → comp_kv_append. Replaces what used to be 6 launches
            // per boundary × ~128 boundaries × 21+ layers in the per-token
            // serial loop (~200ms of launch overhead per chunk).
            let n_boundaries = pos_per_boundary_host.len() as u32;
            if n_boundaries > 0 {
                de.compressor_pool.launch_batched(
                    &de.compute,
                    &mut bd.comp_pooled_batched,
                    &bd.comp_state_kv_snapshots,
                    &bd.comp_state_score_snapshots,
                    N_HEAD_DIM,
                    ratio,
                    n_boundaries,
                )?;
                de.rms_w.launch_weighted_batched(
                    &de.compute,
                    &mut bd.comp_rows_batched,
                    &bd.comp_pooled_batched,
                    &cw.norm,
                    N_HEAD_DIM,
                    RMS_EPS,
                    n_boundaries,
                )?;
                {
                    let mut pv = bd
                        .comp_pos_per_boundary
                        .slice_view_mut(0, n_boundaries as usize);
                    pv.copy_from_host_async(&pos_per_boundary_host, &de.compute)?;
                }
                {
                    let pos_v = bd
                        .comp_pos_per_boundary
                        .slice_view(0, n_boundaries as usize);
                    de.rope.launch_forward_batched(
                        &de.compute,
                        &mut bd.comp_rows_batched,
                        &pos_v,
                        1,
                        N_HEAD_DIM,
                        N_ROT,
                        n_boundaries,
                        &dlw.rope_params,
                    )?;
                }
                de.fp8.launch_batched(
                    &de.compute,
                    &mut bd.comp_rows_batched,
                    N_HEAD_DIM - N_ROT,
                    N_HEAD_DIM,
                    n_boundaries,
                )?;
                de.f16rt.launch(
                    &de.compute,
                    &mut bd.comp_rows_batched,
                    n_boundaries * N_HEAD_DIM,
                )?;
                de.comp_kv_append.launch_batched(
                    &de.compute,
                    &mut cs.comp_kv,
                    &bd.comp_rows_batched,
                    n_comp_start,
                    N_HEAD_DIM,
                    n_boundaries,
                )?;
            }
        } else {
            for _ in 0..b {
                n_comp_after.push(0);
            }
        }

        // ========================================================
        // CSA indexer compressor — second compressor at head_dim=128,
        // only on ratio==4 layers. Same shape as the main compressor's
        // batched-matvec_pair + per-segment-state_write + on-fire-serial-ops
        // pattern, with:
        //   - head_dim = N_INDEXER_HEAD_DIM (128) vs N_HEAD_DIM (512)
        //   - width    = INDEXER_COMP_WIDTH  (256) vs main's (1024)
        //   - NO FP8 step (only valid at head_dim=512 per ds4.c:6702)
        // Scratch (bd.kv_cur, bd.sc_cur, bd.pooled, bd.comp_row) is
        // reused via slice views — main compressor block has already
        // consumed its writes by this point on the same compute stream.
        // row_per_b / pos_mod_per_b reuse — ratio=4 row formula is the
        // same as main's at ratio=4 (`4 + pm`).
        // ics.n_comp tracks identically to cs.n_comp at ratio==4 layers
        // (both fire on the same boundaries) — we don't need a separate
        // n_index_comp_after vector; downstream (Phase 5 mask) can reuse
        // n_comp_after.
        if ratio == 4 {
            let iw = dlw
                .indexer_compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing indexer_compressor weights"))?;
            let ics = ls
                .indexer_compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{layer}: missing indexer_compressor state"))?;
            let icw = INDEXER_COMP_WIDTH; // 256
            let ihd = N_INDEXER_HEAD_DIM; // 128

            // Batched matvec_pair across all B → bd.kv_cur[B,icw] +
            // bd.sc_cur[B,icw] (head of buffers, slice view).
            {
                let mut kv_view =
                    bd.kv_cur.slice_view_mut(0, (b as usize) * (icw as usize));
                let mut sc_view =
                    bd.sc_cur.slice_view_mut(0, (b as usize) * (icw as usize));
                de.f16.matvec_pair_batched(
                    &de.compute,
                    &mut kv_view,
                    &mut sc_view,
                    &iw.wkv.buffer,
                    &iw.wgate.buffer,
                    &bd.attn_input_norm,
                    icw,
                    N_EMBD,
                    b,
                )?;
            }

            // Same snapshot+batched pattern as main compressor above.
            // ratio==4, no fp8 (head_dim=128 != 512).
            let coff_idx: u32 = 2; // ratio==4 ⇒ coff = 2
            let snap_elems_idx = (coff_idx * ratio * icw) as usize;
            let n_idx_comp_start = ics.n_comp;
            let mut pos_per_boundary_idx: Vec<i32> = Vec::new();

            let mut i: u32 = 0;
            while i < b {
                let pos_mod_now = (pos0 + i) % ratio;
                let seg_len = std::cmp::min(ratio - pos_mod_now, b - i);
                let seg_end = i + seg_len;

                let comp_stride = icw as usize;
                let kv_seg = bd.kv_cur.slice_view(
                    (i as usize) * comp_stride,
                    (seg_len as usize) * comp_stride,
                );
                let sc_seg = bd.sc_cur.slice_view(
                    (i as usize) * comp_stride,
                    (seg_len as usize) * comp_stride,
                );
                let row_seg = bd.row_per_b.slice_view(i as usize, seg_len as usize);
                let pm_seg = bd.pos_mod_per_b.slice_view(i as usize, seg_len as usize);
                de.compressor_state_write.launch_batched(
                    &de.compute,
                    &mut ics.state_kv,
                    &mut ics.state_score,
                    &kv_seg,
                    &sc_seg,
                    &iw.ape.buffer,
                    &row_seg,
                    &pm_seg,
                    icw,
                    seg_len,
                )?;

                let comp_fires = (pos0 + seg_end) % ratio == 0;
                if comp_fires {
                    let k = pos_per_boundary_idx.len();
                    let snap_off = k * snap_elems_idx;
                    let mut snap_kv = bd
                        .comp_state_kv_snapshots
                        .slice_view_mut(snap_off, snap_elems_idx);
                    let mut snap_sc = bd
                        .comp_state_score_snapshots
                        .slice_view_mut(snap_off, snap_elems_idx);
                    de.compressor_state_snapshot.launch(
                        &de.compute,
                        &mut snap_kv,
                        &mut snap_sc,
                        &ics.state_kv,
                        &ics.state_score,
                        snap_elems_idx as u32,
                    )?;
                    de.compressor_shuffle.launch(
                        &de.compute,
                        &mut ics.state_kv,
                        &mut ics.state_score,
                        icw,
                    )?;
                    pos_per_boundary_idx.push((pos0 + seg_end - ratio) as i32);
                    ics.n_comp += 1;
                }
                i = seg_end;
            }

            // Batched post-stages for the indexer compressor (no fp8).
            let n_idx_boundaries = pos_per_boundary_idx.len() as u32;
            if n_idx_boundaries > 0 {
                de.compressor_pool.launch_batched(
                    &de.compute,
                    &mut bd.comp_pooled_batched,
                    &bd.comp_state_kv_snapshots,
                    &bd.comp_state_score_snapshots,
                    ihd,
                    ratio,
                    n_idx_boundaries,
                )?;
                de.rms_w.launch_weighted_batched(
                    &de.compute,
                    &mut bd.comp_rows_batched,
                    &bd.comp_pooled_batched,
                    &iw.norm,
                    ihd,
                    RMS_EPS,
                    n_idx_boundaries,
                )?;
                {
                    let mut pv = bd
                        .comp_pos_per_boundary
                        .slice_view_mut(0, n_idx_boundaries as usize);
                    pv.copy_from_host_async(&pos_per_boundary_idx, &de.compute)?;
                }
                {
                    let pos_v = bd
                        .comp_pos_per_boundary
                        .slice_view(0, n_idx_boundaries as usize);
                    de.rope.launch_forward_batched(
                        &de.compute,
                        &mut bd.comp_rows_batched,
                        &pos_v,
                        1,
                        ihd,
                        N_ROT,
                        n_idx_boundaries,
                        &dlw.rope_params,
                    )?;
                }
                de.f16rt.launch(
                    &de.compute,
                    &mut bd.comp_rows_batched,
                    n_idx_boundaries * ihd,
                )?;
                de.comp_kv_append.launch_batched(
                    &de.compute,
                    &mut ics.comp_kv,
                    &bd.comp_rows_batched,
                    n_idx_comp_start,
                    ihd,
                    n_idx_boundaries,
                )?;
            }
        }
        drop(_t_kv_append_comp);

        // ========================================================
        // Stage 5: Attention (BATCHED — grid (n_head, B, 1))
        //
        // Causal: each token i attends to KV prefix [0..n_raw_after[i]]
        // and comp_kv prefix [0..n_comp_after[i]]. Per-token prefix
        // lengths live in bd.n_raw_per / bd.n_comp_per device buffers
        // (uploaded fresh per layer from the host snapshots captured in
        // Stage 4).
        // ========================================================
        let n_raw_per_host: Vec<i32> = n_raw_after.iter().map(|&v| v as i32).collect();
        let n_raw_offset_per_host: Vec<i32> =
            n_raw_offset_after.iter().map(|&v| v as i32).collect();
        let n_comp_per_host: Vec<i32> =
            n_comp_after.iter().map(|&v| v as i32).collect();
        let _t_attn = de.events.stage("dgpu.attn_compute", &de.compute)?;
        // Async copies on de.compute so they FIFO with the subsequent
        // attention launch. Avoids the bulk-sync that copy_from_host
        // would impose (~5us each blocks the host AND fences the device).
        {
            let mut nrp_v = bd.n_raw_per.slice_view_mut(0, b as usize);
            nrp_v.copy_from_host_async(&n_raw_per_host, &de.compute)?;
        }
        {
            let mut nrop_v = bd.n_raw_offset_per.slice_view_mut(0, b as usize);
            nrop_v.copy_from_host_async(&n_raw_offset_per_host, &de.compute)?;
        }
        {
            let mut ncp_v = bd.n_comp_per.slice_view_mut(0, b as usize);
            ncp_v.copy_from_host_async(&n_comp_per_host, &de.compute)?;
        }
        let nrp_view = bd.n_raw_per.slice_view(0, b as usize);
        let nrop_view = bd.n_raw_offset_per.slice_view(0, b as usize);
        let ncp_view = bd.n_comp_per.slice_view(0, b as usize);
        if ratio == 0 {
            let _t = de.events.stage("k.attn.swa", &de.compute)?;
            de.attn_swa.launch_batched(
                &de.compute,
                &mut bd.heads,
                &bd.q_normed,
                &ls.kv_cache,
                &dlw.attn_sinks,
                &nrp_view,
                &nrop_view,
                N_HEAD,
                N_HEAD_DIM,
                b,
            )?;
        } else {
            let cs = ls.compressor.as_ref();
            let any_comp = n_comp_after.iter().any(|&v| v > 0);
            let comp_kv_buf = if any_comp { cs.map(|c| &c.comp_kv) } else { None };
            let n_total_max = n_raw_after
                .iter()
                .zip(n_comp_after.iter())
                .map(|(&r, &c)| r + c)
                .max()
                .unwrap_or(0);
            // Always use the batched split (scores in global, per-row grid,
            // wave-parallel softmax, 16-way ILP wsum). The old monolithic
            // `launch_batched` (LDS scores[2304], used below n_total≤2304) was
            // 12.5× SLOWER at the same B=256 shape — one WG per (head,token)
            // with everything serialized in LDS tanks occupancy — so there is
            // no depth where it wins for batched prefill. It survives only as
            // the correctness oracle in prefill_attention_split_matches_mono.
            // Head-tiled score: one WG per (row, head-group, token) loads the
            // shared MLA latent row once and reuses it across the head group.
            // Score Q·Kᵀ runs as the RDNA4 f16 WMMA GEMM — measured 7.3× over
            // the f32 head-tiled variant on the depth-32k ratio=4 layer.
            // Scores live in DRAM as f16 (the score kernel writes f16,
            // smwsum reads f16). Halves the scores-buffer DRAM round-trip
            // for free; Phase A softmax keeps f32 math. The bd.attn_scores
            // scratch is half-sized accordingly — there is no production
            // f32-scores path. ATTN_FUSED=1 takes the fused-FlashAttention
            // single-kernel path (online softmax, no scores buffer); skips
            // both score and smwsum below.
            // ============================================================
            // CSA indexer mask (per-token, ratio==4 only).
            //
            // For each batch token b at ratio==4 with n_index_comp_per[b] >
            // INDEXER_TOP_K, run matvec(attn_q_b) → RoPE → matvec(proj) →
            // scale → IndexerScore → IndexerTopk → bitpack into a per-
            // token slice of bd.attn_comp_allowed_bits. The score kernel
            // below then sees the mask via its nullable comp_allowed_bits
            // param and stamps -INF for masked comp rows; softmax
            // converts to zero weight.
            //
            // For tokens with n_index_comp_per[b] ≤ INDEXER_TOP_K (early-
            // permit) we set an all-1s mask — leaving the score kernel
            // bit-exact with the pre-mask dense path for that token.
            //
            // For ratio==128 layers and tokens with n_index_comp==0 we
            // pass `None` for the whole-batch mask — the kernel skips
            // the bit test entirely. ratio==128 layers will hit this
            // path because indexer_compressor is None there, so
            // n_index_comp stays 0.
            //
            // The whole pipeline runs serial-per-token on the same
            // compute stream. Cost dominates IndexerScore at long ctx
            // (~70 µs/token at n_index_comp=16K), times B tokens per
            // chunk → maybe ~36 ms added per chunk at B=512, depth 32K.
            // Acceptable per the phase 5 perf budget.
            let need_mask = ratio == 4
                && ls.indexer_compressor.is_some()
                && n_comp_after.iter().any(|&v| v > INDEXER_TOP_K);
            let indexer_fired = if need_mask {
                let _t_ix = de.events.stage("dgpu.prefill_indexer", &de.compute)?;
                let iw = dlw.indexer.as_ref().ok_or_else(|| {
                    eyre!("L{layer}: ratio==4 mask needed but no indexer weights")
                })?;
                let ics = ls.indexer_compressor.as_ref().expect("checked above");
                let n_words_per_b = ((ATTN_MIXED_MAX_KEYS + 31) / 32) as usize;
                let scale = 1.0f32
                    / ((N_INDEXER_HEAD_DIM as f32) * (N_INDEXER_HEAD as f32)).sqrt();
                let wmma = de.indexer_score_wmma.as_ref().ok_or_else(|| {
                    eyre!("L{layer}: batched prefill indexer requires gfx12 IndexerScoreWmma")
                })?;

                // Batched indexer pipeline. One launch per stage replaces
                // the per-token serial loop (was 8 launches × B × 21L =
                // 86K launches per chunk — the regression root cause).
                // Tokens with n_idx ≤ INDEXER_TOP_K are handled correctly
                // by the batched topk degenerating to all-valid selection.
                let n_idx_max: u32 = n_comp_after.iter().copied().max().unwrap_or(0);
                {
                    let mut v = bd
                        .n_index_comp_per_b
                        .slice_view_mut(0, b as usize);
                    v.copy_from_host_async(&n_comp_after, &de.compute)?;
                }
                // q[B, N_INDEXER_HEAD * N_INDEXER_HEAD_DIM] = attn_q_b @ qr[B].
                // LDS-tiled WMMA GEMM: weight loaded once per (m0,n0) tile
                // and reused across BN=64 output cols. matvec_batched was
                // weight-BW-bound on per-batch rereads at 156 ms/chunk.
                {
                    let _t = de.events.stage("k.indexer.matvec_q", &de.compute)?;
                    de.f16.gemm_batched_wmma(
                        &de.compute,
                        &mut bd.indexer_q,
                        &iw.attn_q_b.buffer,
                        &bd.qr_normed,
                        N_INDEXER_HEAD * N_INDEXER_HEAD_DIM,
                        N_LORA_Q,
                        b,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer.rope", &de.compute)?;
                    let pos_v = bd.pos_per_b.slice_view(0, b as usize);
                    de.rope.launch_forward_batched(
                        &de.compute,
                        &mut bd.indexer_q,
                        &pos_v,
                        N_INDEXER_HEAD,
                        N_INDEXER_HEAD_DIM,
                        N_ROT,
                        b,
                        &dlw.rope_params,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer.matvec_proj", &de.compute)?;
                    de.f16.matvec_batched(
                        &de.compute,
                        &mut bd.indexer_head_weights,
                        &iw.proj.buffer,
                        &bd.attn_input_norm,
                        N_INDEXER_HEAD,
                        N_EMBD,
                        b,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer.scale", &de.compute)?;
                    de.vec_scale.launch(
                        &de.compute,
                        &mut bd.indexer_head_weights,
                        scale,
                        b * N_INDEXER_HEAD,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer.score_wmma", &de.compute)?;
                    wmma.launch_batched(
                        &de.compute,
                        &mut bd.indexer_scores,
                        &bd.indexer_q,
                        &bd.indexer_head_weights,
                        &ics.comp_kv,
                        &bd.n_index_comp_per_b,
                        n_idx_max,
                        ATTN_MIXED_MAX_KEYS,
                        b,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer.topk_bitonic", &de.compute)?;
                    de.indexer_topk_bitonic.launch_batched(
                        &de.compute,
                        &mut bd.indexer_selected,
                        &mut bd.attn_comp_allowed_bits,
                        &mut bd.indexer_topk_scratch,
                        &bd.indexer_scores,
                        &bd.n_index_comp_per_b,
                        n_idx_max,
                        ATTN_MIXED_MAX_KEYS,
                        n_words_per_b as u32,
                        INDEXER_TOP_K,
                        b,
                    )?;
                }
                // Gather selected rows from the MAIN compressor's comp_kv
                // (head_dim=512) into a dense per-batch buffer. Attention
                // then reads only top-K rows per token instead of doing
                // dense reads with a -INF mask — 15× DRAM/WMMA savings on
                // both score and smwsum at depth 32K.
                {
                    let _t = de.events.stage("k.indexer.gather", &de.compute)?;
                    let cs_ref = ls
                        .compressor
                        .as_ref()
                        .ok_or_else(|| eyre!("L{layer}: missing compressor state for gather"))?;
                    de.indexer_gather.launch_batched(
                        &de.compute,
                        &mut bd.attn_active_comp_kv,
                        &cs_ref.comp_kv,
                        &bd.indexer_selected,
                        INDEXER_TOP_K,
                        N_HEAD_DIM,
                        b,
                    )?;
                }
                // Re-upload sparse n_comp_per (= min(actual, INDEXER_TOP_K))
                // so score+smwsum iterate only over the gathered top-K rows.
                {
                    let sparse_n_comp_host: Vec<i32> = n_comp_after
                        .iter()
                        .map(|&v| v.min(INDEXER_TOP_K) as i32)
                        .collect();
                    let mut ncp_v = bd.n_comp_per.slice_view_mut(0, b as usize);
                    ncp_v.copy_from_host_async(&sparse_n_comp_host, &de.compute)?;
                }
                let _ = pos0; // pos consumed via pos_per_b
                _t_ix.end()?;
                true
            } else {
                false
            };

            // Sparse-attn switch: when indexer fired, attention reads
            // active_comp_kv (per-batch dense top-K) instead of the
            // 8K+-row dense comp_kv. Score+smwsum get a per-batch stride;
            // the bitmask path becomes unused. n_total_max collapses to
            // raw_window + INDEXER_TOP_K.
            let (eff_comp_kv_buf, eff_n_total_max, eff_comp_kv_batch_stride):
                (Option<&DeviceBuffer<u16>>, u32, u32) = if indexer_fired {
                let sparse_max = n_raw_after.iter().zip(n_comp_after.iter())
                    .map(|(&r, &c)| r + c.min(INDEXER_TOP_K))
                    .max().unwrap_or(0);
                (Some(&bd.attn_active_comp_kv), sparse_max, INDEXER_TOP_K)
            } else {
                (comp_kv_buf, n_total_max, 0u32)
            };

            let fused = std::env::var_os("ATTN_FUSED").is_some();
            let f32_scores = super::batch_scratch::use_f32_scores();
            if !fused {
                let _t = de.events.stage("k.attn.score", &de.compute)?;
                if f32_scores {
                    de.attn_mixed.launch_score_batched_htiled_wmma(
                        &de.compute,
                        &mut bd.attn_scores,
                        &bd.q_normed,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        N_HEAD,
                        N_HEAD_DIM,
                        eff_n_total_max,
                        b,
                    )?;
                } else {
                    de.attn_mixed.launch_score_batched_htiled_wmma_f16s(
                        &de.compute,
                        &mut bd.attn_scores,
                        &bd.q_normed,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        None, // gather handles sparsity; no -INF mask needed
                        N_HEAD,
                        N_HEAD_DIM,
                        eff_n_total_max,
                        b,
                        eff_comp_kv_batch_stride,
                    )?;
                }
            }
            // Head-tiled phase 2: softmax one wave per head + WMMA Phase B
            // W·V via `_ldsv_f16s`. LDS-V staging cooperatively loads each
            // K-tile's 16 V rows once (saves 82.8% of the s_wait_loadcnt
            // stalls the DRAM-V variant ate), and f16 scores halve the
            // Phase A score-DRAM round-trip — together −23% over `_ldsv`
            // at depth 32k, B=256 (13.05 → 10.09 ms p50). The score writer
            // upstream must match the chain (both are f16-only here).
            {
                let _t = de.events.stage(
                    if fused { "k.attn.fused" } else { "k.attn.smwsum" },
                    &de.compute,
                )?;
                if fused {
                    de.attn_mixed.launch_fused_wmma(
                        &de.compute,
                        &mut bd.heads,
                        &bd.q_normed,
                        &dlw.attn_sinks,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        N_HEAD,
                        N_HEAD_DIM,
                        eff_n_total_max,
                        b,
                    )?;
                } else if f32_scores {
                    de.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv(
                        &de.compute,
                        &mut bd.heads,
                        &mut bd.attn_scores,
                        &dlw.attn_sinks,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        N_HEAD,
                        N_HEAD_DIM,
                        b,
                    )?;
                } else {
                    de.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(
                        &de.compute,
                        &mut bd.heads,
                        &mut bd.attn_scores,
                        &dlw.attn_sinks,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        N_HEAD,
                        N_HEAD_DIM,
                        b,
                        eff_comp_kv_batch_stride,
                    )?;
                }
            }
        }
        drop(nrp_view);
        drop(nrop_view);
        drop(ncp_view);
        drop(_t_attn);

        // Post-attention eviction. Cache holds n_raw_during_chunk rows
        // (= n_raw_before + b). Compress back to the SWA invariant:
        //   - if n_raw_during_chunk <= SWA_WINDOW: nothing to do, slots
        //     [0..n_raw_during_chunk) are already the steady state. Update
        //     ls.n_raw = n_raw_during_chunk.
        //   - if n_raw_during_chunk > SWA_WINDOW: copy the LAST SWA_WINDOW
        //     rows down to slots [0..SWA_WINDOW). Set ls.n_raw = SWA_WINDOW.
        //
        // The shift may have source/dest overlap when n_raw_during_chunk is
        // between (SWA_WINDOW, 2*SWA_WINDOW), so route through the kv_ring
        // scratch buffer.
        if n_raw_during_chunk > SWA_WINDOW {
            let src_first_slot = n_raw_during_chunk - SWA_WINDOW;
            let head_dim = N_HEAD_DIM as usize;
            let ring_len = (SWA_WINDOW as usize) * head_dim;
            let src_offset = (src_first_slot as usize) * head_dim;
            // scratch = cache[src_first_slot..src_first_slot+SWA_WINDOW)
            {
                let mut ring_v = bd.kv_ring_scratch.slice_view_mut(0, ring_len);
                let src_v = ls.kv_cache.slice_view(src_offset, ring_len);
                ring_v.copy_from_buffer_async(&src_v, &de.compute)?;
            }
            // cache[0..SWA_WINDOW) = scratch
            {
                let ring_src = bd.kv_ring_scratch.slice_view(0, ring_len);
                let mut dst_v = ls.kv_cache.slice_view_mut(0, ring_len);
                dst_v.copy_from_buffer_async(&ring_src, &de.compute)?;
            }
            ls.n_raw = SWA_WINDOW;
        } else {
            ls.n_raw = n_raw_during_chunk;
        }

        // ========================================================
        // Stage 6: Output projection (rope_inv per b, then BATCHED q8)
        // ========================================================
        let _t_out = de.events.stage("dgpu.output_proj", &de.compute)?;
        {
            let _t = de.events.stage("k.output_proj.rope_inverse", &de.compute)?;
            let pos_v = bd.pos_per_b.slice_view(0, b as usize);
            de.rope.launch_inverse_batched(
                &de.compute,
                &mut bd.heads,
                &pos_v,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                b,
                &dlw.rope_params,
            )?;
        }
        {
            let _t = de.events.stage("k.output_proj.quantize_heads", &de.compute)?;
            de.q8.quantize_input_batched(
                &de.compute,
                &mut bd.heads_xq,
                &mut bd.heads_xscale,
                &bd.heads,
                Q_FLAT,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.output_proj.grouped_matvec", &de.compute)?;
            // LDS-tiled WMMA grouped variant. Per-group shape M=RANK=1024,
            // K=GROUP_DIM=4096 is too small for the legacy per-row GEMV to
            // saturate (1 wave/CU occupancy); the LDS-tiled kernel uses
            // 4 warps × 4 WMMA accs per WG = good occupancy even at small
            // M. Isolated A/B per sub-group at B=512: dp4a 0.42ms vs
            // lds_tiled 0.20ms = 2.1× on each, ~2× on the whole grouped
            // call. Q8_GROUPED_VARIANT=dp4a rolls back.
            let grp_variant = std::env::var("Q8_GROUPED_VARIANT")
                .unwrap_or_else(|_| "lds_tiled".into());
            if grp_variant == "dp4a" {
                de.q8_grouped.matvec_grouped_batched(
                    &de.compute, &mut bd.low, &dlw.attn_output_a.buffer,
                    &bd.heads_xq, &bd.heads_xscale,
                    GROUP_DIM, RANK, N_GROUPS, b,
                )?;
            } else {
                de.q8_wmma.gemm_lds_tiled_grouped(
                    &de.compute, &mut bd.low, &dlw.attn_output_a.buffer,
                    &bd.heads_xq, &bd.heads_xscale,
                    GROUP_DIM, RANK, N_GROUPS, b,
                )?;
            }
        }
        {
            let _t = de.events.stage("k.output_proj.quantize_low", &de.compute)?;
            de.q8.quantize_input_batched(
                &de.compute,
                &mut bd.low_xq,
                &mut bd.low_xscale,
                &bd.low,
                OUT_LOW,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.output_proj.matvec_out", &de.compute)?;
            // Same LDS-tiled WMMA the qb path uses. matvec_out shape
            // (M=N_EMBD=4096, K=OUT_LOW=8192) hits the same s_wait_loadcnt
            // throttle on dp4a; LDS-tiled WMMA wins 6.2× at B=512 isolated
            // (8.82 → 1.42 ms). Q8_OUT_VARIANT=dp4a rolls back.
            let out_variant = std::env::var("Q8_OUT_VARIANT").unwrap_or_else(|_| "lds_tiled".into());
            if out_variant == "dp4a" {
                de.q8.matvec_batched(
                    &de.compute, &mut bd.attn_out, &dlw.attn_output_b.buffer,
                    &bd.low_xq, &bd.low_xscale, N_EMBD, OUT_LOW, b,
                )?;
            } else {
                de.q8_wmma.gemm_lds_tiled(
                    &de.compute, &mut bd.attn_out, &dlw.attn_output_b.buffer,
                    &bd.low_xq, &bd.low_xscale, N_EMBD, OUT_LOW, b,
                )?;
            }
        }
        drop(_t_out);

        // ========================================================
        // Stage 7: mhc_post_attn (BATCHED hc_post_from_split)
        // ========================================================
        let _t_mhc_post = de.events.stage("dgpu.mhc_post_attn", &de.compute)?;
        de.hc_post.launch_from_split_batched(
            &de.compute,
            &mut bd.after_attn_hc,
            &bd.attn_out,
            &bd.residual,
            &bd.split,
            N_HC, // n_w (matches single-token path)
            N_EMBD,
            N_HC,
            b,
        )?;
        drop(_t_mhc_post);

        // ========================================================
        // Stage 8: mhc_pre_ffn (BATCHED, same shape as Stage 1)
        // ========================================================
        let _t_mhc_pre_ffn = de.events.stage("dgpu.mhc_pre_ffn", &de.compute)?;
        {
            let _t = de.events.stage("k.mhc_pre_ffn.rms_nw", &de.compute)?;
            de.rms_nw.launch_batched(
                &de.compute,
                &mut bd.flat,
                &bd.after_attn_hc,
                1,
                HC_DIM,
                RMS_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_ffn.f16_matvec", &de.compute)?;
            de.f16.matvec_narrow_batched(
                &de.compute,
                &mut bd.mix,
                &dlw.hc_ffn_fn.buffer,
                &bd.flat,
                HC_MIX_DIM,
                HC_DIM,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_ffn.sinkhorn", &de.compute)?;
            de.hc_sinkhorn.launch_batched(
                &de.compute,
                &mut bd.split,
                &bd.mix,
                &dlw.hc_ffn_scale,
                &dlw.hc_ffn_base,
                N_HC,
                SINKHORN_ITERS,
                SINKHORN_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_ffn.hc_weighted", &de.compute)?;
            de.hc_weighted.launch_batched(
                &de.compute,
                &mut bd.ffn_cur,
                &bd.after_attn_hc,
                &bd.split,
                N_EMBD,
                N_HC,
                HC_MIX_DIM,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_ffn.rms_w", &de.compute)?;
            de.rms_w.launch_weighted_batched(
                &de.compute,
                &mut bd.ffn_input_norm,
                &bd.ffn_cur,
                &dlw.ffn_norm,
                N_EMBD,
                RMS_EPS,
                b,
            )?;
        }
        drop(_t_mhc_pre_ffn);

        // ========================================================
        // Stage 9: Router (per-batch wide matvec to match single-token
        // float reduction order; batched matvec_narrow has different
        // accumulation order which makes topk pick different experts
        // when logits are near the threshold. f16.matvec dispatches to
        // wide when n_rows >= 64 (N_EXPERT=256 ≥ 64); a future wide-
        // batched variant could remove the per-token launch overhead.)
        // ========================================================
        let _t_router = de.events.stage("dgpu.router", &de.compute)?;
        {
            // Gate projection: one wide batched matvec over all B tokens. The
            // per-row warp reduction is identical to the old per-token loop, so
            // logits are bit-identical — only the launch count drops from B to 1.
            let _t = de.events.stage("k.router.f16_matvec", &de.compute)?;
            // GEMM-tile via LDS-WMMA: M=N_EXPERT=256, K=N_EMBD=4096, N=B.
            // matvec_batched re-reads weight per batch — at B=512 that's
            // weight-BW-bound; tile-shares across BN=64.
            de.f16.gemm_batched_wmma(
                &de.compute,
                &mut bd.router_logits,
                &dlw.ffn_gate_inp.buffer,
                &bd.ffn_input_norm,
                N_EXPERT,
                N_EMBD,
                b,
            )?;
        }
        if !dlw.is_hash_router {
            // Top-k: one block per token in a single launch (B→1 launches).
            let _t = de.events.stage("k.router.topk", &de.compute)?;
            de.router_topk.launch_batched(
                &de.compute,
                &mut bd.d_selected,
                &mut bd.d_ew,
                &bd.router_logits,
                dlw.router_bias_dev.as_ref(),
                N_EXPERT,
                cs_n_used as u32,
                EXPERT_WEIGHT_SCALE,
                ROUTER_WEIGHT_EPS,
                b,
            )?;
        } else {
            // Hash router: readback all B × N_EXPERT logits, run host
            // select per batch element, upload d_selected + d_ew.
            de.compute.synchronize()?;
            bd.router_logits
                .copy_to_host(&mut bd.router_logits_host)?;
            let tid2eid = dlw
                .tid2eid
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: hash router but no tid2eid"))?;
            let mut all_sel: Vec<i32> = Vec::with_capacity(b as usize * cs_n_used);
            let mut all_ew: Vec<f32> = Vec::with_capacity(b as usize * cs_n_used);
            for i in 0..b as usize {
                let logit_slice = &bd.router_logits_host
                    [i * (N_EXPERT as usize)..(i + 1) * (N_EXPERT as usize)];
                let (sel, w) = hash_router_select(tid2eid, tokens[i], logit_slice);
                all_sel.extend_from_slice(&sel);
                all_ew.extend_from_slice(&w);
            }
            // d_selected / d_ew are B_MAX-sized; copy into [0..B*N_USED] view.
            let mut sel_v = bd
                .d_selected
                .slice_view_mut(0, b as usize * cs_n_used);
            sel_v.copy_from_host(&all_sel)?;
            let mut ew_v = bd.d_ew.slice_view_mut(0, b as usize * cs_n_used);
            ew_v.copy_from_host(&all_ew)?;
        }
        drop(_t_router);

        // Stats collection (optional). Copies d_selected to host — sync,
        // fences the device. Don't enable in production prefill.
        if let Some(s) = stats {
            de.compute.synchronize()?;
            let mut sel_host = vec![0i32; (b as usize) * cs_n_used];
            bd.d_selected
                .slice_view(0, (b as usize) * cs_n_used)
                .copy_to_host(&mut sel_host)?;
            s.record_batch(layer as usize, &sel_host, b);
        }

        // ========================================================
        // Stage 10: Shared expert (BATCHED Q8_0 chains)
        // swiglu + vec_add are pure elementwise → stretch n by B
        // ========================================================
        let _t_shared = de.events.stage("dgpu.shared_expert", &de.compute)?;
        {
            let _t = de.events.stage("k.shared_expert.quantize_input", &de.compute)?;
            de.q8.quantize_input_batched(
                &de.compute,
                &mut bd.xq_n_embd,
                &mut bd.xscale_n_embd,
                &bd.ffn_input_norm,
                N_EMBD,
                b,
            )?;
        }
        {
            // LDS-tiled WMMA GEMM: weight tile loaded ONCE per (m0,n0)
            // tile and reused across BN=64 batch cols (matvec_batched
            // re-reads weight per batch — weight-BW-bound at B=512).
            let _t = de.events.stage("k.shared_expert.gate_matvec", &de.compute)?;
            de.q8_wmma.gemm_lds_tiled(
                &de.compute,
                &mut bd.gate_sh,
                &dlw.shared.gate.buffer,
                &bd.xq_n_embd,
                &bd.xscale_n_embd,
                N_FF_SHARED,
                N_EMBD,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.shared_expert.up_matvec", &de.compute)?;
            de.q8_wmma.gemm_lds_tiled(
                &de.compute,
                &mut bd.up_sh,
                &dlw.shared.up.buffer,
                &bd.xq_n_embd,
                &bd.xscale_n_embd,
                N_FF_SHARED,
                N_EMBD,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.shared_expert.swiglu", &de.compute)?;
            // swiglu — elementwise; stretch n to B * N_FF_SHARED.
            de.swiglu.launch(
                &de.compute,
                &mut bd.mid_sh,
                &bd.gate_sh,
                &bd.up_sh,
                b * N_FF_SHARED,
            )?;
        }
        {
            let _t = de.events.stage("k.shared_expert.quantize_mid", &de.compute)?;
            de.q8.quantize_input_batched(
                &de.compute,
                &mut bd.mid_sh_xq,
                &mut bd.mid_sh_xscale,
                &bd.mid_sh,
                N_FF_SHARED,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.shared_expert.down_matvec", &de.compute)?;
            de.q8_wmma.gemm_lds_tiled(
                &de.compute,
                &mut bd.ffn_shared,
                &dlw.shared.down.buffer,
                &bd.mid_sh_xq,
                &bd.mid_sh_xscale,
                N_EMBD,
                N_FF_SHARED,
                b,
            )?;
        }
        drop(_t_shared);

        // ========================================================
        // Stage 11: iGPU routed MoE (batched).
        //
        // One peer-push of [B × N_EMBD] ffn_input_norm + [B × N_USED]
        // d_selected/d_ew, one batched iGPU MoE call chain (q8k_xq →
        // iq2_fused_swiglu → q8k_mid → q2k_down with by-expert dispatch),
        // one peer-push of [B × N_EMBD] ffn_moe back.
        // ========================================================
        let gbpe = ilw.routed.gate_bytes_per_expert as u32;
        let ubpe = ilw.routed.up_bytes_per_expert as u32;
        let dbpe = ilw.routed.down_bytes_per_expert as u32;
        let mid_blocks_bytes = (crate::config::BLOCKS_Q8K_DOWN_IN as usize)
            * crate::q8_k::BLOCK_Q8_K_BYTES;
        // Stage 9 router_topk + Stage 10 shared expert wrote bd.d_selected,
        // bd.d_ew, bd.ffn_input_norm on de.compute. We're about to read
        // them from de.xfer. Use the LayerSyncEvents event chain
        // (selected_ready → wait → push → selected_pushed → igpu waits)
        // instead of a host sync, so de.compute is free to keep queuing
        // the next lane's pre-MoE work while xfer/igpu drain this lane.
        self.set_current_cached(self.dgpu.device)?;
        sev.selected_ready.record(&de.compute)?;
        de.xfer.wait_event(&sev.selected_ready)?;
        // Single batched peer-push of all B activations + routing.
        let ain_v = bd
            .ffn_input_norm
            .slice_view(0, (b as usize) * cs_n_embd);
        let dsel_v = bd.d_selected.slice_view(0, (b as usize) * cs_n_used);
        let dew_v = bd.d_ew.slice_view(0, (b as usize) * cs_n_used);
        let mut bi_ain = bi
            .ffn_input_norm_recv
            .slice_view_mut(0, (b as usize) * cs_n_embd);
        let mut bi_sel = bi
            .d_selected
            .slice_view_mut(0, (b as usize) * cs_n_used);
        let mut bi_ew = bi.d_ew.slice_view_mut(0, (b as usize) * cs_n_used);
        {
            let _t_peer_ain = de.events.stage("dgpu.peer_push_ffn_input_norm", &de.xfer)?;
            {
                let _t = de.events.stage("k.peer_push.ain", &de.xfer)?;
                peer_push_f32(&ain_v, &mut bi_ain, &de.xfer)?;
            }
            {
                let _t = de.events.stage("k.peer_push.d_selected", &de.xfer)?;
                peer_push_i32(&dsel_v, &mut bi_sel, &de.xfer)?;
            }
            {
                let _t = de.events.stage("k.peer_push.d_ew", &de.xfer)?;
                peer_push_f32(&dew_v, &mut bi_ew, &de.xfer)?;
            }
            drop(_t_peer_ain);
        }
        sev.selected_pushed.record(&de.xfer)?;
        drop(bi_ain);
        drop(bi_sel);
        drop(bi_ew);

        // Single batched iGPU MoE call chain. iq2 uses by-expert
        // dispatch (group_builder + work_items pre-pass), q2_k stays
        // by-token (could also be by-expert but smaller perf lever).
        self.set_current_cached(self.igpu.device)?;
        let ie = &self.igpu;
        // Wait for the dGPU→iGPU peer-push to land before any iGPU compute
        // reads the recv buffers. Replaces the old de.xfer.synchronize().
        ie.compute.wait_event(&sev.selected_pushed)?;
        // q8k quantize ain[B*N_EMBD] → d_xq_q8k[B*blocks].
        {
            let _t_q8k_pre = ie.events.stage("igpu.q8k_quantize_pre_iq2", &ie.compute)?;
            ie.q8k.launch(
                &ie.compute,
                &mut bi.d_xq_q8k,
                &bi.ffn_input_norm_recv,
                crate::config::BLOCKS_Q8K_GATE_IN * b,
            )?;
        }
        // Chunked by-expert iq2. Three pre-passes then main kernel:
        //   1. moe_group_builder: invert d_selected → group_count + expert_members.
        //   2. moe_work_items_builder: chunk popular groups → work_items + n_work_items.
        //   3. host sync + readback n_work_items to set main kernel grid.y.
        //   4. iq2 chunked main kernel.
        // Chunk size = how many members per WG the iq2/q2k kernels handle.
        // tile8_row32 caps at 8 (ejpir-style block8 unpack); others use 32.
        let variant_peek = std::env::var("IQ2_VARIANT").unwrap_or_else(|_| "staged".into());
        #[allow(non_snake_case)]
        let CHUNK_SIZE: u32 = if variant_peek == "tile8" { 8 } else { 32 };
        let max_per_expert = bi.max_per_expert();
        bi.group_count.fill_zero()?;
        {
            let _t_grp = ie.events.stage("igpu.moe_group_builder", &ie.compute)?;
            let BatchIgpuScratch {
                group_count,
                expert_members,
                d_selected,
                ..
            } = bi;
            ie.moe_group_builder.launch(
                &ie.compute,
                group_count,
                expert_members,
                d_selected,
                b,
                cs_n_used as u32,
                N_EXPERT,
                max_per_expert,
            )?;
        }
        // IQ2_VARIANT env: "staged" (default), "chunked", "hybrid", or "tile8".
        // - hybrid: split work items by chunk size; staged for large, chunked for small.
        // - tile8:  ejpir-port block8 + tile8_row32 (chunk_size auto-set to 8 above).
        // IQ2_HYBRID_THRESHOLD env: chunk-size cutoff (default 8).
        let variant = variant_peek.clone();
        // Carried out to the q2k_down dispatch below; staged/chunked path
        // assigns it inside its else-branch, hybrid path leaves it 0 (which
        // is fine — Q2K_VARIANT=by_expert is forbidden with hybrid below).
        let mut n_work_items: u32 = 0;
        let threshold: u32 = std::env::var("IQ2_HYBRID_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        if variant == "hybrid" {
            // launch_work_items_split atomicAdds into these counters.
            // Their doc-comment promises "pre-zeroed per layer" — honour
            // that here. Without this, the readback at copy_to_host below
            // returns prev_counter+actual_count, the downstream iq2 grid
            // is overstated, and the staged/chunked kernels read past the
            // real work_items[] tail into uninit slots.
            bi.n_staged_work_items.fill_zero()?;
            bi.n_chunked_work_items.fill_zero()?;
            {
                let BatchIgpuScratch {
                    staged_work_items,
                    chunked_work_items,
                    n_staged_work_items,
                    n_chunked_work_items,
                    group_count,
                    ..
                } = bi;
                let _t_wis = ie.events.stage("igpu.moe_work_items_split", &ie.compute)?;
                let max_items = staged_work_items.len() as u32;
                ie.moe_group_builder.launch_work_items_split(
                    &ie.compute,
                    staged_work_items,
                    chunked_work_items,
                    n_staged_work_items,
                    n_chunked_work_items,
                    group_count,
                    N_EXPERT,
                    CHUNK_SIZE,
                    threshold,
                    max_items,
                )?;
            }
            ie.compute.synchronize()?;
            let mut counts = [0i32; 1];
            bi.n_staged_work_items.copy_to_host(&mut counts)?;
            let n_staged = counts[0] as u32;
            bi.n_chunked_work_items.copy_to_host(&mut counts)?;
            let n_chunked = counts[0] as u32;
            {
                let BatchIgpuScratch {
                    d_mid_cat,
                    d_xq_q8k,
                    d_ew,
                    group_count,
                    expert_members,
                    staged_work_items,
                    chunked_work_items,
                    ..
                } = bi;
                if n_staged > 0 {
                    let _t_st = ie.events.stage("igpu.iq2_staged", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_chunked_staged(
                        &ie.compute, d_mid_cat,
                        &ilw.routed.gate.buffer, &ilw.routed.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, staged_work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_staged,
                    )?;
                }
                if n_chunked > 0 {
                    let _t_ch = ie.events.stage("igpu.iq2_chunked", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_chunked(
                        &ie.compute, d_mid_cat,
                        &ilw.routed.gate.buffer, &ilw.routed.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, chunked_work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_chunked,
                    )?;
                }
            }
        } else {
            bi.n_work_items.fill_zero()?;
            {
                let BatchIgpuScratch {
                    work_items,
                    n_work_items,
                    group_count,
                    ..
                } = bi;
                let _t_wi = ie.events.stage("igpu.moe_work_items", &ie.compute)?;
                ie.moe_group_builder.launch_work_items(
                    &ie.compute,
                    work_items,
                    n_work_items,
                    group_count,
                    N_EXPERT,
                    CHUNK_SIZE,
                    work_items.len() as u32,
                )?;
            }
            ie.compute.synchronize()?;
            let mut n_wi_host = [0i32; 1];
            bi.n_work_items.copy_to_host(&mut n_wi_host)?;
            n_work_items = n_wi_host[0] as u32;
            {
                let BatchIgpuScratch {
                    d_mid_cat,
                    d_xq_q8k,
                    d_ew,
                    group_count,
                    expert_members,
                    work_items,
                    ..
                } = bi;
                if variant == "tile8" {
                    let _t_t8 = ie.events.stage("igpu.iq2_tile8", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_tile8_row32(
                        &ie.compute, d_mid_cat,
                        &ilw.routed.gate.buffer, &ilw.routed.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_work_items,
                    )?;
                } else if variant == "staged_v2" {
                    let _t_s2 = ie.events.stage("igpu.iq2_staged_v2", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_chunked_staged_v2(
                        &ie.compute, d_mid_cat,
                        &ilw.routed.gate.buffer, &ilw.routed.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_work_items,
                    )?;
                } else if variant != "chunked" {
                    let _t_st = ie.events.stage("igpu.iq2_staged", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_chunked_staged(
                        &ie.compute, d_mid_cat,
                        &ilw.routed.gate.buffer, &ilw.routed.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_work_items,
                    )?;
                } else {
                    let _t_ch = ie.events.stage("igpu.iq2_chunked", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_chunked(
                        &ie.compute, d_mid_cat,
                        &ilw.routed.gate.buffer, &ilw.routed.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_work_items,
                    )?;
                }
            }
        }
        {
            let _t_q8k_post = ie.events.stage("igpu.q8k_quantize_post_iq2", &ie.compute)?;
            ie.q8k.launch(
                &ie.compute,
                &mut bi.d_midq_cat,
                &bi.d_mid_cat,
                crate::config::BLOCKS_Q8K_DOWN_IN * (cs_n_used as u32) * b,
            )?;
        }
        {
            let _t_q2k = ie.events.stage("igpu.q2k_down", &ie.compute)?;
            // Default `by_expert`: invert (B, expert) iteration so each
            // expert's row-tile is read once instead of B*n_used times.
            // Was: 94% DRAM-BW-bound on redundant weight reads (PMC L2 hit
            // 4%, MemUnitBusy 99.98%). Reuses the iq2 group arrays
            // (group_count / expert_members / work_items) built once per
            // layer by moe_group_builder — no extra pre-pass. Writes per-
            // (b, slot) partials, then a tiny reduce kernel sums across
            // slots to produce out — fully deterministic (no atomicAdd).
            // E2E: +14-23% prefill throughput at B_MAX=512 across all
            // depths (4K..64K); ~flat at B≤64; small-batch regression for
            // pathological B<32 prefills.
            // Q2K_VARIANT=bxn rolls back to the original kernel.
            // Hybrid IQ2 is incompatible (splits work_items into two
            // buckets); error out if both are requested simultaneously.
            let q2k_variant = std::env::var("Q2K_VARIANT")
                .unwrap_or_else(|_| "by_expert".into());
            let use_by_expert = q2k_variant == "by_expert";
            if use_by_expert && variant == "hybrid" {
                return Err(eyre!(
                    "Q2K_VARIANT=by_expert not supported with IQ2_VARIANT=hybrid \
                     (would need to combine staged_+chunked_work_items). \
                     Set Q2K_VARIANT=bxn to opt out."
                ));
            }
            if use_by_expert {
                // Zero partials: unwritten (b, slot) pairs must stay 0 so
                // the reduce-sum is correct. The by_expert kernel only writes
                // the slots in expert_members.
                bi.q2k_partials.fill_zero()?;
                ie.q2k.launch_by_expert(
                    &ie.compute,
                    &mut bi.q2k_partials,
                    &ilw.routed.down.buffer,
                    &bi.d_midq_cat,
                    &bi.group_count,
                    &bi.expert_members,
                    &bi.work_items,
                    dbpe,
                    mid_blocks_bytes as u32,
                    cs_n_used as u32,
                    max_per_expert,
                    CHUNK_SIZE,
                    N_EMBD,
                    crate::config::BLOCKS_Q8K_DOWN_IN,
                    n_work_items,
                )?;
                ie.q2k.launch_reduce_partials(
                    &ie.compute,
                    &mut bi.ffn_moe,
                    &bi.q2k_partials,
                    cs_n_used as u32,
                    N_EMBD,
                    b,
                )?;
            } else {
                ie.q2k.launch_batched_bxn(
                    &ie.compute,
                    &mut bi.ffn_moe,
                    &ilw.routed.down.buffer,
                    &bi.d_midq_cat,
                    &bi.d_selected,
                    dbpe,
                    mid_blocks_bytes as u32,
                    cs_n_used as u32,
                    N_EMBD,
                    crate::config::BLOCKS_Q8K_DOWN_IN,
                    b,
                )?;
            }
        }
        // Record MoE-done so the iGPU xfer can wait without a host sync.
        sev.moe_done.record(&ie.compute)?;
        ie.xfer.wait_event(&sev.moe_done)?;

        // Single batched peer-push back of bi.ffn_moe[B*N_EMBD].
        let bi_ffn_view = bi.ffn_moe.slice_view(0, (b as usize) * cs_n_embd);
        let mut bd_moe_dst = bd
            .ffn_moe_recv
            .slice_view_mut(0, (b as usize) * cs_n_embd);
        {
            let _t_peer_moe = ie.events.stage("igpu.peer_push_ffn_moe", &ie.xfer)?;
            peer_push_f32(&bi_ffn_view, &mut bd_moe_dst, &ie.xfer)?;
        }
        // moe_arrived fires once ffn_moe has landed on dGPU; ffn_combine
        // below waits on it instead of the old ie.xfer.synchronize().
        sev.moe_arrived.record(&ie.xfer)?;
        drop(bi_ffn_view);
        drop(bd_moe_dst);
        self.set_current_cached(self.dgpu.device)?;
        // moe_arrived is the post-MoE handoff event; forward_layer_post_moe_v2
        // queues the wait + ffn_combine. Pre-MoE returns here without
        // touching de.compute again, so the caller can interleave another
        // lane's pre-MoE before queueing this lane's ffn_combine.
        Ok(())
    }

    /// Stage 12: dGPU ffn_combine. Separated from the pre-MoE body so the
    /// pipelined caller can queue both lanes' pre-MoE work on de.compute
    /// before any ffn_combine reads its post-MoE inputs. `sev.moe_arrived`
    /// must be the same LayerSyncEvents passed to the matching pre-MoE call.
    fn forward_layer_post_moe_v2(
        &self,
        bd: &mut BatchDgpuScratch,
        b: u32,
        sev: &super::engine::LayerSyncEvents,
    ) -> eyre::Result<()> {
        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;
        de.compute.wait_event(&sev.moe_arrived)?;
        let _t_combine = de.events.stage("dgpu.ffn_combine", &de.compute)?;
        {
            let _t = de.events.stage("k.ffn_combine.vec_add", &de.compute)?;
            de.vec_add.launch(
                &de.compute,
                &mut bd.ffn_moe_recv,
                &bd.ffn_shared,
                b * N_EMBD,
            )?;
        }
        {
            let _t = de.events.stage("k.ffn_combine.hc_post", &de.compute)?;
            de.hc_post.launch_from_split_batched(
                &de.compute,
                &mut bd.residual_next,
                &bd.ffn_moe_recv,
                &bd.after_attn_hc,
                &bd.split,
                N_HC,
                N_EMBD,
                N_HC,
                b,
            )?;
        }
        Ok(())
    }
}
