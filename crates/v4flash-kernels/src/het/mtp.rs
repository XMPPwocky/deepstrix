//! DSpark drafter: state and the entry projection.
//!
//! The drafter is a 3-layer model (see `docs/v41/DSPARK_DESIGN.md`) run
//! autoregressively to emit K draft tokens, which the main model then verifies in
//! one batched pass. This module holds the state it needs and the ENTRY step:
//! turning the main model's residuals into the drafter's input.
//!
//! Entry, per the reference `forward_spec`:
//!
//!     main_x = main_norm( main_proj( cat( mean_over_hc( residual ENTERING
//!                                         layers 37, 38, 39 ) ) ) )
//!
//! `mean_over_hc` is `hc_weighted` with uniform weights — the same kernel the
//! head uses to collapse the 4 hyper-connection copies, just with 1/N_HC instead
//! of learned weights. `cat` is free: the three means are written into adjacent
//! slices of one `[3 * N_EMBD]` buffer.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, Stream};

use crate::config::{
    GROUP_DIM, N_EMBD, N_GROUPS, N_HC, N_HEAD, N_HEAD_DIM, N_LORA_Q, N_ROT, OUT_LOW, Q_FLAT,
    RANK, RMS_EPS,
};
use crate::het::engine::DeviceEngine;
use crate::het::weights::MtpWeights;

// ---------------------------------------------------------------------------
// Drafter geometry. All from the checkpoint's own config.json (`text_config`)
// and `inference/model.py`, NOT inferred — several of these differ from the main
// model in ways that would be silently wrong if guessed.
// ---------------------------------------------------------------------------

/// `dspark_target_layer_ids` — main-model layers whose ENTERING residual feeds
/// the drafter, in order.
pub const MTP_SRC_LAYERS: [i32; 3] = [37, 38, 39];
/// `dspark_block_size` — draft tokens emitted per speculation step.
pub const MTP_BLOCK: usize = 5;
/// `sliding_window` — the drafter's KV ring depth. It attends over a window of
/// main-model-derived KV, never the full context.
pub const MTP_WINDOW: usize = 128;
/// `dspark_num_experts_per_tok` — top-3 of 128, where the main model is top-6
/// of 384. `router_topk` takes both at runtime, so no kernel change.
pub const MTP_TOPK: u32 = 3;
/// `dspark_markov_rank` — the auxiliary n-gram head's hidden width.
pub const MTP_MARKOV_RANK: usize = 256;
/// `dspark_noise_token_id`.
pub const MTP_NOISE_TOKEN: i32 = 128799;

/// Per-request drafter state. Allocated once; `main_x` is refilled every token.
pub struct MtpState {
    /// `cat(mean_over_hc(residual@37), ..@38, ..@39)` — `[3 * N_EMBD]`.
    pub main_x: DeviceBuffer<f32>,
    /// Uniform `1/N_HC`, so `hc_weighted` computes a mean.
    hc_mean: DeviceBuffer<f32>,
    /// Q8 staging for the `main_proj` matvec — `[3 * N_EMBD]`, distinct from the
    /// per-layer `xq`/`xscale` which are `[N_EMBD]`.
    proj_xq: DeviceBuffer<i8>,
    proj_xscale: DeviceBuffer<f32>,
    /// `main_proj` output and its norm — the drafter's layer-0 input.
    pub proj: DeviceBuffer<f32>,
    pub x: DeviceBuffer<f32>,
    /// Which of `MTP_SRC_LAYERS` have been captured this token. Guards against
    /// running the entry on a stale or partial `main_x`.
    captured: [bool; MTP_SRC_LAYERS.len()],

    // ---- per-layer attention state ----
    /// KV ring per drafter layer: `[MTP_WINDOW + 1] x N_HEAD_DIM` f16.
    ///
    /// One slot wider than the window on purpose. `attn_swa` cannot be used here
    /// (its LDS `scores[ATTN_SWA_MAX_KV]` caps at 128 and this needs 129+, and
    /// this codebase has a documented LDS-occupancy trap around that kernel), so
    /// attention runs through `attn_mixed` raw-only, which needs the keys
    /// CONTIGUOUS. The ring's valid entries are always a prefix
    /// `[0, min(win, start_pos+1))`, so the current token's own KV is written at
    /// index `n_valid` — right after that prefix — and `n_kv = n_valid + 1`.
    pub rings: Vec<DeviceBuffer<u16>>,

    // ---- scratch, shared across the three layers (they run in sequence) ----
    xq: DeviceBuffer<i8>,
    xscale: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    qr_normed: DeviceBuffer<f32>,
    qr_xq: DeviceBuffer<i8>,
    qr_xscale: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    kv_raw: DeviceBuffer<f32>,
    kv_normed: DeviceBuffer<f32>,
    /// Ring index this step's own KV is written at — `n_valid(pos)`. Device-side
    /// because `launch_kv_post_fused` takes the slot on-device.
    slot_dev: DeviceBuffer<u32>,
    scores: DeviceBuffer<f32>,
    heads: DeviceBuffer<f32>,
    heads_xq: DeviceBuffer<i8>,
    heads_xscale: DeviceBuffer<f32>,
    low: DeviceBuffer<f32>,
    low_xq: DeviceBuffer<i8>,
    low_xscale: DeviceBuffer<f32>,
    /// The layer's attention output, `[N_EMBD]`.
    pub attn_out: DeviceBuffer<f32>,
}

impl MtpState {
    pub fn alloc(device_id: i32) -> eyre::Result<Self> {
        let k = MTP_SRC_LAYERS.len() * N_EMBD as usize;
        let mut hc_mean = DeviceBuffer::<f32>::new(device_id, N_HC as usize)?;
        hc_mean.copy_from_host(&vec![1.0f32 / N_HC as f32; N_HC as usize])?;
        let ne = N_EMBD as usize;
        let mut rings = Vec::with_capacity(MTP_SRC_LAYERS.len());
        for _ in 0..MTP_SRC_LAYERS.len() {
            rings.push(DeviceBuffer::<u16>::new(
                device_id,
                (MTP_WINDOW + 1) * N_HEAD_DIM as usize,
            )?);
        }
        Ok(Self {
            main_x: DeviceBuffer::new(device_id, k)?,
            hc_mean,
            proj_xq: DeviceBuffer::new(device_id, k)?,
            proj_xscale: DeviceBuffer::new(device_id, k.div_ceil(32))?,
            proj: DeviceBuffer::new(device_id, ne)?,
            x: DeviceBuffer::new(device_id, ne)?,
            captured: [false; MTP_SRC_LAYERS.len()],
            rings,
            xq: DeviceBuffer::new(device_id, ne)?,
            xscale: DeviceBuffer::new(device_id, ne.div_ceil(32))?,
            qr: DeviceBuffer::new(device_id, N_LORA_Q as usize)?,
            qr_normed: DeviceBuffer::new(device_id, N_LORA_Q as usize)?,
            qr_xq: DeviceBuffer::new(device_id, N_LORA_Q as usize)?,
            qr_xscale: DeviceBuffer::new(device_id, (N_LORA_Q as usize).div_ceil(32))?,
            q: DeviceBuffer::new(device_id, Q_FLAT as usize)?,
            q_normed: DeviceBuffer::new(device_id, Q_FLAT as usize)?,
            kv_raw: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            kv_normed: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            slot_dev: DeviceBuffer::new(device_id, 1)?,
            scores: DeviceBuffer::new(device_id, N_HEAD as usize * (MTP_WINDOW + 1))?,
            heads: DeviceBuffer::new(device_id, Q_FLAT as usize)?,
            heads_xq: DeviceBuffer::new(device_id, Q_FLAT as usize)?,
            heads_xscale: DeviceBuffer::new(device_id, (Q_FLAT as usize).div_ceil(32))?,
            low: DeviceBuffer::new(device_id, OUT_LOW as usize)?,
            low_xq: DeviceBuffer::new(device_id, OUT_LOW as usize)?,
            low_xscale: DeviceBuffer::new(device_id, (OUT_LOW as usize).div_ceil(32))?,
            attn_out: DeviceBuffer::new(device_id, ne)?,
        })
    }

    /// Start a new token: forget last token's captures.
    pub fn begin_token(&mut self) {
        self.captured = [false; MTP_SRC_LAYERS.len()];
    }

    /// Slot for `layer` if it feeds the drafter.
    #[inline]
    pub fn src_slot(layer: i32) -> Option<usize> {
        MTP_SRC_LAYERS.iter().position(|&l| l == layer)
    }

    /// Capture the residual ENTERING `layer`, collapsed over the hc copies.
    ///
    /// Call from the decode path BEFORE the layer runs. A no-op for layers the
    /// drafter does not read, so the caller can invoke it unconditionally.
    pub fn capture(
        &mut self,
        e: &DeviceEngine,
        s: &Stream,
        layer: i32,
        residual: &DeviceBuffer<f32>,
    ) -> eyre::Result<()> {
        let Some(slot) = Self::src_slot(layer) else { return Ok(()) };
        let mut dst = self.main_x.slice_view_mut(slot * N_EMBD as usize, N_EMBD as usize);
        e.hc_weighted.launch(s, &mut dst, residual, &self.hc_mean, N_EMBD, N_HC)?;
        self.captured[slot] = true;
        Ok(())
    }

    /// `main_norm(main_proj(main_x))` — the drafter's layer-0 input.
    ///
    /// Errors if any source layer was missed: running on a partial `main_x`
    /// would silently draft from a stale residual, which is precisely the class
    /// of bug that made DSpark acceptance read 0.44 for weeks.
    pub fn entry(&mut self, e: &DeviceEngine, s: &Stream, w: &MtpWeights) -> eyre::Result<()> {
        if let Some(i) = self.captured.iter().position(|c| !c) {
            return Err(eyre!(
                "mtp entry: residual for layer {} was never captured this token",
                MTP_SRC_LAYERS[i]
            ));
        }
        let k = (MTP_SRC_LAYERS.len() * N_EMBD as usize) as u32;
        crate::het::dispatch::dense_matvec(
            e, s, &mut self.proj, &w.main_proj, &self.main_x, &self.proj_xq, &self.proj_xscale,
            N_EMBD, k,
        )?;
        e.rms_w
            .launch_weighted(s, &mut self.x, &self.proj, &w.main_norm, N_EMBD, RMS_EPS)?;
        Ok(())
    }
}

impl MtpState {
    /// Number of valid ring entries at `pos`, and therefore the index the current
    /// token's own KV is written at. The ring's valid entries are always the
    /// prefix `[0, min(MTP_WINDOW, pos + 1))`.
    #[inline]
    pub fn n_valid(pos: u32) -> usize {
        (MTP_WINDOW).min(pos as usize + 1)
    }

    /// One drafter layer's attention.
    ///
    /// Mirrors `DSparkAttention.forward(x, start_pos, main_x)` from the reference:
    ///
    ///   * the RING is fed from `main_x` (the main model's residuals), roped at
    ///     the main position and written at `pos % MTP_WINDOW`;
    ///   * the QUERY comes from `x` (the draft stream), and so does this step's
    ///     own KV, roped at the draft position and placed right after the valid
    ///     ring prefix;
    ///   * attention is DENSE over `[ring prefix ++ own kv]` — the reference's
    ///     `sparse_attn` + `get_dspark_topk_idxs` selects exactly that set for
    ///     every query, so there is no sparsity to implement;
    ///   * an INVERSE rope is applied to the attention output before `wo_a`.
    ///
    /// `attn_mixed` raw-only rather than `attn_swa`: the latter's LDS
    /// `scores[ATTN_SWA_MAX_KV]` caps at 128 and this needs 129+, and widening it
    /// would touch the main model's SWA occupancy.
    #[allow(clippy::too_many_arguments)]
    pub fn attn(
        &mut self,
        e: &DeviceEngine,
        s: &Stream,
        w: &crate::het::weights::MtpLayerWeights,
        li: usize,
        x: &DeviceBuffer<f32>,
        pos_dev: &DeviceBuffer<u32>,
        rope: &crate::RopeParams,
        pos: u32,
    ) -> eyre::Result<()> {
        let n_valid = Self::n_valid(pos);
        let n_kv = (n_valid + 1) as u32;

        // --- query stream, from x ---
        e.q8.quantize_input(s, &mut self.xq, &mut self.xscale, x, N_EMBD)?;
        crate::het::dispatch::dense_matvec(
            e, s, &mut self.qr, &w.attn_q_a, x, &self.xq, &self.xscale, N_LORA_Q, N_EMBD,
        )?;
        e.rms_w.launch_weighted_quantize_q8(
            s, &mut self.qr_normed, &mut self.qr_xq, &mut self.qr_xscale, &self.qr, &w.q_a_norm,
            N_LORA_Q, RMS_EPS,
        )?;
        e.q8.matvec(
            s, &mut self.q, &w.attn_q_b.buffer, &self.qr_xq, &self.qr_xscale, Q_FLAT, N_LORA_Q,
        )?;
        // V4.1 has no per-head q RMSNorm after wq_b (same as the main model).
        self.q_normed.copy_from_buffer_async(&self.q, s)?;
        e.rope.launch_forward_pdev(
            s, &mut self.q_normed, pos_dev, N_HEAD, N_HEAD_DIM, N_ROT, rope,
        )?;

        // --- this step's own KV, from x, placed right after the ring prefix ---
        e.q8.matvec(
            s, &mut self.kv_raw, &w.attn_kv.buffer, &self.xq, &self.xscale, N_HEAD_DIM, N_EMBD,
        )?;
        self.slot_dev.copy_from_host(&[n_valid as u32])?;
        e.fp8.launch_kv_post_fused(
            s, &mut self.kv_normed, &mut self.rings[li], &self.kv_raw, &w.kv_a_norm, pos_dev,
            &self.slot_dev, N_HEAD_DIM, N_ROT, RMS_EPS, rope,
        )?;

        // --- attention over [ring prefix ++ own], then the inverse rope ---
        e.attn_mixed.launch_score(
            s, &mut self.scores, &self.q_normed, &self.rings[li], None, N_HEAD, N_HEAD_DIM,
            n_kv, 0,
        )?;
        e.attn_mixed.launch_softmax_wsum(
            s, &mut self.heads, &mut self.scores, &w.attn_sinks, &self.rings[li], None, N_HEAD,
            N_HEAD_DIM, n_kv, 0,
        )?;
        e.rope.launch_inverse_pdev(
            s, &mut self.heads, pos_dev, N_HEAD, N_HEAD_DIM, N_ROT, rope,
        )?;

        // --- output projection: grouped wo_a, then wo_b ---
        e.q8.quantize_input(s, &mut self.heads_xq, &mut self.heads_xscale, &self.heads, Q_FLAT)?;
        e.q8_grouped.matvec_grouped(
            s, &mut self.low, &w.attn_output_a.buffer, &self.heads_xq, &self.heads_xscale,
            GROUP_DIM, RANK, N_GROUPS,
        )?;
        e.q8.quantize_input(s, &mut self.low_xq, &mut self.low_xscale, &self.low, OUT_LOW)?;
        e.q8.matvec(
            s, &mut self.attn_out, &w.attn_output_b.buffer, &self.low_xq, &self.low_xscale,
            N_EMBD, OUT_LOW,
        )?;
        Ok(())
    }
}
