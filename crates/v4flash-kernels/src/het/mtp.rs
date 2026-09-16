//! DSpark drafter: state, the entry projection, and the three-layer draft pass.
//!
//! The drafter is a 3-layer model (see `docs/v41/DSPARK_DESIGN.md`). It is NOT
//! autoregressive: `forward_embed` builds
//! `[real_token, noise, noise, noise, noise]` and ONE pass over the three layers
//! emits all `MTP_BLOCK` drafts, so every buffer here is `[MTP_BLOCK, ...]`.
//! The only sequential part is the exit's markov head, which is not in this
//! module.
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
//!
//! A drafter layer is structurally the MAIN model's layer — the same mHC
//! sandwich around an attention and a MoE sub-block — so it reuses the main
//! model's kernels throughout, at batch `MTP_BLOCK` via the `_batched` variants
//! that the M50 prefill path added.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, Stream};

use crate::config::{
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, EXPERT_WEIGHT_SCALE, GROUP_DIM, HC_DIM, HC_MIX_DIM,
    N_EMBD, N_FF_EXP, N_FF_SHARED, N_GROUPS, N_HC, N_HEAD, N_HEAD_DIM, N_LORA_Q, N_ROT, OUT_LOW,
    Q_FLAT, RANK, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS, SWIGLU_CLAMP_EXP, N_VOCAB,
};
use v4flash_core::hf_v41::MTP_N_EXPERT;

/// Matches `forward_layer.rs`'s private constant of the same name — the router's
/// weight floor. Duplicated rather than made public: it is a property of the
/// router kernel's contract, and the drafter must use the identical value or its
/// expert weights diverge from the reference.
const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;
use crate::het::engine::DeviceEngine;
use crate::het::remote_experts::{MIDQ_BYTES_PER_SLOT, SENTINEL_EXPERT, XQ_BYTES_PER_TOKEN};
use crate::het::weights::{MtpExitWeights, MtpLayerWeights, MtpWeights};

// ---------------------------------------------------------------------------
// Drafter geometry. All from the checkpoint's own config.json (`text_config`)
// and `inference/model.py`, NOT inferred — several of these differ from the main
// model in ways that would be silently wrong if guessed.
// ---------------------------------------------------------------------------

/// `dspark_target_layer_ids` — main-model layers whose ENTERING residual feeds
/// the drafter, in order.
pub const MTP_SRC_LAYERS: [i32; 3] = [37, 38, 39];
/// `dspark_block_size` — draft tokens emitted per speculation step, and the
/// batch every drafter kernel runs at.
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

/// The drafter's rope parameters.
///
/// `DSparkAttention.forward` opens with `assert self.compress_ratio == 0`, so
/// the drafter always takes the main model's DENSE-layer arm of
/// `rope_for_layer`: base 10000, no yarn extension, no scaling. Defined here so
/// the drafter cannot silently drift onto the compressed-layer parameters.
pub fn mtp_rope() -> crate::RopeParams {
    crate::RopeParams {
        freq_base: 10000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        n_ctx_orig: 0,
    }
}

/// Ablations for localising a weak drafter. `V41_DSPARK_NO_ATTN=1` zeroes the
/// attention sub-block, `V41_DSPARK_NO_ROUTED=1` drops the routed experts and
/// keeps only the shared one. If acceptance is UNCHANGED with a part disabled,
/// that part was contributing nothing.
fn no_attn() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("V41_DSPARK_NO_ATTN").as_deref(), Ok("1")))
}
fn debug_moe() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("V41_DSPARK_DEBUG_MOE").as_deref(), Ok("1")))
}
static MOE_DBG_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn no_routed() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("V41_DSPARK_NO_ROUTED").as_deref(), Ok("1")))
}

/// Batch every drafter kernel runs at.
const B: u32 = MTP_BLOCK as u32;
/// Ring slots: the window, plus the block's own transient KV packed right after
/// the valid prefix. See `attn`.
const RING_SLOTS: usize = MTP_WINDOW + MTP_BLOCK;
/// `attention_mixed_score` indexes `scores[h * ATTN_MIXED_MAX_KEYS + k]` — the
/// stride is the kernel's compile-time cap, NOT `n_kv`. Sizing this buffer to
/// the actual key count (as an earlier draft did) writes ~600x past its end.
/// The five queries run in sequence on one stream, so they share one buffer.
const SCORES_ELEMS: usize = N_HEAD as usize * crate::attention::ATTN_MIXED_MAX_KEYS as usize;

/// Per-request drafter state. Allocated once; refilled every token.
pub struct MtpState {
    /// `cat(mean_over_hc(residual@37), ..@38, ..@39)` — `[3 * N_EMBD]`.
    pub main_x: DeviceBuffer<f32>,
    /// Uniform `1/N_HC`, so `hc_weighted` computes a mean.
    hc_mean: DeviceBuffer<f32>,
    /// Q8 staging for the `main_proj` matvec — `[3 * N_EMBD]`, distinct from the
    /// per-layer `xq`/`xscale` which are `[B, N_EMBD]`.
    proj_xq: DeviceBuffer<i8>,
    proj_xscale: DeviceBuffer<f32>,
    /// `main_proj` output and its norm. `x` is `main_x` in the reference's
    /// naming: the KV SOURCE for every drafter layer, one row, not the residual
    /// stream.
    pub proj: DeviceBuffer<f32>,
    pub x: DeviceBuffer<f32>,
    /// Which of `MTP_SRC_LAYERS` have been captured this token. Guards against
    /// running the entry on a stale or partial `main_x`.
    captured: [bool; MTP_SRC_LAYERS.len()],
    /// How many `main_kv` rows have actually been written into the rings.
    ///
    /// The reference indexes the window by `start_pos % window` and treats
    /// `min(window, start_pos + 1)` entries as valid, because its prefill seeds
    /// the window with the prompt's last `window` rows. Ours does not (yet), so
    /// deriving validity from the ABSOLUTE position claims 128 live keys when
    /// only a handful were written, and attention averages over ~120
    /// uninitialised ones. Counting real writes keeps the valid set a true
    /// prefix, re-based to whenever the drafter started.
    ring_writes: usize,

    // ---- residual stream ----
    /// The drafter's hyper-connection residual, `[B, N_HC, N_EMBD]`. `hc_post`
    /// cannot write its own residual input, so the two ping-pong.
    pub h: DeviceBuffer<f32>,
    h_next: DeviceBuffer<f32>,
    /// `pre_mix` carried between sub-blocks: each `hc_mixes` produces the
    /// collapse weights for the NEXT sub-block. `[B, HC_MIX_DIM]`, of which
    /// `hc_weighted` reads the first `N_HC` per row.
    pre_carry: DeviceBuffer<f32>,
    split: DeviceBuffer<f32>,
    flat: DeviceBuffer<f32>,
    mix: DeviceBuffer<f32>,
    /// `hc_pre` output — the sub-block's single-copy input, `[B, N_EMBD]`.
    cur: DeviceBuffer<f32>,
    /// `attn_norm`/`ffn_norm` of `cur`, `[B, N_EMBD]`.
    normed: DeviceBuffer<f32>,

    // ---- per-layer attention state ----
    /// KV ring per drafter layer: `[RING_SLOTS] x N_HEAD_DIM` f16.
    ///
    /// Slots `[0, n_valid)` hold `kv_norm(wkv(main_x))` — ONE row per accepted
    /// token, roped at the main position. Slots `[n_valid, n_valid + B)` hold
    /// the block's own transient KV. See `attn` for why that packing is legal.
    pub rings: Vec<DeviceBuffer<u16>>,

    // ---- scratch, shared across the three layers (they run in sequence) ----
    xq: DeviceBuffer<i8>,
    xscale: DeviceBuffer<f32>,
    /// Q8 staging for the single-row `main_x`, whose width is N_EMBD but whose
    /// batch is 1, not B.
    main_xq: DeviceBuffer<i8>,
    main_xscale: DeviceBuffer<f32>,
    main_kv_raw: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    qr_normed: DeviceBuffer<f32>,
    qr_xq: DeviceBuffer<i8>,
    qr_xscale: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    kv_raw: DeviceBuffer<f32>,
    kv_normed: DeviceBuffer<f32>,
    /// Ring slot each `kv_post_fused` call writes at, and the position it ropes
    /// with. One entry per call site; sliced one element at a time.
    slot_dev: DeviceBuffer<u32>,
    pos_dev: DeviceBuffer<u32>,
    /// Draft positions `pos+1 ..= pos+B` as i32, for the batched rope.
    pos_per_b: DeviceBuffer<i32>,
    scores: DeviceBuffer<f32>,
    heads: DeviceBuffer<f32>,
    heads_xq: DeviceBuffer<i8>,
    heads_xscale: DeviceBuffer<f32>,
    low: DeviceBuffer<f32>,
    low_xq: DeviceBuffer<i8>,
    low_xscale: DeviceBuffer<f32>,
    /// The layer's attention output, `[B, N_EMBD]`.
    pub attn_out: DeviceBuffer<f32>,

    // ---- MoE ----
    ffn_xq: DeviceBuffer<u8>,
    router_logits: DeviceBuffer<f32>,
    d_selected: DeviceBuffer<i32>,
    d_ew: DeviceBuffer<f32>,
    /// Identity remap over the drafter's 128 resident experts: `-(e)-1` for
    /// `e < MTP_N_EXPERT`, 0 (= "not ours") for the sentinel. Built once — the
    /// drafter never pages, so residency never changes.
    remap: DeviceBuffer<i32>,
    mid: DeviceBuffer<f32>,
    midq: DeviceBuffer<u8>,
    gate_sh: DeviceBuffer<f32>,
    up_sh: DeviceBuffer<f32>,
    mid_sh: DeviceBuffer<f32>,
    mid_sh_xq: DeviceBuffer<i8>,
    mid_sh_xscale: DeviceBuffer<f32>,
    ffn_shared: DeviceBuffer<f32>,
    /// Routed-expert output (shared expert added in), `[B, N_EMBD]`.
    pub ffn_out: DeviceBuffer<f32>,
}

impl MtpState {
    pub fn alloc(device_id: i32) -> eyre::Result<Self> {
        let k = MTP_SRC_LAYERS.len() * N_EMBD as usize;
        let mut hc_mean = DeviceBuffer::<f32>::new(device_id, N_HC as usize)?;
        hc_mean.copy_from_host(&vec![1.0f32 / N_HC as f32; N_HC as usize])?;
        // Every drafter expert is resident, so the remap is a constant identity.
        // Sentinel (and any id past 128) stays 0 = "not ours", which is what makes
        // an unused pick slot a no-op in the MoE kernel.
        let mut remap_host = vec![0i32; SENTINEL_EXPERT as usize + 1];
        for (e, r) in remap_host.iter_mut().enumerate().take(MTP_N_EXPERT) {
            *r = -(e as i32) - 1;
        }
        let mut remap = DeviceBuffer::<i32>::new(device_id, remap_host.len())?;
        remap.copy_from_host(&remap_host)?;
        let ne = N_EMBD as usize;
        let b = MTP_BLOCK;
        let mut rings = Vec::with_capacity(MTP_SRC_LAYERS.len());
        for _ in 0..MTP_SRC_LAYERS.len() {
            rings.push(DeviceBuffer::<u16>::new(
                device_id,
                RING_SLOTS * N_HEAD_DIM as usize,
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
            ring_writes: 0,

            h: DeviceBuffer::new(device_id, b * N_HC as usize * ne)?,
            h_next: DeviceBuffer::new(device_id, b * N_HC as usize * ne)?,
            pre_carry: DeviceBuffer::new(device_id, b * HC_MIX_DIM as usize)?,
            split: DeviceBuffer::new(device_id, b * HC_MIX_DIM as usize)?,
            flat: DeviceBuffer::new(device_id, b * HC_DIM as usize)?,
            mix: DeviceBuffer::new(device_id, b * HC_MIX_DIM as usize)?,
            cur: DeviceBuffer::new(device_id, b * ne)?,
            normed: DeviceBuffer::new(device_id, b * ne)?,

            rings,
            xq: DeviceBuffer::new(device_id, b * ne)?,
            xscale: DeviceBuffer::new(device_id, b * ne.div_ceil(32))?,
            main_xq: DeviceBuffer::new(device_id, ne)?,
            main_xscale: DeviceBuffer::new(device_id, ne.div_ceil(32))?,
            main_kv_raw: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            qr: DeviceBuffer::new(device_id, b * N_LORA_Q as usize)?,
            qr_normed: DeviceBuffer::new(device_id, b * N_LORA_Q as usize)?,
            qr_xq: DeviceBuffer::new(device_id, b * N_LORA_Q as usize)?,
            qr_xscale: DeviceBuffer::new(device_id, b * (N_LORA_Q as usize).div_ceil(32))?,
            q: DeviceBuffer::new(device_id, b * Q_FLAT as usize)?,
            kv_raw: DeviceBuffer::new(device_id, b * N_HEAD_DIM as usize)?,
            kv_normed: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            slot_dev: DeviceBuffer::new(device_id, b + 1)?,
            pos_dev: DeviceBuffer::new(device_id, b + 1)?,
            pos_per_b: DeviceBuffer::new(device_id, b)?,
            scores: DeviceBuffer::new(device_id, SCORES_ELEMS)?,
            heads: DeviceBuffer::new(device_id, b * Q_FLAT as usize)?,
            heads_xq: DeviceBuffer::new(device_id, b * Q_FLAT as usize)?,
            heads_xscale: DeviceBuffer::new(device_id, b * (Q_FLAT as usize).div_ceil(32))?,
            low: DeviceBuffer::new(device_id, b * OUT_LOW as usize)?,
            low_xq: DeviceBuffer::new(device_id, b * OUT_LOW as usize)?,
            low_xscale: DeviceBuffer::new(device_id, b * (OUT_LOW as usize).div_ceil(32))?,
            attn_out: DeviceBuffer::new(device_id, b * ne)?,

            ffn_xq: DeviceBuffer::new(device_id, b * XQ_BYTES_PER_TOKEN)?,
            router_logits: DeviceBuffer::new(device_id, b * MTP_N_EXPERT)?,
            d_selected: DeviceBuffer::new(device_id, b * MTP_TOPK as usize)?,
            d_ew: DeviceBuffer::new(device_id, b * MTP_TOPK as usize)?,
            remap,
            mid: DeviceBuffer::new(device_id, b * MTP_TOPK as usize * N_FF_EXP as usize)?,
            midq: DeviceBuffer::new(device_id, b * MTP_TOPK as usize * MIDQ_BYTES_PER_SLOT)?,
            gate_sh: DeviceBuffer::new(device_id, b * N_FF_SHARED as usize)?,
            up_sh: DeviceBuffer::new(device_id, b * N_FF_SHARED as usize)?,
            mid_sh: DeviceBuffer::new(device_id, b * N_FF_SHARED as usize)?,
            mid_sh_xq: DeviceBuffer::new(device_id, b * N_FF_SHARED as usize)?,
            mid_sh_xscale: DeviceBuffer::new(device_id, b * (N_FF_SHARED as usize).div_ceil(32))?,
            ffn_shared: DeviceBuffer::new(device_id, b * ne)?,
            ffn_out: DeviceBuffer::new(device_id, b * ne)?,
        })
    }

    /// Start a new token: forget last token's captures.
    pub fn begin_token(&mut self) {
        self.captured = [false; MTP_SRC_LAYERS.len()];
    }

    /// Inject `main_x` directly, bypassing capture. For validating `entry`
    /// against a reference dump, where the capture inputs come from a file
    /// rather than from a live decode.
    #[doc(hidden)]
    pub fn inject_main_hidden(&mut self, host: &[f32]) -> eyre::Result<()> {
        let want = MTP_SRC_LAYERS.len() * N_EMBD as usize;
        if host.len() != want {
            return Err(eyre!("inject_main_hidden: {} floats, want {want}", host.len()));
        }
        self.main_x.copy_from_host(host)?;
        self.captured = [true; MTP_SRC_LAYERS.len()];
        Ok(())
    }

    /// The pre-mix carried out of the last drafter layer. `forward_head` needs
    /// it to collapse the final residual — the reference's
    /// `blocks[-1].hc_pre(h, pre_mix)`.
    pub fn pre_carry(&self) -> &DeviceBuffer<f32> {
        &self.pre_carry
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

    /// `main_norm(main_proj(main_x))` — the KV source for every drafter layer.
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
        // `main_proj` is Q8_0, so `dense_matvec` reads the QUANTIZED input, not
        // the f32 one. Skipping this leaves xq/xscale zeroed and the projection
        // silently returns all zeros.
        e.q8.quantize_input(s, &mut self.proj_xq, &mut self.proj_xscale, &self.main_x, k)?;
        crate::het::dispatch::dense_matvec(
            e, s, &mut self.proj, &w.main_proj, &self.main_x, &self.proj_xq, &self.proj_xscale,
            N_EMBD, k,
        )?;
        e.rms_w
            .launch_weighted(s, &mut self.x, &self.proj, &w.main_norm, N_EMBD, RMS_EPS)?;
        Ok(())
    }

    /// Valid ring entries once this step's `main_kv` is written, and the slot
    /// it goes in. Driven by writes actually performed, not by the absolute
    /// position — see `ring_writes`.
    #[inline]
    fn ring_geom(&self) -> (usize, usize) {
        (
            MTP_WINDOW.min(self.ring_writes + 1),
            self.ring_writes % MTP_WINDOW,
        )
    }

    /// Ring rows written so far, capped at the window.
    pub fn ring_filled(&self) -> usize {
        self.ring_writes.min(MTP_WINDOW)
    }

    /// `forward_embed`: seed the residual stream from the accepted token.
    ///
    /// The reference embeds `[token, noise, noise, noise, noise]` and repeats
    /// each row across the `N_HC` hyper-connection copies. `token_embd` is not
    /// device-resident (M57 — the server embeds host-side), so the caller passes
    /// the two rows it has already looked up and this assembles the block. Both
    /// rows are `HC_DIM` wide — exactly what `embed::embed_lookup` writes, which
    /// already does the `N_HC` replication.
    ///
    /// `pre_mix` is reset to `make_identity_pre_mix`: one-hot on copy 0.
    pub fn embed(
        &mut self,
        s: &Stream,
        token_row: &[f32],
        noise_row: &[f32],
    ) -> eyre::Result<()> {
        let hcd = HC_DIM as usize;
        if token_row.len() != hcd || noise_row.len() != hcd {
            return Err(eyre!(
                "mtp embed: rows {}/{} floats, want {hcd}",
                token_row.len(),
                noise_row.len()
            ));
        }
        let mut host = vec![0.0f32; MTP_BLOCK * hcd];
        for j in 0..MTP_BLOCK {
            let row = if j == 0 { token_row } else { noise_row };
            host[j * hcd..(j + 1) * hcd].copy_from_slice(row);
        }
        self.h.copy_from_host(&host)?;
        let mut pre = vec![0.0f32; MTP_BLOCK * HC_MIX_DIM as usize];
        for j in 0..MTP_BLOCK {
            pre[j * HC_MIX_DIM as usize] = 1.0;
        }
        self.pre_carry.copy_from_host(&pre)?;
        let _ = s;
        Ok(())
    }

    /// `hc_mixes` for one sub-block: normalise the flattened hc stream, project
    /// it through `hc_*_fn`, and Sinkhorn-split into `self.split`.
    fn hc_mixes(
        &mut self,
        e: &DeviceEngine,
        s: &Stream,
        fnw: &crate::weights::DeviceWeight,
        scale: &DeviceBuffer<f32>,
        base: &DeviceBuffer<f32>,
    ) -> eyre::Result<()> {
        // The reference computes `F.linear(x, fn) * rsqrt(mean(x^2))`; scaling
        // the input instead is identical (rsqrt is one scalar per row) and is
        // what the main model's path does.
        e.rms_nw
            .launch_batched(s, &mut self.flat, &self.h, 1, HC_DIM, RMS_EPS, B)?;
        // Per-row `matvec`, NOT `gemm_batched_wmma`: that kernel's BM/BN are 64
        // and it is only correct at prefill-sized batches. At B=5 it writes rows
        // 1..4 as ZEROS and leaves row 4's result sitting in row 0 (see
        // tests/f16_gemm_batched_small_b.rs). Silently, with no error — which
        // fed the drafter constant mHC mixes and a uniform router.
        for j in 0..MTP_BLOCK {
            let fj = self.flat.slice_view(j * HC_DIM as usize, HC_DIM as usize);
            let mut mj = self
                .mix
                .slice_view_mut(j * HC_MIX_DIM as usize, HC_MIX_DIM as usize);
            e.f16.matvec(s, &mut mj, &fnw.buffer, &fj, HC_MIX_DIM, HC_DIM)?;
        }
        e.hc_sinkhorn.launch_batched(
            s, &mut self.split, &self.mix, scale, base, N_HC, SINKHORN_ITERS, SINKHORN_EPS, B,
        )?;
        Ok(())
    }

    /// Collapse `h` with the carried pre-mix, then carry this sub-block's own.
    ///
    /// V4.1's single-pass mHC: each `hc_mixes` produces the collapse weights for
    /// the NEXT sub-block, so attention uses what the previous FFN produced and
    /// the FFN uses what this attention produced.
    fn hc_pre_and_carry(&mut self, e: &DeviceEngine, s: &Stream) -> eyre::Result<()> {
        e.hc_weighted.launch_batched(
            s, &mut self.cur, &self.h, &self.pre_carry, N_EMBD, N_HC, HC_MIX_DIM, B,
        )?;
        self.pre_carry.copy_from_buffer_async(&self.split, s)?;
        Ok(())
    }

    /// Expand a sub-block output back to `N_HC` copies and mix the residual in.
    /// Writes `h_next` and swaps, because `hc_post` cannot alias its residual.
    fn hc_post(&mut self, e: &DeviceEngine, s: &Stream, out: bool) -> eyre::Result<()> {
        {
            let block_out = if out { &self.attn_out } else { &self.ffn_out };
            e.hc_post.launch_from_split_batched(
                s, &mut self.h_next, block_out, &self.h, &self.split, N_HC, N_EMBD, N_HC, B,
            )?;
        }
        std::mem::swap(&mut self.h, &mut self.h_next);
        Ok(())
    }

    /// One drafter layer: mHC + attention, then mHC + MoE.
    #[allow(clippy::too_many_arguments)]
    pub fn layer(
        &mut self,
        e: &DeviceEngine,
        s: &Stream,
        w: &MtpLayerWeights,
        li: usize,
        rope: &crate::RopeParams,
        pos: u32,
    ) -> eyre::Result<()> {
        self.hc_mixes(e, s, &w.hc_attn_fn, &w.hc_attn_scale, &w.hc_attn_base)?;
        self.hc_pre_and_carry(e, s)?;
        e.rms_w.launch_weighted_batched(
            s, &mut self.normed, &self.cur, &w.attn_norm, N_EMBD, RMS_EPS, B,
        )?;
        if no_attn() {
            let z = vec![0.0f32; self.attn_out.len()];
            self.attn_out.copy_from_host(&z)?;
        } else {
            self.attn(e, s, w, li, rope, pos)?;
        }
        self.hc_post(e, s, true)?;

        self.hc_mixes(e, s, &w.hc_ffn_fn, &w.hc_ffn_scale, &w.hc_ffn_base)?;
        self.hc_pre_and_carry(e, s)?;
        e.rms_w.launch_weighted_batched(
            s, &mut self.normed, &self.cur, &w.ffn_norm, N_EMBD, RMS_EPS, B,
        )?;
        self.moe(e, s, w)?;
        self.hc_post(e, s, false)?;
        Ok(())
    }

    /// The whole draft pass: entry, embed, three layers.
    ///
    /// `pos` is the main model's position for the token being drafted from, i.e.
    /// the reference's `start_pos`. Leaves the collapsed, un-normed stream in
    /// `h`/`pre_carry` for the exit head.
    pub fn forward(
        &mut self,
        e: &DeviceEngine,
        s: &Stream,
        w: &MtpWeights,
        rope: &crate::RopeParams,
        pos: u32,
        token_row: &[f32],
        noise_row: &[f32],
    ) -> eyre::Result<()> {
        if pos == 0 {
            return Err(eyre!("mtp forward: pos 0 is prefill-seed only"));
        }
        self.entry(e, s, w)?;
        self.embed(s, token_row, noise_row)?;
        for li in 0..w.layers.len() {
            self.layer(e, s, &w.layers[li], li, rope, pos)?;
        }
        // One `main_kv` row per token, shared by all three layers.
        self.ring_writes += 1;
        Ok(())
    }

    /// Ring-write-ONLY: advance the drafter's KV ring by one MAIN position using
    /// `main_hidden` (the main model's residual at `pos`), WITHOUT drafting a
    /// block. The cheap primitive the tech report's SWA replay implies — for
    /// prefill window seeding and dense accept-mode ring maintenance. Costs one
    /// entry + one `attn_kv` matvec per layer + the ring write; no embed, no
    /// block query/attention/MoE, no exit.
    pub fn advance_ring(
        &mut self,
        e: &DeviceEngine,
        s: &Stream,
        w: &MtpWeights,
        rope: &crate::RopeParams,
        pos: u32,
        main_hidden: &[f32],
    ) -> eyre::Result<()> {
        if pos == 0 {
            return Err(eyre!("mtp advance_ring: pos 0 is prefill-seed only"));
        }
        self.inject_main_hidden(main_hidden)?;
        self.entry(e, s, w)?;
        let (_, main_slot) = self.ring_geom();
        self.slot_dev.slice_view_mut(0, 1).copy_from_host(&[main_slot as u32])?;
        self.pos_dev.slice_view_mut(0, 1).copy_from_host(&[pos])?;
        // self.x (entry output) is the same KV source for every layer; quantise once.
        e.q8
            .quantize_input(s, &mut self.main_xq, &mut self.main_xscale, &self.x, N_EMBD)?;
        for li in 0..w.layers.len() {
            let lw = &w.layers[li];
            e.q8.matvec(
                s, &mut self.main_kv_raw, &lw.attn_kv.buffer, &self.main_xq, &self.main_xscale,
                N_HEAD_DIM, N_EMBD,
            )?;
            e.fp8.launch_kv_post_fused(
                s, &mut self.kv_normed, &mut self.rings[li], &self.main_kv_raw, &lw.kv_a_norm,
                &self.pos_dev.slice_view(0, 1), &self.slot_dev.slice_view(0, 1), N_HEAD_DIM, N_ROT,
                RMS_EPS, rope,
            )?;
        }
        self.ring_writes += 1;
        Ok(())
    }

    /// One drafter layer's attention.
    ///
    /// Mirrors `DSparkAttention.forward(x, start_pos, main_x)`:
    ///
    ///   * the RING is fed from `main_x` — the main model's residuals, NOT the
    ///     draft stream — roped at the MAIN position and written at
    ///     `pos % MTP_WINDOW`. One row per accepted token, and the same source
    ///     for every layer;
    ///   * the QUERY comes from the draft stream, and so does the block's own
    ///     KV, roped at draft positions `pos+1 ..= pos+B`. That KV is TRANSIENT:
    ///     the reference concatenates it after the window rather than caching
    ///     it, so nothing here needs rolling back on rejection;
    ///   * attention is DENSE over `[valid ring prefix ++ block KV]`.
    ///     `get_dspark_topk_idxs` expands ONE index vector over every query, so
    ///     the block is bidirectional and `n_kv` is the same for all B;
    ///   * an INVERSE rope is applied to the attention output before `wo_a`.
    ///
    /// The block's KV is packed at `[n_valid, n_valid + B)` instead of the
    /// reference's fixed `[window, window + B)`. RoPE is baked into each key at
    /// cache time, so attention is permutation-invariant over the KV set and the
    /// attended set is identical — this just keeps the keys contiguous, which is
    /// what the batched score kernel requires. Anything at or past `n_valid` is
    /// scratch, so the next step's `main_kv` write legitimately overwrites it.
    #[allow(clippy::too_many_arguments)]
    fn attn(
        &mut self,
        e: &DeviceEngine,
        s: &Stream,
        w: &MtpLayerWeights,
        li: usize,
        rope: &crate::RopeParams,
        pos: u32,
    ) -> eyre::Result<()> {
        let (n_valid, main_slot) = self.ring_geom();
        let n_kv = (n_valid + MTP_BLOCK) as u32;
        if n_kv > crate::attention::ATTN_MIXED_MAX_KEYS {
            return Err(eyre!("mtp attn: n_kv={n_kv} exceeds the attention key cap"));
        }

        // --- the ring row, from main_x at the MAIN position ---
        e.q8
            .quantize_input(s, &mut self.main_xq, &mut self.main_xscale, &self.x, N_EMBD)?;
        e.q8.matvec(
            s, &mut self.main_kv_raw, &w.attn_kv.buffer, &self.main_xq, &self.main_xscale,
            N_HEAD_DIM, N_EMBD,
        )?;
        // Slot 0 of the scratch pair carries the main row, 1..=B the block's.
        let mut slots = vec![0u32; MTP_BLOCK + 1];
        let mut poss = vec![0u32; MTP_BLOCK + 1];
        slots[0] = main_slot as u32;
        poss[0] = pos;
        for j in 0..MTP_BLOCK {
            slots[j + 1] = (n_valid + j) as u32;
            poss[j + 1] = pos + 1 + j as u32;
        }
        self.slot_dev.copy_from_host(&slots)?;
        self.pos_dev.copy_from_host(&poss)?;
        e.fp8.launch_kv_post_fused(
            s, &mut self.kv_normed, &mut self.rings[li], &self.main_kv_raw, &w.kv_a_norm,
            &self.pos_dev.slice_view(0, 1), &self.slot_dev.slice_view(0, 1), N_HEAD_DIM, N_ROT,
            RMS_EPS, rope,
        )?;

        // --- the block's own KV, from the draft stream ---
        e.q8
            .quantize_input_batched(s, &mut self.xq, &mut self.xscale, &self.normed, N_EMBD, B)?;
        e.q8.matvec_batched(
            s, &mut self.kv_raw, &w.attn_kv.buffer, &self.xq, &self.xscale, N_HEAD_DIM, N_EMBD, B,
        )?;
        for j in 0..MTP_BLOCK {
            let row = self.kv_raw.slice_view(j * N_HEAD_DIM as usize, N_HEAD_DIM as usize);
            e.fp8.launch_kv_post_fused(
                s, &mut self.kv_normed, &mut self.rings[li], &row, &w.kv_a_norm,
                &self.pos_dev.slice_view(j + 1, 1), &self.slot_dev.slice_view(j + 1, 1),
                N_HEAD_DIM, N_ROT, RMS_EPS, rope,
            )?;
        }

        // --- queries ---
        for j in 0..MTP_BLOCK {
            // `attn_q_a` is dtype-dispatched, and `dense_matvec` has no batched
            // twin; B is 5, so the loop is cheaper than a new kernel.
            let xr = self.normed.slice_view(j * N_EMBD as usize, N_EMBD as usize);
            let xqr = self.xq.slice_view(j * N_EMBD as usize, N_EMBD as usize);
            let xsr = self
                .xscale
                .slice_view(j * (N_EMBD as usize).div_ceil(32), (N_EMBD as usize).div_ceil(32));
            let mut qo = self.qr.slice_view_mut(j * N_LORA_Q as usize, N_LORA_Q as usize);
            crate::het::dispatch::dense_matvec(
                e, s, &mut qo, &w.attn_q_a, &xr, &xqr, &xsr, N_LORA_Q, N_EMBD,
            )?;
        }
        e.rms_w.launch_weighted_batched(
            s, &mut self.qr_normed, &self.qr, &w.q_a_norm, N_LORA_Q, RMS_EPS, B,
        )?;
        e.q8.quantize_input_batched(
            s, &mut self.qr_xq, &mut self.qr_xscale, &self.qr_normed, N_LORA_Q, B,
        )?;
        e.q8.matvec_batched(
            s, &mut self.q, &w.attn_q_b.buffer, &self.qr_xq, &self.qr_xscale, Q_FLAT, N_LORA_Q, B,
        )?;
        // V4.1 has no per-head q RMSNorm after wq_b (same as the main model).
        let dpos: Vec<i32> = (0..MTP_BLOCK).map(|j| (pos + 1 + j as u32) as i32).collect();
        self.pos_per_b.copy_from_host(&dpos)?;
        e.rope.launch_forward_batched(
            s, &mut self.q, &self.pos_per_b, N_HEAD, N_HEAD_DIM, N_ROT, B, rope,
        )?;

        // --- attention over [ring prefix ++ block], then the inverse rope ---
        // Per-query, NOT one batched launch. Every `*_batched_htiled_wmma*`
        // variant guards its phase B behind `__gfx1200__ || __gfx1201__`, so on
        // the iGPU (gfx1151) the softmax lands and the weighted sum is compiled
        // out — `out` is left untouched and attention silently returns zeros.
        // `attention_mixed_score` / `attention_mixed_softmax_wsum` carry no arch
        // guard. B is 5, so the loop costs 10 small launches per layer.
        for j in 0..MTP_BLOCK {
            let qj = self.q.slice_view(j * Q_FLAT as usize, Q_FLAT as usize);
            e.attn_mixed.launch_score(
                s, &mut self.scores, &qj, &self.rings[li], None, N_HEAD, N_HEAD_DIM, n_kv, 0,
            )?;
            let mut hj = self.heads.slice_view_mut(j * Q_FLAT as usize, Q_FLAT as usize);
            e.attn_mixed.launch_softmax_wsum(
                s, &mut hj, &mut self.scores, &w.attn_sinks, &self.rings[li], None, N_HEAD,
                N_HEAD_DIM, n_kv, 0,
            )?;
        }
        e.rope.launch_inverse_batched(
            s, &mut self.heads, &self.pos_per_b, N_HEAD, N_HEAD_DIM, N_ROT, B, rope,
        )?;

        // --- output projection: grouped wo_a, then wo_b ---
        e.q8.quantize_input_batched(
            s, &mut self.heads_xq, &mut self.heads_xscale, &self.heads, Q_FLAT, B,
        )?;
        e.q8_grouped.matvec_grouped_batched(
            s, &mut self.low, &w.attn_output_a.buffer, &self.heads_xq, &self.heads_xscale,
            GROUP_DIM, RANK, N_GROUPS, B,
        )?;
        e.q8.quantize_input_batched(
            s, &mut self.low_xq, &mut self.low_xscale, &self.low, OUT_LOW, B,
        )?;
        e.q8.matvec_batched(
            s, &mut self.attn_out, &w.attn_output_b.buffer, &self.low_xq, &self.low_xscale,
            N_EMBD, OUT_LOW, B,
        )?;
        Ok(())
    }

    /// One drafter layer's MoE: router (top-3 of 128), routed experts, and the
    /// shared expert added in.
    ///
    /// Same kernels as box 2's resident executor, which is the cleanest example
    /// of a non-paged MoE in the tree. `router_topk` takes `n_expert` and
    /// `n_used` at runtime, so 128/3 needs no kernel change against the main
    /// model's 384/6.
    ///
    /// `cap = MTP_TOPK` makes the sentinel path a no-op exactly as it does for
    /// box 2: a resident expert has `remap < 0` so the kernel claims it, and the
    /// sentinel has `remap == 0` with `res_rank < cap` so it is skipped.
    fn moe(&mut self, e: &DeviceEngine, s: &Stream, w: &MtpLayerWeights) -> eyre::Result<()> {
        // Per-row, for the same reason as `hc_mixes`: at B=5 the batched WMMA
        // GEMM returned a constant, so every token routed to experts [0, 1, 2]
        // with weight 0.5 — a uniform softmax wearing a router's clothes.
        for j in 0..MTP_BLOCK {
            let xj = self.normed.slice_view(j * N_EMBD as usize, N_EMBD as usize);
            let mut lj = self
                .router_logits
                .slice_view_mut(j * MTP_N_EXPERT, MTP_N_EXPERT);
            e.f16.matvec(s, &mut lj, &w.ffn_gate_inp.buffer, &xj, MTP_N_EXPERT as u32, N_EMBD)?;
        }
        for j in 0..MTP_BLOCK {
            let lg = self.router_logits.slice_view(j * MTP_N_EXPERT, MTP_N_EXPERT);
            let mut sel = self.d_selected.slice_view_mut(j * MTP_TOPK as usize, MTP_TOPK as usize);
            let mut ew = self.d_ew.slice_view_mut(j * MTP_TOPK as usize, MTP_TOPK as usize);
            e.router_topk.launch(
                s, &mut sel, &mut ew, &lg, Some(&w.exp_probs_b), MTP_N_EXPERT as u32, MTP_TOPK,
                EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS,
            )?;
        }
        e.q8k
            .launch(s, &mut self.ffn_xq, &self.normed, BLOCKS_Q8K_GATE_IN * B)?;
        // `moe_*_hetsplit`'s `n_rows` is the OUTPUT WIDTH, not a token batch —
        // these kernels batch over the `n_used` expert slots of ONE token (box
        // 2's executor passes N_FF_EXP / N_EMBD there). So the block's rows are
        // a loop, not a batch dimension. Passing B=5 here reached the MXFP4
        // pair kernel as `n_rows=5` and tripped its `% 8` check.
        let tk = MTP_TOPK as usize;
        let ffe = N_FF_EXP as usize;
        let ne = N_EMBD as usize;
        for j in 0..(if no_routed() { 0 } else { MTP_BLOCK }) {
            let xq_j = self.ffn_xq.slice_view(j * XQ_BYTES_PER_TOKEN, XQ_BYTES_PER_TOKEN);
            let ew_j = self.d_ew.slice_view(j * tk, tk);
            let sel_j = self.d_selected.slice_view(j * tk, tk);
            let mut mid_j = self.mid.slice_view_mut(j * tk * ffe, tk * ffe);
            crate::het::dispatch::moe_gate_up_batch_hetsplit(
                e, w.routed.gate.dtype, s, &mut mid_j, &w.routed.gate.buffer,
                &w.routed.up.buffer, &xq_j, &ew_j, &sel_j, &self.remap, 0, MTP_TOPK,
                w.routed.gate_bytes_per_expert as u32, w.routed.up_bytes_per_expert as u32,
                MTP_TOPK, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            )?;
            let mut midq_j = self
                .midq
                .slice_view_mut(j * tk * MIDQ_BYTES_PER_SLOT, tk * MIDQ_BYTES_PER_SLOT);
            e.q8k
                .launch(s, &mut midq_j, &mid_j, BLOCKS_Q8K_DOWN_IN * MTP_TOPK)?;
            let mut out_j = self.ffn_out.slice_view_mut(j * ne, ne);
            crate::het::dispatch::moe_down_batched_hetsplit(
                e, w.routed.down.dtype, s, &mut out_j, &w.routed.down.buffer, &midq_j, &sel_j,
                &self.remap, 0, MTP_TOPK, w.routed.down_bytes_per_expert as u32,
                MIDQ_BYTES_PER_SLOT as u32, MTP_TOPK, N_EMBD, BLOCKS_Q8K_DOWN_IN,
            )?;
        }

        if no_routed() {
            let z = vec![0.0f32; self.ffn_out.len()];
            self.ffn_out.copy_from_host(&z)?;
        }
        let dbg = debug_moe()
            && MOE_DBG_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6;
        let routed_rms = if dbg {
            s.synchronize()?;
            let mut sel = vec![0i32; MTP_BLOCK * MTP_TOPK as usize];
            self.d_selected.copy_to_host(&mut sel)?;
            let mut ew = vec![0.0f32; MTP_BLOCK * MTP_TOPK as usize];
            self.d_ew.copy_to_host(&mut ew)?;
            let mut o = vec![0.0f32; self.ffn_out.len()];
            self.ffn_out.copy_to_host(&mut o)?;
            let rms = (o.iter().map(|v| v * v).sum::<f32>() / o.len() as f32).sqrt();
            tracing::info!(
                selected = ?&sel[..MTP_TOPK as usize * 2],
                weights = ?&ew[..MTP_TOPK as usize * 2],
                routed_rms = format!("{rms:.5}"),
                "mtp.moe.dbg"
            );
            Some(rms)
        } else {
            None
        };
        // Shared expert, same swiglu clamp as the routed path.
        for j in 0..MTP_BLOCK {
            let xr = self.normed.slice_view(j * ne, ne);
            let xqr = self.xq.slice_view(j * ne, ne);
            let xsr = self.xscale.slice_view(j * ne.div_ceil(32), ne.div_ceil(32));
            let nf = N_FF_SHARED as usize;
            let mut g = self.gate_sh.slice_view_mut(j * nf, nf);
            crate::het::dispatch::dense_matvec(
                e, s, &mut g, &w.shared.gate, &xr, &xqr, &xsr, N_FF_SHARED, N_EMBD,
            )?;
            let mut u = self.up_sh.slice_view_mut(j * nf, nf);
            crate::het::dispatch::dense_matvec(
                e, s, &mut u, &w.shared.up, &xr, &xqr, &xsr, N_FF_SHARED, N_EMBD,
            )?;
        }
        e.swiglu.launch_clamped(
            s, &mut self.mid_sh, &self.gate_sh, &self.up_sh, N_FF_SHARED * B, SWIGLU_CLAMP_EXP,
        )?;
        e.q8.quantize_input_batched(
            s, &mut self.mid_sh_xq, &mut self.mid_sh_xscale, &self.mid_sh, N_FF_SHARED, B,
        )?;
        for j in 0..MTP_BLOCK {
            let nf = N_FF_SHARED as usize;
            let mr = self.mid_sh.slice_view(j * nf, nf);
            let mq = self.mid_sh_xq.slice_view(j * nf, nf);
            let ms = self.mid_sh_xscale.slice_view(j * nf.div_ceil(32), nf.div_ceil(32));
            let mut o = self.ffn_shared.slice_view_mut(j * ne, ne);
            crate::het::dispatch::dense_matvec(
                e, s, &mut o, &w.shared.down, &mr, &mq, &ms, N_EMBD, N_FF_SHARED,
            )?;
        }
        if let Some(rr) = routed_rms {
            s.synchronize()?;
            let mut sh = vec![0.0f32; self.ffn_shared.len()];
            self.ffn_shared.copy_to_host(&mut sh)?;
            let srms = (sh.iter().map(|v| v * v).sum::<f32>() / sh.len() as f32).sqrt();
            tracing::info!(
                routed_rms = format!("{rr:.5}"),
                shared_rms = format!("{srms:.5}"),
                ratio = format!("{:.3}", rr / srms.max(1e-9)),
                "mtp.moe.dbg.shared"
            );
        }
        e.vec_add
            .launch(s, &mut self.ffn_out, &self.ffn_shared, N_EMBD * B)?;
        Ok(())
    }
}


// ---------------------------------------------------------------------------
// Exit stage
// ---------------------------------------------------------------------------

/// The drafter's exit: collapse, norm, the TIED head, and the markov loop.
///
/// Mirrors `DSparkBlock.forward_head`:
///
///     x      = hc_pre(h, pre_mix)                     [B, N_EMBD]
///     logits = head(norm(x), full_logits=True)        [B, N_VOCAB]
///     ids[0] = the accepted token
///     for i in 0..B:                       # SEQUENTIAL, and the only place
///         bias, memb = markov_head(ids[i]) # the block's positions talk
///         logits[i] += bias
///         ids[i+1]   = sample(logits[i])
///
/// The transformer produced all B positions in one bidirectional pass over
/// noise placeholders, so position i has no idea what was drafted at i-1. The
/// markov head — a rank-256 factorisation of a full `[vocab, vocab]` bigram
/// table — is the only channel that carries it, which is why this loop is
/// serial while everything above it is not.
///
/// `head` is the MAIN model's `output` weight: `Transformer.__init__` assigns
/// `mtp[s].head = self.head` for every stage, so the drafter adds no vocab
/// projection of its own.
pub struct MtpExit {
    /// The drafter's final residual and carried pre-mix, staged on THIS device.
    /// The layer stack runs on the iGPU and the exit on the dGPU (that is where
    /// the tied head lives), so they are handed across through the host: 410 KB
    /// a step, which at decode rates is far below the noise floor and avoids
    /// the peer-copy stream rule entirely.
    h: DeviceBuffer<f32>,
    pre_carry: DeviceBuffer<f32>,
    /// `hc_pre` output, PRE-norm — what the confidence head reads.
    pub x: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    xq: DeviceBuffer<i8>,
    xscale: DeviceBuffer<f32>,
    /// `[B, N_VOCAB]`, markov bias already folded in.
    pub logits: DeviceBuffer<f32>,
    memb: DeviceBuffer<f32>,
    memb_xq: DeviceBuffer<i8>,
    memb_xscale: DeviceBuffer<f32>,
    bias: DeviceBuffer<f32>,
    tok_dev: DeviceBuffer<i32>,
}

impl MtpExit {
    pub fn alloc(device_id: i32) -> eyre::Result<Self> {
        let b = MTP_BLOCK;
        let ne = N_EMBD as usize;
        let nv = N_VOCAB as usize;
        let mr = MTP_MARKOV_RANK;
        Ok(Self {
            h: DeviceBuffer::new(device_id, b * HC_DIM as usize)?,
            pre_carry: DeviceBuffer::new(device_id, b * HC_MIX_DIM as usize)?,
            x: DeviceBuffer::new(device_id, b * ne)?,
            xn: DeviceBuffer::new(device_id, b * ne)?,
            xq: DeviceBuffer::new(device_id, b * ne)?,
            xscale: DeviceBuffer::new(device_id, b * ne.div_ceil(32))?,
            logits: DeviceBuffer::new(device_id, b * nv)?,
            memb: DeviceBuffer::new(device_id, mr)?,
            memb_xq: DeviceBuffer::new(device_id, mr)?,
            memb_xscale: DeviceBuffer::new(device_id, mr.div_ceil(32))?,
            bias: DeviceBuffer::new(device_id, nv)?,
            tok_dev: DeviceBuffer::new(device_id, 1)?,
        })
    }

    /// One markov-embedding row, host-side. Kept off-device for the same reason
    /// `token_embd` is: the ids are produced one at a time by the loop below,
    /// so a device gather would serialise on the same readback anyway.
    fn markov_row(
        bytes: &[u8],
        dtype: v4flash_core::gguf::GgufType,
        id: i32,
        out: &mut [f32],
    ) -> eyre::Result<()> {
        use v4flash_core::gguf::GgufType;
        let k = MTP_MARKOV_RANK;
        match dtype {
            GgufType::F16 => {
                let off = id as usize * k * 2;
                let row = bytes
                    .get(off..off + k * 2)
                    .ok_or_else(|| eyre!("markov row: token {id} out of range"))?;
                for i in 0..k {
                    out[i] =
                        crate::iq2_xxs_tables::f16_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
                }
                Ok(())
            }
            other => Err(eyre!("markov embed dtype {other:?} unsupported")),
        }
    }

    /// Run the exit and return the `MTP_BLOCK` draft tokens.
    ///
    /// Greedy (`argmax`) only for now: the end-to-end gate is that speculative
    /// decode reproduces non-speculative output byte for byte, which is a greedy
    /// comparison. Temperature sampling needs the drafter and the verifier to
    /// share a random stream and is a separate change.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        e: &DeviceEngine,
        s: &Stream,
        h_host: &[f32],
        pre_host: &[f32],
        w: &MtpExitWeights,
        head: &crate::weights::DeviceWeight,
        markov_embd: &[u8],
        markov_dtype: v4flash_core::gguf::GgufType,
        first_token: i32,
    ) -> eyre::Result<([i32; MTP_BLOCK], [i32; MTP_BLOCK])> {
        let ne = N_EMBD as usize;
        let nv = N_VOCAB as usize;
        if h_host.len() != MTP_BLOCK * HC_DIM as usize
            || pre_host.len() != MTP_BLOCK * HC_MIX_DIM as usize
        {
            return Err(eyre!(
                "mtp exit: h {} / pre {} floats, want {} / {}",
                h_host.len(),
                pre_host.len(),
                MTP_BLOCK * HC_DIM as usize,
                MTP_BLOCK * HC_MIX_DIM as usize
            ));
        }
        self.h.copy_from_host(h_host)?;
        self.pre_carry.copy_from_host(pre_host)?;

        e.hc_weighted.launch_batched(
            s, &mut self.x, &self.h, &self.pre_carry, N_EMBD, N_HC, HC_MIX_DIM, B,
        )?;
        e.rms_w
            .launch_weighted_batched(s, &mut self.xn, &self.x, &w.norm, N_EMBD, RMS_EPS, B)?;
        e.q8
            .quantize_input_batched(s, &mut self.xq, &mut self.xscale, &self.xn, N_EMBD, B)?;
        for j in 0..MTP_BLOCK {
            let xj = self.xn.slice_view(j * ne, ne);
            let qj = self.xq.slice_view(j * ne, ne);
            let sj = self.xscale.slice_view(j * ne.div_ceil(32), ne.div_ceil(32));
            let mut lj = self.logits.slice_view_mut(j * nv, nv);
            crate::het::dispatch::dense_matvec(e, s, &mut lj, head, &xj, &qj, &sj, N_VOCAB, N_EMBD)?;
        }

        // Transformer-only drafts, before any markov bias. The markov head is
        // the one component with no counterpart in the main model, so scoring
        // both tells us which half is wrong without a second run.
        let mut plain = [0i32; MTP_BLOCK];
        let mut got = [0i32; 1];
        for j in 0..MTP_BLOCK {
            let lj = self.logits.slice_view(j * nv, nv);
            e.sampler.launch_argmax(s, &mut self.tok_dev, &lj, N_VOCAB)?;
            s.synchronize()?;
            self.tok_dev.copy_to_host(&mut got)?;
            plain[j] = got[0];
        }

        let mut ids = [0i32; MTP_BLOCK];
        let mut prev = first_token;
        let mut row = vec![0.0f32; MTP_MARKOV_RANK];
        for i in 0..MTP_BLOCK {
            Self::markov_row(markov_embd, markov_dtype, prev, &mut row)?;
            self.memb.copy_from_host(&row)?;
            e.q8.quantize_input(
                s, &mut self.memb_xq, &mut self.memb_xscale, &self.memb,
                MTP_MARKOV_RANK as u32,
            )?;
            crate::het::dispatch::dense_matvec(
                e, s, &mut self.bias, &w.markov_head, &self.memb, &self.memb_xq,
                &self.memb_xscale, N_VOCAB, MTP_MARKOV_RANK as u32,
            )?;
            let mut lj = self.logits.slice_view_mut(i * nv, nv);
            e.vec_add.launch(s, &mut lj, &self.bias, N_VOCAB)?;
            e.sampler.launch_argmax(s, &mut self.tok_dev, &lj, N_VOCAB)?;
            s.synchronize()?;
            self.tok_dev.copy_to_host(&mut got)?;
            ids[i] = got[0];
            prev = got[0];
        }
        Ok((ids, plain))
    }
}


/// dGPU-side capture of the residuals the drafter eats.
///
/// `MtpState::capture` collapses the hc copies where the residual already is,
/// which during decode is the dGPU. The drafter itself is iGPU-resident, so the
/// two are joined by a 60 KB readback rather than by running the entry on the
/// wrong device. Sized `[3 * N_EMBD]` — the concatenation is free because the
/// three means are written into adjacent slices of one buffer.
pub struct MtpCapture {
    src: DeviceBuffer<f32>,
    hc_mean: DeviceBuffer<f32>,
    captured: [bool; MTP_SRC_LAYERS.len()],
}

impl MtpCapture {
    pub fn alloc(device_id: i32) -> eyre::Result<Self> {
        let mut hc_mean = DeviceBuffer::<f32>::new(device_id, N_HC as usize)?;
        hc_mean.copy_from_host(&vec![1.0f32 / N_HC as f32; N_HC as usize])?;
        Ok(Self {
            src: DeviceBuffer::new(device_id, MTP_SRC_LAYERS.len() * N_EMBD as usize)?,
            hc_mean,
            captured: [false; MTP_SRC_LAYERS.len()],
        })
    }

    /// Start a token. Call before the layer loop.
    pub fn begin(&mut self) {
        self.captured = [false; MTP_SRC_LAYERS.len()];
    }

    /// Capture the residual ENTERING `layer`. A no-op for layers the drafter
    /// does not read, so the layer loop can call it unconditionally.
    pub fn on_layer(
        &mut self,
        e: &DeviceEngine,
        s: &Stream,
        layer: i32,
        residual: &DeviceBuffer<f32>,
    ) -> eyre::Result<()> {
        let Some(slot) = MtpState::src_slot(layer) else { return Ok(()) };
        let mut dst = self.src.slice_view_mut(slot * N_EMBD as usize, N_EMBD as usize);
        e.hc_weighted.launch(s, &mut dst, residual, &self.hc_mean, N_EMBD, N_HC)?;
        self.captured[slot] = true;
        Ok(())
    }

    /// Read the concatenated residuals back for `MtpState::inject_main_hidden`.
    /// Errors if a source layer was missed — drafting from a stale residual is
    /// exactly the failure that read as 0.44 acceptance for weeks.
    pub fn read(&self, out: &mut Vec<f32>) -> eyre::Result<()> {
        if let Some(i) = self.captured.iter().position(|c| !c) {
            return Err(eyre!(
                "mtp capture: residual for layer {} was never captured this token",
                MTP_SRC_LAYERS[i]
            ));
        }
        out.resize(MTP_SRC_LAYERS.len() * N_EMBD as usize, 0.0);
        self.src.copy_to_host(out)?;
        Ok(())
    }
}
