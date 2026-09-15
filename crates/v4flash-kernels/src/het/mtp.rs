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

use crate::config::{N_EMBD, N_HC, RMS_EPS};
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
    /// Q8 staging for the `main_proj` matvec.
    xq: DeviceBuffer<i8>,
    xscale: DeviceBuffer<f32>,
    /// `main_proj` output and its norm — the drafter's layer-0 input.
    pub proj: DeviceBuffer<f32>,
    pub x: DeviceBuffer<f32>,
    /// Which of `MTP_SRC_LAYERS` have been captured this token. Guards against
    /// running the entry on a stale or partial `main_x`.
    captured: [bool; MTP_SRC_LAYERS.len()],
}

impl MtpState {
    pub fn alloc(device_id: i32) -> eyre::Result<Self> {
        let k = MTP_SRC_LAYERS.len() * N_EMBD as usize;
        let mut hc_mean = DeviceBuffer::<f32>::new(device_id, N_HC as usize)?;
        hc_mean.copy_from_host(&vec![1.0f32 / N_HC as f32; N_HC as usize])?;
        Ok(Self {
            main_x: DeviceBuffer::new(device_id, k)?,
            hc_mean,
            xq: DeviceBuffer::new(device_id, k)?,
            xscale: DeviceBuffer::new(device_id, k.div_ceil(32))?,
            proj: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            x: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            captured: [false; MTP_SRC_LAYERS.len()],
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
            e, s, &mut self.proj, &w.main_proj, &self.main_x, &self.xq, &self.xscale, N_EMBD, k,
        )?;
        e.rms_w
            .launch_weighted(s, &mut self.x, &self.proj, &w.main_norm, N_EMBD, RMS_EPS)?;
        Ok(())
    }
}
