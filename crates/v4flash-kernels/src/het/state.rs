//! Per-layer state for the het orchestrator (KV cache + compressor state).
//!
//! In M13.1 everything lives on the dGPU (the device that runs attention
//! and reads the KV cache). M13.5 will migrate compressor state to the
//! iGPU and stream `comp_row` back to the dGPU on boundary tokens via a
//! tiny peer push.

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer};

use crate::config::{COMPRESS_RATIOS, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LAYER, NEG_INF, SWA_WINDOW};
use crate::het::batch_scratch::B_MAX;

/// During batched prefill we need to hold the prior SWA-window AND the
/// current chunk's freshly-computed KVs together in cache so each token in
/// the batch can attend to its causally-valid window (which spans both)
/// — see the n_raw_offset_per attention parameter in forward_prefill.rs.
/// Outside of prefill chunks only the first SWA_WINDOW rows are used.
pub const KV_CACHE_ROWS: usize = SWA_WINDOW as usize + B_MAX;

/// Per-layer compressor state. After M13.5, `state_kv` and `state_score`
/// live on the **iGPU** (where the compressor kernels run), while
/// `comp_kv` stays on the **dGPU** (where `attn_mixed` reads it). On
/// each boundary the iGPU compressor emits a `comp_row` that's
/// peer-pushed to the dGPU's `comp_kv`.
pub struct HetCompressorState {
    /// iGPU-resident: compressor sliding state.
    pub state_kv: DeviceBuffer<f32>,
    pub state_score: DeviceBuffer<f32>,
    /// dGPU-resident: cumulative pooled comp-KV cache consumed by
    /// `attn_mixed`.
    pub comp_kv: DeviceBuffer<f32>,
    pub n_comp: u32,
    pub width: u32,
    pub head_dim: u32,
}

impl HetCompressorState {
    pub fn alloc(
        igpu_device: Device,
        dgpu_device: Device,
        ratio: u32,
        head_dim: u32,
        n_kv_max: u32,
    ) -> eyre::Result<Self> {
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let state_rows = ratio * coff;
        let n_state = (state_rows as usize) * (width as usize);
        let zeros = vec![0f32; n_state];
        let neg_inf = vec![NEG_INF; n_state];

        // State on iGPU.
        igpu_device.set_current()?;
        let mut state_kv: DeviceBuffer<f32> = DeviceBuffer::new(igpu_device.id, n_state)?;
        let mut state_score: DeviceBuffer<f32> = DeviceBuffer::new(igpu_device.id, n_state)?;
        state_kv.copy_from_host(&zeros)?;
        state_score.copy_from_host(&neg_inf)?;

        // comp_kv on dGPU.
        dgpu_device.set_current()?;
        let max_n_comp = (n_kv_max + ratio - 1) / ratio;
        let comp_kv_capacity = (max_n_comp as usize) * (head_dim as usize);
        let comp_kv: DeviceBuffer<f32> = DeviceBuffer::new(dgpu_device.id, comp_kv_capacity)?;
        Ok(Self {
            state_kv,
            state_score,
            comp_kv,
            n_comp: 0,
            width,
            head_dim,
        })
    }
}

pub struct HetLayerState {
    pub kv_cache: DeviceBuffer<f32>,
    pub n_raw: u32,
    pub compressor: Option<HetCompressorState>,
    pub indexer_compressor: Option<HetCompressorState>,
}

pub struct HetModelState {
    pub layers: Vec<HetLayerState>,
    pub n_kv_max: u32,
}

impl HetModelState {
    pub fn alloc(dgpu_device: Device, igpu_device: Device, n_kv_max: u32) -> eyre::Result<Self> {
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for layer in 0..N_LAYER {
            let ratio = COMPRESS_RATIOS[layer as usize];
            let compressor = if ratio > 0 {
                // M14L: attn compressor state moved to dGPU so the compressor
                // can run there alongside attn_input_norm with no peer push.
                Some(HetCompressorState::alloc(
                    dgpu_device,
                    dgpu_device,
                    ratio,
                    N_HEAD_DIM,
                    n_kv_max,
                )?)
            } else {
                None
            };
            let indexer_compressor = if ratio == 4 {
                Some(HetCompressorState::alloc(
                    igpu_device,
                    dgpu_device,
                    ratio,
                    N_INDEXER_HEAD_DIM,
                    n_kv_max,
                )?)
            } else {
                None
            };
            // Raw KV cache is sized SWA_WINDOW + B_MAX rows. The first
            // SWA_WINDOW slots hold the steady-state SWA-window contents
            // (which is all that decode/single-token attention sees).
            // During batched prefill we additionally use slots
            // [SWA_WINDOW .. SWA_WINDOW + chunk_b) for the chunk's
            // freshly-computed KVs so that each token's per-token
            // n_raw_offset_per can see its causally-valid window across
            // the prior+current boundary. After each chunk the last
            // SWA_WINDOW rows are evicted back down to slot [0..W).
            dgpu_device.set_current()?;
            let raw_rows = KV_CACHE_ROWS as u32;
            layers.push(HetLayerState {
                kv_cache: DeviceBuffer::new(
                    dgpu_device.id,
                    (raw_rows as usize) * (N_HEAD_DIM as usize),
                )?,
                n_raw: 0,
                compressor,
                indexer_compressor,
            });
        }
        Ok(Self {
            layers,
            n_kv_max,
        })
    }
}
