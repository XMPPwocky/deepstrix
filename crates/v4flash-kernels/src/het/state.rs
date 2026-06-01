//! Per-layer state for the het orchestrator (KV cache + compressor state).
//!
//! All state lives on the dGPU (the device that runs attention and reads
//! the KV cache). The compressor's iGPU-resident era was rolled back —
//! see [`HetCompressorState`].

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer};

use crate::config::{COMPRESS_RATIOS, N_HEAD_DIM, N_LAYER, NEG_INF, SWA_WINDOW};
use crate::het::batch_scratch::B_MAX;

/// During batched prefill we need to hold the prior SWA-window AND the
/// current chunk's freshly-computed KVs together in cache so each token in
/// the batch can attend to its causally-valid window (which spans both)
/// — see the n_raw_offset_per attention parameter in forward_prefill.rs.
/// Outside of prefill chunks only the first SWA_WINDOW rows are used.
pub const KV_CACHE_ROWS: usize = SWA_WINDOW as usize + B_MAX;

/// Per-layer compressor state. All buffers live on the dGPU: the
/// compressor kernels run alongside attn_input_norm on dGPU, so
/// `state_kv` / `state_score` (sliding compressor state) and `comp_kv`
/// (the cumulative pooled cache `attn_mixed` reads) are all local —
/// no peer push needed when the compressor fires at a boundary.
pub struct HetCompressorState {
    /// iGPU-resident: compressor sliding state.
    pub state_kv: DeviceBuffer<f32>,
    pub state_score: DeviceBuffer<f32>,
    /// dGPU-resident: cumulative pooled comp-KV cache consumed by
    /// `attn_mixed`. Stored as f16 (held as u16) — V values come out of
    /// the compressor as f32, get cast at the comp_kv_append store,
    /// halving DRAM bw for the dominant V-read cost in long-context attention.
    pub comp_kv: DeviceBuffer<u16>,
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
        let comp_kv: DeviceBuffer<u16> = DeviceBuffer::new(dgpu_device.id, comp_kv_capacity)?;
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
    /// SWA raw KV cache. f16-stored (see `HetCompressorState::comp_kv` rationale).
    pub kv_cache: DeviceBuffer<u16>,
    pub n_raw: u32,
    pub compressor: Option<HetCompressorState>,
}

pub struct HetModelState {
    pub layers: Vec<HetLayerState>,
    pub n_kv_max: u32,
}

impl HetModelState {
    pub fn alloc(dgpu_device: Device, _igpu_device: Device, n_kv_max: u32) -> eyre::Result<Self> {
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for layer in 0..N_LAYER {
            let ratio = COMPRESS_RATIOS[layer as usize];
            let compressor = if ratio > 0 {
                // Attn compressor state lives on dGPU alongside attn_input_norm
                // (no peer push needed for the boundary `comp_row` write).
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
                kv_cache: DeviceBuffer::<u16>::new(
                    dgpu_device.id,
                    (raw_rows as usize) * (N_HEAD_DIM as usize),
                )?,
                n_raw: 0,
                compressor,
            });
        }
        Ok(Self {
            layers,
            n_kv_max,
        })
    }
}
