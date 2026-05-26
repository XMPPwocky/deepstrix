//! M50: per-batch scratch for prefill.
//!
//! Phase 1 implementation: a `Vec<DgpuScratch>` + `Vec<IgpuScratch>` of
//! length `B_MAX`, each element being a complete single-token scratch.
//! `forward_prompt_batch` calls the existing single-token `forward_layer`
//! in a loop over batch elements, using `scratches[b]` per element.
//!
//! Memory cost at `B_MAX = 64`:
//! * dGPU: ~64 × 2 MB = ~128 MB (each `DgpuScratch` ~2 MB)
//! * iGPU: ~64 × 250 KB = ~16 MB
//!
//! Phase 2 will replace `Vec<DgpuScratch>` with a single contiguous
//! `BatchDgpuScratch` whose buffers are sized `[B_MAX × per_token_size]`,
//! and Phase 2 batched kernels read/write with per-batch offsets.

use color_eyre::eyre;
use v4flash_hip::Device;

use super::scratch::{DgpuScratch, IgpuScratch};

/// Max prefill batch size. Sized for ds4-parity (chunk=64 tokens).
pub const B_MAX: usize = 64;

/// Per-batch dGPU + iGPU scratch for prefill.
///
/// Phase 1: Holds ONE shared `DgpuScratch`/`IgpuScratch` used by the
/// engine's existing `forward_layer` (so captured HIP graphs replay
/// consistently — captures bake in scratch buffer pointers, so we
/// can't naively spread them across B independent scratches), plus B
/// small per-token residual buffers swapped in/out around each
/// per-layer call. KV cache + compressor state live in `HetModelState`
/// (per-layer, shared across batch).
///
/// In Phase 2 the shared scratch goes away and per-batch fields are
/// batch-extended (`[B × original_size]`) within a single struct, so
/// batched kernels can index directly without copies. For now this
/// keeps Phase 1 small and lets us validate the layer-major schedule.
pub struct BatchScratch {
    pub shared_dgpu: DgpuScratch,
    pub shared_igpu: IgpuScratch,
    /// Per-token residual buffers ping-ponged into shared scratch
    /// around each `forward_layer` call.
    pub per_token_residual: Vec<v4flash_hip::DeviceBuffer<f32>>,
    /// Per-token residual_next (post-layer-N output buffer).
    pub per_token_residual_next: Vec<v4flash_hip::DeviceBuffer<f32>>,
}

impl BatchScratch {
    pub fn alloc(dgpu_device: Device, igpu_device: Device) -> eyre::Result<Self> {
        use crate::forward::HC_DIM;
        let shared_dgpu = DgpuScratch::alloc(dgpu_device)?;
        let shared_igpu = IgpuScratch::alloc(igpu_device)?;
        dgpu_device.set_current()?;
        let mut per_token_residual = Vec::with_capacity(B_MAX);
        let mut per_token_residual_next = Vec::with_capacity(B_MAX);
        for _ in 0..B_MAX {
            per_token_residual.push(v4flash_hip::DeviceBuffer::new(
                dgpu_device.id,
                HC_DIM as usize,
            )?);
            per_token_residual_next.push(v4flash_hip::DeviceBuffer::new(
                dgpu_device.id,
                HC_DIM as usize,
            )?);
        }
        Ok(Self {
            shared_dgpu,
            shared_igpu,
            per_token_residual,
            per_token_residual_next,
        })
    }

    pub fn b_max(&self) -> usize {
        B_MAX
    }
}
