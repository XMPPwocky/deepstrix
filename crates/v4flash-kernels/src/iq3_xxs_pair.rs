//! IQ3_XXS gate+up fused-SwiGLU matvec — blk.26 of the unsloth UD-Q2_K_XL
//! mix (every other layer's gate/up is IQ2_XS).
//!
//! Distinct from [`crate::iq3_xxs`], which is the same format on the DOWN
//! projection: down is a single matrix, gate/up are a fused pair.
//!
//! Contract mirrors [`crate::iq2_xxs::Iq2XxsPairMatvec`]'s three
//! load-bearing variants; the prefill `chunked` kernel takes the same
//! work-items arrays as the iq2/q2k/iq3 by-expert family and writes the
//! SwiGLU-fused `mid` directly (no partials/reduce). CPU reference:
//! [`crate::iq3_xxs_tables::cpu_dot_iq3_xxs_q8_k`].

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const IQ3_XXS_GFX1201: &[u8] = include_bytes!(env!("KERNEL_IQ3_XXS_PAIR_MATVEC_GFX1201"));
const IQ3_XXS_GFX1151: &[u8] = include_bytes!(env!("KERNEL_IQ3_XXS_PAIR_MATVEC_GFX1151"));

pub const BLOCK_IQ3_XXS_PAIR_BYTES: usize = 98;

pub struct Iq3XxsPairMatvec {
    module: Module,
}

impl Iq3XxsPairMatvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            IQ3_XXS_GFX1201
        } else if arch.starts_with("gfx1151") {
            IQ3_XXS_GFX1151
        } else {
            return Err(eyre!("unsupported arch for iq3_xxs_pair_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Decode: fused gate+up+SwiGLU over the n_used selected experts.
    /// Contract identical to `Iq2XxsPairMatvec::launch_fused_swiglu_batch`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_batch(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<i32>,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("iq2_s fused_batch: n_rows={n_rows} not %8"));
        }
        if mid.len() < (n_used as usize) * (n_rows as usize) {
            return Err(eyre!("iq2_s mid: len {} < n_used*n_rows", mid.len()));
        }
        let function = self
            .module
            .get_function("iq3_xxs_pair_matvec_fused_swiglu_batch")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_used, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(),
            expert_w.raw(), selected.raw(), gate_bpe, up_bpe, clamp, n_rows, n_blocks
        ])
    }

    /// Decode het-split; contract identical to
    /// `Iq2XxsPairMatvec::launch_fused_swiglu_batch_hetsplit`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_batch_hetsplit(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<i32>,
        remap: &DeviceBuffer<i32>,
        mode: u32,
        dgpu_cap: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("iq2_s hetsplit: n_rows={n_rows} not %8"));
        }
        if remap.len() < 256 {
            return Err(eyre!("iq2_s hetsplit: remap len {} < 256", remap.len()));
        }
        let function = self
            .module
            .get_function("iq3_xxs_pair_matvec_fused_swiglu_batch_hetsplit")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_used, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(),
            expert_w.raw(), selected.raw(), remap.raw(), mode, dgpu_cap,
            gate_bpe, up_bpe, clamp, n_rows, n_blocks
        ])
    }

    /// Prefill: chunked by-expert (work-items interface), SwiGLU-fused
    /// output straight into `mid[B, n_used, n_rows]` — the (b, slot) pairs
    /// not present in `expert_members` must be pre-zeroed by the caller
    /// (same invariant as the down-family partials).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_chunked(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("iq2_s chunked: n_rows={n_rows} not %8"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("iq3_xxs_pair_matvec_fused_swiglu_chunked")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(),
            expert_w.raw(), group_count.raw(), expert_members.raw(), work_items.raw(),
            gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp,
            n_rows, n_blocks
        ])
    }
}
