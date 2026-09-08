//! IQ2_XS gate+up fused-SwiGLU matvec — 42 of 43 layers in the unsloth
//! UD-Q2_K_XL mix (blk.26's gate/up is IQ3_XXS).
//!
//! Contract mirrors [`crate::iq2_xxs::Iq2XxsPairMatvec`]'s three
//! load-bearing variants; the prefill `chunked` kernel takes the same
//! work-items arrays as the iq2/q2k/iq3 by-expert family and writes the
//! SwiGLU-fused `mid` directly (no partials/reduce). CPU reference:
//! [`crate::iq2_xs_tables::cpu_dot_iq2_xs_q8_k`].

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const IQ2_XS_GFX1201: &[u8] = include_bytes!(env!("KERNEL_IQ2_XS_PAIR_MATVEC_GFX1201"));
const IQ2_XS_GFX1151: &[u8] = include_bytes!(env!("KERNEL_IQ2_XS_PAIR_MATVEC_GFX1151"));

pub const BLOCK_IQ2_XS_BYTES: usize = 74;

pub struct Iq2XsPairMatvec {
    module: Module,
    rdna3: bool,
}

impl Iq2XsPairMatvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            IQ2_XS_GFX1201
        } else if arch.starts_with("gfx1151") {
            IQ2_XS_GFX1151
        } else {
            return Err(eyre!("unsupported arch for iq2_xs_pair_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module, rdna3: arch.starts_with("gfx11") })
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
            .get_function("iq2_xs_pair_matvec_fused_swiglu_batch")?;
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
            .get_function("iq2_xs_pair_matvec_fused_swiglu_batch_hetsplit")?;
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
            .get_function("iq2_xs_pair_matvec_fused_swiglu_chunked")?;
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

    /// Prefill kwide (M51-structure port, 2026-08-15): dequant each lane's
    /// 16 gate+up weights ONCE per super-block pair, amortized across all
    /// chunk members (the `_chunked` kernel re-dequants per member).
    /// Same work-items contract as `launch_fused_swiglu_chunked`;
    /// additionally requires `n_blocks % 2 == 0` and `chunk_size <= 32`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_kwide(
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
            return Err(eyre!("iq2_xs kwide: n_rows={n_rows} not %8"));
        }
        if n_blocks % 2 != 0 {
            return Err(eyre!("iq2_xs kwide: n_blocks={n_blocks} must be even"));
        }
        if chunk_size > 32 {
            return Err(eyre!(
                "iq2_xs kwide: chunk_size={chunk_size} exceeds IQ2XS_KW_MAX_CHUNK=32"
            ));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("iq2_xs_pair_matvec_fused_swiglu_kwide")?;
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

    /// f16 WMMA batched-prefill gate/up (2026-09-08; gfx11 only). Same
    /// work-item contract as [`Self::launch_fused_swiglu_kwide`] but takes
    /// f16 activations `x16` = [B, n_blocks*256] instead of Q8_K. `mode` 0
    /// is the kernel; other modes are bench-only twins (see below). Isolated-bench + oracle only so far.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_wmma(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        x16: &DeviceBuffer<u16>,
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
        mode: u32,
    ) -> eyre::Result<()> {
        if !self.rdna3 {
            return Err(eyre!("iq2_xs wmma: RDNA3 (gfx11) WMMA layout only"));
        }
        if n_rows % 16 != 0 {
            return Err(eyre!("iq2_xs wmma: n_rows={n_rows} not %16"));
        }
        if chunk_size > 32 {
            return Err(eyre!(
                "iq2_xs wmma: chunk_size={chunk_size} exceeds IQ2XS_WM_MAX_CHUNK=32"
            ));
        }
        if x16.len() < (n_blocks as usize) * 256 {
            return Err(eyre!("iq2_xs wmma: x16 too small for n_blocks={n_blocks}"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        // mode: 0 = the kernel (magic-number dequant); bench-only twins:
        // 1 = per-byte cvt dequant, 6 = f16-LUT dequant, 2..=5 = ablations 1..4
        // (no WMMA / no dequant / no x staging / no B loads) — garbage output.
        let name = match mode {
            0 => "iq2_xs_pair_matvec_fused_swiglu_wmma",
            1 => "iq2_xs_pair_matvec_fused_swiglu_wmma_cvt",
            6 => "iq2_xs_pair_matvec_fused_swiglu_wmma_lut",
            2 => "iq2_xs_pair_matvec_fused_swiglu_wmma_abl1",
            3 => "iq2_xs_pair_matvec_fused_swiglu_wmma_abl2",
            4 => "iq2_xs_pair_matvec_fused_swiglu_wmma_abl3",
            5 => "iq2_xs_pair_matvec_fused_swiglu_wmma_abl4",
            m => return Err(eyre!("iq2_xs wmma: unknown mode {m}")),
        };
        let function = self.module.get_function(name)?;
        let cfg = LaunchConfig {
            grid: (n_rows.div_ceil(128), n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid.raw(), gate_w_base.raw(), up_w_base.raw(), x16.raw(),
            expert_w.raw(), group_count.raw(), expert_members.raw(), work_items.raw(),
            gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp,
            n_rows, n_blocks
        ])
    }

    /// Bench-only: gfx11 WMMA operand-lane probe (see kernel banner).
    pub fn launch_upper_lane_probe(&self, stream: &Stream, out: &mut DeviceBuffer<f32>) -> eyre::Result<()> {
        if out.len() < 9 * 256 { return Err(eyre!("probe: out < 2304")); }
        let function = self.module.get_function("iq2_xs_wmma_upper_lane_probe")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [out.raw()])
    }

    /// Production form of [`Self::launch_fused_swiglu_wmma`]: writes f16 mid
    /// `[B*n_used][n_rows]` (the down kernel's B operand). gfx11 only.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_wmma_f16out(
        &self,
        stream: &Stream,
        mid16: &mut DeviceBuffer<u16>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        x16: &DeviceBuffer<u16>,
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
        if !self.rdna3 {
            return Err(eyre!("iq2_xs wmma: RDNA3 (gfx11) WMMA layout only"));
        }
        if n_rows % 16 != 0 {
            return Err(eyre!("iq2_xs wmma: n_rows={n_rows} not %16"));
        }
        if chunk_size > 32 {
            return Err(eyre!("iq2_xs wmma: chunk_size={chunk_size} exceeds 32"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self.module.get_function("iq2_xs_pair_matvec_fused_swiglu_wmma_h")?;
        let cfg = LaunchConfig {
            grid: (n_rows.div_ceil(128), n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid16.raw(), gate_w_base.raw(), up_w_base.raw(), x16.raw(),
            expert_w.raw(), group_count.raw(), expert_members.raw(), work_items.raw(),
            gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp,
            n_rows, n_blocks
        ])
    }
}
