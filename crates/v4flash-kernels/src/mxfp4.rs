//! MXFP4 × Q8_K matvec family — routed-MoE down projection for blk.26 +
//! blk.42 of the unsloth UD mix (the official checkpoint's native expert
//! format — plausibly lossless weights on those layers).
//!
//! Same contract as the q2_k/iq3_xxs down family (Q8_K activations,
//! slot-major midq, q2_k_reduce_partials-compatible partials). NOTE:
//! `n_blocks_in` counts Q8_K superblocks (256 elems = 136 MXFP4 bytes).
//! CPU reference: [`crate::mxfp4_tables::cpu_dot_mxfp4_q8_k`].

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MXFP4_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MXFP4_MATVEC_GFX1201"));
const MXFP4_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MXFP4_MATVEC_GFX1151"));

pub const SUPER_MXFP4_BYTES: usize = 136;

pub struct Mxfp4Matvec {
    module: Module,
}

impl Mxfp4Matvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            MXFP4_GFX1201
        } else if arch.starts_with("gfx1151") {
            MXFP4_GFX1151
        } else {
            return Err(eyre!("unsupported arch for mxfp4_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Decode batched over `n_used` selected experts (graph-captured core).
    /// Contract identical to `Q2KAccumulateMatvec::launch_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        n_rows: u32,
        n_blocks_in: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4_matvec_par_batched: n_rows={n_rows} not %8"));
        }
        if out.len() < n_rows as usize {
            return Err(eyre!("mxfp4 batched out: len {} < n_rows {n_rows}", out.len()));
        }
        if (selected.len() as u32) < n_used {
            return Err(eyre!("selected len {} < n_used {n_used}", selected.len()));
        }
        let function = self.module.get_function("mxfp4_matvec_par_batched")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(), selected.raw(),
            dbpe, xq_slot_stride, n_used, n_rows, n_blocks_in
        ])
    }

    /// Decode het-split; contract identical to
    /// `Q2KAccumulateMatvec::launch_batched_hetsplit`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_hetsplit(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<i32>,
        remap: &DeviceBuffer<i32>,
        mode: u32,
        dgpu_cap: u32,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        n_rows: u32,
        n_blocks_in: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4 hetsplit: n_rows={n_rows} not %8"));
        }
        if remap.len() < 256 {
            return Err(eyre!("mxfp4 hetsplit: remap len {} < 256", remap.len()));
        }
        let function = self
            .module
            .get_function("mxfp4_matvec_par_batched_hetsplit")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(), selected.raw(), remap.raw(), mode, dgpu_cap,
            dbpe, xq_slot_stride, n_used, n_rows, n_blocks_in
        ])
    }

    /// Prefill by-expert kwide2 (production analog of
    /// `Q2KAccumulateMatvec::launch_by_expert_kwide2`): grid
    /// `(n_rows/16, n_work_items)`, 2 rows per warp, members in halves of
    /// 16, weights unpacked once per (row, block). Caller zeroes
    /// `partials` and pairs with `q2_k_reduce_partials`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_by_expert_kwide2(
        &self,
        stream: &Stream,
        partials: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        n_rows: u32,
        n_blocks_in: u32,
    ) -> eyre::Result<()> {
        if n_rows % 16 != 0 {
            return Err(eyre!("mxfp4 kwide2: n_rows={n_rows} not %16"));
        }
        if chunk_size == 0 || chunk_size > 32 {
            return Err(eyre!("mxfp4 kwide2: chunk_size={chunk_size} not in 1..=32"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("mxfp4_matvec_par_by_expert_kwide2")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 16, n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            partials.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(), work_items.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, chunk_size,
            n_rows, n_blocks_in
        ])
    }
}
