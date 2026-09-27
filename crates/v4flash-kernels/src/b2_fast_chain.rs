//! Box 2's small-batch decode chain kernels (kernels/b2_fast_chain.hip), used
//! only by `het::remote_experts::MoeExecutor` under `V41_B2_FAST_CHAIN`:
//!
//! * [`B2FastChain::launch_builder`] -- the group + work-item build of one
//!   by-expert pass in ONE work-group (replaces two memsets and two kernels);
//! * [`B2FastChain::launch_reduce_zero`] -- `q2_k_reduce_partials_hetsplit`,
//!   bit for bit, that also re-zeroes the partial rows it consumes and can
//!   write the f16 result straight into host-mapped pinned memory.
//!
//! See the .hip header for the contracts and why they are output-identical to
//! the chain they replace.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, PinnedBuffer, Stream};

use crate::moe_group_builder::MAX_MOE_GROUP_IDS;

const B2_FAST_CHAIN_GFX1201: &[u8] = include_bytes!(env!("KERNEL_B2_FAST_CHAIN_GFX1201"));
const B2_FAST_CHAIN_GFX1151: &[u8] = include_bytes!(env!("KERNEL_B2_FAST_CHAIN_GFX1151"));

/// Work-group size of the fused builder (`B2FB_MAX` in the kernel): the most
/// picks (`batch * n_used`) one launch can take.
pub const B2_BUILDER_MAX_PICKS: usize = 128;

pub struct B2FastChain {
    module: Module,
}

impl B2FastChain {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            B2_FAST_CHAIN_GFX1201
        } else if arch.starts_with("gfx1151") {
            B2_FAST_CHAIN_GFX1151
        } else {
            return Err(eyre!("unsupported arch for b2_fast_chain: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// `moe_group_builder_hetsplit` + `moe_work_items_builder` (and the two
    /// memsets before them) for one pass of `batch * n_used <=
    /// B2_BUILDER_MAX_PICKS` picks. Writes `group_count` / `expert_members` for
    /// the groups this pass touches, `work_items[..n]` and `n_work_items[0] =
    /// n`; nothing needs zeroing first. Consumers must bound their work-item
    /// reads by the device count (or an exact host copy of it), which every
    /// by-expert consumer already does.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_builder(
        &self,
        stream: &Stream,
        group_count: &mut DeviceBuffer<i32>,
        expert_members: &mut DeviceBuffer<i32>,
        work_items: &mut DeviceBuffer<i32>,
        n_work_items: &mut DeviceBuffer<i32>,
        d_selected: &DeviceBuffer<i32>,
        remap: &DeviceBuffer<i32>,
        mode: u32,
        cap: u32,
        batch: u32,
        n_used: u32,
        n_expert: u32,
        max_per_expert: u32,
        chunk_size: u32,
        max_items: u32,
    ) -> eyre::Result<()> {
        let total = batch as usize * n_used as usize;
        if total == 0 || total > B2_BUILDER_MAX_PICKS {
            return Err(eyre!("b2_moe_group_wi_builder: {total} picks outside 1..={B2_BUILDER_MAX_PICKS}"));
        }
        // Same packing limits as the builders it replaces (`MAX_MOE_GROUP_IDS`).
        if n_expert as usize > MAX_MOE_GROUP_IDS || max_per_expert as usize > MAX_MOE_GROUP_IDS {
            return Err(eyre!("b2_moe_group_wi_builder: n_expert={n_expert} / max_per_expert={max_per_expert} exceed the work-item packing"));
        }
        if chunk_size == 0 {
            return Err(eyre!("b2_moe_group_wi_builder: chunk_size 0"));
        }
        if group_count.len() < n_expert as usize
            || expert_members.len() < n_expert as usize * max_per_expert as usize
            || d_selected.len() < total
            || work_items.len() < max_items as usize
            || n_work_items.is_empty()
        {
            return Err(eyre!("b2_moe_group_wi_builder: buffers too small"));
        }
        let function = self.module.get_function("b2_moe_group_wi_builder")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (B2_BUILDER_MAX_PICKS as u32, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            group_count.raw(), expert_members.raw(), work_items.raw(), n_work_items.raw(),
            d_selected.raw(), remap.raw(), mode, cap, batch, n_used, n_expert, max_per_expert,
            chunk_size, max_items
        ])
    }

    /// `q2_k_reduce_partials_hetsplit` into `out` (same per-row sums, 8 rows
    /// per thread; `n_rows % 8 == 0`), then
    /// zero every consumed partial slot and every slot whose pick is a real id
    /// (`< zero_below`); with `out16`, also the f16 result (RNE, as
    /// `f32_to_f16_cast`) into that host-mapped pinned buffer, host-visible
    /// once a later event on `stream` completes.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_reduce_zero(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        out16: Option<&mut PinnedBuffer<u16>>,
        partials: &mut DeviceBuffer<f32>,
        d_selected: &DeviceBuffer<i32>,
        remap: &DeviceBuffer<i32>,
        mode: u32,
        cap: u32,
        n_used: u32,
        n_rows: u32,
        batch: u32,
        zero_below: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if n_rows % 8 != 0 {
            return Err(eyre!("b2_reduce_partials_zero: n_rows={n_rows} not a multiple of 8"));
        }
        let n_out = batch as usize * n_rows as usize;
        if out.len() < n_out
            || partials.len() < n_out * n_used as usize
            || d_selected.len() < batch as usize * n_used as usize
            || out16.as_ref().is_some_and(|o| o.len() < n_out)
        {
            return Err(eyre!("b2_reduce_partials_zero: buffers too small for batch={batch}"));
        }
        let out16_ptr = match out16 {
            Some(o) => o.device_ptr(),
            None => std::ptr::null_mut(),
        };
        let function = self.module.get_function("b2_reduce_partials_zero")?;
        let block: u32 = 256;
        // 8 rows per thread.
        let cfg = LaunchConfig { grid: ((n_out / 8).div_ceil(block as usize) as u32, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            out.raw(), out16_ptr, partials.raw(), d_selected.raw(), remap.raw(),
            mode, cap, n_used, n_rows, batch, zero_below
        ])
    }
}
