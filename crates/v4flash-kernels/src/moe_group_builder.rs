//! M50 by-expert MoE pre-pass — invert `d_selected[B, n_used]` into
//! per-expert (token, slot) lists so the by-expert iq2 kernel can
//! amortize each expert's weight reads across all tokens that picked it.
//!
//! Caller convention:
//! 1. `group_count[n_expert]` must be ZEROED before each launch.
//! 2. `expert_members[n_expert * max_per_expert]` is written; only
//!    `expert_members[e * max_per_expert + 0..group_count[e]]` after
//!    the launch is meaningful.
//! 3. `max_per_expert >= B` (worst-case all tokens pick same expert in
//!    same slot — actually `max_per_expert >= B * n_used` covers any
//!    pathological case but `B` is sufficient for V4-Flash where each
//!    of n_used slots picks a distinct expert).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MOE_GROUP_BUILDER_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MOE_GROUP_BUILDER_GFX1201"));
const MOE_GROUP_BUILDER_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MOE_GROUP_BUILDER_GFX1151"));
const MOE_WORK_ITEMS_BUILDER_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_MOE_WORK_ITEMS_BUILDER_GFX1201"));
const MOE_WORK_ITEMS_BUILDER_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_MOE_WORK_ITEMS_BUILDER_GFX1151"));

pub struct MoeGroupBuilder {
    module: Module,
    work_items_module: Module,
}

impl MoeGroupBuilder {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let (image, wi_image): (&[u8], &[u8]) = if arch.starts_with("gfx1201") {
            (MOE_GROUP_BUILDER_GFX1201, MOE_WORK_ITEMS_BUILDER_GFX1201)
        } else if arch.starts_with("gfx1151") {
            (MOE_GROUP_BUILDER_GFX1151, MOE_WORK_ITEMS_BUILDER_GFX1151)
        } else {
            return Err(eyre!("unsupported arch for moe_group_builder: {arch}"));
        };
        let module = Module::load_data(image)?;
        let work_items_module = Module::load_data(wi_image)?;
        Ok(Self {
            module,
            work_items_module,
        })
    }

    /// Phase 7.2: build work_items list from group_count for chunked
    /// by-expert dispatch. Each work item is `(expert_id << 16) | member_start`.
    /// Returns the kernel; the caller is responsible for copying
    /// `n_work_items[0]` to host (sync) to read the actual count for
    /// the main kernel's grid_y.
    ///
    /// Caller MUST zero `n_work_items` before this call. `work_items` is
    /// sized for the worst case; only the first `n_work_items[0]` entries
    /// are valid after the call.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_work_items(
        &self,
        stream: &Stream,
        work_items: &mut DeviceBuffer<i32>,
        n_work_items: &mut DeviceBuffer<i32>,
        group_count: &DeviceBuffer<i32>,
        n_expert: u32,
        chunk_size: u32,
        max_items: u32,
    ) -> eyre::Result<()> {
        let function = self
            .work_items_module
            .get_function("moe_work_items_builder")?;
        let block_x = 256u32;
        let grid_x = n_expert.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            work_items.raw(), n_work_items.raw(), group_count.raw(), n_expert, chunk_size, max_items
        ])
    }

    /// M50 hybrid dispatch: split-builder variant. Emits work items into two
    /// disjoint arrays based on actual chunk size, AND sorts each by expert.
    /// Chunks with size ≥ threshold go to `staged_work_items`; smaller ones
    /// go to `chunked_work_items`. Within each, expert e's chunks precede
    /// expert e+1's (consecutive WGs in the launch hit the same expert →
    /// L2 reuse). `n_staged` and `n_chunked` are written directly (no
    /// pre-zero required). Caller syncs + reads both before launching
    /// the respective iq2 kernels.
    ///
    /// Launch geometry: one block of `n_expert` threads (in-block prefix sum).
    /// Requires `n_expert ≤ 1024`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_work_items_split(
        &self,
        stream: &Stream,
        staged_work_items: &mut DeviceBuffer<i32>,
        chunked_work_items: &mut DeviceBuffer<i32>,
        n_staged: &mut DeviceBuffer<i32>,
        n_chunked: &mut DeviceBuffer<i32>,
        group_count: &DeviceBuffer<i32>,
        n_expert: u32,
        chunk_size: u32,
        threshold: u32,
        max_items: u32,
    ) -> eyre::Result<()> {
        let function = self
            .work_items_module
            .get_function("moe_work_items_builder_split")?;
        if n_expert > 1024 {
            return Err(eyre!(
                "moe_work_items_builder_split: n_expert={n_expert} exceeds 1024 (in-block prefix sum limit)"
            ));
        }
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (n_expert, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            staged_work_items.raw(), chunked_work_items.raw(), n_staged.raw(), n_chunked.raw(),
            group_count.raw(), n_expert, chunk_size, threshold, max_items
        ])
    }

    /// Build per-expert (token, slot) groups from `d_selected[B, n_used]`.
    /// Caller MUST zero `group_count` before this call.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &Stream,
        group_count: &mut DeviceBuffer<i32>,
        expert_members: &mut DeviceBuffer<i32>,
        d_selected: &DeviceBuffer<i32>,
        batch: u32,
        n_used: u32,
        n_expert: u32,
        max_per_expert: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let total = batch * n_used;
        if group_count.len() < n_expert as usize {
            return Err(eyre!("group_count too small"));
        }
        if expert_members.len() < (n_expert as usize) * (max_per_expert as usize) {
            return Err(eyre!("expert_members too small"));
        }
        if d_selected.len() < total as usize {
            return Err(eyre!("d_selected too small"));
        }
        let function = self.module.get_function("moe_group_builder")?;
        // 384 threads = B=64 * n_used=6; round up to 512 to give some headroom.
        let block_x = 512u32;
        let grid_x = total.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            group_count.raw(), expert_members.raw(), d_selected.raw(),
            batch, n_used, n_expert, max_per_expert
        ])
    }

    /// M61 prefill het-split: residency-aware group builder. `remap[e] >= 0`
    /// marks a dGPU-resident expert; a token's resident slots are ranked in
    /// slot order, ranks < `cap` go to the dGPU. `mode=0` keeps the
    /// complement in ORIGINAL id space (iGPU); `mode=1` keeps the hits in
    /// DENSE remap space (dGPU — pass `n_expert = n_hot`). Caller MUST zero
    /// `group_count` (stream-ordered) before this call.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_hetsplit(
        &self,
        stream: &Stream,
        group_count: &mut DeviceBuffer<i32>,
        expert_members: &mut DeviceBuffer<i32>,
        d_selected: &DeviceBuffer<i32>,
        remap: &DeviceBuffer<i32>,
        mode: u32,
        cap: u32,
        batch: u32,
        n_used: u32,
        n_expert: u32,
        max_per_expert: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let total = batch * n_used;
        if group_count.len() < n_expert as usize {
            return Err(eyre!("group_count too small"));
        }
        if expert_members.len() < (n_expert as usize) * (max_per_expert as usize) {
            return Err(eyre!("expert_members too small"));
        }
        if d_selected.len() < total as usize {
            return Err(eyre!("d_selected too small"));
        }
        let function = self.module.get_function("moe_group_builder_hetsplit")?;
        let block_x = 512u32;
        let grid_x = total.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            group_count.raw(), expert_members.raw(), d_selected.raw(), remap.raw(),
            mode, cap, batch, n_used, n_expert, max_per_expert
        ])
    }
}
