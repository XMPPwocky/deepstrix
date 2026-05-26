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

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const MOE_GROUP_BUILDER_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MOE_GROUP_BUILDER_GFX1201"));
const MOE_GROUP_BUILDER_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MOE_GROUP_BUILDER_GFX1151"));

pub struct MoeGroupBuilder {
    module: Module,
}

impl MoeGroupBuilder {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            MOE_GROUP_BUILDER_GFX1201
        } else if arch.starts_with("gfx1151") {
            MOE_GROUP_BUILDER_GFX1151
        } else {
            return Err(eyre!("unsupported arch for moe_group_builder: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
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
        let mut gc_ptr = group_count.raw();
        let mut em_ptr = expert_members.raw();
        let mut ds_ptr = d_selected.raw();
        let mut b = batch;
        let mut nu = n_used;
        let mut ne = n_expert;
        let mut mpe = max_per_expert;
        let mut args: [*mut c_void; 7] = [
            &mut gc_ptr as *mut _ as *mut c_void,
            &mut em_ptr as *mut _ as *mut c_void,
            &mut ds_ptr as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut mpe as *mut _ as *mut c_void,
        ];
        // 384 threads = B=64 * n_used=6; round up to 512 to give some headroom.
        let block_x = 512u32;
        let grid_x = total.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
