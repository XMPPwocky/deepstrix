//! M62 — device-side expert-selection histogram (see kernels/expert_sel_count.hip).
//!
//! Accumulates router picks into a persistent per-bank
//! `u32[N_LAYER × N_EXPERT]` buffer on the dGPU. Prefill and decode each
//! get their own bank; the server harvests and zeroes the banks at the
//! existing snapshot-save points (turn end / shutdown).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const EXPERT_SEL_COUNT_GFX1201: &[u8] = include_bytes!(env!("KERNEL_EXPERT_SEL_COUNT_GFX1201"));
const EXPERT_SEL_COUNT_GFX1151: &[u8] = include_bytes!(env!("KERNEL_EXPERT_SEL_COUNT_GFX1151"));

pub struct ExpertSelCount {
    module: Module,
}

impl ExpertSelCount {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            EXPERT_SEL_COUNT_GFX1201
        } else if arch.starts_with("gfx1151") {
            EXPERT_SEL_COUNT_GFX1151
        } else {
            return Err(eyre!("unsupported arch for expert_sel_count: {arch}"));
        };
        Ok(Self {
            module: Module::load_data(image)?,
        })
    }

    /// Accumulate `d_selected[0..total]` picks into bank slice
    /// `counts[layer*n_expert ..][e]`. `total = B × n_used`.
    pub fn launch(
        &self,
        stream: &Stream,
        counts: &mut DeviceBuffer<u32>,
        d_selected: &DeviceBuffer<i32>,
        layer: u32,
        n_expert: u32,
        total: u32,
    ) -> eyre::Result<()> {
        if total == 0 {
            return Ok(());
        }
        if counts.len() < ((layer + 1) * n_expert) as usize {
            return Err(eyre!("expert_sel_count: counts bank too small"));
        }
        if d_selected.len() < total as usize {
            return Err(eyre!("expert_sel_count: d_selected too small"));
        }
        let function = self.module.get_function("expert_sel_count")?;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid: (total.div_ceil(block), 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            counts.raw(), d_selected.raw(), layer * n_expert, n_expert, total
        ])
    }
}
