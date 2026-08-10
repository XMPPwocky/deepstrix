//! Laguna heterogeneous decode MoE split — the `laguna_moe_hot_cold_split`
//! partition kernel. Splits the dGPU-computed top-K routing into a HOT (dGPU-
//! resident) and COLD (iGPU-resident) subset so the two devices can compute
//! independent partial MoE sums concurrently. See `kernels/laguna_het_moe.hip`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const LAGUNA_HET_MOE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_LAGUNA_HET_MOE_GFX1201"));
const LAGUNA_HET_MOE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_LAGUNA_HET_MOE_GFX1151"));

pub struct LagunaHetMoeSplit {
    module: Module,
}

impl LagunaHetMoeSplit {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            LAGUNA_HET_MOE_GFX1201
        } else if arch.starts_with("gfx1151") {
            LAGUNA_HET_MOE_GFX1151
        } else {
            return Err(eyre!("unsupported arch for laguna_het_moe: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// Partition `sel`/`ew` (top-`n_used` global routing) into hot (dGPU) and
    /// cold (iGPU) selections using the per-layer `hot_map` (global expert id ->
    /// local dGPU slot, or -1). Both outputs are `[n_used]`, padded with the
    /// sentinel -1 in the slots owned by the other device.
    #[allow(clippy::too_many_arguments)]
    pub fn partition(
        &self,
        stream: &Stream,
        sel: &DeviceBuffer<i32>,
        ew: &DeviceBuffer<f32>,
        hot_map: &DeviceBuffer<i32>,
        hot_sel: &mut DeviceBuffer<i32>,
        hot_ew: &mut DeviceBuffer<f32>,
        cold_sel: &mut DeviceBuffer<i32>,
        cold_ew: &mut DeviceBuffer<f32>,
        n_used: u32,
    ) -> eyre::Result<()> {
        let f = self.module.get_function("laguna_moe_hot_cold_split")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (n_used.max(1), 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [
            sel.raw(), ew.raw(), hot_map.raw(),
            hot_sel.raw(), hot_ew.raw(), cold_sel.raw(), cold_ew.raw(), n_used
        ])
    }
}
