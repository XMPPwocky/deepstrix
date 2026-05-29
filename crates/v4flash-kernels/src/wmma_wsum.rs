//! WMMA wsum de-risk prototype bindings — see kernels/wmma_wsum.hip.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const WMMA_WSUM_GFX1201: &[u8] = include_bytes!(env!("KERNEL_WMMA_WSUM_GFX1201"));

pub struct WmmaWsum {
    module: Module,
}

impl WmmaWsum {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        if !arch.starts_with("gfx1201") {
            return Err(eyre!("wmma_wsum prototype only supports gfx1201, got {arch}"));
        }
        Ok(Self {
            module: Module::load_data(WMMA_WSUM_GFX1201)?,
        })
    }

    /// O[M,N] = inv[m] * sum_k W[m,k] * V[k,n]. W/V are f16 bits (u16).
    /// Grid: (ceil(N/16), ceil(M/16)); one wave32 warp per output 16x16 tile.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_wmma(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w: &DeviceBuffer<u16>,
        v: &DeviceBuffer<u16>,
        inv: &DeviceBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("wsum_wmma_f16")?;
        let cfg = LaunchConfig {
            grid: (n.div_ceil(16), m.div_ceil(16), 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), w.raw(), v.raw(), inv.raw(), m, n, k])
    }

    /// One-shot fragment-layout calibration. Writes 32*8 floats.
    pub fn launch_layout_probe(
        &self,
        stream: &Stream,
        raw_out: &mut DeviceBuffer<f32>,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("wmma_layout_probe")?;
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [raw_out.raw()])
    }

    /// Faithful f32 baseline (tuned htiled Phase B). Grid (ceil(M/16),), block 512.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_f32_ref(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        inv: &DeviceBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("wsum_f32_ref")?;
        let cfg = LaunchConfig {
            grid: (m.div_ceil(16), 1, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), w.raw(), v.raw(), inv.raw(), m, n, k])
    }
}
