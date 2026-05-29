//! WMMA throughput probe — see kernels/wmma_probe.hip.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const WMMA_PROBE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_WMMA_PROBE_GFX1201"));
const WMMA_PROBE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_WMMA_PROBE_GFX1151"));

pub struct WmmaProbe {
    module: Module,
}

impl WmmaProbe {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            WMMA_PROBE_GFX1201
        } else if arch.starts_with("gfx1151") {
            WMMA_PROBE_GFX1151
        } else {
            return Err(eyre!("unsupported arch for wmma_probe: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Run the WMMA probe. Each warp does `n_iters` back-to-back WMMA
    /// fragment ops on the same accumulator (serial dependency chain
    /// stressing per-warp issue rate). Total ops emitted =
    /// `n_blocks × (block_threads / 32) × n_iters × 8192`.
    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        a_in: &DeviceBuffer<u16>, // f16 bits
        b_in: &DeviceBuffer<u16>,
        n_iters: u32,
        n_blocks: u32,
        block_threads: u32,
    ) -> eyre::Result<()> {
        self.launch_named(stream, "wmma_f16_throughput_probe", out, a_in, b_in, n_iters, n_blocks, block_threads)
    }

    /// Parallel-accumulator variant: each iter does 8 independent WMMAs.
    /// Measures peak throughput (not latency-bound rate). Effective
    /// ops/iter = 8 × 8192 = 65536 (use this multiplier when computing
    /// TFLOPs from this variant's wall).
    pub fn launch_parallel(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        a_in: &DeviceBuffer<u16>,
        b_in: &DeviceBuffer<u16>,
        n_iters: u32,
        n_blocks: u32,
        block_threads: u32,
    ) -> eyre::Result<()> {
        self.launch_named(stream, "wmma_f16_throughput_probe_parallel", out, a_in, b_in, n_iters, n_blocks, block_threads)
    }

    /// IU8 (int8 → int32) WMMA, parallel accumulators. Same shape as
    /// f16 variant. Each WMMA emits 8192 int-ops (4096 mul + 4096 add).
    pub fn launch_iu8_parallel(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<i32>,
        a_in: &DeviceBuffer<i8>,
        b_in: &DeviceBuffer<i8>,
        n_iters: u32,
        n_blocks: u32,
        block_threads: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("wmma_iu8_throughput_probe_parallel")?;
        let cfg = LaunchConfig {
            grid: (n_blocks, 1, 1),
            block: (block_threads, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), a_in.raw(), b_in.raw(), n_iters])
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_named(
        &self,
        stream: &Stream,
        kernel_name: &str,
        out: &mut DeviceBuffer<f32>,
        a_in: &DeviceBuffer<u16>,
        b_in: &DeviceBuffer<u16>,
        n_iters: u32,
        n_blocks: u32,
        block_threads: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function(kernel_name)?;
        let cfg = LaunchConfig {
            grid: (n_blocks, 1, 1),
            block: (block_threads, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), a_in.raw(), b_in.raw(), n_iters])
    }
}
