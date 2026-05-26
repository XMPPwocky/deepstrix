//! WMMA throughput probe — see kernels/wmma_probe.hip.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

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
        let mut o_ptr = out.raw();
        let mut a_ptr = a_in.raw();
        let mut b_ptr = b_in.raw();
        let mut ni = n_iters;
        let mut args: [*mut c_void; 4] = [
            &mut o_ptr as *mut _ as *mut c_void,
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_blocks, 1, 1),
            block: (block_threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
