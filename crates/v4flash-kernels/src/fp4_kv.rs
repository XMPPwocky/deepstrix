//! Launcher for `kernels/fp4_kv_quant.hip` — the V4.1 compressed-KV fake
//! quantisation (E2M1 values, one E4M3 scale per 16) applied in place to
//! post-RoPE latent rows before they enter the f16 comp_kv store.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const FP4KV_GFX1201: &[u8] = include_bytes!(env!("KERNEL_FP4_KV_QUANT_GFX1201"));
const FP4KV_GFX1151: &[u8] = include_bytes!(env!("KERNEL_FP4_KV_QUANT_GFX1151"));

pub struct Fp4KvQuant {
    module: Module,
}

impl Fp4KvQuant {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            FP4KV_GFX1201
        } else if arch.starts_with("gfx1151") {
            FP4KV_GFX1151
        } else {
            return Err(eyre!("unsupported arch for fp4_kv_quant: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// In-place fake quantisation of `n_rows` rows of `width` floats
    /// (`width` a multiple of 16, at most 1024).
    pub fn launch(&self, stream: &Stream, x: &mut DeviceBuffer<f32>, n_rows: u32, width: u32) -> eyre::Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        if width == 0 || width % 16 != 0 || width > 1024 {
            return Err(eyre!("fp4_kv_quant: width={width} must be a multiple of 16 ≤ 1024"));
        }
        if x.len() < (n_rows * width) as usize {
            return Err(eyre!("fp4_kv_quant: buffer has {} floats, need {}", x.len(), n_rows * width));
        }
        let function = self.module.get_function("fp4_kv_quant_inplace")?;
        let cfg = LaunchConfig { grid: (n_rows, 1, 1), block: (width, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [x.raw(), n_rows, width])
    }

    /// V4.1 window-KV fake quantisation in place: E4M3 values with one
    /// power-of-two (ue8m0) scale per 32 over the whole row (`width` a
    /// multiple of 32, at most 1024).
    pub fn launch_fp8_window(&self, stream: &Stream, x: &mut DeviceBuffer<f32>, n_rows: u32, width: u32) -> eyre::Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        if width == 0 || width % 32 != 0 || width > 1024 {
            return Err(eyre!("fp8_act_quant: width={width} must be a multiple of 32 ≤ 1024"));
        }
        if x.len() < (n_rows * width) as usize {
            return Err(eyre!("fp8_act_quant: buffer has {} floats, need {}", x.len(), n_rows * width));
        }
        let function = self.module.get_function("fp8_act_quant_inplace")?;
        let cfg = LaunchConfig { grid: (n_rows, 1, 1), block: (width, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [x.raw(), n_rows, width])
    }
}
