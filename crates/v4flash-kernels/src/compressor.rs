//! V4 Flash compressor — produces compressed-KV rows for the CSA path
//! (`layer_attention_mixed_one`). Mirrors ds4's
//! `compressor_decode_one_decode_scratch` (ds4.c:6589) by composing four
//! kernels: F16 matvec (×2) → APE-add → per-dim attention pool → RMS-norm →
//! forward RoPE → FP8 E4M3FN quantize → F16 roundtrip-on-push.
//!
//! M7 builds these kernels piece-by-piece. This module currently exposes:
//! - `CompressorPool` — the per-output-dim softmax-weighted-average kernel.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const COMPRESSOR_POOL_GFX1201: &[u8] = include_bytes!(env!("KERNEL_COMPRESSOR_POOL_GFX1201"));
const COMPRESSOR_POOL_GFX1151: &[u8] = include_bytes!(env!("KERNEL_COMPRESSOR_POOL_GFX1151"));

/// Per-output-dim softmax-weighted average over the compressor state
/// buffer. One workgroup, head_dim threads.
pub struct CompressorPool {
    module: Module,
}

impl CompressorPool {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            COMPRESSOR_POOL_GFX1201
        } else if arch.starts_with("gfx1151") {
            COMPRESSOR_POOL_GFX1151
        } else {
            return Err(eyre!("unsupported arch for compressor_pool: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Pool the `state_kv` / `state_score` buffer into `out`.
    ///
    /// - `out`:         `[head_dim]`
    /// - `state_kv`:    `[ratio * coff, width]` where coff = 2 if ratio==4 else 1, width = coff*head_dim
    /// - `state_score`: same shape as state_kv
    /// - `head_dim`:    512 for main compressor, 128 for indexer compressor
    /// - `compress_ratio`: 4 or 128
    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        state_kv: &DeviceBuffer<f32>,
        state_score: &DeviceBuffer<f32>,
        head_dim: u32,
        compress_ratio: u32,
    ) -> eyre::Result<()> {
        if compress_ratio != 4 && compress_ratio != 128 {
            return Err(eyre!(
                "compressor_pool: compress_ratio={compress_ratio} must be 4 or 128"
            ));
        }
        let coff: u32 = if compress_ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let state_rows = compress_ratio * coff;
        let needed = (state_rows as usize) * (width as usize);
        if state_kv.len() < needed || state_score.len() < needed {
            return Err(eyre!(
                "compressor_pool: state buffers have {}/{} elems, need {} (rows={state_rows}, width={width})",
                state_kv.len(),
                state_score.len(),
                needed
            ));
        }
        if out.len() < head_dim as usize {
            return Err(eyre!(
                "compressor_pool: out has {} elems, need head_dim={head_dim}",
                out.len()
            ));
        }

        let function = self.module.get_function("compressor_pool")?;

        let mut out_ptr = out.raw();
        let mut kv_ptr = state_kv.raw();
        let mut sc_ptr = state_score.raw();
        let mut head_dim_v = head_dim;
        let mut compress_ratio_v = compress_ratio;
        let mut args: [*mut c_void; 5] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut kv_ptr as *mut _ as *mut c_void,
            &mut sc_ptr as *mut _ as *mut c_void,
            &mut head_dim_v as *mut _ as *mut c_void,
            &mut compress_ratio_v as *mut _ as *mut c_void,
        ];

        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
