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

const FP8_E4M3FN_GFX1201: &[u8] = include_bytes!(env!("KERNEL_FP8_E4M3FN_GFX1201"));
const FP8_E4M3FN_GFX1151: &[u8] = include_bytes!(env!("KERNEL_FP8_E4M3FN_GFX1151"));

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

/// FP8 E4M3FN in-place quantize over the non-RoPE portion (first
/// head_dim-n_rot elements) of a compressed-KV row. Mirrors ds4's
/// `dsv4_fp8_kv_quantize_row_inplace_cpu` (ds4.c:1635). Block size 64
/// per FP8 quantisation block; one workgroup per block. Only invoked
/// when head_dim == DS4_N_HEAD_DIM == 512 — the indexer skips this op.
pub struct Fp8E4m3fnQuantize {
    module: Module,
}

impl Fp8E4m3fnQuantize {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            FP8_E4M3FN_GFX1201
        } else if arch.starts_with("gfx1151") {
            FP8_E4M3FN_GFX1151
        } else {
            return Err(eyre!("unsupported arch for fp8_e4m3fn: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// In-place E4M3FN round-trip over `x[0..n_nope]`. `n_nope` must be
    /// a multiple of 64 (the quantisation block size).
    pub fn launch(
        &self,
        stream: &Stream,
        x: &mut DeviceBuffer<f32>,
        n_nope: u32,
    ) -> eyre::Result<()> {
        if n_nope % 64 != 0 {
            return Err(eyre!("fp8_e4m3fn: n_nope={n_nope} not a multiple of 64"));
        }
        if x.len() < n_nope as usize {
            return Err(eyre!(
                "fp8_e4m3fn: x has {} elems, need at least n_nope={n_nope}",
                x.len()
            ));
        }
        let function = self.module.get_function("fp8_e4m3fn_quantize")?;

        let mut x_ptr = x.raw();
        let mut n_nope_v = n_nope;
        let mut args: [*mut c_void; 2] = [
            &mut x_ptr as *mut _ as *mut c_void,
            &mut n_nope_v as *mut _ as *mut c_void,
        ];

        let cfg = LaunchConfig {
            grid: (n_nope / 64, 1, 1),
            block: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
