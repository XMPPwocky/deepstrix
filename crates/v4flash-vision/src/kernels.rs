//! Launch wrappers for `kernels/vit.hip` (see the header comment there for
//! the kernel contracts). All launches are async on the caller's stream.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const VIT_GFX1201: &[u8] = include_bytes!(env!("KERNEL_VIT_GFX1201"));
const VIT_GFX1151: &[u8] = include_bytes!(env!("KERNEL_VIT_GFX1151"));

pub const GEMM_TILE: u32 = 64;
pub const GEMM_THREADS: u32 = 128;
pub const GEMM_BK: u32 = 32;
pub const ATT_BR: u32 = 64;

pub const FLAG_ACCUM: u32 = 1;
pub const FLAG_GELU: u32 = 2;

/// Which GEMM kernel [`VitKernels::gemm`] dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmPath {
    /// `vit_gemm`: RDNA3 WMMA on gfx11xx, scalar body elsewhere.
    Wmma,
    /// `vit_gemm_scalar`: portable scalar tile kernel (A/B + non-gfx11 fallback).
    Scalar,
}

pub struct VitKernels {
    module: Module,
    pub arch: String,
    pub gemm_path: GemmPath,
}

fn null() -> *mut c_void {
    std::ptr::null_mut()
}

impl VitKernels {
    /// Load the code object for `arch` (the current device's `gcn_arch_name`).
    /// `VIT_GEMM=scalar` forces the scalar GEMM.
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            VIT_GFX1201
        } else if arch.starts_with("gfx1151") {
            VIT_GFX1151
        } else {
            return Err(eyre!("v4flash-vision: no kernel image for arch {arch}"));
        };
        let module = Module::load_data(image)?;
        let forced_scalar = std::env::var("VIT_GEMM").map(|v| v == "scalar").unwrap_or(false);
        let gemm_path = if arch.starts_with("gfx115") && !forced_scalar { GemmPath::Wmma } else { GemmPath::Scalar };
        Ok(Self { module, arch: arch.to_string(), gemm_path })
    }

    /// `out[n_tok, m] = act(x[n_tok, k] · w[m, k]^T + bias)`; `x`/`w` f16 bits,
    /// `k % 32 == 0`. Exactly one of `out_f32`/`out_f16` is `Some`.
    /// `FLAG_ACCUM` adds into `out_f32` (residual stream); `FLAG_GELU`
    /// applies GELU(erf) before the store.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        stream: &Stream,
        out_f32: Option<&mut DeviceBuffer<f32>>,
        out_f16: Option<&mut DeviceBuffer<u16>>,
        x: &DeviceBuffer<u16>,
        w: &DeviceBuffer<u16>,
        bias: Option<&DeviceBuffer<f32>>,
        n_tok: u32,
        k: u32,
        m: u32,
        flags: u32,
    ) -> eyre::Result<()> {
        if n_tok == 0 {
            return Ok(());
        }
        if k % GEMM_BK != 0 {
            return Err(eyre!("vit gemm: k={k} not a multiple of {GEMM_BK}"));
        }
        if x.len() < (n_tok as usize) * (k as usize) {
            return Err(eyre!("vit gemm: x len {} < {n_tok}x{k}", x.len()));
        }
        if w.len() != (m as usize) * (k as usize) {
            return Err(eyre!("vit gemm: w len {} != {m}x{k}", w.len()));
        }
        if let Some(b) = bias {
            if b.len() != m as usize {
                return Err(eyre!("vit gemm: bias len {} != {m}", b.len()));
            }
        }
        let need = (n_tok as usize) * (m as usize);
        let (o32, o16) = match (out_f32, out_f16) {
            (Some(o), None) => {
                if o.len() < need {
                    return Err(eyre!("vit gemm: out_f32 len {} < {need}", o.len()));
                }
                (o.raw(), null())
            }
            (None, Some(o)) => {
                if o.len() < need {
                    return Err(eyre!("vit gemm: out_f16 len {} < {need}", o.len()));
                }
                (null(), o.raw())
            }
            _ => return Err(eyre!("vit gemm: exactly one of out_f32/out_f16 must be given")),
        };
        let bias_ptr = bias.map(|b| b.raw()).unwrap_or_else(null);
        let name = match self.gemm_path {
            GemmPath::Wmma => "vit_gemm",
            GemmPath::Scalar => "vit_gemm_scalar",
        };
        let function = self.module.get_function(name)?;
        let cfg = LaunchConfig {
            grid: (m.div_ceil(GEMM_TILE), n_tok.div_ceil(GEMM_TILE), 1),
            block: (GEMM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [o32, o16, x.raw(), w.raw(), bias_ptr, n_tok, k, m, flags])
    }

    /// Raw WMMA fragment probe: `out[lane*8 + i]` (32×8 f32).
    pub fn wmma_probe(&self, stream: &Stream, out: &mut DeviceBuffer<f32>) -> eyre::Result<()> {
        if out.len() != 256 {
            return Err(eyre!("wmma_probe: out len {} != 256", out.len()));
        }
        let function = self.module.get_function("vit_wmma_probe")?;
        launch_kernel!(function, LaunchConfig::simple(1, 32), stream, [out.raw()])
    }

    /// RMSNorm rows (`[n_tok, dim]` f32) → f16.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_f16(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u16>,
        x: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        n_tok: u32,
        dim: u32,
        eps: f32,
    ) -> eyre::Result<()> {
        if n_tok == 0 {
            return Ok(());
        }
        let need = (n_tok as usize) * (dim as usize);
        if out.len() < need || x.len() < need || weight.len() != dim as usize {
            return Err(eyre!("rmsnorm_f16: len mismatch (out {}, x {}, w {}, need {need})", out.len(), x.len(), weight.len()));
        }
        let function = self.module.get_function("vit_rmsnorm_f16")?;
        launch_kernel!(function, LaunchConfig::simple(n_tok, 256), stream, [out.raw(), x.raw(), weight.raw(), dim, eps])
    }

    /// 2-D RoPE on the fused qkv rows (`[n_tok, 3072]` f32) → f16 q/k/v `[n_tok, 16, 64]`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_split(
        &self,
        stream: &Stream,
        q: &mut DeviceBuffer<u16>,
        k: &mut DeviceBuffer<u16>,
        v: &mut DeviceBuffer<u16>,
        qkv: &DeviceBuffer<f32>,
        cos: &DeviceBuffer<f32>,
        sin: &DeviceBuffer<f32>,
        n_tok: u32,
    ) -> eyre::Result<()> {
        if n_tok == 0 {
            return Ok(());
        }
        let n = n_tok as usize;
        if q.len() < n * 1024 || k.len() < n * 1024 || v.len() < n * 1024 || qkv.len() < n * 3072 || cos.len() < n * 32 || sin.len() < n * 32 {
            return Err(eyre!("rope_split: buffer too small for n_tok={n_tok}"));
        }
        let function = self.module.get_function("vit_rope_split")?;
        launch_kernel!(function, LaunchConfig::simple(n_tok, 512), stream, [q.raw(), k.raw(), v.raw(), qkv.raw(), cos.raw(), sin.raw()])
    }

    /// Bidirectional 16-head MHA over f16 `[n_tok, 16, 64]` q/k/v → f16 out.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u16>,
        q: &DeviceBuffer<u16>,
        k: &DeviceBuffer<u16>,
        v: &DeviceBuffer<u16>,
        n_tok: u32,
        scale: f32,
    ) -> eyre::Result<()> {
        if n_tok == 0 {
            return Ok(());
        }
        let need = (n_tok as usize) * 1024;
        if out.len() < need || q.len() < need || k.len() < need || v.len() < need {
            return Err(eyre!("attention: buffer too small for n_tok={n_tok}"));
        }
        let function = self.module.get_function("vit_attention")?;
        let cfg = LaunchConfig { grid: (16, n_tok.div_ceil(ATT_BR), 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [out.raw(), q.raw(), k.raw(), v.raw(), n_tok, scale])
    }

    /// `h = silu(gate) * up` from the fused gate|up rows (`[n_tok, 2*ffn]` f32) → f16 `[n_tok, ffn]`.
    pub fn swiglu_f16(&self, stream: &Stream, out: &mut DeviceBuffer<u16>, gu: &DeviceBuffer<f32>, n_tok: u32, ffn: u32) -> eyre::Result<()> {
        if n_tok == 0 {
            return Ok(());
        }
        let n = (n_tok as usize) * (ffn as usize);
        if out.len() < n || gu.len() < 2 * n {
            return Err(eyre!("swiglu_f16: buffer too small"));
        }
        let function = self.module.get_function("vit_swiglu_f16")?;
        launch_kernel!(function, LaunchConfig::simple((n as u32).div_ceil(256), 256), stream, [out.raw(), gu.raw(), n_tok, ffn])
    }

    /// Aligner 3×3 unfold (channel-major, zero padded) f16 → f16.
    #[allow(clippy::too_many_arguments)]
    pub fn unfold(
        &self,
        stream: &Stream,
        dst: &mut DeviceBuffer<u16>,
        src: &DeviceBuffer<u16>,
        n_h: u32,
        n_w: u32,
        n_llm_h: u32,
        n_llm_w: u32,
        dim: u32,
    ) -> eyre::Result<()> {
        let n_llm = n_llm_h * n_llm_w;
        if n_llm == 0 {
            return Ok(());
        }
        if dst.len() < (n_llm as usize) * (dim as usize) * 9 || src.len() < (n_h as usize) * (n_w as usize) * (dim as usize) {
            return Err(eyre!("unfold: buffer too small"));
        }
        let function = self.module.get_function("vit_unfold")?;
        launch_kernel!(function, LaunchConfig::simple(n_llm, 256), stream, [dst.raw(), src.raw(), n_h, n_w, n_llm_w, dim])
    }
}
