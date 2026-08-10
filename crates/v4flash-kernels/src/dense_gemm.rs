//! Dense K-quant GEMM (dp4a, register-tiled) for the het/ prefill path —
//! Q4_K / Q5_K / Q6_K weights × Q8_K activations at B tokens.
//!
//! Kernels live in `laguna_moe_tiled.hip` (`*_dense_gemm_dp4a_r32_nolds`);
//! this wrapper lifts the Laguna-shape caps: DENSE_MAXB=2 covers
//! n_blk ≤ 16 (K ≤ 4096) for every variant — the old per-type caps
//! (Q4_K ≤ 12, Q6_K ≤ 4) were the caller's shapes, not kernel limits.
//! Q5_K is the unsloth-UD addition (shexp gate/up + attn_q_a).
//!
//! Q5_K spike context: matvec_batched at B=512 measured 11.5× the Q8 WMMA
//! GEMM (weight re-read per token); this register-tiled GEMM reads each
//! row's weights once per launch.

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::GgufType;
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const LAGUNA_MOE_TILED_GFX1201: &[u8] = include_bytes!(env!("KERNEL_LAGUNA_MOE_TILED_GFX1201"));
const LAGUNA_MOE_TILED_GFX1151: &[u8] = include_bytes!(env!("KERNEL_LAGUNA_MOE_TILED_GFX1151"));

pub struct DenseGemmDp4a {
    module: Module,
}

impl DenseGemmDp4a {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            LAGUNA_MOE_TILED_GFX1201
        } else if arch.starts_with("gfx1151") {
            LAGUNA_MOE_TILED_GFX1151
        } else {
            return Err(eyre!("unsupported arch for dense_gemm_dp4a: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `out[b, r] = Σ_k dequant(W[r, k]) * dequant(xq[b, k])` — W in `dt`,
    /// xq Q8_K token-major (`[B, n_blk*292]`). `n_rows % 32 == 0`,
    /// `n_blk ≤ 16`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        stream: &Stream,
        dt: GgufType,
        out: &mut DeviceBuffer<f32>,
        w: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        b: u32,
        n_rows: u32,
        n_blk: u32,
    ) -> eyre::Result<()> {
        if b == 0 {
            return Ok(());
        }
        if n_rows % 32 != 0 {
            return Err(eyre!("dense_gemm_dp4a: n_rows={n_rows} not %32"));
        }
        if n_blk == 0 || n_blk > 16 {
            return Err(eyre!("dense_gemm_dp4a: n_blk={n_blk} not in 1..=16"));
        }
        let (fname, block_bytes) = match dt {
            GgufType::Q4_K => ("q4_k_dense_gemm_dp4a_r32_nolds", 144usize),
            GgufType::Q5_K => ("q5_k_dense_gemm_dp4a_r32_nolds", 176),
            GgufType::Q6_K => ("q6_k_dense_gemm_dp4a_r32_nolds", 210),
            other => return Err(eyre!("dense_gemm_dp4a: unsupported dtype {other:?}")),
        };
        let need_w = (n_rows as usize) * (n_blk as usize) * block_bytes;
        if w.byte_len() < need_w {
            return Err(eyre!(
                "dense_gemm_dp4a: weight bytes {} < {need_w} ({dt:?}, n_rows={n_rows}, n_blk={n_blk})",
                w.byte_len()
            ));
        }
        if out.len() < (b as usize) * (n_rows as usize) {
            return Err(eyre!("dense_gemm_dp4a: out len {} < B*n_rows", out.len()));
        }
        let function = self.module.get_function(fname)?;
        let cfg = LaunchConfig {
            grid: (n_rows / 32, 1, 1),
            block: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w.raw(), xq.raw(), b, n_rows, n_blk
        ])
    }
}
