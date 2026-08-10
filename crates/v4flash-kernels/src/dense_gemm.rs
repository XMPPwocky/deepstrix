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
const KQ_WMMA_GFX1201: &[u8] = include_bytes!(env!("KERNEL_KQUANT_GEMM_WMMA_GFX1201"));
const KQ_WMMA_GFX1151: &[u8] = include_bytes!(env!("KERNEL_KQUANT_GEMM_WMMA_GFX1151"));

pub struct DenseGemmDp4a {
    module: Module,
    /// WMMA GEMM (gfx12 only): dequant-to-f16-in-LDS + matrix cores.
    wmma: Option<Module>,
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
        let wmma = if arch.starts_with("gfx1201") {
            Some(Module::load_data(KQ_WMMA_GFX1201)?)
        } else {
            let _ = KQ_WMMA_GFX1151; // stub image; iGPU never dispatches these
            None
        };
        Ok(Self { module, wmma })
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
        // WMMA path (gfx12) beats dp4a whenever the matrix cores can be
        // fed — B must fill at least one BN=64 tile to pay for the LDS
        // dequant, so B=1 decode keeps the dp4a kernel.
        // KQ_GEMM=dp4a forces the register-tiled path.
        static FORCE_DP4A: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var("KQ_GEMM").map(|v| v == "dp4a").unwrap_or(false)
        });
        let use_wmma = !*FORCE_DP4A && b >= 64 && self.wmma.is_some();
        let (fname, block_bytes) = match (dt, use_wmma) {
            (GgufType::Q4_K, false) => ("q4_k_dense_gemm_dp4a_r32_nolds", 144usize),
            (GgufType::Q5_K, false) => ("q5_k_dense_gemm_dp4a_r32_nolds", 176),
            (GgufType::Q6_K, false) => ("q6_k_dense_gemm_dp4a_r32_nolds", 210),
            (GgufType::Q4_K, true) => ("q4_k_gemm_wmma_lds_tiled", 144),
            (GgufType::Q5_K, true) => ("q5_k_gemm_wmma_lds_tiled", 176),
            (GgufType::Q6_K, true) => ("q6_k_gemm_wmma_lds_tiled", 210),
            (other, _) => return Err(eyre!("dense_gemm: unsupported dtype {other:?}")),
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
        if use_wmma {
            let function = self.wmma.as_ref().expect("checked").get_function(fname)?;
            let cfg = LaunchConfig {
                grid: (n_rows.div_ceil(64), b.div_ceil(64), 1),
                block: (128, 1, 1), // 4 warps × wave32
                shared_mem_bytes: 0,
            };
            let k = n_blk * 256;
            return launch_kernel!(function, cfg, stream, [
                out.raw(), w.raw(), xq.raw(), k, n_rows, b, n_blk
            ]);
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
