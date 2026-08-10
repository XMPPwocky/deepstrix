//! Dense (non-MoE) Q4_K matvec — `out[i] = sum_k dequant(W[i,k]) * x[k]`.
//!
//! For the Laguna model port: attention Q/K/V/O projections are dense Q4_K
//! weights. The existing [`crate::q4_k::Q4KMatvec`] is MoE-batched-only
//! (expert-indexed, Q8_K activations). This provides the plain dense path.
//!
//! CORRECTNESS-FIRST: the activation vector `x` stays f32 (no quantization),
//! and the kernel does a scalar f32 dequant-and-dot. No WMMA, no dp4a, no
//! tiling — perf is a later phase. Mirrors [`crate::q8_0::Q8_0Matvec`] for
//! the binding shape (`for_arch`, `matvec`, `matvec_batched`) and the
//! exhaustive length/byte validation style.
//!
//! Q4_K block layout (144 bytes per 256 elements) is identical to
//! llama.cpp/ggml `block_q4_K` and matches what the GGUF loader feeds:
//!   d (ggml_half, 2 B) | dmin (ggml_half, 2 B) | scales[12] | qs[128]

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const Q4_K_DENSE_MATVEC_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_Q4_K_DENSE_MATVEC_GFX1201"));
const Q4_K_DENSE_MATVEC_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_Q4_K_DENSE_MATVEC_GFX1151"));

/// Q4_K superblock = 256 elements packed in 144 bytes.
pub const Q4_K_DENSE_BLOCK_ELEMS: u32 = 256;
pub const Q4_K_DENSE_BLOCK_BYTES: u32 = 144;

/// One workgroup processes 8 output rows; one warp (32 lanes) per row.
const GEMV_ROWS_PER_BLOCK: u32 = 8;
const GEMV_WARP_LANES: u32 = 32;

#[allow(non_camel_case_types)]
pub struct Q4_KDenseMatvec {
    module: Module,
}

impl Q4_KDenseMatvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q4_K_DENSE_MATVEC_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q4_K_DENSE_MATVEC_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q4_k dense matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `out[i] = sum_k dequant(W[i,k]) * x[k]` for i in 0..n_rows.
    /// Weight is `[n_rows, K]` Q4_K, row-major, row pitch `(K/256)*144`
    /// bytes. Activation `x` is `[K]` f32, out is `[n_rows]` f32.
    /// K must be a multiple of 256.
    pub fn matvec(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        if k % Q4_K_DENSE_BLOCK_ELEMS != 0 {
            return Err(eyre!("q4_k dense matvec: k={k} not a multiple of 256"));
        }
        let n_super = k / Q4_K_DENSE_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (n_super as usize) * (Q4_K_DENSE_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q4_k dense matvec weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if out.len() != n_rows as usize {
            return Err(eyre!(
                "q4_k dense matvec out len: have {}, expected n_rows={n_rows}",
                out.len()
            ));
        }
        if x.len() != k as usize {
            return Err(eyre!(
                "q4_k dense matvec x len: have {}, expected k={k}",
                x.len()
            ));
        }

        let function = self.module.get_function("q4_k_dense_gemv")?;
        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES; // 8 × 32 = 256
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), x.raw(), k, n_rows, n_super])
    }

    /// Batched dense Q4_K GEMV with `grid.z = B`. Same per-row math as
    /// [`Self::matvec`]; B parallel WGs run concurrently, one per batch
    /// element. `x[B,K]`, `out[B,n_rows]` — row-major. Weight `[n_rows,K]`
    /// Q4_K is shared across the batch.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if k % Q4_K_DENSE_BLOCK_ELEMS != 0 {
            return Err(eyre!("q4_k dense matvec_batched: k={k} not a multiple of 256"));
        }
        let n_super = k / Q4_K_DENSE_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (n_super as usize) * (Q4_K_DENSE_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q4_k dense matvec_batched weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        let expected_out = (batch as usize) * (n_rows as usize);
        if out.len() < expected_out {
            return Err(eyre!(
                "q4_k dense matvec_batched out len: have {}, expected {}",
                out.len(),
                expected_out
            ));
        }
        let expected_x = (batch as usize) * (k as usize);
        if x.len() < expected_x {
            return Err(eyre!(
                "q4_k dense matvec_batched x len: have {} (need {expected_x}), n_rows={n_rows}, k={k}, batch={batch}",
                x.len()
            ));
        }

        let function = self.module.get_function("q4_k_dense_gemv_batched")?;
        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES; // 8 × 32 = 256
        let cfg = LaunchConfig {
            grid: (grid_x, 1, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), x.raw(), k, n_rows, n_super])
    }
}
