//! Q8_0 GEMV — mirrors ds4's `matvec_q8_0_decode_scratch` /
//! `dot_q8_0_row` (`ds4.c:2920`).
//!
//! Two-kernel pipeline:
//!   1. `q8_0_quantize_f32`: pre-quantize the f32 input vector to int8 + per-
//!      32-block f32 scale. ds4 does the equivalent in CPU before each matvec.
//!   2. `q8_0_gemv_warp8`: one warp per output row; each lane handles a
//!      block via AMD `__builtin_amdgcn_sudot4` 4-byte int8 SIMD dot product;
//!      warp-reduces the partial sums.
//!
//! Used in V4 Flash for: the output projection (`output.weight` in the GGUF —
//! Q8_0 `[n_embd=4096, n_vocab=129280]`), and (in future milestones) every
//! Q8_0-format attention projection and shared-expert MLP.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const Q8_0_MATVEC_GFX1201: &[u8] = include_bytes!(env!("KERNEL_Q8_0_MATVEC_GFX1201"));
const Q8_0_MATVEC_GFX1151: &[u8] = include_bytes!(env!("KERNEL_Q8_0_MATVEC_GFX1151"));

const Q8_0_GROUPED_MATVEC_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_Q8_0_GROUPED_MATVEC_GFX1201"));
const Q8_0_GROUPED_MATVEC_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_Q8_0_GROUPED_MATVEC_GFX1151"));

const Q8_0_MATVEC_WMMA_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_Q8_0_MATVEC_WMMA_GFX1201"));
const Q8_0_MATVEC_WMMA_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_Q8_0_MATVEC_WMMA_GFX1151"));

/// Q8_0 packs 32 int8 quants per 2-byte f16 scale → 34 bytes per block,
/// identical layout to ds4 / llama.cpp.
pub const Q8_0_BLOCK_ELEMS: u32 = 32;
pub const Q8_0_BLOCK_BYTES: u32 = 34;

/// One workgroup processes 8 output rows in the gemv kernel.
const GEMV_ROWS_PER_BLOCK: u32 = 8;
const GEMV_WARP_LANES: u32 = 32;

#[allow(non_camel_case_types)]
pub struct Q8_0Matvec {
    module: Module,
}

impl Q8_0Matvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q8_0_MATVEC_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q8_0_MATVEC_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q8_0 matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Pre-quantize a length-n f32 vector into int8 + per-block f32 scale.
    /// `xq.len()` must be `n`; `xscale.len()` must be `n / 32` (n must be a
    /// multiple of 32). One workgroup per 32-element block, 32 threads each.
    pub fn quantize_input(
        &self,
        stream: &Stream,
        xq: &mut DeviceBuffer<i8>,
        xscale: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        n: u32,
    ) -> eyre::Result<()> {
        if n % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 quantize: n={n} not a multiple of 32"));
        }
        let blocks = n / Q8_0_BLOCK_ELEMS;
        if x.len() != n as usize
            || xq.len() != n as usize
            || xscale.len() != blocks as usize
        {
            return Err(eyre!(
                "q8_0 quantize len mismatch: n={n}, x={}, xq={}, xscale={}",
                x.len(),
                xq.len(),
                xscale.len()
            ));
        }
        let function = self.module.get_function("q8_0_quantize_f32")?;
        let cfg = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [xq.raw(), xscale.raw(), x.raw(), blocks])
    }

    /// M50 Phase 2: batched quantize. Equivalent to `quantize_input` over
    /// `B × n` contiguous elements. The kernel has no batch concept — it
    /// just processes `B × blocks` blocks. Buffers must be at least
    /// `B × n` (xq), `B × n/32` (xscale), `B × n` (x).
    pub fn quantize_input_batched(
        &self,
        stream: &Stream,
        xq: &mut DeviceBuffer<i8>,
        xscale: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        n: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if n % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 quantize_batched: n={n} not %32"));
        }
        let total_blocks = (n / Q8_0_BLOCK_ELEMS) * batch;
        let total_n = (n as usize) * (batch as usize);
        if x.len() < total_n || xq.len() < total_n || xscale.len() < total_blocks as usize {
            return Err(eyre!(
                "q8_0 quantize_batched: buffer too small (n*B={total_n}, blocks*B={total_blocks}, x={}, xq={}, xscale={})",
                x.len(), xq.len(), xscale.len()
            ));
        }
        let function = self.module.get_function("q8_0_quantize_f32")?;
        let cfg = LaunchConfig {
            grid: (total_blocks, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [xq.raw(), xscale.raw(), x.raw(), total_blocks])
    }

    /// `out[i] = sum_b f16_scale_w[i, b] * xscale[b] * dot_i8x32(qs_w[i, b], xq[b])`
    /// for i in 0..n_rows. The Q8_0 weight buffer holds `n_rows` rows of
    /// `(k/32) * 34` bytes each, row-major.
    pub fn matvec(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 matvec: k={k} not a multiple of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 matvec weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if out.len() != n_rows as usize {
            return Err(eyre!(
                "q8_0 matvec out len: have {}, expected n_rows={n_rows}",
                out.len()
            ));
        }
        if xq.len() != k as usize || xscale.len() != blocks as usize {
            return Err(eyre!(
                "q8_0 matvec xq/xscale len: xq={}, xscale={}, expected k={k}, blocks={blocks}",
                xq.len(),
                xscale.len()
            ));
        }

        let function = self.module.get_function("q8_0_gemv_warp8")?;

        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES; // 8 × 32 = 256
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, blocks])
    }

    /// M50 Phase 2: batched GEMV with `grid.z = B`. Same per-row math
    /// as `matvec`; B parallel WGs run concurrently, one per batch
    /// element. `xq[B, K]`, `xscale[B, K/32]`, `out[B, n_rows]` —
    /// row-major. Weight `[n_rows, K]` Q8_0 is shared across batch.
    ///
    /// v0 of batching: no W amortization across batch (each WG re-reads
    /// W independently). A v1 kernel will pack multiple batch elements
    /// per WG to amortize W reads.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 matvec_batched: k={k} not a multiple of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 matvec_batched weight bytes: have {}, expected {}",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        let expected_out = (batch as usize) * (n_rows as usize);
        if out.len() < expected_out {
            return Err(eyre!(
                "q8_0 matvec_batched out len: have {}, expected {}",
                out.len(),
                expected_out
            ));
        }
        let expected_xq = (batch as usize) * (k as usize);
        let expected_xscale = (batch as usize) * (blocks as usize);
        if xq.len() < expected_xq || xscale.len() < expected_xscale {
            return Err(eyre!(
                "q8_0 matvec_batched xq/xscale len: xq={} (need {expected_xq}), xscale={} (need {expected_xscale})",
                xq.len(),
                xscale.len()
            ));
        }

        let function = self.module.get_function("q8_0_gemv_batched_warp8")?;
        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES; // 8 × 32 = 256
        let cfg = LaunchConfig {
            grid: (grid_x, 1, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, blocks])
    }

    /// M40-P4.5: 2-wide pair GEMV. Same as `matvec` but processes TWO input
    /// columns against ONE weight matrix per call. Reads W once and computes
    /// both outputs in the same kernel pass — halves W bandwidth vs two
    /// independent calls. Used by pair-forward where t0 and t1 share weights
    /// but have different activations.
    ///
    /// `out[i] = sum_b f16(scale_w[i,b]) * xscale[b] * dot_i8x32(qs_w[i,b], xq[b])`
    /// for both `a` and `b` columns simultaneously.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_pair(
        &self,
        stream: &Stream,
        out_a: &mut DeviceBuffer<f32>,
        out_b: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq_a: &DeviceBuffer<i8>,
        xq_b: &DeviceBuffer<i8>,
        xscale_a: &DeviceBuffer<f32>,
        xscale_b: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 matvec_pair: k={k} not a multiple of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 matvec_pair weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if out_a.len() != n_rows as usize || out_b.len() != n_rows as usize {
            return Err(eyre!(
                "q8_0 matvec_pair out lens: a={}, b={}, expected n_rows={n_rows}",
                out_a.len(),
                out_b.len()
            ));
        }
        if xq_a.len() != k as usize
            || xq_b.len() != k as usize
            || xscale_a.len() != blocks as usize
            || xscale_b.len() != blocks as usize
        {
            return Err(eyre!(
                "q8_0 matvec_pair xq/xscale lens: xq_a={}, xq_b={}, xs_a={}, xs_b={}, expected k={k}, blocks={blocks}",
                xq_a.len(), xq_b.len(), xscale_a.len(), xscale_b.len()
            ));
        }

        let function = self.module.get_function("q8_0_gemv_pair_warp8")?;

        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out_a.raw(), out_b.raw(), weight.raw(), xq_a.raw(), xq_b.raw(),
            xscale_a.raw(), xscale_b.raw(), k, n_rows, blocks
        ])
    }
}

/// Q8_0 int8-WMMA GEMM (gfx12 only): `out[B,M] = W[M,K] · Xq[B,K]^T`, with
/// both Q8_0 dequant scales folded into the f16 WMMA operands at load time.
/// One wave per 16-row M-tile; batch N-tiles loop inside each K-tile so the
/// weight A-fragment is read once per K-tile and reused across all batch
/// columns. Same numeric result as `Q8_0Matvec::matvec_batched`, just via the
/// matrix cores instead of the grid.z=B dp4a path. On non-gfx12 the kernel
/// has a scalar fallback (so the gfx1151 build succeeds); it should only ever
/// be launched on the dGPU.
#[allow(non_camel_case_types)]
pub struct Q8_0MatvecWmma {
    module: Module,
}

impl Q8_0MatvecWmma {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q8_0_MATVEC_WMMA_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q8_0_MATVEC_WMMA_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q8_0 wmma matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `out[b,m] = sum_k (qW[m,k]·wscale[m,k/32]) · (qX[b,k]·xscale[b,k/32])`.
    /// `weight` is `[n_rows=M, K]` Q8_0 (row pitch `blocks*34`), `xq` is
    /// `[B, K]` int8, `xscale` is `[B, blocks]` f32, `out` is `[B, M]` f32.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32, // M
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 wmma gemm: k={k} not a multiple of 32"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 wmma gemm weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        let expected_out = (batch as usize) * (n_rows as usize);
        if out.len() < expected_out {
            return Err(eyre!(
                "q8_0 wmma gemm out len: have {}, expected {}",
                out.len(),
                expected_out
            ));
        }
        let expected_xq = (batch as usize) * (k as usize);
        let expected_xscale = (batch as usize) * (blocks as usize);
        if xq.len() < expected_xq || xscale.len() < expected_xscale {
            return Err(eyre!(
                "q8_0 wmma gemm xq/xscale len: xq={} (need {expected_xq}), xscale={} (need {expected_xscale})",
                xq.len(),
                xscale.len()
            ));
        }

        let function = self.module.get_function("q8_0_gemm_wmma_i8")?;
        let grid_x = n_rows.div_ceil(16);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, batch, blocks
        ])
    }

    /// Grouped LDS-tiled GEMM (RDNA4 only). 8-group block-diagonal matmul
    /// where the per-group inputs/outputs are interleaved per-batch
    /// (xq/xscale/out all strided by n_groups within the batch dim).
    /// Used for output_proj.grouped_matvec.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_lds_tiled_grouped(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,         // [B, n_groups*rank]
        weight: &DeviceBuffer<u8>,           // [n_groups*rank, group_dim Q8_0]
        xq: &DeviceBuffer<i8>,               // [B, n_groups*group_dim]
        xscale: &DeviceBuffer<f32>,          // [B, n_groups*blocks]
        group_dim: u32,
        rank: u32,
        n_groups: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_groups == 0 {
            return Ok(());
        }
        if group_dim % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!(
                "gemm_lds_tiled_grouped: group_dim={group_dim} not %32"
            ));
        }
        if rank % 64 != 0 {
            return Err(eyre!(
                "gemm_lds_tiled_grouped: rank={rank} not %64 (BM)"
            ));
        }
        let blocks = group_dim / Q8_0_BLOCK_ELEMS;
        let function = self
            .module
            .get_function("q8_0_gemm_wmma_lds_tiled_grouped")?;
        let cfg = LaunchConfig {
            grid: (rank.div_ceil(64), batch.div_ceil(64), n_groups),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(),
            group_dim, rank, n_groups, batch, blocks
        ])
    }

    /// LDS-tiled Q8_0 GEMM (RDNA4 only). Same semantics as `gemm()` but
    /// cooperatively stages BM×BK A + BK×BN B into LDS once per K-outer
    /// iter, then runs the inner WMMA loop entirely from LDS — amortizes
    /// global weight loads across many compute ops to kill the
    /// `s_wait_loadcnt` latency stall that bottlenecks both `gemm()` and
    /// the dp4a `matvec_batched()` at long-K shapes (matvec_out, etc).
    /// Requires `M % 64 == 0` and `K % 32 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_lds_tiled(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        n_rows: u32, // M
        k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if k % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!("q8_0 gemm_lds_tiled: k={k} not a multiple of 32"));
        }
        if n_rows % 64 != 0 {
            return Err(eyre!("q8_0 gemm_lds_tiled: n_rows={n_rows} not a multiple of 64 (BM)"));
        }
        let blocks = k / Q8_0_BLOCK_ELEMS;
        let expected_weight_bytes =
            (n_rows as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 gemm_lds_tiled weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(), expected_weight_bytes
            ));
        }
        let function = self.module.get_function("q8_0_gemm_wmma_lds_tiled")?;
        let grid_x = n_rows.div_ceil(64);          // BM=64
        let grid_y = batch.div_ceil(64);           // BN=64
        let cfg = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (128, 1, 1),                    // 4 warps × wave32
            shared_mem_bytes: 0,                   // declared static in kernel
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(), k, n_rows, batch, blocks
        ])
    }
}

/// Grouped Q8_0 GEMV — each output row `idx` in `[0, n_groups*rank)` reads
/// input from group `g = idx/rank`'s 4096-element slice. Mirrors ds4's
/// `matvec_q8_0_grouped_rows_decode_scratch` (ds4.c:3618). Used by V4 Flash
/// for the attention output's first-stage projection (`attn_output_a`,
/// `[n_groups*rank=8192, group_dim=4096]` Q8_0).
///
/// Input quantisation: the flat `n_groups*group_dim` f32 input can be
/// quantised with the regular `Q8_0Matvec::quantize_input` (block boundaries
/// align with group boundaries because `group_dim % 32 == 0`), producing
/// a flat `[n_groups*group_dim]` int8 buffer and `[n_groups*blocks_per_group]`
/// scales — bit-identical to ds4's per-group quantisation loop.
#[allow(non_camel_case_types)]
pub struct Q8_0GroupedMatvec {
    module: Module,
}

impl Q8_0GroupedMatvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q8_0_GROUPED_MATVEC_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q8_0_GROUPED_MATVEC_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q8_0_grouped matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `out[idx] = sum_b f16(scale_w[idx,b]) * xscale[g,b] * dot_i8x32(qs_w[idx,b], xq[g,b])`
    /// for `idx` in `0..n_groups*rank`, where `g = idx/rank`.
    pub fn matvec_grouped(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        group_dim: u32,
        rank: u32,
        n_groups: u32,
    ) -> eyre::Result<()> {
        if group_dim % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!(
                "q8_0 grouped matvec: group_dim={group_dim} not a multiple of 32"
            ));
        }
        let blocks_per_group = group_dim / Q8_0_BLOCK_ELEMS;
        let out_dim = n_groups * rank;
        let expected_weight_bytes =
            (out_dim as usize) * (blocks_per_group as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 grouped matvec weight bytes: have {}, expected {} (out_dim={out_dim}, group_dim={group_dim})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        let in_total = (n_groups as usize) * (group_dim as usize);
        let scales_total = (n_groups as usize) * (blocks_per_group as usize);
        if out.len() < out_dim as usize {
            return Err(eyre!(
                "q8_0 grouped matvec out len: have {}, expected {}",
                out.len(),
                out_dim
            ));
        }
        if xq.len() < in_total || xscale.len() < scales_total {
            return Err(eyre!(
                "q8_0 grouped matvec xq/xscale len: xq={}, xscale={}, expected xq={}, xscale={}",
                xq.len(),
                xscale.len(),
                in_total,
                scales_total
            ));
        }

        let function = self.module.get_function("q8_0_grouped_gemv")?;

        let grid_x = out_dim.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(),
            group_dim, rank, blocks_per_group, n_groups
        ])
    }

    /// M50 Phase 2: batched grouped GEMV with `grid.z = B`. Per-batch
    /// xq[B, n_groups*group_dim], xscale[B, n_groups*blocks_per_group],
    /// out[B, n_groups*rank]. Weight shared across batch.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_grouped_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<i8>,
        xscale: &DeviceBuffer<f32>,
        group_dim: u32,
        rank: u32,
        n_groups: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if group_dim % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!(
                "q8_0 grouped matvec_batched: group_dim={group_dim} not %32"
            ));
        }
        let blocks_per_group = group_dim / Q8_0_BLOCK_ELEMS;
        let out_dim = n_groups * rank;
        let expected_weight_bytes =
            (out_dim as usize) * (blocks_per_group as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 grouped matvec_batched weight bytes: {}!={expected_weight_bytes}",
                weight.byte_len()
            ));
        }
        let per_batch_in = (n_groups as usize) * (group_dim as usize);
        let per_batch_scales = (n_groups as usize) * (blocks_per_group as usize);
        if xq.len() < (batch as usize) * per_batch_in
            || xscale.len() < (batch as usize) * per_batch_scales
            || out.len() < (batch as usize) * (out_dim as usize)
        {
            return Err(eyre!(
                "q8_0 grouped matvec_batched: buffer too small for batch={batch} (xq {} xs {} out {})",
                xq.len(), xscale.len(), out.len()
            ));
        }

        let function = self.module.get_function("q8_0_grouped_gemv_batched")?;
        let grid_x = out_dim.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), weight.raw(), xq.raw(), xscale.raw(),
            group_dim, rank, blocks_per_group, n_groups
        ])
    }

    /// M40-P4.5: 2-wide pair variant of grouped GEMV. Same weight, two
    /// input vectors → two outputs in one launch. Halves W BW vs running
    /// twice.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_grouped_pair(
        &self,
        stream: &Stream,
        out_a: &mut DeviceBuffer<f32>,
        out_b: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        xq_a: &DeviceBuffer<i8>,
        xq_b: &DeviceBuffer<i8>,
        xscale_a: &DeviceBuffer<f32>,
        xscale_b: &DeviceBuffer<f32>,
        group_dim: u32,
        rank: u32,
        n_groups: u32,
    ) -> eyre::Result<()> {
        if group_dim % Q8_0_BLOCK_ELEMS != 0 {
            return Err(eyre!(
                "q8_0 grouped matvec_pair: group_dim={group_dim} not a multiple of 32"
            ));
        }
        let blocks_per_group = group_dim / Q8_0_BLOCK_ELEMS;
        let out_dim = n_groups * rank;
        let expected_weight_bytes =
            (out_dim as usize) * (blocks_per_group as usize) * (Q8_0_BLOCK_BYTES as usize);
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "q8_0 grouped matvec_pair weight bytes: have {}, expected {} (out_dim={out_dim}, group_dim={group_dim})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        let in_total = (n_groups as usize) * (group_dim as usize);
        let scales_total = (n_groups as usize) * (blocks_per_group as usize);
        if out_a.len() < out_dim as usize || out_b.len() < out_dim as usize {
            return Err(eyre!(
                "q8_0 grouped matvec_pair out lens: a={}, b={}, expected {}",
                out_a.len(),
                out_b.len(),
                out_dim
            ));
        }
        if xq_a.len() < in_total
            || xq_b.len() < in_total
            || xscale_a.len() < scales_total
            || xscale_b.len() < scales_total
        {
            return Err(eyre!(
                "q8_0 grouped matvec_pair xq/xscale lens: xq_a={}, xq_b={}, xs_a={}, xs_b={}, expected xq={}, xscale={}",
                xq_a.len(), xq_b.len(), xscale_a.len(), xscale_b.len(),
                in_total, scales_total
            ));
        }

        let function = self.module.get_function("q8_0_grouped_gemv_pair")?;

        let grid_x = out_dim.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out_a.raw(), out_b.raw(), weight.raw(),
            xq_a.raw(), xq_b.raw(), xscale_a.raw(), xscale_b.raw(),
            group_dim, rank, blocks_per_group, n_groups
        ])
    }
}
