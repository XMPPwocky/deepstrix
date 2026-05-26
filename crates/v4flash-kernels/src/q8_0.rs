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

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const Q8_0_MATVEC_GFX1201: &[u8] = include_bytes!(env!("KERNEL_Q8_0_MATVEC_GFX1201"));
const Q8_0_MATVEC_GFX1151: &[u8] = include_bytes!(env!("KERNEL_Q8_0_MATVEC_GFX1151"));

const Q8_0_GROUPED_MATVEC_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_Q8_0_GROUPED_MATVEC_GFX1201"));
const Q8_0_GROUPED_MATVEC_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_Q8_0_GROUPED_MATVEC_GFX1151"));

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

        let mut xq_ptr = xq.raw();
        let mut xscale_ptr = xscale.raw();
        let mut x_ptr = x.raw();
        let mut blocks_val = blocks;
        let mut args: [*mut c_void; 4] = [
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut xscale_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut blocks_val as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
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

        let mut out_ptr = out.raw();
        let mut w_ptr = weight.raw();
        let mut xq_ptr = xq.raw();
        let mut xs_ptr = xscale.raw();
        let mut in_dim = k;
        let mut out_dim = n_rows;
        let mut n_blocks = blocks;
        let mut args: [*mut c_void; 7] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut xs_ptr as *mut _ as *mut c_void,
            &mut in_dim as *mut _ as *mut c_void,
            &mut out_dim as *mut _ as *mut c_void,
            &mut n_blocks as *mut _ as *mut c_void,
        ];

        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES; // 8 × 32 = 256
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
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
        let mut out_ptr = out.raw();
        let mut w_ptr = weight.raw();
        let mut xq_ptr = xq.raw();
        let mut xs_ptr = xscale.raw();
        let mut in_dim = k;
        let mut out_dim = n_rows;
        let mut n_blocks = blocks;
        let mut args: [*mut c_void; 7] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut xs_ptr as *mut _ as *mut c_void,
            &mut in_dim as *mut _ as *mut c_void,
            &mut out_dim as *mut _ as *mut c_void,
            &mut n_blocks as *mut _ as *mut c_void,
        ];

        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES; // 8 × 32 = 256
        let cfg = LaunchConfig {
            grid: (grid_x, 1, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
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

        let mut out_a_ptr = out_a.raw();
        let mut out_b_ptr = out_b.raw();
        let mut w_ptr = weight.raw();
        let mut xq_a_ptr = xq_a.raw();
        let mut xq_b_ptr = xq_b.raw();
        let mut xs_a_ptr = xscale_a.raw();
        let mut xs_b_ptr = xscale_b.raw();
        let mut in_dim = k;
        let mut out_dim = n_rows;
        let mut n_blocks = blocks;
        let mut args: [*mut c_void; 10] = [
            &mut out_a_ptr as *mut _ as *mut c_void,
            &mut out_b_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut xq_a_ptr as *mut _ as *mut c_void,
            &mut xq_b_ptr as *mut _ as *mut c_void,
            &mut xs_a_ptr as *mut _ as *mut c_void,
            &mut xs_b_ptr as *mut _ as *mut c_void,
            &mut in_dim as *mut _ as *mut c_void,
            &mut out_dim as *mut _ as *mut c_void,
            &mut n_blocks as *mut _ as *mut c_void,
        ];

        let grid_x = n_rows.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
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

        let mut out_ptr = out.raw();
        let mut w_ptr = weight.raw();
        let mut xq_ptr = xq.raw();
        let mut xs_ptr = xscale.raw();
        let mut group_dim_v = group_dim;
        let mut rank_v = rank;
        let mut blocks_per_group_v = blocks_per_group;
        let mut n_groups_v = n_groups;
        let mut args: [*mut c_void; 8] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut xs_ptr as *mut _ as *mut c_void,
            &mut group_dim_v as *mut _ as *mut c_void,
            &mut rank_v as *mut _ as *mut c_void,
            &mut blocks_per_group_v as *mut _ as *mut c_void,
            &mut n_groups_v as *mut _ as *mut c_void,
        ];

        let grid_x = out_dim.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
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

        let mut out_a_ptr = out_a.raw();
        let mut out_b_ptr = out_b.raw();
        let mut w_ptr = weight.raw();
        let mut xq_a_ptr = xq_a.raw();
        let mut xq_b_ptr = xq_b.raw();
        let mut xs_a_ptr = xscale_a.raw();
        let mut xs_b_ptr = xscale_b.raw();
        let mut group_dim_v = group_dim;
        let mut rank_v = rank;
        let mut blocks_per_group_v = blocks_per_group;
        let mut n_groups_v = n_groups;
        let mut args: [*mut c_void; 11] = [
            &mut out_a_ptr as *mut _ as *mut c_void,
            &mut out_b_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut xq_a_ptr as *mut _ as *mut c_void,
            &mut xq_b_ptr as *mut _ as *mut c_void,
            &mut xs_a_ptr as *mut _ as *mut c_void,
            &mut xs_b_ptr as *mut _ as *mut c_void,
            &mut group_dim_v as *mut _ as *mut c_void,
            &mut rank_v as *mut _ as *mut c_void,
            &mut blocks_per_group_v as *mut _ as *mut c_void,
            &mut n_groups_v as *mut _ as *mut c_void,
        ];

        let grid_x = out_dim.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
