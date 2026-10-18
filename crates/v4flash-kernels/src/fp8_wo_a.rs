//! `V41_ATTN_O_FP8`: the attention output projection's first stage (`wo_a`) FP8 x FP8, as the
//! DeepSeek-V4.1-Flash-1017 reference computes it (kernels/fp8_wo_a.hip; numerics in
//! [`crate::act_quant_fp8`]).
//!
//! Operands:
//!   * `o`: e4m3 codes `[b, 64 x 512]` (bytes, carried in the engine's `heads_xq: DeviceBuffer<i8>`)
//!     and one power-of-two f32 scale per 32 dims (`heads_xscale`) -- the reference's
//!     `act_quant(o, 32, "ue8m0")`, written by the epilogue that used to write the Q8_0 input
//!     (`RopeTail::launch_inverse_quant_fp8`, `AttnDecG3` / `AttnDecG3o` with `o8`, or
//!     [`Fp8WoA::act_quant`]);
//!   * `wo_a`: the checkpoint's e4m3 weight `[8192, 4096]` followed by its e8m0 block scales
//!     `[256, 128]`, verbatim, in one device buffer ([`crate::act_quant_fp8::fp8_weight_bytes`];
//!     loaded by `het::weights` from the HF view's `blk.N.attn_output_a_fp8.weight`).
//!
//! Kernels (gfx1201; the dot / WMMA bodies are stubs on gfx1151, which has no FP8 dot):
//!   * [`Fp8WoA::matvec`] -- `fp8_grouped_gemv_tB{1..8}` / `fp8_grouped_gemv_rt`: one wave per output
//!     row, 8 x `v_dot4_f32_fp8_fp8` per 32-block, the block's `s_w * s_a` applied to the block sum;
//!     the decode / verify arm (<= 16 rows; any b in 16-row grid.z slices);
//!   * [`Fp8WoA::gemm`] -- `fp8_grouped_gemm_wmma`: 128 x 128 tiles of `v_wmma_f32_16x16x16_fp8_fp8`,
//!     the prefill arm (where the Q8_0 path ran the f16x GEMM);
//!   * [`Fp8WoA::quantize_q8_o8`] -- the DSpark drafter's input (iGPU): the rounding, then the Q8_0
//!     quantize of the rounded values (its wo_a stays Q8_0).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

use crate::act_quant_fp8::fp8_weight_bytes;

const FP8_WO_A_GFX1201: &[u8] = include_bytes!(env!("KERNEL_FP8_WO_A_GFX1201"));
const FP8_WO_A_GFX1151: &[u8] = include_bytes!(env!("KERNEL_FP8_WO_A_GFX1151"));

/// Rows per GEMV grid.z slice (`FW_GEMV_MAX_B`).
pub const FP8_GEMV_SLICE: u32 = 16;
/// The GEMM tile (`FWG_BM` x `FWG_BN`).
pub const FP8_GEMM_TILE: u32 = 128;

/// `V41_ATTN_O_FP8` (see `knobs::ATTN_O_FP8`): static, read once.
pub fn attn_o_fp8() -> bool {
    crate::knobs::ATTN_O_FP8.on()
}

pub struct Fp8WoA {
    module: Module,
    arch_fp8: bool,
}

impl Fp8WoA {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let (image, arch_fp8): (&[u8], bool) = if arch.starts_with("gfx1201") {
            (FP8_WO_A_GFX1201, true)
        } else if arch.starts_with("gfx1151") {
            (FP8_WO_A_GFX1151, false)
        } else {
            return Err(eyre!("unsupported arch for fp8_wo_a: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)?, arch_fp8 })
    }

    /// The loaded module (tests: explicit-symbol launches).
    pub fn module(&self) -> &Module {
        &self.module
    }

    fn quant_args(x: &DeviceBuffer<f32>, xq_len: usize, xs_len: usize, n: u32, batch: u32, what: &str) -> eyre::Result<u32> {
        if n % 32 != 0 {
            return Err(eyre!("{what}: n={n} not a multiple of 32"));
        }
        let total = n as usize * batch as usize;
        if x.len() < total || xq_len < total || xs_len < total / 32 {
            return Err(eyre!("{what}: buffers too small for {batch} x {n} (x {}, xq {xq_len}, xscale {xs_len})", x.len()));
        }
        Ok((total / 32) as u32)
    }

    /// `xq` / `xscale` := the reference's act_quant of `x[batch, n]` (e4m3 codes as bytes, one
    /// power-of-two scale per 32). Same grid as the wave Q8_0 quantize (8 blocks per 256-thread WG,
    /// + the `V41_Q8_QUANT_GRID_PAD` idle WG).
    pub fn act_quant(&self, stream: &Stream, xq: &mut DeviceBuffer<i8>, xscale: &mut DeviceBuffer<f32>, x: &DeviceBuffer<f32>, n: u32, batch: u32) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let blocks = Self::quant_args(x, xq.len(), xscale.len(), n, batch, "act_quant_fp8")?;
        let function = self.module.get_function("act_quant_fp8_wave")?;
        let cfg = LaunchConfig { grid: (blocks.div_ceil(8) + crate::q8_0::q8_quant_grid_pad(), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [xq.raw(), xscale.raw(), x.raw(), blocks])
    }

    /// The drafter's wo_a input: act_quant rounding of `x` then `q8_0_quantize_f32_wave`'s Q8_0 of
    /// the rounded values (= `act_quant_inplace` + `Q8_0Matvec::quantize_input_batched`), one launch.
    pub fn quantize_q8_o8(&self, stream: &Stream, xq: &mut DeviceBuffer<i8>, xscale: &mut DeviceBuffer<f32>, x: &DeviceBuffer<f32>, n: u32, batch: u32) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let blocks = Self::quant_args(x, xq.len(), xscale.len(), n, batch, "q8_0_quantize_o8")?;
        let function = self.module.get_function("q8_0_quantize_o8_wave")?;
        let cfg = LaunchConfig { grid: (blocks.div_ceil(8) + crate::q8_0::q8_quant_grid_pad(), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [xq.raw(), xscale.raw(), x.raw(), blocks])
    }

    /// Tests: `x[0..n]` := `act_quant_inplace(x)` on the device (the rounding arithmetic alone).
    pub fn act_quant_inplace(&self, stream: &Stream, x: &mut DeviceBuffer<f32>, n: u32) -> eyre::Result<()> {
        if n % 32 != 0 || x.len() < n as usize {
            return Err(eyre!("act_quant_fp8_inplace: n={n} (x {}) must be a multiple of 32 within x", x.len()));
        }
        let blocks = n / 32;
        let function = self.module.get_function("act_quant_fp8_inplace_wave")?;
        let cfg = LaunchConfig { grid: (blocks.div_ceil(8), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [x.raw(), blocks])
    }

    #[allow(clippy::too_many_arguments)]
    fn check_mv(&self, what: &str, out: &DeviceBuffer<f32>, weight: &DeviceBuffer<u8>, xq: &DeviceBuffer<i8>, xscale: &DeviceBuffer<f32>,
                group_dim: u32, rank: u32, n_groups: u32, batch: u32) -> eyre::Result<()> {
        if !self.arch_fp8 {
            return Err(eyre!("{what}: this arch has no FP8 dot (gfx1201 only); the FP8 wo_a is dGPU-only"));
        }
        if group_dim % 32 != 0 || rank % 32 != 0 {
            return Err(eyre!("{what}: group_dim={group_dim} / rank={rank} must be multiples of 32"));
        }
        let out_dim = (n_groups * rank) as usize;
        let need_w = fp8_weight_bytes(out_dim, group_dim as usize);
        if weight.byte_len() != need_w {
            return Err(eyre!("{what}: weight has {} bytes, the FP8 wo_a ([{out_dim}, {group_dim}] e4m3 + e8m0 32x32 scales) is {need_w} \
                              (a Q8_0 wo_a here means V41_ATTN_O_FP8 changed after load)", weight.byte_len()));
        }
        let per_in = (n_groups * group_dim) as usize;
        if xq.len() < batch as usize * per_in || xscale.len() < batch as usize * per_in / 32 || out.len() < batch as usize * out_dim {
            return Err(eyre!("{what}: buffers too small for batch={batch} (xq {} xscale {} out {})", xq.len(), xscale.len(), out.len()));
        }
        Ok(())
    }

    fn gemv_symbol(batch: u32) -> &'static str {
        match batch {
            1 => "fp8_grouped_gemv_tB1",
            2 => "fp8_grouped_gemv_tB2",
            3 => "fp8_grouped_gemv_tB3",
            4 => "fp8_grouped_gemv_tB4",
            5 => "fp8_grouped_gemv_tB5",
            6 => "fp8_grouped_gemv_tB6",
            7 => "fp8_grouped_gemv_tB7",
            8 => "fp8_grouped_gemv_tB8",
            _ => "fp8_grouped_gemv_rt",
        }
    }

    fn gemv_cfg(out_dim: u32, batch: u32) -> LaunchConfig {
        let z = if batch <= 8 { 1 } else { batch.div_ceil(FP8_GEMV_SLICE) };
        LaunchConfig { grid: (out_dim.div_ceil(8), 1, z), block: (256, 1, 1), shared_mem_bytes: 0 }
    }

    /// `out[b, g * rank + r] = sum_kb s_w * s_a * sum_k w[r, k] * o[b, g, k]` for `b < batch`:
    /// the grouped FP8 GEMV (decode / verify). `xq` holds e4m3 codes.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec(&self, stream: &Stream, out: &mut DeviceBuffer<f32>, weight: &DeviceBuffer<u8>, xq: &DeviceBuffer<i8>, xscale: &DeviceBuffer<f32>,
                  group_dim: u32, rank: u32, n_groups: u32, batch: u32) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        self.check_mv("fp8_grouped_gemv", out, weight, xq, xscale, group_dim, rank, n_groups, batch)?;
        let function = self.module.get_function(Self::gemv_symbol(batch))?;
        let cfg = Self::gemv_cfg(n_groups * rank, batch);
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), xq.raw(), xscale.raw(), group_dim, rank, n_groups, batch])
    }

    /// [`Self::matvec`] through its `_ind` twin (docs/v41/GRAPH_KEYS_DESIGN.md 2.3): operands 0..3
    /// (out, weight, xq, xscale) marked in `ind` come from the arena context; the buffers passed
    /// are the real ones (checked as for a direct launch).
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_ind(&self, stream: &Stream, ind: crate::het::arena_ctx::Ind, out: &mut DeviceBuffer<f32>, weight: &DeviceBuffer<u8>,
                      xq: &DeviceBuffer<i8>, xscale: &DeviceBuffer<f32>, group_dim: u32, rank: u32, n_groups: u32, batch: u32) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        self.check_mv("fp8_grouped_gemv_ind", out, weight, xq, xscale, group_dim, rank, n_groups, batch)?;
        let function = self.module.get_function(&ind.symbol(Self::gemv_symbol(batch)))?;
        let cfg = Self::gemv_cfg(n_groups * rank, batch);
        let (p_out, p_w, p_xq, p_xs) = (
            ind.ptr(0, out.raw() as u64),
            ind.ptr(1, weight.raw() as u64),
            ind.ptr(2, xq.raw() as u64),
            ind.ptr(3, xscale.raw() as u64),
        );
        crate::het::arena_ctx::vet_ind(&ind, &[out.raw() as u64, weight.raw() as u64, xq.raw() as u64, xscale.raw() as u64]);
        launch_kernel!(function, cfg, stream, [ind.mask(), ind.canary, ind.tag, p_out, p_w, p_xq, p_xs, group_dim, rank, n_groups, batch])
    }

    /// The same product as [`Self::matvec`] by FP8 WMMA tiles (prefill). Needs rank % 128 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&self, stream: &Stream, out: &mut DeviceBuffer<f32>, weight: &DeviceBuffer<u8>, xq: &DeviceBuffer<i8>, xscale: &DeviceBuffer<f32>,
                group_dim: u32, rank: u32, n_groups: u32, batch: u32) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        self.check_mv("fp8_grouped_gemm_wmma", out, weight, xq, xscale, group_dim, rank, n_groups, batch)?;
        if rank % FP8_GEMM_TILE != 0 || group_dim % 32 != 0 {
            return Err(eyre!("fp8_grouped_gemm_wmma: rank={rank} must be a multiple of {FP8_GEMM_TILE}"));
        }
        let function = self.module.get_function("fp8_grouped_gemm_wmma")?;
        // o-row tiles fastest (the weight tile is shared by the WGs that run together)
        let cfg = LaunchConfig { grid: (batch.div_ceil(FP8_GEMM_TILE), rank / FP8_GEMM_TILE, n_groups), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [out.raw(), weight.raw(), xq.raw(), xscale.raw(), group_dim, rank, n_groups, batch])
    }
}
