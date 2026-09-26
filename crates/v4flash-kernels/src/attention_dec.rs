//! Decode-shape twin of the arena attention score kernel
//! (kernels/attention_dec.hip): same arguments and BIT-IDENTICAL output as
//! [`crate::AttentionMixed::launch_score_batched_htiled_wmma_f16s_rows`], with
//! the head tile as a grid dimension and each warp's K-chunk loads issued in
//! groups (tests/attention_dec_bitexact.rs). gfx12 only (WMMA).
//!
//! Also the decode attention PAIR (score + `_softmax_wsum_batched_htiled_wmma_
//! ldsv_f16s`) as ONE launch, [`AttentionDec::launch_fused_f16s_rows`]
//! (kernels/attention_dec_fused.hip), BIT-IDENTICAL to the pair for rows of
//! <= [`ATTN_DEC_FUSED_MAX_KEYS`] keys (2026-09-26 sweep, x2.3 at b = 4).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const ATTENTION_DEC_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_DEC_GFX1201"));
const ATTENTION_DEC_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_DEC_GFX1151"));
const ATTENTION_DEC_FUSED_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_DEC_FUSED_GFX1201"));
const ATTENTION_DEC_FUSED_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_DEC_FUSED_GFX1151"));

/// Most keys (`n_raw + n_comp`) per row `launch_fused_f16s_rows` accepts:
/// the kernel's `DEC_MAX_KEYS` (its LDS weight rows are 664 f16 and it has NO
/// device-side guard; the launcher errors above this). Decode rows are
/// 128 window + <= 512 gathered keys; the call site must fall back to the
/// pair when a row could be longer.
pub const ATTN_DEC_FUSED_MAX_KEYS: u32 = 640;

/// Which decode score kernel `launch_score_f16s_rows_with` runs. Both write
/// bit-identical scores; they differ only in warps per WG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecScoreKernel {
    /// `attention_dec_score_htiled_wmma_f16s`: 16 warps x 16 keys per WG,
    /// grid.x = ceil(n_total_max / 256). The 2026-09-13 kernel.
    Blk256,
    /// `attention_dec_score_blk128`: 4 warps x 16 keys per WG, grid.x =
    /// ceil(n_total_max / 64). 2026-09-26 sweep (D_attention/score_blk128):
    /// 15.4 -> 11.7 us at b = 4 (x1.32), x1.42 at b = 1, ~x1.05 at b >= 5.
    Blk128,
}

pub struct AttentionDec {
    module: Module,
    fused: Module,
}

impl AttentionDec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let (image, fused_image): (&[u8], &[u8]) = if arch.starts_with("gfx1201") {
            (ATTENTION_DEC_GFX1201, ATTENTION_DEC_FUSED_GFX1201)
        } else if arch.starts_with("gfx1151") {
            (ATTENTION_DEC_GFX1151, ATTENTION_DEC_FUSED_GFX1151)
        } else {
            return Err(eyre!("unsupported arch for attention_dec: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)?, fused: Module::load_data(fused_image)? })
    }

    /// Same contract as `launch_score_batched_htiled_wmma_f16s_rows` (f16
    /// scores); requires `head_dim == 512`, `n_head % 16 == 0`. The
    /// 16-warp kernel (`DecScoreKernel::Blk256`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_score_f16s_rows(
        &self,
        stream: &Stream,
        scores_g: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        comp_allowed_bits: Option<&DeviceBuffer<u32>>,
        n_head: u32,
        head_dim: u32,
        n_total_max: u32,
        batch: u32,
        comp_kv_batch_stride: u32,
        scores_stride: u32,
        comp_base_per: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        self.launch_score_f16s_rows_with(
            DecScoreKernel::Blk256,
            stream,
            scores_g,
            q,
            raw_kv,
            comp_kv,
            n_raw_per,
            n_raw_offset_per,
            n_comp_per,
            comp_allowed_bits,
            n_head,
            head_dim,
            n_total_max,
            batch,
            comp_kv_batch_stride,
            scores_stride,
            comp_base_per,
        )
    }

    /// `launch_score_f16s_rows` with the kernel chosen by `kernel`
    /// (bit-identical scores either way; tests/attention_dec_bitexact.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_score_f16s_rows_with(
        &self,
        kernel: DecScoreKernel,
        stream: &Stream,
        scores_g: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        comp_allowed_bits: Option<&DeviceBuffer<u32>>,
        n_head: u32,
        head_dim: u32,
        n_total_max: u32,
        batch: u32,
        comp_kv_batch_stride: u32,
        scores_stride: u32,
        comp_base_per: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        if batch == 0 || n_total_max == 0 {
            return Ok(());
        }
        if head_dim != 512 || n_head % 16 != 0 || n_head == 0 {
            return Err(eyre!("attention_dec score: head_dim {head_dim} / n_head {n_head} (needs 512 / a multiple of 16)"));
        }
        if n_total_max > scores_stride {
            return Err(eyre!("attention_dec score: n_total_max={n_total_max} exceeds scores stride {scores_stride}"));
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        // (symbol, keys per WG, threads per WG): 16 or 4 warps x 16 keys.
        let (sym, keys_per_blk, block_x) = match kernel {
            DecScoreKernel::Blk256 => ("attention_dec_score_htiled_wmma_f16s", 256u32, 512u32),
            DecScoreKernel::Blk128 => ("attention_dec_score_blk128", 64u32, 128u32),
        };
        let function = self.module.get_function(sym)?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let mask_ptr = comp_allowed_bits.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let comp_base_ptr = comp_base_per.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let max_keys_words: u32 = (crate::ATTN_MIXED_MAX_KEYS + 31) / 32;
        let cfg = LaunchConfig {
            grid: (n_total_max.div_ceil(keys_per_blk), n_head / 16, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores_g.raw(), q.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            mask_ptr, max_keys_words, n_head, scores_stride, kq_scale,
            comp_kv_batch_stride, comp_base_ptr
        ])
    }

    /// The decode attention pair -- `launch_score_f16s_rows` followed by
    /// `AttentionMixed::launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows`
    /// -- as ONE launch of `attention_dec_fused_vt_qreg_sp_d4`
    /// (kernels/attention_dec_fused.hip), BIT-IDENTICAL `out` for every row
    /// (tests/attention_dec_bitexact.rs). 2026-09-26 sweep (dGPU, 640 keys,
    /// 64 heads): pair 46.9 -> 20.4 us at b = 4 (x2.30), 41.9 -> 19.9 at b = 1.
    ///
    /// Differences from the pair the caller must accept: no scores buffer is
    /// written (the pair left the softmax weights in it; nothing reads them),
    /// and there is no `comp_allowed_bits` mask path (decode passes none).
    /// Requires `head_dim == 512`, `n_head % 16 == 0` and every row's
    /// `n_raw + n_comp <= n_total_max <= ATTN_DEC_FUSED_MAX_KEYS` (hard error:
    /// the kernel would overflow its LDS silently). `comp_kv` None /
    /// `comp_base_per` / `comp_kv_batch_stride` have the pair's semantics.
    /// Grid (n_head / 16, batch) x 512, 62 KB static LDS.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_f16s_rows(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        n_total_max: u32,
        batch: u32,
        comp_kv_batch_stride: u32,
        comp_base_per: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if head_dim != 512 || n_head % 16 != 0 || n_head == 0 {
            return Err(eyre!("attention_dec fused: head_dim {head_dim} / n_head {n_head} (needs 512 / a multiple of 16)"));
        }
        if n_total_max > ATTN_DEC_FUSED_MAX_KEYS {
            return Err(eyre!(
                "attention_dec fused: n_total_max={n_total_max} exceeds the kernel's {ATTN_DEC_FUSED_MAX_KEYS} keys (caller must use the pair)"
            ));
        }
        // n_total_max == 0 still launches: the pair's smwsum writes zeros for
        // an empty row and so does the fused kernel.
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self.fused.get_function("attention_dec_fused_vt_qreg_sp_d4")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let comp_base_ptr = comp_base_per.map(|b| b.raw() as *const i32).unwrap_or(std::ptr::null());
        let cfg = LaunchConfig {
            grid: (n_head / 16, batch, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), sinks.raw(), q.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            n_head, kq_scale, comp_kv_batch_stride, comp_base_ptr
        ])
    }
}
