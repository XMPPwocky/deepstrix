//! Decode-shape twin of the arena attention score kernel
//! (kernels/attention_dec.hip): same arguments and BIT-IDENTICAL output as
//! [`crate::AttentionMixed::launch_score_batched_htiled_wmma_f16s_rows`], with
//! the head tile as a grid dimension and each warp's K-chunk loads issued in
//! groups (tests/attention_dec_bitexact.rs). gfx12 only (WMMA).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const ATTENTION_DEC_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_DEC_GFX1201"));
const ATTENTION_DEC_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_DEC_GFX1151"));

pub struct AttentionDec {
    module: Module,
}

impl AttentionDec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ATTENTION_DEC_GFX1201
        } else if arch.starts_with("gfx1151") {
            ATTENTION_DEC_GFX1151
        } else {
            return Err(eyre!("unsupported arch for attention_dec: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// Same contract as `launch_score_batched_htiled_wmma_f16s_rows` (f16
    /// scores); requires `head_dim == 512`, `n_head % 16 == 0`.
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
        let function = self.module.get_function("attention_dec_score_htiled_wmma_f16s")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let mask_ptr = comp_allowed_bits.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let comp_base_ptr = comp_base_per.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let max_keys_words: u32 = (crate::ATTN_MIXED_MAX_KEYS + 31) / 32;
        let cfg = LaunchConfig {
            grid: (n_total_max.div_ceil(256), n_head / 16, batch),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores_g.raw(), q.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            mask_ptr, max_keys_words, n_head, scores_stride, kq_scale,
            comp_kv_batch_stride, comp_base_ptr
        ])
    }
}
