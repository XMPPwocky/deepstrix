//! Packed FP8 storage of the ratio-4 compressed-KV rows (lever 2 of
//! `docs/VRAM_FREE_PLAN_2026-09.md`, build notes in
//! `docs/FP8_KV_IMPL_2026-09.md`).
//!
//! The compressor E4M3-rounds the 448 non-RoPE dims of every stored row
//! with a power-of-two block-of-64 scale before the f16 store, so the f16
//! cache holds exactly `half_rn(sign * table[idx] * 2^e)` per value.
//! Storing `(code, e)` and expanding on read is bit-identical and cuts a
//! row from 1024 B to [`FP8_KV_ROW_BYTES`] (592, -42%).
//!
//! Row layout (byte offsets): `[0,448)` codes (`sign<<7 | idx`),
//! `[448,456)` i8 block exponents (7 used, byte 455 = 0), `[456,584)`
//! 64 x f16 RoPE dims, `[584,592)` zero pad.
//!
//! Device kernels live in `kernels/comp_kv_fp8.hip`; the numerics they
//! share with the historical quantiser are in
//! `kernels/fp8_e4m3fn_common.inc`. The host functions here
//! ([`expand_half_bits_host`], [`unpack_row_host`]) are the reference the
//! tests compare the device expand against.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

use crate::weight_contract::f32_to_f16_bits;

const COMP_KV_FP8_GFX1201: &[u8] = include_bytes!(env!("KERNEL_COMP_KV_FP8_GFX1201"));
const COMP_KV_FP8_GFX1151: &[u8] = include_bytes!(env!("KERNEL_COMP_KV_FP8_GFX1151"));

/// Compressed-KV row width the packed format is defined for.
pub const FP8_KV_HEAD_DIM: usize = 512;
/// Non-RoPE dims per row (7 blocks of 64).
pub const FP8_KV_N_NOPE: usize = 448;
/// RoPE dims per row, stored f16 verbatim.
pub const FP8_KV_N_ROT: usize = 64;
/// Exponent blocks per row.
pub const FP8_KV_N_BLOCKS: usize = 7;
/// Packed row stride in bytes (584 used, padded to a 16-B multiple).
pub const FP8_KV_ROW_BYTES: usize = 592;
/// Byte offset of the exponent bytes within a row.
pub const FP8_KV_OFF_EXP: usize = 448;
/// Byte offset of the f16 RoPE tail within a row.
pub const FP8_KV_OFF_ROPE: usize = 456;
/// Rows kept as an f16 shadow at the head of the cache for the dense
/// attention path (decode below the indexer's early-permit boundary and
/// prefill chunks entirely below it). Equals `INDEXER_TOP_K`: the dense
/// path is taken only while `n_comp <= INDEXER_TOP_K`.
pub const FP8_KV_HEAD_ROWS: usize = 512;

/// The 128-entry E4M3FN magnitude table (index 127 unused). Mirrors
/// `c_e4m3fn_values` in `kernels/fp8_e4m3fn_common.inc`; asserted equal
/// on device by `tests/fp8_kv_format.rs`.
pub fn e4m3fn_magnitude(idx: u8) -> f32 {
    let ex = (idx >> 3) as i32;
    let mant = (idx & 7) as f32;
    if ex == 0 {
        mant * (1.0 / 512.0)
    } else {
        (8.0 + mant) * (1.0 / 8.0) * 2f32.powi(ex - 7)
    }
}

/// Host expand of one packed value: the f16 bits the cache holds.
/// f32 product then round-to-nearest-even to f16
/// (`weight_contract::f32_to_f16_bits`, denormal-correct IEEE RN; device
/// `__float2half_rn` is the same rounding — the device is the authority,
/// see module docs).
pub fn expand_half_bits_host(code: u8, e: i8) -> u16 {
    expand_half_bits_scaled(code, 2f32.powi(e as i32))
}

/// [`expand_half_bits_host`] with the block scale `2^e` precomputed.
/// Sign applied to the f16 bits (see the device expand for why).
#[inline]
fn expand_half_bits_scaled(code: u8, scale: f32) -> u16 {
    let idx = code & 0x7F;
    let m = if idx < 127 { e4m3fn_magnitude(idx) } else { 0.0 };
    f32_to_f16_bits(m * scale) | ((code as u16 & 0x80) << 8)
}

/// Unpack one packed row into 512 f16 bits (host reference).
pub fn unpack_row_host(packed: &[u8], out: &mut [u16]) {
    assert!(packed.len() >= FP8_KV_ROW_BYTES && out.len() >= FP8_KV_HEAD_DIM);
    for d in 0..FP8_KV_N_NOPE {
        let e = packed[FP8_KV_OFF_EXP + (d >> 6)] as i8;
        out[d] = expand_half_bits_host(packed[d], e);
    }
    for r in 0..FP8_KV_N_ROT {
        let o = FP8_KV_OFF_ROPE + r * 2;
        out[FP8_KV_N_NOPE + r] = u16::from_le_bytes([packed[o], packed[o + 1]]);
    }
}

/// Kernel handle for the packed-FP8 compressed-KV path.
pub struct CompKvFp8 {
    module: Module,
}

impl CompKvFp8 {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            COMP_KV_FP8_GFX1201
        } else if arch.starts_with("gfx1151") {
            COMP_KV_FP8_GFX1151
        } else {
            return Err(eyre!("unsupported arch for comp_kv_fp8: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    fn check_packed(packed: &DeviceBuffer<u8>, rows_needed: usize, what: &str) -> eyre::Result<()> {
        let need = rows_needed * FP8_KV_ROW_BYTES;
        if packed.len() < need {
            return Err(eyre!(
                "{what}: packed comp_kv has {} bytes, need {} ({} rows x {})",
                packed.len(),
                need,
                rows_needed,
                FP8_KV_ROW_BYTES
            ));
        }
        Ok(())
    }

    fn check_head(head: &DeviceBuffer<u16>, head_rows: u32, what: &str) -> eyre::Result<()> {
        if head.len() < (head_rows as usize) * FP8_KV_HEAD_DIM {
            return Err(eyre!(
                "{what}: head shadow has {} f16, need {} rows x {}",
                head.len(),
                head_rows,
                FP8_KV_HEAD_DIM
            ));
        }
        Ok(())
    }

    /// Append one post-RoPE f32 row (`[512]`) at index `n_comp`: packed row
    /// + f16 head shadow when `n_comp < head_rows`. Replaces the
    /// `fp8_e4m3fn_quantize -> f16_roundtrip -> comp_kv_append` chain.
    pub fn launch_append(
        &self,
        stream: &Stream,
        packed: &mut DeviceBuffer<u8>,
        head: &mut DeviceBuffer<u16>,
        row: &DeviceBuffer<f32>,
        n_comp: u32,
        head_rows: u32,
    ) -> eyre::Result<()> {
        Self::check_packed(packed, n_comp as usize + 1, "comp_kv_append_fp8")?;
        Self::check_head(head, head_rows, "comp_kv_append_fp8")?;
        if row.len() < FP8_KV_HEAD_DIM {
            return Err(eyre!("comp_kv_append_fp8: row len {} < {}", row.len(), FP8_KV_HEAD_DIM));
        }
        let function = self.module.get_function("comp_kv_append_fp8")?;
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (FP8_KV_HEAD_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            packed.raw(), head.raw(), row.raw(), n_comp, head_rows
        ])
    }

    /// Batched append: rows `rows_b[k*512]` land at `n_comp_start + k`.
    pub fn launch_append_batched(
        &self,
        stream: &Stream,
        packed: &mut DeviceBuffer<u8>,
        head: &mut DeviceBuffer<u16>,
        rows_b: &DeviceBuffer<f32>,
        n_comp_start: u32,
        n_boundaries: u32,
        head_rows: u32,
    ) -> eyre::Result<()> {
        if n_boundaries == 0 {
            return Ok(());
        }
        Self::check_packed(packed, (n_comp_start + n_boundaries) as usize, "comp_kv_append_fp8_batched")?;
        Self::check_head(head, head_rows, "comp_kv_append_fp8_batched")?;
        if rows_b.len() < (n_boundaries as usize) * FP8_KV_HEAD_DIM {
            return Err(eyre!(
                "comp_kv_append_fp8_batched: rows_b len {} < {}",
                rows_b.len(),
                (n_boundaries as usize) * FP8_KV_HEAD_DIM
            ));
        }
        let function = self.module.get_function("comp_kv_append_fp8_batched")?;
        let cfg = LaunchConfig {
            grid: (1, n_boundaries, 1),
            block: (FP8_KV_HEAD_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            packed.raw(), head.raw(), rows_b.raw(), n_comp_start, head_rows
        ])
    }

    /// `active_comp_kv[i, :] = expand(packed[selected[i], :])`, sentinel
    /// `selected[i] < 0` skipped. Same contract as `IndexerGather::launch`.
    pub fn launch_gather(
        &self,
        stream: &Stream,
        active_comp_kv: &mut DeviceBuffer<u16>,
        packed: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<i32>,
        top_k: u32,
    ) -> eyre::Result<()> {
        if top_k == 0 {
            return Ok(());
        }
        if active_comp_kv.len() < (top_k as usize) * FP8_KV_HEAD_DIM {
            return Err(eyre!(
                "indexer_gather_fp8: active_comp_kv has {} f16, need {}",
                active_comp_kv.len(),
                (top_k as usize) * FP8_KV_HEAD_DIM
            ));
        }
        if selected.len() < top_k as usize {
            return Err(eyre!("indexer_gather_fp8: selected has {} slots, need {top_k}", selected.len()));
        }
        let function = self.module.get_function("indexer_gather_fp8")?;
        let cfg = LaunchConfig {
            grid: (top_k, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            active_comp_kv.raw(), packed.raw(), selected.raw(), top_k
        ])
    }

    /// Batched gather: `selected_b` at stride `top_k`, destination at stride
    /// `top_k * 512`. Same contract as `IndexerGather::launch_batched`.
    pub fn launch_gather_batched(
        &self,
        stream: &Stream,
        active_comp_kv_b: &mut DeviceBuffer<u16>,
        packed: &DeviceBuffer<u8>,
        selected_b: &DeviceBuffer<i32>,
        top_k: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || top_k == 0 {
            return Ok(());
        }
        let need = (batch as usize) * (top_k as usize) * FP8_KV_HEAD_DIM;
        if active_comp_kv_b.len() < need {
            return Err(eyre!(
                "indexer_gather_fp8_batched: active_comp_kv_b has {} f16, need {} (B={batch})",
                active_comp_kv_b.len(),
                need
            ));
        }
        if selected_b.len() < (batch as usize) * (top_k as usize) {
            return Err(eyre!(
                "indexer_gather_fp8_batched: selected_b too small (have {}, need {})",
                selected_b.len(),
                (batch as usize) * (top_k as usize)
            ));
        }
        let function = self.module.get_function("indexer_gather_fp8_batched")?;
        let cfg = LaunchConfig {
            grid: (top_k, batch, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            active_comp_kv_b.raw(), packed.raw(), selected_b.raw(), top_k
        ])
    }

    /// `dst[r, :] = expand(packed[r, :])` for `r in 0..n_rows` (head-shadow
    /// rebuild on restore, oracle dump, snapshot conversion check).
    pub fn launch_expand(
        &self,
        stream: &Stream,
        dst: &mut DeviceBuffer<u16>,
        packed: &DeviceBuffer<u8>,
        n_rows: u32,
    ) -> eyre::Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        Self::check_packed(packed, n_rows as usize, "comp_kv_fp8_expand")?;
        if dst.len() < (n_rows as usize) * FP8_KV_HEAD_DIM {
            return Err(eyre!(
                "comp_kv_fp8_expand: dst has {} f16, need {}",
                dst.len(),
                (n_rows as usize) * FP8_KV_HEAD_DIM
            ));
        }
        let function = self.module.get_function("comp_kv_fp8_expand")?;
        let cfg = LaunchConfig {
            grid: (n_rows, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [dst.raw(), packed.raw(), n_rows])
    }

    /// Test kernel: `out_table[i] = c_e4m3fn_values[i]`,
    /// `out_decode[i] = e4m3fn_decode_index(i)` for `i in 0..128`.
    pub fn launch_table_check(
        &self,
        stream: &Stream,
        out_table: &mut DeviceBuffer<f32>,
        out_decode: &mut DeviceBuffer<f32>,
    ) -> eyre::Result<()> {
        if out_table.len() < 128 || out_decode.len() < 128 {
            return Err(eyre!("fp8_kv_table_check: outputs need 128 slots"));
        }
        let function = self.module.get_function("fp8_kv_table_check")?;
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out_table.raw(), out_decode.raw()])
    }

    /// Test kernel: the device expand for every `(code, e)` with
    /// `e in e_lo..e_lo+n_e`, laid out `[(e - e_lo) * 256 + code]`.
    pub fn launch_expand_exhaustive(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u16>,
        e_lo: i32,
        n_e: u32,
    ) -> eyre::Result<()> {
        let n = (n_e as usize) * 256;
        if out.len() < n {
            return Err(eyre!("fp8_kv_expand_exhaustive: output needs {n} slots"));
        }
        let function = self.module.get_function("fp8_kv_expand_exhaustive")?;
        let cfg = LaunchConfig {
            grid: ((n as u32 + 255) / 256, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), e_lo, n_e])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnitude_table_matches_reference_points() {
        assert_eq!(e4m3fn_magnitude(0), 0.0);
        assert_eq!(e4m3fn_magnitude(1), 0.001953125);
        assert_eq!(e4m3fn_magnitude(8), 0.015625);
        assert_eq!(e4m3fn_magnitude(56), 1.0);
        assert_eq!(e4m3fn_magnitude(63), 1.875);
        assert_eq!(e4m3fn_magnitude(126), 448.0);
    }

    #[test]
    fn negative_zero_code_is_preserved() {
        assert_eq!(expand_half_bits_host(0x80, -5), 0x8000);
        assert_eq!(expand_half_bits_host(0x00, -5), 0x0000);
    }
}

// The dense attention path is taken only while `n_comp <= INDEXER_TOP_K`;
// the head shadow must cover exactly that.
const _: () = assert!(FP8_KV_HEAD_ROWS == crate::config::INDEXER_TOP_K as usize);
