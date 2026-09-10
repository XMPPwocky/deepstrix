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
//! ([`pack_row_from_f16_host`], [`unpack_row_host`]) are the reference used by the
//! tests and by the v3 -> v4 snapshot conversion; the device expand is the
//! authority (the conversion re-verifies on device).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

use crate::iq2_xxs_tables::f16_to_f32;
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
    let idx = (code & 0x7F) as usize;
    let m = if idx < 127 { magnitude_table()[idx] } else { 0.0 };
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

/// Recover a packed row from an f16 row that was produced by the f16 path
/// (`fp8_e4m3fn_quantize -> f16_roundtrip -> comp_kv_append`). Used by the
/// v3 -> v4 snapshot conversion. Returns `None` when some value in some
/// block is not exactly `half_rn(±table[idx] * 2^e')` for the recovered
/// `e'` — the caller must then refuse the row (the amax floor case
/// `e = -22` can flush to f16 subnormal/zero and is not recoverable in
/// general). The recovered exponent is `ceil(log2(amax/448))` computed
/// from the STORED values, which is `e0` or `e0 - 1` (the stored block
/// max is at most the original amax); both are tried.
///
/// A row that converts here must still be verified through the device
/// expand before it is trusted — host and device f16 rounding agree on
/// normal values, and the device is what production reads.
pub fn pack_row_from_f16_host(row: &[u16], out: &mut [u8]) -> Option<()> {
    assert!(row.len() >= FP8_KV_HEAD_DIM && out.len() >= FP8_KV_ROW_BYTES);
    for blk in 0..FP8_KV_N_BLOCKS {
        let vals = &row[blk * 64..blk * 64 + 64];
        let amax = vals
            .iter()
            .map(|&b| f16_to_f32(b).abs())
            .fold(0f32, f32::max);
        // The producer's floor (amax < 1e-4 -> 1e-4) gives e = -22 at most
        // negative; a stored all-zero block therefore came from e = -22
        // (or from a block whose values all flushed). Try the direct
        // recovery first, then e0 - 1.
        let amax_eff = if amax < 1.0e-4 { 1.0e-4 } else { amax };
        let e_direct = (amax_eff / 448.0).log2().ceil() as i32;
        let mut found = false;
        for e_try in [e_direct, e_direct + 1, e_direct - 1] {
            if !(-128..=127).contains(&e_try) {
                continue;
            }
            let mut ok = true;
            let mut codes = [0u8; 64];
            let scale = 2f32.powi(e_try);
            let inv_scale = 2f32.powi(-e_try);
            for (j, &b) in vals.iter().enumerate() {
                match code_for_half(b, scale, inv_scale) {
                    Some(c) => codes[j] = c,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                out[blk * 64..blk * 64 + 64].copy_from_slice(&codes);
                out[FP8_KV_OFF_EXP + blk] = e_try as i8 as u8;
                found = true;
                break;
            }
        }
        if !found {
            return None;
        }
    }
    out[FP8_KV_OFF_EXP + FP8_KV_N_BLOCKS] = 0;
    for r in 0..FP8_KV_N_ROT {
        let o = FP8_KV_OFF_ROPE + r * 2;
        out[o..o + 2].copy_from_slice(&row[FP8_KV_N_NOPE + r].to_le_bytes());
    }
    for b in out.iter_mut().take(FP8_KV_ROW_BYTES).skip(FP8_KV_OFF_ROPE + FP8_KV_N_ROT * 2) {
        *b = 0;
    }
    Some(())
}

/// The 127 reachable E4M3FN magnitudes, ascending (index = table index).
fn magnitude_table() -> &'static [f32; 127] {
    static T: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 127];
        for (i, v) in t.iter_mut().enumerate() {
            *v = e4m3fn_magnitude(i as u8);
        }
        t
    })
}

/// The code whose expansion at exponent `e` is exactly the f16 `bits`,
/// if any. Sign of zero is preserved (`0x8000` -> code `0x80`).
///
/// O(1) per value (a 7-step binary search plus a verification window of
/// five candidates): a 192K session converts in seconds, not minutes.
/// The magnitude `|v| * 2^-e` is exactly a table value whenever the store
/// did not round (every normal-range value); the window covers the
/// f16-subnormal cases where several neighbouring codes round to the
/// same stored value.
fn code_for_half(bits: u16, scale: f32, inv_scale: f32) -> Option<u8> {
    let sign = (bits & 0x8000) != 0;
    let mag = f16_to_f32(bits & 0x7FFF) * inv_scale;
    let table = magnitude_table();
    // Largest index with table[idx] <= mag (partition_point gives the
    // first index with table[idx] > mag).
    let hi = table.partition_point(|&t| t <= mag);
    let base = hi.saturating_sub(1) as i32;
    for delta in [0i32, 1, -1, 2, -2] {
        let idx = base + delta;
        if !(0..127).contains(&idx) {
            continue;
        }
        let code = if sign { idx as u8 | 0x80 } else { idx as u8 };
        if expand_half_bits_scaled(code, scale) == bits {
            return Some(code);
        }
    }
    None
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
    fn host_pack_unpack_round_trip_from_f16() {
        // Build an f16 row the way the producer would: codes at a known e.
        let mut packed = vec![0u8; FP8_KV_ROW_BYTES];
        for d in 0..FP8_KV_N_NOPE {
            packed[d] = ((d * 37) % 127) as u8 | if d % 3 == 0 { 0x80 } else { 0 };
        }
        for blk in 0..FP8_KV_N_BLOCKS {
            packed[FP8_KV_OFF_EXP + blk] = (-(blk as i8) - 5) as u8;
        }
        for r in 0..FP8_KV_N_ROT {
            let bits = f32_to_f16_bits(0.25 * r as f32 - 3.0);
            packed[FP8_KV_OFF_ROPE + r * 2..FP8_KV_OFF_ROPE + r * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        let mut row = vec![0u16; FP8_KV_HEAD_DIM];
        unpack_row_host(&packed, &mut row);
        let mut repacked = vec![0u8; FP8_KV_ROW_BYTES];
        pack_row_from_f16_host(&row, &mut repacked).expect("recoverable");
        let mut row2 = vec![0u16; FP8_KV_HEAD_DIM];
        unpack_row_host(&repacked, &mut row2);
        assert_eq!(row, row2, "expansion must be identical after recovery");
    }

    #[test]
    fn negative_zero_code_is_preserved() {
        assert_eq!(expand_half_bits_host(0x80, -5), 0x8000);
        assert_eq!(expand_half_bits_host(0x00, -5), 0x0000);
        let (sc, inv) = (2f32.powi(-5), 2f32.powi(5));
        assert_eq!(code_for_half(0x8000, sc, inv), Some(0x80));
        assert_eq!(code_for_half(0x0000, sc, inv), Some(0x00));
    }
}

// The dense attention path is taken only while `n_comp <= INDEXER_TOP_K`;
// the head shadow must cover exactly that.
const _: () = assert!(FP8_KV_HEAD_ROWS == crate::config::INDEXER_TOP_K as usize);

#[cfg(test)]
mod conversion_speed {
    use super::*;

    /// Host-only: how long the v3 -> v4 recovery takes per row, so the
    /// restore cost of an old snapshot is known (192K = 21 x 49152 rows).
    #[test]
    fn recovery_throughput() {
        let mut rows: Vec<u16> = Vec::new();
        let mut packed = vec![0u8; FP8_KV_ROW_BYTES];
        let mut seed = 0x1234_5678_9abc_def0u64;
        let n = 2000usize;
        for _ in 0..n {
            for d in 0..FP8_KV_N_NOPE {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                packed[d] = (seed % 127) as u8 | if seed & 0x100 != 0 { 0x80 } else { 0 };
            }
            for blk in 0..FP8_KV_N_BLOCKS {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                packed[FP8_KV_OFF_EXP + blk] = ((seed % 20) as i8 - 14) as u8;
            }
            let mut row = vec![0u16; FP8_KV_HEAD_DIM];
            unpack_row_host(&packed, &mut row);
            rows.extend_from_slice(&row);
        }
        let mut out = vec![0u8; FP8_KV_ROW_BYTES];
        let t0 = std::time::Instant::now();
        let mut ok = 0;
        for r in 0..n {
            if pack_row_from_f16_host(&rows[r * FP8_KV_HEAD_DIM..(r + 1) * FP8_KV_HEAD_DIM], &mut out).is_some() {
                ok += 1;
            }
        }
        let dt = t0.elapsed();
        let per_row = dt.as_secs_f64() / n as f64;
        eprintln!(
            "recovery: {ok}/{n} rows, {:.1} us/row -> 192K session (21 x 49152 rows) = {:.1} s",
            per_row * 1e6,
            per_row * 21.0 * 49152.0
        );
        assert_eq!(ok, n);
    }
}
