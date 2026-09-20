//! Packed E2M1 (FP4) storage of the ratio-4 indexer key rows (lever 3 of
//! `docs/VRAM_FREE_PLAN_2026-09.md`, build notes in
//! `docs/E2M1_INDEXER_KEYS_2026-09.md`).
//!
//! `indexer_qat` snaps every key value to `±m x 2^e` (`m` from an 8-entry
//! table, one power-of-two `e` per 32-element block) before the f16 store,
//! so the f16 key cache holds exactly `half_rn(±m x 2^e)`. Storing the
//! nibble `(sign<<3 | idx)` plus the block exponent and expanding on read
//! is bit-identical and cuts a 128-dim row from 256 B to
//! [`E2M1_KEY_ROW_BYTES`] (80).
//!
//! Row layout (byte offsets): `[0,64)` nibbles (element `i` in byte `i/2`,
//! low nibble for even `i`), `[64,68)` i8 block exponents, `[68,80)` zero
//! pad. Device kernels: `kernels/index_kv_e2m1.hip`; shared numerics with
//! the QAT kernel: `kernels/e2m1_key_common.inc`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

use crate::weight_contract::f32_to_f16_bits;

const INDEX_KV_E2M1_GFX1201: &[u8] = include_bytes!(env!("KERNEL_INDEX_KV_E2M1_GFX1201"));
const INDEX_KV_E2M1_GFX1151: &[u8] = include_bytes!(env!("KERNEL_INDEX_KV_E2M1_GFX1151"));

/// Key row width the packed format is defined for (`N_INDEXER_HEAD_DIM`).
pub const E2M1_KEY_DIM: usize = 128;
/// Exponent blocks per row (32 elements each).
pub const E2M1_KEY_N_BLOCKS: usize = 4;
/// Packed row stride in bytes (68 used, padded to a 16-B multiple).
pub const E2M1_KEY_ROW_BYTES: usize = 80;
/// Byte offset of the exponent bytes within a row.
pub const E2M1_KEY_OFF_EXP: usize = 64;

/// The E2M1 magnitude table (index = the low 3 bits of a nibble).
pub const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Host expand of one nibble at block exponent `e`: the f16 bits the cache
/// holds. Sign applied to the bits (see the device expand for why).
pub fn expand_half_bits_host(nib: u8, e: i8) -> u16 {
    let mag = E2M1_MAGNITUDES[(nib & 7) as usize] * 2f32.powi(e as i32);
    f32_to_f16_bits(mag) | (((nib & 8) as u16) << 12)
}

/// Unpack one packed row into 128 f16 bits (host reference).
pub fn unpack_row_host(packed: &[u8], out: &mut [u16]) {
    assert!(packed.len() >= E2M1_KEY_ROW_BYTES && out.len() >= E2M1_KEY_DIM);
    for i in 0..E2M1_KEY_DIM {
        let byte = packed[i / 2];
        let nib = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
        let e = packed[E2M1_KEY_OFF_EXP + i / 32] as i8;
        out[i] = expand_half_bits_host(nib, e);
    }
}

/// Kernel handle for the packed-E2M1 indexer-key path.
pub struct IndexKvE2m1 {
    module: Module,
}

impl IndexKvE2m1 {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            INDEX_KV_E2M1_GFX1201
        } else if arch.starts_with("gfx1151") {
            INDEX_KV_E2M1_GFX1151
        } else {
            return Err(eyre!("unsupported arch for index_kv_e2m1: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    fn check_packed(packed: &DeviceBuffer<u8>, rows_needed: usize, what: &str) -> eyre::Result<()> {
        let need = rows_needed * E2M1_KEY_ROW_BYTES;
        if packed.len() < need {
            return Err(eyre!(
                "{what}: packed index_comp_kv has {} bytes, need {} ({} rows x {})",
                packed.len(),
                need,
                rows_needed,
                E2M1_KEY_ROW_BYTES
            ));
        }
        Ok(())
    }

    /// Append one QAT'd f32 row (`[128]`) at index `n_comp`. Replaces the
    /// `f16_roundtrip -> comp_kv_append` pair after `indexer_qat`.
    pub fn launch_append(
        &self,
        stream: &Stream,
        packed: &mut DeviceBuffer<u8>,
        row: &DeviceBuffer<f32>,
        n_comp: u32,
    ) -> eyre::Result<()> {
        Self::check_packed(packed, n_comp as usize + 1, "index_kv_append_e2m1")?;
        if row.len() < E2M1_KEY_DIM {
            return Err(eyre!("index_kv_append_e2m1: row len {} < {}", row.len(), E2M1_KEY_DIM));
        }
        let function = self.module.get_function("index_kv_append_e2m1")?;
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (E2M1_KEY_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [packed.raw(), row.raw(), n_comp])
    }

    /// Batched append: rows `rows_b[k*128]` land at `n_comp_start + k`.
    /// Single-sequence form of [`Self::launch_append_batched_rows`] (per-row base = none).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_append_batched(
        &self,
        stream: &Stream,
        packed: &mut DeviceBuffer<u8>,
        rows_b: &DeviceBuffer<f32>,
        n_comp_start: u32,
        n_boundaries: u32,
    ) -> eyre::Result<()> {
        self.launch_append_batched_rows(stream, packed, rows_b, n_comp_start, n_boundaries, None)
    }

    pub fn launch_append_batched_rows(
        &self,
        stream: &Stream,
        packed: &mut DeviceBuffer<u8>,
        rows_b: &DeviceBuffer<f32>,
        n_comp_start: u32,
        n_boundaries: u32,
        dst_row_per: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        let dst_row_per_ptr = dst_row_per.map(|b| b.raw() as *const i32).unwrap_or(std::ptr::null());
        if n_boundaries == 0 {
            return Ok(());
        }
        if dst_row_per.is_none() {
            Self::check_packed(packed, (n_comp_start + n_boundaries) as usize, "index_kv_append_e2m1_batched")?;
        }
        if rows_b.len() < (n_boundaries as usize) * E2M1_KEY_DIM {
            return Err(eyre!(
                "index_kv_append_e2m1_batched: rows_b len {} < {}",
                rows_b.len(),
                (n_boundaries as usize) * E2M1_KEY_DIM
            ));
        }
        let function = self.module.get_function("index_kv_append_e2m1_batched")?;
        let cfg = LaunchConfig {
            grid: (1, n_boundaries, 1),
            block: (E2M1_KEY_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [packed.raw(), rows_b.raw(), n_comp_start, dst_row_per_ptr])
    }

    /// `dst[r, :] = expand(packed[r, :])` for `r in 0..n_rows`.
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
        Self::check_packed(packed, n_rows as usize, "index_kv_e2m1_expand")?;
        if dst.len() < (n_rows as usize) * E2M1_KEY_DIM {
            return Err(eyre!(
                "index_kv_e2m1_expand: dst has {} f16, need {}",
                dst.len(),
                (n_rows as usize) * E2M1_KEY_DIM
            ));
        }
        let function = self.module.get_function("index_kv_e2m1_expand")?;
        let cfg = LaunchConfig {
            grid: (n_rows, 1, 1),
            block: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [dst.raw(), packed.raw(), n_rows])
    }

    /// Test kernel: the device's E2M1 magnitude table.
    pub fn launch_table_check(&self, stream: &Stream, out: &mut DeviceBuffer<f32>) -> eyre::Result<()> {
        if out.len() < 8 {
            return Err(eyre!("e2m1_key_table_check: output needs 8 slots"));
        }
        let function = self.module.get_function("e2m1_key_table_check")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (8, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [out.raw()])
    }

    /// Test kernel: the fast 8-nibble expansion (`e2m1_key_expand8`, the
    /// score kernels' path) versus the general per-nibble expansion, for
    /// every word in `words` at every `e in e_lo..e_lo+n_e`. Outputs are
    /// `[n_e * n_words * 4]` u32 (four u32 = eight f16 per (e, word)).
    pub fn launch_expand8_check(
        &self,
        stream: &Stream,
        out_fast: &mut DeviceBuffer<u32>,
        out_ref: &mut DeviceBuffer<u32>,
        words: &DeviceBuffer<u32>,
        e_lo: i32,
        n_e: u32,
    ) -> eyre::Result<()> {
        let n_words = words.len() as u32;
        let n = (n_e as usize) * (n_words as usize) * 4;
        if out_fast.len() < n || out_ref.len() < n {
            return Err(eyre!("e2m1_key_expand8_check: outputs need {n} slots"));
        }
        let function = self.module.get_function("e2m1_key_expand8_check")?;
        let total = n_e * n_words;
        let cfg = LaunchConfig {
            grid: ((total + 255) / 256, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out_fast.raw(), out_ref.raw(), words.raw(), n_words, e_lo, n_e])
    }

    /// Test kernel: the device expand for every `(nibble, e)`,
    /// `e in e_lo..e_lo+n_e`, laid out `[(e - e_lo) * 16 + nibble]`.
    pub fn launch_expand_exhaustive(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u16>,
        e_lo: i32,
        n_e: u32,
    ) -> eyre::Result<()> {
        let n = (n_e as usize) * 16;
        if out.len() < n {
            return Err(eyre!("e2m1_key_expand_exhaustive: output needs {n} slots"));
        }
        let function = self.module.get_function("e2m1_key_expand_exhaustive")?;
        let cfg = LaunchConfig {
            grid: ((n as u32 + 255) / 256, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), e_lo, n_e])
    }
}

// The indexer key rows are N_INDEXER_HEAD_DIM wide.
const _: () = assert!(E2M1_KEY_DIM == crate::config::N_INDEXER_HEAD_DIM as usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_zero_nibble_is_preserved() {
        assert_eq!(expand_half_bits_host(0x8, -3), 0x8000);
        assert_eq!(expand_half_bits_host(0x0, -3), 0x0000);
        // 6 x 2^0 = 6.0 -> f16 0x4600; -1.5 x 2^1 = -3.0 -> 0xC200
        assert_eq!(expand_half_bits_host(0x7, 0), 0x4600);
        assert_eq!(expand_half_bits_host(0xB, 1), 0xC200);
    }
}
