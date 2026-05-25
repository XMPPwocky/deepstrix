//! F16 weight matvec — mirrors ds4's `matvec_any` fallback for F16 weights.
//! Used by V4 Flash for every compressor + indexer projection (all F16
//! in our model). Same launch geometry as `Q8_0Matvec` (8 rows/workgroup,
//! warp-per-row) without the per-block dequant.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const F16_MATVEC_GFX1201: &[u8] = include_bytes!(env!("KERNEL_F16_MATVEC_GFX1201"));
const F16_MATVEC_GFX1151: &[u8] = include_bytes!(env!("KERNEL_F16_MATVEC_GFX1151"));
const F16_MATVEC_NARROW_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_F16_MATVEC_NARROW_GFX1201"));
const F16_MATVEC_NARROW_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_F16_MATVEC_NARROW_GFX1151"));
const F16_MATVEC_PAIR_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_F16_MATVEC_PAIR_GFX1201"));
const F16_MATVEC_PAIR_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_F16_MATVEC_PAIR_GFX1151"));
const F16_MATVEC_PAIR_STATE_WRITE_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_F16_MATVEC_PAIR_STATE_WRITE_GFX1201"));
const F16_MATVEC_PAIR_STATE_WRITE_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_F16_MATVEC_PAIR_STATE_WRITE_GFX1151"));

const GEMV_ROWS_PER_BLOCK: u32 = 8;
const GEMV_WARP_LANES: u32 = 32;
const NARROW_BLOCK_THREADS: u32 = 256;

/// Below this `n_rows` the original 8-rows-per-block kernel under-fills
/// the GPU (e.g. n_rows=24 → 3 blocks) and the per-row latency chain
/// dominates. The narrow variant (1 row per block, 256 threads
/// cooperating) is faster. Calibrated against the mhc_pre_* calls
/// (n_rows=24, k=16384) on gfx1201; threshold conservative enough that
/// the larger compressor matvecs (n_rows≥256) still take the wide path.
const NARROW_ROWS_THRESHOLD: u32 = 64;

pub struct F16Matvec {
    wide: Module,
    narrow: Module,
    pair: Module,
    pair_state_write: Module,
}

impl F16Matvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let (wide_img, narrow_img, pair_img, psw_img): (&[u8], &[u8], &[u8], &[u8]) =
            if arch.starts_with("gfx1201") {
                (
                    F16_MATVEC_GFX1201,
                    F16_MATVEC_NARROW_GFX1201,
                    F16_MATVEC_PAIR_GFX1201,
                    F16_MATVEC_PAIR_STATE_WRITE_GFX1201,
                )
            } else if arch.starts_with("gfx1151") {
                (
                    F16_MATVEC_GFX1151,
                    F16_MATVEC_NARROW_GFX1151,
                    F16_MATVEC_PAIR_GFX1151,
                    F16_MATVEC_PAIR_STATE_WRITE_GFX1151,
                )
            } else {
                return Err(eyre!("unsupported arch for f16_matvec: {arch}"));
            };
        let wide = Module::load_data(wide_img)?;
        let narrow = Module::load_data(narrow_img)?;
        let pair = Module::load_data(pair_img)?;
        let pair_state_write = Module::load_data(psw_img)?;
        Ok(Self {
            wide,
            narrow,
            pair,
            pair_state_write,
        })
    }

    /// Paired matvec: `kv[r] = W_kv[r] · x`, `gate[r] = W_gate[r] · x` for
    /// r in 0..n_rows. Single launch; activation reads shared in cache;
    /// half2/float2-vectorized loads with two independent accumulators
    /// per output (M14h).
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_pair(
        &self,
        stream: &Stream,
        kv: &mut DeviceBuffer<f32>,
        gate: &mut DeviceBuffer<f32>,
        kv_w: &DeviceBuffer<u8>,
        gate_w: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        let expected = (n_rows as usize) * (k as usize) * 2;
        if kv_w.byte_len() != expected || gate_w.byte_len() != expected {
            return Err(eyre!(
                "f16 matvec_pair: weight bytes mismatch (kv={}, gate={}, expected={})",
                kv_w.byte_len(),
                gate_w.byte_len(),
                expected
            ));
        }
        if kv.len() < n_rows as usize || gate.len() < n_rows as usize {
            return Err(eyre!(
                "f16 matvec_pair: out len short (kv={}, gate={}, n_rows={n_rows})",
                kv.len(),
                gate.len()
            ));
        }
        if x.len() < k as usize {
            return Err(eyre!("f16 matvec_pair: x len {} < k {k}", x.len()));
        }
        if k % 2 != 0 {
            return Err(eyre!("f16 matvec_pair: k={k} must be even for half2 loads"));
        }

        let function = self.pair.get_function("f16_matvec_pair")?;
        let mut kv_ptr = kv.raw();
        let mut g_ptr = gate.raw();
        let mut kvw_ptr = kv_w.raw();
        let mut gw_ptr = gate_w.raw();
        let mut x_ptr = x.raw();
        let mut k_v = k;
        let mut nr_v = n_rows;
        let mut args: [*mut c_void; 7] = [
            &mut kv_ptr as *mut _ as *mut c_void,
            &mut g_ptr as *mut _ as *mut c_void,
            &mut kvw_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut nr_v as *mut _ as *mut c_void,
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

    /// Fused paired matvec + compressor state write. Equivalent to
    /// `matvec_pair(kv_cur, sc_cur, ...)` followed by
    /// `compressor_state_write(state_kv, state_score, kv_cur, sc_cur,
    /// ape, width, row, pos_mod)`, but as one kernel launch. No
    /// intermediate kv_cur / sc_cur scratch buffers needed.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_pair_state_write(
        &self,
        stream: &Stream,
        state_kv: &mut DeviceBuffer<f32>,
        state_score: &mut DeviceBuffer<f32>,
        kv_w: &DeviceBuffer<u8>,
        gate_w: &DeviceBuffer<u8>,
        ape: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        k: u32,
        width: u32,
        row: u32,
        pos_mod: u32,
    ) -> eyre::Result<()> {
        let expected_w = (width as usize) * (k as usize) * 2;
        if kv_w.byte_len() != expected_w || gate_w.byte_len() != expected_w {
            return Err(eyre!(
                "f16 matvec_pair_state_write: weight bytes mismatch (kv={}, gate={}, expected={})",
                kv_w.byte_len(),
                gate_w.byte_len(),
                expected_w
            ));
        }
        if x.len() < k as usize {
            return Err(eyre!(
                "f16 matvec_pair_state_write: x len {} < k {k}",
                x.len()
            ));
        }
        if k % 4 != 0 {
            return Err(eyre!(
                "f16 matvec_pair_state_write: k={k} must be divisible by 4 for float4 loads"
            ));
        }

        let function = self
            .pair_state_write
            .get_function("f16_matvec_pair_state_write")?;
        let mut sk_ptr = state_kv.raw();
        let mut ss_ptr = state_score.raw();
        let mut kvw_ptr = kv_w.raw();
        let mut gw_ptr = gate_w.raw();
        let mut ape_ptr = ape.raw();
        let mut x_ptr = x.raw();
        let mut k_v = k;
        let mut w_v = width;
        let mut row_v = row;
        let mut pm_v = pos_mod;
        let mut args: [*mut c_void; 10] = [
            &mut sk_ptr as *mut _ as *mut c_void,
            &mut ss_ptr as *mut _ as *mut c_void,
            &mut kvw_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut ape_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut w_v as *mut _ as *mut c_void,
            &mut row_v as *mut _ as *mut c_void,
            &mut pm_v as *mut _ as *mut c_void,
        ];
        let grid_x = width.div_ceil(GEMV_ROWS_PER_BLOCK);
        let block_x = GEMV_ROWS_PER_BLOCK * GEMV_WARP_LANES;
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// `out[r] = sum_i f32(weight[r, i]) * x[i]` for `r in 0..n_rows`.
    /// Weight is F16 row-major `[n_rows, k]`, passed as a `DeviceBuffer<u8>`
    /// holding raw F16 bytes (mirrors how Q8_0 weights are passed).
    pub fn matvec(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        k: u32,
    ) -> eyre::Result<()> {
        let expected_weight_bytes = (n_rows as usize) * (k as usize) * 2;
        if weight.byte_len() != expected_weight_bytes {
            return Err(eyre!(
                "f16 matvec weight bytes: have {}, expected {} (n_rows={n_rows}, k={k})",
                weight.byte_len(),
                expected_weight_bytes
            ));
        }
        if out.len() < n_rows as usize {
            return Err(eyre!(
                "f16 matvec out len: have {}, expected n_rows={n_rows}",
                out.len()
            ));
        }
        if x.len() < k as usize {
            return Err(eyre!(
                "f16 matvec x len: have {}, expected k={k}",
                x.len()
            ));
        }

        let mut out_ptr = out.raw();
        let mut w_ptr = weight.raw();
        let mut x_ptr = x.raw();
        let mut k_v = k;
        let mut n_rows_v = n_rows;
        let mut args: [*mut c_void; 5] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut n_rows_v as *mut _ as *mut c_void,
        ];

        if n_rows < NARROW_ROWS_THRESHOLD {
            let function = self.narrow.get_function("f16_matvec_narrow")?;
            let cfg = LaunchConfig {
                grid: (n_rows, 1, 1),
                block: (NARROW_BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { function.launch_raw(cfg, stream, &mut args) }
        } else {
            let function = self.wide.get_function("f16_matvec")?;
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
}
