//! F16 weight matvec — mirrors ds4's `matvec_any` fallback for F16 weights.
//! Used by V4 Flash for every compressor + indexer projection (all F16
//! in our model). Same launch geometry as `Q8_0Matvec` (8 rows/workgroup,
//! warp-per-row) without the per-block dequant.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const F16_MATVEC_GFX1201: &[u8] = include_bytes!(env!("KERNEL_F16_MATVEC_GFX1201"));
const F16_MATVEC_GFX1151: &[u8] = include_bytes!(env!("KERNEL_F16_MATVEC_GFX1151"));

const GEMV_ROWS_PER_BLOCK: u32 = 8;
const GEMV_WARP_LANES: u32 = 32;

pub struct F16Matvec {
    module: Module,
}

impl F16Matvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            F16_MATVEC_GFX1201
        } else if arch.starts_with("gfx1151") {
            F16_MATVEC_GFX1151
        } else {
            return Err(eyre!("unsupported arch for f16_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
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

        let function = self.module.get_function("f16_matvec")?;

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
