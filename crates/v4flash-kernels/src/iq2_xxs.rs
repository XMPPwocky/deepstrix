//! IQ2_XXS paired matvec: gate[r] and up[r] from two IQ2_XXS weight rows
//! sharing one Q8_K-quantized activation. Mirrors ds4
//! `matvec_iq2_xxs_experts_mid_prequant` inner kernel (ds4.c:3879).

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const IQ2_XXS_PAIR_GFX1201: &[u8] = include_bytes!(env!("KERNEL_IQ2_XXS_PAIR_MATVEC_GFX1201"));
const IQ2_XXS_PAIR_GFX1151: &[u8] = include_bytes!(env!("KERNEL_IQ2_XXS_PAIR_MATVEC_GFX1151"));

pub const BLOCK_IQ2_XXS_BYTES: usize = 66;

pub struct Iq2XxsPairMatvec {
    module: Module,
}

impl Iq2XxsPairMatvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            IQ2_XXS_PAIR_GFX1201
        } else if arch.starts_with("gfx1151") {
            IQ2_XXS_PAIR_GFX1151
        } else {
            return Err(eyre!("unsupported arch for iq2_xxs_pair_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `gate[r] = sum_b dot_iq2xxs(gate_w[r,b], xq[b])` and analogous for `up`,
    /// for `r in 0..n_rows`. Each weight row is `n_blocks` × 66 B blocks; the
    /// activation `xq` is `n_blocks` × 292 B Q8_K blocks. `n_rows` must be a
    /// multiple of 8.
    pub fn launch(
        &self,
        stream: &Stream,
        gate: &mut DeviceBuffer<f32>,
        up: &mut DeviceBuffer<f32>,
        gate_w: &DeviceBuffer<u8>,
        up_w: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("iq2_xxs_pair: n_rows={n_rows} must be multiple of 8"));
        }
        let row_bytes = (n_blocks as usize) * BLOCK_IQ2_XXS_BYTES;
        if gate_w.byte_len() < (n_rows as usize) * row_bytes {
            return Err(eyre!(
                "gate_w bytes: have {}, need {}",
                gate_w.byte_len(),
                (n_rows as usize) * row_bytes
            ));
        }
        if up_w.byte_len() < (n_rows as usize) * row_bytes {
            return Err(eyre!(
                "up_w bytes: have {}, need {}",
                up_w.byte_len(),
                (n_rows as usize) * row_bytes
            ));
        }

        let function = self.module.get_function("iq2_xxs_pair_matvec")?;
        let mut g_ptr = gate.raw();
        let mut u_ptr = up.raw();
        let mut gw_ptr = gate_w.raw();
        let mut uw_ptr = up_w.raw();
        let mut xq_ptr = xq.raw();
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 7] = [
            &mut g_ptr as *mut _ as *mut c_void,
            &mut u_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
