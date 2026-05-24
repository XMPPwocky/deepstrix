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
        self.launch_with_offsets(stream, gate, up, gate_w, 0, up_w, 0, xq, n_rows, n_blocks)
    }

    /// Same as [`launch`] but reads the gate/up weights starting at a
    /// per-buffer byte offset. Used by the forward orchestrator to point
    /// into a single resident routed-expert tensor (256 experts of
    /// `n_rows * n_blocks * BLOCK_IQ2_XXS_BYTES` bytes each) without
    /// allocating per-slot scratch buffers.
    pub fn launch_with_offsets(
        &self,
        stream: &Stream,
        gate: &mut DeviceBuffer<f32>,
        up: &mut DeviceBuffer<f32>,
        gate_w: &DeviceBuffer<u8>,
        gate_w_offset: usize,
        up_w: &DeviceBuffer<u8>,
        up_w_offset: usize,
        xq: &DeviceBuffer<u8>,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("iq2_xxs_pair: n_rows={n_rows} must be multiple of 8"));
        }
        let row_bytes = (n_blocks as usize) * BLOCK_IQ2_XXS_BYTES;
        let need = (n_rows as usize) * row_bytes;
        if gate_w.byte_len() < gate_w_offset + need {
            return Err(eyre!(
                "gate_w bytes: have {}, need offset {} + {} = {}",
                gate_w.byte_len(),
                gate_w_offset,
                need,
                gate_w_offset + need
            ));
        }
        if up_w.byte_len() < up_w_offset + need {
            return Err(eyre!(
                "up_w bytes: have {}, need offset {} + {} = {}",
                up_w.byte_len(),
                up_w_offset,
                need,
                up_w_offset + need
            ));
        }

        let function = self.module.get_function("iq2_xxs_pair_matvec")?;
        let mut g_ptr = gate.raw();
        let mut u_ptr = up.raw();
        // SAFETY: bounds-checked above; pointer math within the allocation.
        let mut gw_ptr = unsafe { (gate_w.raw() as *mut u8).add(gate_w_offset) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut uw_ptr = unsafe { (up_w.raw() as *mut u8).add(up_w_offset) }
            as v4flash_hip::sys::hipDeviceptr_t;
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
