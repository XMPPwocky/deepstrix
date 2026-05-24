//! Q8_K activation quantization — packs an F32 vector into 256-element
//! Q8_K blocks (292 bytes each) consumed by routed-expert IQ2_XXS/Q2_K
//! matvecs. Mirrors `ds4_quantize_row_q8_K` (ds4.c:1655).

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const Q8_K_QUANTIZE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_Q8_K_QUANTIZE_GFX1201"));
const Q8_K_QUANTIZE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_Q8_K_QUANTIZE_GFX1151"));

pub const QK_K: u32 = 256;
pub const BLOCK_Q8_K_BYTES: usize = 292;

pub struct Q8KQuantize {
    module: Module,
}

impl Q8KQuantize {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q8_K_QUANTIZE_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q8_K_QUANTIZE_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q8_k_quantize: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        let needed_x = (n_blocks as usize) * (QK_K as usize);
        let needed_out = (n_blocks as usize) * BLOCK_Q8_K_BYTES;
        if x.len() < needed_x {
            return Err(eyre!(
                "q8_k_quantize x len: have {}, need {}",
                x.len(),
                needed_x
            ));
        }
        if out.byte_len() < needed_out {
            return Err(eyre!(
                "q8_k_quantize out bytes: have {}, need {}",
                out.byte_len(),
                needed_out
            ));
        }

        let function = self.module.get_function("q8_k_quantize")?;
        let mut out_ptr = out.raw();
        let mut x_ptr = x.raw();
        let mut n = n_blocks;
        let mut args: [*mut c_void; 3] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
